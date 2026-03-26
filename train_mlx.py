"""
Autoresearch pretraining script — Apple Silicon MLX backend.
Single-GPU, single-file (plus backends/ for optimizer and hardware detection).
Usage: AUTORESEARCH_BACKEND=mlx uv run train.py
   or: uv run train_mlx.py
"""

import os
import gc
import math
import time
from dataclasses import dataclass, asdict

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_map

import numpy as np

from backends import get_hardware_info, suggest_hyperparameters, get_peak_flops, get_peak_memory_mb
from backends.muon_mlx import MuonAdamWMLX, build_param_groups
from prepare import MAX_SEQ_LEN, TIME_BUDGET, Tokenizer, make_dataloader, evaluate_bpb

# ---------------------------------------------------------------------------
# GPT Model (MLX)
# ---------------------------------------------------------------------------

@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6
    n_kv_head: int = 6
    n_embd: int = 768
    window_pattern: str = "SSSL"


def norm(x):
    """RMS norm."""
    return x * mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + 1e-5)


def has_ve(layer_idx, n_layer):
    """Returns True if layer should have Value Embedding."""
    return layer_idx % 2 == (n_layer - 1) % 2


def create_causal_mask(seq_len, dtype=mx.float32):
    """Create standard causal attention mask."""
    indices = mx.arange(seq_len)
    blocked = indices[None, :] > indices[:, None]
    return mx.where(blocked, mx.array(float("-inf"), dtype=dtype), mx.array(0.0, dtype=dtype))


def create_sliding_window_mask(seq_len, window_size, dtype=mx.float32):
    """Create causal attention mask with sliding window."""
    indices = mx.arange(seq_len)
    causal = indices[None, :] > indices[:, None]
    too_far = (indices[:, None] - indices[None, :]) >= window_size
    blocked = causal | too_far
    return mx.where(blocked, mx.array(float("-inf"), dtype=dtype), mx.array(0.0, dtype=dtype))


class RotaryEmbedding:
    """Precomputed rotary embeddings."""
    def __init__(self, head_dim, max_seq_len, base=10000):
        channel_range = mx.arange(0, head_dim, 2, dtype=mx.float32)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        t = mx.arange(max_seq_len, dtype=mx.float32)
        freqs = mx.outer(t, inv_freq)
        self.cos = freqs.cos().astype(mx.bfloat16)
        self.sin = freqs.sin().astype(mx.bfloat16)

    def apply(self, x, offset=0):
        """Apply rotary embeddings. x: (B, T, n_heads, head_dim)"""
        T = x.shape[1]
        cos = self.cos[offset:offset + T]  # (T, head_dim//2)
        sin = self.sin[offset:offset + T]
        # Reshape for broadcasting: (1, T, 1, head_dim//2)
        cos = cos[None, :, None, :]
        sin = sin[None, :, None, :]
        d = x.shape[-1] // 2
        x1, x2 = x[..., :d], x[..., d:]
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return mx.concatenate([y1, y2], axis=-1)


class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.ve_gate_channels = 32
        self.has_ve = has_ve(layer_idx, config.n_layer)
        if self.has_ve:
            self.ve_gate = nn.Linear(self.ve_gate_channels, self.n_kv_head, bias=False)

    def __call__(self, x, ve, rotary, mask):
        B, T, C = x.shape
        q = self.c_q(x).reshape(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).reshape(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).reshape(B, T, self.n_kv_head, self.head_dim)

        # Value residual
        if ve is not None and self.has_ve:
            ve = ve.reshape(B, T, self.n_kv_head, self.head_dim)
            gate = 2 * mx.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
            v = v + mx.expand_dims(gate, axis=-1) * ve

        # Apply rotary embeddings
        q = rotary.apply(q)
        k = rotary.apply(k)
        q, k = norm(q), norm(k)

        # Expand KV heads for GQA
        if self.n_kv_head < self.n_head:
            rep = self.n_head // self.n_kv_head
            k = mx.repeat(k, rep, axis=2)
            v = mx.repeat(v, rep, axis=2)

        # Transpose to (B, n_head, T, head_dim)
        q = mx.swapaxes(q, 1, 2)
        k = mx.swapaxes(k, 1, 2)
        v = mx.swapaxes(v, 1, 2)

        # Scaled dot-product attention
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = (q @ mx.swapaxes(k, -2, -1)) * scale
        if mask is not None:
            scores = scores + mask
        weights = mx.softmax(scores, axis=-1)
        y = weights @ v

        # Back to (B, T, n_embd)
        y = mx.swapaxes(y, 1, 2).reshape(B, T, -1)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def __call__(self, x):
        x = self.c_fc(x)
        x = mx.maximum(x, 0) ** 2  # ReluSquared
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def __call__(self, x, ve, rotary, mask):
        x = x + self.attn(norm(x), ve, rotary, mask)
        x = x + self.mlp(norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.window_sizes = self._compute_window_sizes(config)
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.blocks = [Block(config, i) for i in range(config.n_layer)]
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.resid_lambdas = mx.ones(config.n_layer)
        self.x0_lambdas = mx.zeros(config.n_layer)

        # Value embeddings
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.value_embeds = {}
        for i in range(config.n_layer):
            if has_ve(i, config.n_layer):
                self.value_embeds[str(i)] = nn.Embedding(config.vocab_size, kv_dim)

        # Rotary embeddings (not a parameter)
        self.rotary = RotaryEmbedding(head_dim, config.sequence_len * 10)

        # Precompute attention masks
        self._masks = {}
        for ws in set(tuple(w) if isinstance(w, (list, tuple)) else (w,) for w in self.window_sizes):
            window = ws[0] if isinstance(ws, tuple) else ws
            T = config.sequence_len
            if window > 0 and window < T:
                self._masks[window] = create_sliding_window_mask(T, window, mx.float32)
            else:
                self._masks[window] = create_causal_mask(T, mx.float32)

    def init_weights(self):
        """Initialize weights matching the original."""
        n_embd = self.config.n_embd
        s = 3**0.5 * n_embd**-0.5

        # Embeddings
        self.wte.weight = mx.random.normal(self.wte.weight.shape).astype(mx.bfloat16)
        self.lm_head.weight = mx.random.normal(self.lm_head.weight.shape) * 0.001

        # Transformer blocks
        for block in self.blocks:
            block.attn.c_q.weight = mx.random.uniform(-s, s, block.attn.c_q.weight.shape)
            block.attn.c_k.weight = mx.random.uniform(-s, s, block.attn.c_k.weight.shape)
            block.attn.c_v.weight = mx.random.uniform(-s, s, block.attn.c_v.weight.shape)
            block.attn.c_proj.weight = mx.zeros_like(block.attn.c_proj.weight)
            block.mlp.c_fc.weight = mx.random.uniform(-s, s, block.mlp.c_fc.weight.shape)
            block.mlp.c_proj.weight = mx.zeros_like(block.mlp.c_proj.weight)
            if block.attn.has_ve:
                block.attn.ve_gate.weight = mx.zeros_like(block.attn.ve_gate.weight)

        # Per-layer scalars
        self.resid_lambdas = mx.ones(self.config.n_layer)
        self.x0_lambdas = mx.full(self.config.n_layer, 0.1)

        # Value embeddings
        for ve in self.value_embeds.values():
            ve.weight = mx.random.uniform(-s, s, ve.weight.shape).astype(mx.bfloat16)

    def _compute_window_sizes(self, config):
        pattern = config.window_pattern.upper()
        long_window = config.sequence_len
        short_window = long_window // 2
        char_to_window = {"L": (long_window, 0), "S": (short_window, 0)}
        window_sizes = []
        for layer_idx in range(config.n_layer):
            char = pattern[layer_idx % len(pattern)]
            window_sizes.append(char_to_window[char])
        window_sizes[-1] = (long_window, 0)
        return window_sizes

    def estimate_flops(self):
        """Estimated FLOPs per token (forward + backward)."""
        flat = dict(tree_flatten(self.parameters()))
        nparams = sum(p.size for p in flat.values())
        value_embeds_numel = sum(self.value_embeds[k].weight.size for k in self.value_embeds)
        nparams_exclude = (self.wte.weight.size + value_embeds_numel +
                          self.resid_lambdas.size + self.x0_lambdas.size)
        h = self.config.n_head
        q = self.config.n_embd // self.config.n_head
        t = self.config.sequence_len
        attn_flops = 0
        for window_size in self.window_sizes:
            window = window_size[0]
            effective_seq = t if window < 0 else min(window, t)
            attn_flops += 12 * h * q * effective_seq
        return 6 * (nparams - nparams_exclude) + attn_flops

    def num_scaling_params(self):
        flat = dict(tree_flatten(self.parameters()))
        wte = self.wte.weight.size
        value_embeds = sum(self.value_embeds[k].weight.size for k in self.value_embeds)
        lm_head = self.lm_head.weight.size
        block_params = sum(p.size for name, p in flat.items() if 'blocks' in name)
        scalars = self.resid_lambdas.size + self.x0_lambdas.size
        total = wte + value_embeds + lm_head + block_params + scalars
        return {
            'wte': wte, 'value_embeds': value_embeds, 'lm_head': lm_head,
            'transformer_matrices': block_params, 'scalars': scalars, 'total': total,
        }

    def __call__(self, idx, targets=None, reduction='mean'):
        B, T = idx.shape

        x = self.wte(idx)
        x = norm(x)
        x0 = x

        for i, block in enumerate(self.blocks):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve_key = str(i)
            ve = self.value_embeds[ve_key](idx) if ve_key in self.value_embeds else None
            window = self.window_sizes[i][0]
            mask = self._masks.get(window)
            if mask is not None and mask.shape[0] > T:
                mask = mask[:T, :T]
            x = block(x, ve, self.rotary, mask)

        x = norm(x)

        softcap = 15
        logits = self.lm_head(x)
        logits = logits.astype(mx.float32)
        logits = softcap * mx.tanh(logits / softcap)

        if targets is not None:
            # Cross-entropy loss
            logits_flat = logits.reshape(-1, logits.shape[-1])
            targets_flat = targets.reshape(-1)
            if reduction == 'none':
                loss = nn.losses.cross_entropy(logits_flat, targets_flat, reduction='none')
                return loss.reshape(B, T)
            else:
                loss = nn.losses.cross_entropy(logits_flat, targets_flat, reduction='mean')
                return loss
        return logits


# ---------------------------------------------------------------------------
# Hyperparameters (edit these directly, no CLI flags needed)
# ---------------------------------------------------------------------------

_hw_info = get_hardware_info()
_hp_defaults = suggest_hyperparameters(_hw_info)

# Model architecture
ASPECT_RATIO = 64
HEAD_DIM = 128
WINDOW_PATTERN = "SSSL"

# Optimization
TOTAL_BATCH_SIZE = _hp_defaults['total_batch_size']
EMBEDDING_LR = 0.6
UNEMBEDDING_LR = 0.004
MATRIX_LR = 0.04
SCALAR_LR = 0.5
WEIGHT_DECAY = 0.2
ADAM_BETAS = (0.8, 0.95)
WARMUP_RATIO = 0.0
WARMDOWN_RATIO = 0.5
FINAL_LR_FRAC = 0.0

# Model size
DEPTH = _hp_defaults['depth']
DEVICE_BATCH_SIZE = _hp_defaults['device_batch_size']

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

t_start = time.time()
mx.random.seed(42)

PEAK_FLOPS = get_peak_flops(_hw_info)
print(f"Backend: MLX ({_hw_info['chip_name']})")
print(f"Peak bf16 FLOPS: {PEAK_FLOPS:.2e}")

tokenizer = Tokenizer.from_directory()
vocab_size = tokenizer.get_vocab_size()
print(f"Vocab size: {vocab_size:,}")

def build_model_config(depth):
    base_dim = depth * ASPECT_RATIO
    model_dim = ((base_dim + HEAD_DIM - 1) // HEAD_DIM) * HEAD_DIM
    num_heads = model_dim // HEAD_DIM
    return GPTConfig(
        sequence_len=MAX_SEQ_LEN, vocab_size=vocab_size,
        n_layer=depth, n_head=num_heads, n_kv_head=num_heads, n_embd=model_dim,
        window_pattern=WINDOW_PATTERN,
    )

config = build_model_config(DEPTH)
print(f"Model config: {asdict(config)}")

model = GPT(config)
model.init_weights()
mx.eval(model.parameters())

param_counts = model.num_scaling_params()
print("Parameter counts:")
for key, value in param_counts.items():
    print(f"  {key:24s}: {value:,}")
num_params = param_counts['total']
num_flops_per_token = model.estimate_flops()
print(f"Estimated FLOPs per token: {num_flops_per_token:e}")

tokens_per_fwdbwd = DEVICE_BATCH_SIZE * MAX_SEQ_LEN
assert TOTAL_BATCH_SIZE % tokens_per_fwdbwd == 0
grad_accum_steps = TOTAL_BATCH_SIZE // tokens_per_fwdbwd

# Build optimizer with Muon
model_dim = config.n_embd
param_groups = build_param_groups(model, {
    'model_dim': model_dim,
    'matrix_lr': MATRIX_LR,
    'embedding_lr': EMBEDDING_LR,
    'unembedding_lr': UNEMBEDDING_LR,
    'scalar_lr': SCALAR_LR,
    'adam_betas': ADAM_BETAS,
    'weight_decay': WEIGHT_DECAY,
})
optimizer = MuonAdamWMLX(param_groups)

# Loss + gradient function
def loss_fn(model, x, y):
    loss = model(x, y, reduction='mean')
    return loss

loss_grad_fn = nn.value_and_grad(model, loss_fn)

train_loader = make_dataloader(tokenizer, DEVICE_BATCH_SIZE, MAX_SEQ_LEN, "train", backend="mlx")
x, y, epoch = next(train_loader)

print(f"Time budget: {TIME_BUDGET}s")
print(f"Gradient accumulation steps: {grad_accum_steps}")

# Schedules
def get_lr_multiplier(progress):
    if progress < WARMUP_RATIO:
        return progress / WARMUP_RATIO if WARMUP_RATIO > 0 else 1.0
    elif progress < 1.0 - WARMDOWN_RATIO:
        return 1.0
    else:
        cooldown = (1.0 - progress) / WARMDOWN_RATIO
        return cooldown * 1.0 + (1 - cooldown) * FINAL_LR_FRAC

def get_muon_momentum(step):
    frac = min(step / 300, 1)
    return (1 - frac) * 0.85 + frac * 0.95

def get_weight_decay(progress):
    return WEIGHT_DECAY * (1 - progress)

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

t_start_training = time.time()
smooth_train_loss = 0
total_training_time = 0
step = 0

while True:
    t0 = time.time()

    # Gradient accumulation
    accum_loss = mx.array(0.0)
    accum_grads = None

    for micro_step in range(grad_accum_steps):
        loss, grads = loss_grad_fn(model, x, y)
        accum_loss = accum_loss + loss

        if accum_grads is None:
            accum_grads = grads
        else:
            accum_grads = tree_map(lambda a, b: a + b, accum_grads, grads)

        x, y, epoch = next(train_loader)

    # Average gradients
    if grad_accum_steps > 1:
        accum_grads = tree_map(lambda g: g / grad_accum_steps, accum_grads)

    train_loss_val = (accum_loss / grad_accum_steps)

    # Progress and schedules
    progress = min(total_training_time / TIME_BUDGET, 1.0)
    lrm = get_lr_multiplier(progress)
    muon_momentum = get_muon_momentum(step)
    muon_weight_decay = get_weight_decay(progress)

    # Update optimizer schedules
    for group in optimizer.param_groups:
        group['lr'] = group['initial_lr'] * lrm
        if group['kind'] == 'muon':
            group['momentum'] = muon_momentum
            group['weight_decay'] = muon_weight_decay

    optimizer.update(model, accum_grads)
    mx.eval(model.parameters(), train_loss_val)

    train_loss_f = train_loss_val.item()

    # Fast fail
    if math.isnan(train_loss_f) or train_loss_f > 100:
        print("FAIL")
        exit(1)

    t1 = time.time()
    dt = t1 - t0

    if step > 10:
        total_training_time += dt

    # Logging
    ema_beta = 0.9
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta**(step + 1))
    pct_done = 100 * progress
    tok_per_sec = int(TOTAL_BATCH_SIZE / dt)
    mfu = 100 * num_flops_per_token * TOTAL_BATCH_SIZE / dt / PEAK_FLOPS
    remaining = max(0, TIME_BUDGET - total_training_time)

    print(f"\rstep {step:05d} ({pct_done:.1f}%) | loss: {debiased_smooth_loss:.6f} | lrm: {lrm:.2f} | dt: {dt*1000:.0f}ms | tok/sec: {tok_per_sec:,} | mfu: {mfu:.1f}% | epoch: {epoch} | remaining: {remaining:.0f}s    ", end="", flush=True)

    # GC management
    if step == 0:
        gc.collect()
        gc.freeze()
        gc.disable()
    elif (step + 1) % 5000 == 0:
        gc.collect()

    step += 1

    if step > 10 and total_training_time >= TIME_BUDGET:
        break

print()

total_tokens = step * TOTAL_BATCH_SIZE

# Final eval
val_bpb = evaluate_bpb(model, tokenizer, DEVICE_BATCH_SIZE, backend="mlx")

# Final summary
t_end = time.time()
steady_state_mfu = 100 * num_flops_per_token * TOTAL_BATCH_SIZE * (step - 10) / total_training_time / PEAK_FLOPS if total_training_time > 0 else 0
peak_vram_mb = get_peak_memory_mb("mlx")

print("---")
print(f"val_bpb:          {val_bpb:.6f}")
print(f"training_seconds: {total_training_time:.1f}")
print(f"total_seconds:    {t_end - t_start:.1f}")
print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
print(f"mfu_percent:      {steady_state_mfu:.2f}")
print(f"total_tokens_M:   {total_tokens / 1e6:.1f}")
print(f"num_steps:        {step}")
print(f"num_params_M:     {num_params / 1e6:.1f}")
print(f"depth:            {DEPTH}")
print(f"backend:          mlx")
print(f"chip:             {_hw_info['chip_name']}")
