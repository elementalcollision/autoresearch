"""
MuonAdamW optimizer ported to MLX for Apple Silicon.
This is a novel port — no public MLX Muon implementation exists.

Implements the full Muon optimizer:
- Newton-Schulz (Polar Express) orthogonalization
- Nesterov momentum
- NorMuon variance reduction
- Cautious weight decay
- Combined with AdamW for non-matrix parameters
"""

import math
import mlx.core as mx
from mlx.utils import tree_flatten

# ---------------------------------------------------------------------------
# Polar Express coefficients (Newton-Schulz orthogonalization)
# ---------------------------------------------------------------------------

POLAR_EXPRESS_COEFFS = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]


def newton_schulz_orthogonalize(X, ns_steps=5):
    """
    Polar express: approximate the orthogonal polar factor of X
    using Newton-Schulz iterations with precomputed optimal coefficients.

    X: shape (..., M, N) in bfloat16
    Returns: orthogonalized X of same shape
    """
    X = X.astype(mx.bfloat16)
    # Normalize
    # Compute Frobenius norm over last two dims
    norms = mx.sqrt((X * X).sum(axis=(-2, -1), keepdims=True))
    X = X / (norms * 1.02 + 1e-6)

    M, N = X.shape[-2], X.shape[-1]
    if M > N:
        # Tall matrices: use X^T X (smaller N×N matrix)
        for a, b, c in POLAR_EXPRESS_COEFFS[:ns_steps]:
            Xt = mx.swapaxes(X, -2, -1)
            A = Xt @ X
            B = b * A + c * (A @ A)
            X = a * X + X @ B
    else:
        # Wide/square matrices: use X X^T (smaller M×M matrix)
        for a, b, c in POLAR_EXPRESS_COEFFS[:ns_steps]:
            Xt = mx.swapaxes(X, -2, -1)
            A = X @ Xt
            B = b * A + c * (A @ A)
            X = a * X + B @ X

    return X


# ---------------------------------------------------------------------------
# Helper to set a nested parameter by path
# ---------------------------------------------------------------------------

def _set_param_by_path(model, path, value):
    """Set a parameter in an MLX model by its dot-separated path."""
    parts = path.split(".")
    obj = model
    for part in parts[:-1]:
        if part.isdigit():
            obj = obj[int(part)]
        else:
            obj = getattr(obj, part)
    last = parts[-1]
    if last.isdigit():
        obj[int(last)] = value
    else:
        setattr(obj, last, value)


def _get_param_by_path(model, path):
    """Get a parameter from an MLX model by its dot-separated path."""
    parts = path.split(".")
    obj = model
    for part in parts:
        if part.isdigit():
            obj = obj[int(part)]
        else:
            obj = getattr(obj, part)
    return obj


# ---------------------------------------------------------------------------
# MuonAdamW for MLX
# ---------------------------------------------------------------------------

class MuonAdamWMLX:
    """
    Combined optimizer: Muon for 2D matrix params, AdamW for others.
    Full MLX port of the original PyTorch MuonAdamW.

    param_groups: list of dicts, each with:
        kind: 'adamw' or 'muon'
        paths: list of parameter paths (dot-separated)
        lr: learning rate
        For adamw: betas, eps, weight_decay
        For muon: momentum, ns_steps, beta2, weight_decay
    """

    def __init__(self, param_groups):
        self.param_groups = param_groups
        self.state = {}
        self._step_count = 0

        # Store initial LRs for schedule support
        for group in self.param_groups:
            group["initial_lr"] = group["lr"]

    def _step_adamw(self, path, grad, param, group):
        """Standard AdamW update for a single parameter."""
        grad_f32 = grad.astype(mx.float32)
        param_f32 = param.astype(mx.float32)
        lr = group['lr']
        beta1, beta2 = group['betas']
        eps = group['eps']
        wd = group['weight_decay']

        state = self.state.setdefault(path, {
            'm': mx.zeros_like(grad_f32),
            'v': mx.zeros_like(grad_f32),
            't': 0,
        })
        state['t'] += 1
        t = state['t']

        state['m'] = beta1 * state['m'] + (1 - beta1) * grad_f32
        state['v'] = beta2 * state['v'] + (1 - beta2) * (grad_f32 * grad_f32)

        bias1 = 1 - beta1 ** t
        bias2 = 1 - beta2 ** t
        denom = mx.sqrt(state['v'] / bias2) + eps
        step_size = lr / bias1

        param_f32 = param_f32 * (1 - lr * wd)
        param_f32 = param_f32 - step_size * (state['m'] / denom)
        return param_f32.astype(param.dtype)

    def _step_muon(self, stacked_grads, stacked_params, group):
        """
        Muon update for a group of same-shape 2D parameters (stacked).
        Implements: Nesterov momentum → Newton-Schulz → NorMuon → cautious weight decay.
        """
        momentum_val = group['momentum']
        lr = group['lr']
        wd = group['weight_decay']
        beta2 = group.get('beta2', 0.95)
        ns_steps = group.get('ns_steps', 5)
        group_id = group['_group_id']

        shape = stacked_grads.shape
        M, N = shape[-2], shape[-1]
        red_dim = -1 if M >= N else -2

        state = self.state.setdefault(group_id, {})

        # Initialize momentum buffers
        if 'momentum_buffer' not in state:
            state['momentum_buffer'] = mx.zeros_like(stacked_grads)
        if 'second_momentum_buffer' not in state:
            if M >= N:
                smb_shape = list(shape[:-1]) + [1]
            else:
                smb_shape = list(shape[:-2]) + [1, N]
            state['second_momentum_buffer'] = mx.zeros(smb_shape, dtype=mx.float32)

        # Nesterov momentum
        momentum = mx.array(momentum_val, dtype=stacked_grads.dtype)
        state['momentum_buffer'] = momentum * state['momentum_buffer'] + (1 - momentum) * stacked_grads
        g = (1 - momentum) * stacked_grads + momentum * state['momentum_buffer']

        # Polar express orthogonalization
        g = newton_schulz_orthogonalize(g, ns_steps)

        # NorMuon variance reduction — all in float32 to avoid bf16 overflow
        g_f32 = g.astype(mx.float32)
        beta2_f32 = mx.array(beta2, dtype=mx.float32)
        v_mean = (g_f32 ** 2).mean(axis=red_dim, keepdims=True)
        red_dim_size = g.shape[red_dim]
        v_norm_sq = v_mean.sum(axis=(-2, -1), keepdims=True) * red_dim_size
        v_norm = mx.sqrt(v_norm_sq)

        smb_f32 = state['second_momentum_buffer']  # already float32
        smb_f32 = smb_f32 + (1 - beta2_f32) * (v_mean - smb_f32)
        state['second_momentum_buffer'] = smb_f32

        step_size = mx.rsqrt(mx.maximum(smb_f32, mx.array(1e-10)))
        scaled_sq_sum = (v_mean * red_dim_size) * (step_size ** 2)
        v_norm_new = mx.sqrt(scaled_sq_sum.sum(axis=(-2, -1), keepdims=True))
        final_scale = step_size * (v_norm / mx.maximum(v_norm_new, mx.array(1e-10)))
        g = (g_f32 * final_scale).astype(g.dtype)

        # Cautious weight decay + parameter update
        lr_val = mx.array(lr, dtype=g.dtype)
        wd_val = mx.array(wd, dtype=g.dtype)
        mask = (g * stacked_params) >= 0
        updated = stacked_params - lr_val * g - lr_val * wd_val * stacked_params * mask

        return updated

    def update(self, model, grads):
        """Apply one optimization step. grads is the gradient tree from nn.value_and_grad."""
        self._step_count += 1
        flat_grads = dict(tree_flatten(grads))
        flat_params = dict(tree_flatten(model.parameters()))

        # AdamW groups
        for group in self.param_groups:
            if group['kind'] != 'adamw':
                continue
            for path in group['paths']:
                if path not in flat_grads:
                    continue
                new_param = self._step_adamw(
                    path, flat_grads[path], flat_params[path], group)
                _set_param_by_path(model, path, new_param)

        # Muon groups (stacked by shape for batched operations)
        for group in self.param_groups:
            if group['kind'] != 'muon':
                continue
            paths = group['paths']
            if not paths:
                continue

            # Stack parameters and gradients for batched Muon step
            grads_list = [flat_grads[p] for p in paths if p in flat_grads]
            params_list = [flat_params[p] for p in paths if p in flat_grads]
            if not grads_list:
                continue

            stacked_g = mx.stack(grads_list)
            stacked_p = mx.stack(params_list)

            # Scale LR by aspect ratio (same as original)
            shape = params_list[0].shape
            group['lr'] = group['initial_lr'] * max(1.0, shape[-2] / shape[-1]) ** 0.5

            updated = self._step_muon(stacked_g, stacked_p, group)

            # Unstack and write back
            active_paths = [p for p in paths if p in flat_grads]
            for i, path in enumerate(active_paths):
                _set_param_by_path(model, path, updated[i])

    def set_lr_multiplier(self, multiplier):
        """Scale all learning rates by multiplier (for warmup/cooldown)."""
        for group in self.param_groups:
            group['lr'] = group['initial_lr'] * multiplier


def build_param_groups(model, config_dict):
    """
    Build param_groups for MuonAdamWMLX from a model and configuration.

    config_dict should have:
        matrix_lr, embedding_lr, unembedding_lr, scalar_lr,
        adam_betas, weight_decay, model_dim
    """
    flat_params = dict(tree_flatten(model.parameters()))
    model_dim = config_dict['model_dim']
    dmodel_lr_scale = (model_dim / 768) ** -0.5

    # Classify parameters by path
    embedding_paths = []
    value_embed_paths = []
    lm_head_paths = []
    resid_paths = []
    x0_paths = []
    matrix_paths_by_shape = {}

    for path, param in flat_params.items():
        if 'wte' in path and 'weight' in path:
            embedding_paths.append(path)
        elif 'value_embeds' in path:
            value_embed_paths.append(path)
        elif 'lm_head' in path:
            lm_head_paths.append(path)
        elif 'resid_lambdas' in path:
            resid_paths.append(path)
        elif 'x0_lambdas' in path:
            x0_paths.append(path)
        elif 'blocks' in path and param.ndim == 2:
            shape = param.shape
            if shape not in matrix_paths_by_shape:
                matrix_paths_by_shape[shape] = []
            matrix_paths_by_shape[shape].append(path)
        elif param.ndim == 2:
            # Other 2D params (e.g., ve_gate) — treat as matrix/muon
            shape = param.shape
            if shape not in matrix_paths_by_shape:
                matrix_paths_by_shape[shape] = []
            matrix_paths_by_shape[shape].append(path)

    adam_betas = config_dict.get('adam_betas', (0.8, 0.95))
    param_groups = []

    # AdamW groups
    if lm_head_paths:
        param_groups.append(dict(
            kind='adamw', paths=lm_head_paths,
            lr=config_dict['unembedding_lr'] * dmodel_lr_scale,
            betas=adam_betas, eps=1e-10, weight_decay=0.0,
        ))
    if embedding_paths:
        param_groups.append(dict(
            kind='adamw', paths=embedding_paths,
            lr=config_dict['embedding_lr'] * dmodel_lr_scale,
            betas=adam_betas, eps=1e-10, weight_decay=0.0,
        ))
    if value_embed_paths:
        param_groups.append(dict(
            kind='adamw', paths=value_embed_paths,
            lr=config_dict['embedding_lr'] * dmodel_lr_scale,
            betas=adam_betas, eps=1e-10, weight_decay=0.0,
        ))
    if resid_paths:
        param_groups.append(dict(
            kind='adamw', paths=resid_paths,
            lr=config_dict['scalar_lr'] * 0.01,
            betas=adam_betas, eps=1e-10, weight_decay=0.0,
        ))
    if x0_paths:
        param_groups.append(dict(
            kind='adamw', paths=x0_paths,
            lr=config_dict['scalar_lr'],
            betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0,
        ))

    # Muon groups (one per unique shape)
    for i, (shape, paths) in enumerate(sorted(matrix_paths_by_shape.items())):
        param_groups.append(dict(
            kind='muon', paths=paths,
            lr=config_dict['matrix_lr'],
            momentum=0.95, ns_steps=5, beta2=0.95,
            weight_decay=config_dict.get('weight_decay', 0.0),
            _group_id=f'muon_shape_{i}_{shape[0]}x{shape[1]}',
        ))

    # Assign group IDs to AdamW groups too
    for i, group in enumerate(param_groups):
        if '_group_id' not in group:
            group['_group_id'] = f'adamw_{i}'

    return param_groups
