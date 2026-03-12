# autoresearch-arm

Apple Silicon dual-backend port of [karpathy/autoresearch](https://github.com/karpathy/autoresearch) with full Muon optimizer support on both PyTorch MPS and MLX.

![MLX training run on Apple M1 Max](mlx_run.png)

> **Latest results**: [25 experiments on M1 Max](https://github.com/elementalcollision/autoresearch/wiki/Experiment-Results-Mar-11-2026) — val_bpb **2.094 → 1.621** (−22.6%) via batch size tuning and MLP ratio optimization.

## What is this?

[Autoresearch](https://github.com/karpathy/autoresearch) is Karpathy's framework for autonomous AI-driven LLM training experiments. An AI agent modifies the training code, runs a 5-minute experiment, checks if results improved, keeps or discards, and repeats overnight.

The original requires an NVIDIA GPU (H100) with CUDA, FlashAttention-3, and `torch.compile`. This fork ports everything to Apple Silicon, supporting both **PyTorch MPS** and **MLX** backends. It targets M-series Max and Ultra chips (64-192GB unified memory), though it runs on any Apple Silicon Mac.

### Key features

- **Dual backend**: PyTorch MPS and Apple MLX, auto-detected or manually selected
- **Full Muon optimizer on both backends**: Newton-Schulz (Polar Express) orthogonalization, Nesterov momentum, NorMuon variance reduction, cautious weight decay. The MLX port is a novel implementation that doesn't exist in any public fork.
- **Hardware auto-detection**: Identifies chip generation (M1-M4), tier (base/Pro/Max/Ultra), GPU core count, and memory. Scales hyperparameters accordingly.
- **Hardware-adaptive defaults**: Batch size, model depth, and total batch size tuned per chip tier
- **No CUDA dependencies**: Pure Apple Silicon. FlashAttention-3 replaced with PyTorch SDPA (MPS) and native attention (MLX).

## Quick start

**Requirements**: Apple Silicon Mac (M1 or later), Python 3.10+, [uv](https://docs.astral.sh/uv/)

```bash
# 1. Install uv (if needed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install dependencies (pick your backend)
uv pip install -e '.[mlx]'    # MLX only (recommended)
uv pip install -e '.[mps]'    # PyTorch MPS only
uv pip install -e '.[all]'    # Both backends

# 3. Download data and train tokenizer (one-time, ~2 min)
uv run prepare.py

# 4. Run a training experiment (~5 min)
uv run train.py
```

## Backend selection

The system auto-detects the best backend (prefers MLX). Override with an environment variable:

```bash
# Auto-detect (default: prefers MLX)
uv run train.py

# Force MLX
AUTORESEARCH_BACKEND=mlx uv run train.py

# Force MPS
AUTORESEARCH_BACKEND=mps uv run train.py

# Run MLX directly
uv run train_mlx.py
```

Check your detected hardware and suggested config:

```bash
uv run -c "from backends import print_hardware_summary; print_hardware_summary()"
```

## Project structure

```
prepare.py              Data prep, tokenizer, dataloader, evaluation (do not modify)
train.py                MPS training script + backend dispatch (agent modifies this)
train_mlx.py            MLX training script (agent modifies this)
program.md              Agent instructions for autonomous experiments
backends/
  __init__.py           Hardware detection, chip tier, hyperparameter suggestions
  muon_mps.py           Muon+AdamW optimizer for PyTorch MPS
  muon_mlx.py           Muon+AdamW optimizer for MLX (novel port)
pyproject.toml          Dependencies with optional groups
```

**What the agent edits**: `train.py` (MPS) or `train_mlx.py` (MLX). Everything is fair game: architecture, optimizer settings, hyperparameters, batch size, model depth.

**What is fixed**: `prepare.py` (evaluation, data loading, constants), `backends/` (optimizer, hardware detection).

## Running autonomous experiments

Point your AI agent (Claude, Codex, etc.) at this repo and prompt:

```
Hi, have a look at program.md and let's kick off a new experiment! Let's do the setup first.
```

The agent reads `program.md`, establishes a baseline, then enters an autonomous loop: modify code, train 5 minutes, compare results, keep or discard, repeat. See `program.md` for full details.

## Hardware recommendations

| Chip tier | Memory | Model depth | Device batch | Total batch | Expected perf |
|-----------|--------|-------------|-------------|-------------|---------------|
| Base (M1-M4) | 8-24 GB | 4 | 8 | 32K tokens | Functional, slower |
| Pro | 18-36 GB | 6 | 16 | 64K tokens | Good |
| Max | 36-128 GB | 8 | 32 | 128K tokens | Recommended |
| Ultra | 64-192 GB | 10 | 64 | 256K tokens | Best |

These are the auto-detected defaults. The agent can change them during experimentation.

## Differences from the original

| Feature | Original (CUDA) | This fork (Apple Silicon) |
|---------|-----------------|---------------------------|
| Attention | FlashAttention-3 | PyTorch SDPA (MPS) / native (MLX) |
| Compilation | `torch.compile` | Eager mode (MPS) / `mx.compile` (MLX) |
| Memory model | Discrete GPU VRAM | Unified CPU/GPU memory |
| MFU metric | Exact (known H100 FLOPS) | Approximate (estimated per-chip FLOPS) |
| Optimizer | Muon+AdamW (CUDA) | Muon+AdamW on both backends |
| Backends | Single (CUDA) | Dual (MPS + MLX) |
| Precision | bf16 via autocast | bf16 with manual casting (MPS) / native (MLX) |

## Output format

After a 5-minute run, the script prints:

```
---
val_bpb:          0.997900
training_seconds: 300.1
total_seconds:    325.9
peak_vram_mb:     8192.0
mfu_percent:      45.20
total_tokens_M:   125.0
num_steps:        250
num_params_M:     50.3
depth:            8
backend:          mlx
chip:             Apple M4 Max
```

The key metric is **val_bpb** (validation bits per byte) -- lower is better.

## Technical notes

### MPS backend
- No `torch.compile` (not supported on MPS)
- All optimizer arithmetic done in float32 to avoid MPS mixed-dtype crashes
- Nesterov momentum uses explicit `mul_/add_` instead of `lerp_` (MPS dtype issue)
- Sliding window attention via manual mask + SDPA

### MLX backend
- Uses `mx.compile` for kernel fusion on the training step
- Newton-Schulz orthogonalization uses `mx.swapaxes` for matrix transpose
- Gradient accumulation via `tree_map`
- Explicit `mx.eval()` calls for lazy evaluation control
- `nn.value_and_grad()` replaces PyTorch's `.backward()`

### Muon optimizer
The Muon optimizer combines Newton-Schulz orthogonalization (Polar Express) with Nesterov momentum, NorMuon variance reduction, and cautious weight decay. It is applied to 2D matrix parameters in transformer blocks, while embeddings and scalars use standard AdamW. The MLX implementation is a complete port of the original CUDA version, adapted for MLX's lazy evaluation model.

## Acknowledgments

- [Andrej Karpathy](https://github.com/karpathy/autoresearch) -- original autoresearch framework and design philosophy
- [miolini/autoresearch-macos](https://github.com/miolini/autoresearch-macos) -- reference MPS port
- [trevin-creator/autoresearch-mlx](https://github.com/trevin-creator/autoresearch-mlx) -- reference MLX port
- [Jordan Keller](https://kellerjordan.github.io/posts/muon/) -- Muon optimizer

## License

MIT
