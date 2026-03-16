"""
Hardware auto-detection, chip tier classification, and hyperparameter suggestions
for Apple Silicon Macs. Supports both PyTorch MPS and MLX backends.
"""

import os
import sys
import subprocess
import re


def detect_backend():
    """
    Auto-detect best available backend.
    Priority: MLX > MPS (MLX is generally faster for training on Apple Silicon).
    Override with AUTORESEARCH_BACKEND env var: 'mlx', 'mps', or 'auto'.
    """
    override = os.environ.get("AUTORESEARCH_BACKEND", "auto").lower()
    if override not in ("auto", "mlx", "mps"):
        raise ValueError(f"AUTORESEARCH_BACKEND must be 'auto', 'mlx', or 'mps', got '{override}'")

    if override == "mlx":
        try:
            import mlx.core as mx
            if mx.metal.is_available():
                return "mlx"
            raise RuntimeError("MLX is installed but Metal is not available")
        except ImportError:
            raise RuntimeError("AUTORESEARCH_BACKEND=mlx but mlx is not installed. "
                               "Install with: uv pip install 'autoresearch-arm[mlx]'")

    if override == "mps":
        try:
            import torch
            if torch.backends.mps.is_available():
                return "mps"
            raise RuntimeError("PyTorch is installed but MPS is not available")
        except ImportError:
            raise RuntimeError("AUTORESEARCH_BACKEND=mps but torch is not installed. "
                               "Install with: uv pip install 'autoresearch-arm[mps]'")

    # Auto-detect: prefer MLX
    if sys.platform != "darwin":
        raise RuntimeError("autoresearch-arm requires macOS with Apple Silicon")

    try:
        import mlx.core as mx
        if mx.metal.is_available():
            return "mlx"
    except ImportError:
        pass

    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass

    raise RuntimeError(
        "No compatible backend found. Install at least one:\n"
        "  MLX:  uv pip install 'autoresearch-arm[mlx]'\n"
        "  MPS:  uv pip install 'autoresearch-arm[mps]'\n"
        "  Both: uv pip install 'autoresearch-arm[all]'"
    )


def get_hardware_info():
    """
    Returns hardware info dict with keys:
      memory_gb, chip_name, chip_tier, gpu_cores (estimated)
    """
    info = {
        "memory_gb": 0,
        "chip_name": "unknown",
        "chip_tier": "unknown",
        "gpu_cores": 0,
    }

    # Memory
    try:
        mem_bytes = int(subprocess.check_output(
            ["sysctl", "-n", "hw.memsize"], text=True).strip())
        info["memory_gb"] = mem_bytes / (1024 ** 3)
    except (subprocess.CalledProcessError, ValueError):
        pass

    # Chip name
    try:
        brand = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
        info["chip_name"] = brand
    except subprocess.CalledProcessError:
        pass

    # Classify chip tier
    chip = info["chip_name"].lower()
    if "ultra" in chip:
        info["chip_tier"] = "ultra"
        info["gpu_cores"] = 80  # M4 Ultra estimate
    elif "max" in chip:
        info["chip_tier"] = "max"
        info["gpu_cores"] = 40  # M4 Max estimate
    elif "pro" in chip:
        info["chip_tier"] = "pro"
        info["gpu_cores"] = 18
    elif "m1" in chip or "m2" in chip or "m3" in chip or "m4" in chip:
        info["chip_tier"] = "base"
        info["gpu_cores"] = 10
    else:
        info["chip_tier"] = "unknown"

    # Refine GPU core estimates by specific chip model
    m = re.search(r"(m[1-5])\s*(pro|max|ultra)?", chip)
    if m:
        gen = m.group(1)
        tier = m.group(2) or "base"
        # Rough estimates for GPU cores by generation/tier
        gpu_core_map = {
            ("m1", "base"): 8, ("m1", "pro"): 16, ("m1", "max"): 32, ("m1", "ultra"): 64,
            ("m2", "base"): 10, ("m2", "pro"): 19, ("m2", "max"): 38, ("m2", "ultra"): 76,
            ("m3", "base"): 10, ("m3", "pro"): 18, ("m3", "max"): 40, ("m3", "ultra"): 76,
            ("m4", "base"): 10, ("m4", "pro"): 20, ("m4", "max"): 40, ("m4", "ultra"): 80,
        }
        info["gpu_cores"] = gpu_core_map.get((gen, tier), info["gpu_cores"])

    return info


def get_peak_flops(hw_info=None):
    """
    Estimate peak bf16 TFLOPS for MFU calculation.
    Returns FLOPS (not TFLOPS) for direct use in MFU computation.
    """
    if hw_info is None:
        hw_info = get_hardware_info()

    chip = hw_info["chip_name"].lower()
    gpu_cores = hw_info["gpu_cores"]

    # Rough bf16 TFLOPS estimates per GPU core (varies by generation)
    # M3/M4 cores are ~30% faster than M1/M2 cores
    m = re.search(r"(m[1-5])", chip)
    gen = m.group(1) if m else "m4"

    flops_per_core = {
        "m1": 0.5e12,   # ~0.5 TFLOPS/core bf16
        "m2": 0.55e12,
        "m3": 0.65e12,
        "m4": 0.7e12,
    }.get(gen, 0.65e12)

    return gpu_cores * flops_per_core


def suggest_hyperparameters(hw_info=None):
    """
    Suggest hyperparameters based on hardware tier.
    Returns dict with: depth, device_batch_size, total_batch_size, eval_tokens_multiplier
    """
    if hw_info is None:
        hw_info = get_hardware_info()

    mem_gb = hw_info["memory_gb"]
    tier = hw_info["chip_tier"]

    # Defaults validated by characterization sessions:
    #   M1 Max 64GB (mar11):     batch=16K, dev=8,  depth=8  → val_bpb=1.621
    #   M4 Pro 24GB (mar14):     batch=8K,  dev=8,  depth=6  → val_bpb=1.429
    #   M5 Max 64GB (mar14-m5):  batch=32K, dev=16, depth=8  → val_bpb=1.320
    #
    # Key finding: larger batches cause memory-pressure swapping even on 64GB.
    # More gradient steps (smaller batches) consistently beats model capacity.

    if tier == "ultra" or mem_gb >= 128:
        return {
            "depth": 10,
            "device_batch_size": 32,
            "total_batch_size": 2**16,  # 64K tokens
            "eval_tokens_multiplier": 10,
        }
    elif tier == "max" or mem_gb >= 48:
        return {
            "depth": 8,
            "device_batch_size": 16,
            "total_batch_size": 2**15,  # 32K tokens
            "eval_tokens_multiplier": 10,
        }
    elif tier == "pro" or mem_gb >= 18:
        return {
            "depth": 6,
            "device_batch_size": 8,
            "total_batch_size": 2**13,  # 8K tokens
            "eval_tokens_multiplier": 5,
        }
    else:
        # Base M-series or low memory
        return {
            "depth": 4,
            "device_batch_size": 4,
            "total_batch_size": 2**12,  # 4K tokens
            "eval_tokens_multiplier": 3,
        }


def sync_device(device_type):
    """Synchronize device for accurate timing."""
    if device_type == "cuda":
        import torch
        torch.cuda.synchronize()
    elif device_type == "mps":
        import torch
        torch.mps.synchronize()
    # MLX: no explicit sync needed (mx.eval() handles it)


def get_peak_memory_mb(device_type):
    """Get peak memory usage in MB."""
    if device_type == "cuda":
        import torch
        return torch.cuda.max_memory_allocated() / 1024 / 1024
    elif device_type == "mps":
        import torch
        try:
            return torch.mps.driver_allocated_memory() / 1024 / 1024
        except AttributeError:
            return 0.0
    elif device_type == "mlx":
        import mlx.core as mx
        try:
            return mx.get_peak_memory() / 1024 / 1024
        except AttributeError:
            return 0.0
    return 0.0


def print_hardware_summary():
    """Print a summary of detected hardware and suggested config."""
    hw = get_hardware_info()
    hp = suggest_hyperparameters(hw)
    peak_flops = get_peak_flops(hw)

    print(f"Hardware: {hw['chip_name']}")
    print(f"  Memory: {hw['memory_gb']:.0f} GB unified")
    print(f"  GPU cores: {hw['gpu_cores']} (estimated)")
    print(f"  Chip tier: {hw['chip_tier']}")
    print(f"  Peak bf16 FLOPS: {peak_flops:.2e}")
    print(f"Suggested config:")
    print(f"  Depth: {hp['depth']}")
    print(f"  Device batch size: {hp['device_batch_size']}")
    print(f"  Total batch size: {hp['total_batch_size']:,}")


if __name__ == "__main__":
    print_hardware_summary()
    print(f"\nDetected backend: {detect_backend()}")
