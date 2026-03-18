"""Apple Silicon hardware info for TUI display."""

import subprocess
import re


def get_hardware_summary() -> dict:
    """Get hardware info without importing ML frameworks.

    Returns dict with: chip_name, gpu_cores, total_memory_gb, peak_tflops
    """
    info = {
        'chip_name': 'Unknown',
        'gpu_cores': 0,
        'total_memory_gb': 0,
        'peak_tflops': 0.0,
    }

    # Get chip name
    try:
        result = subprocess.run(
            ['/usr/sbin/sysctl', '-n', 'machdep.cpu.brand_string'],
            capture_output=True, text=True, timeout=5,
        )
        info['chip_name'] = result.stdout.strip()
    except Exception:
        pass

    # Get total memory
    try:
        result = subprocess.run(
            ['/usr/sbin/sysctl', '-n', 'hw.memsize'],
            capture_output=True, text=True, timeout=5,
        )
        info['total_memory_gb'] = int(result.stdout.strip()) / (1024 ** 3)
    except Exception:
        pass

    # Get GPU cores
    try:
        result = subprocess.run(
            ['/usr/sbin/sysctl', '-n', 'hw.perflevel0.gpucount'],
            capture_output=True, text=True, timeout=5,
        )
        info['gpu_cores'] = int(result.stdout.strip())
    except Exception:
        # Fallback: try to infer from chip name
        chip = info['chip_name'].lower()
        gpu_core_map = {
            ("m1", "base"): 8, ("m1", "pro"): 16, ("m1", "max"): 32, ("m1", "ultra"): 64,
            ("m2", "base"): 10, ("m2", "pro"): 19, ("m2", "max"): 38, ("m2", "ultra"): 76,
            ("m3", "base"): 10, ("m3", "pro"): 18, ("m3", "max"): 40, ("m3", "ultra"): 80,
            ("m4", "base"): 10, ("m4", "pro"): 20, ("m4", "max"): 40, ("m4", "ultra"): 80,
            ("m5", "base"): 10, ("m5", "pro"): 20, ("m5", "max"): 40, ("m5", "ultra"): 80,
        }
        gen_match = re.search(r'(m[1-9])', chip)
        if gen_match:
            gen = gen_match.group(1)
            tier = "ultra" if "ultra" in chip else "max" if "max" in chip else "pro" if "pro" in chip else "base"
            info['gpu_cores'] = gpu_core_map.get((gen, tier), 0)

    # Estimate peak TFLOPS
    chip = info['chip_name'].lower()
    flops_per_core = {
        "m1": 0.5e12, "m2": 0.55e12, "m3": 0.65e12,
        "m4": 0.7e12, "m5": 0.85e12,
    }
    gen_match = re.search(r'(m[1-9])', chip)
    if gen_match and info['gpu_cores'] > 0:
        gen = gen_match.group(1)
        fpc = flops_per_core.get(gen, 0.5e12)
        info['peak_tflops'] = info['gpu_cores'] * fpc / 1e12

    return info
