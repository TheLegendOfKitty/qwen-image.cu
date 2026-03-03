#!/usr/bin/env python3
"""Compare per-step latent dumps between CUDA and sd.cpp implementations.

Usage:
  1. Run sd.cpp with:  DUMP_STEP_LATENTS=/tmp/step_dumps ./sd ...
  2. Run CUDA with:    DUMP_STEP_LATENTS=/tmp/step_dumps ./qwen_image ...
  3. python3 compare_steps.py /tmp/step_dumps [num_steps]
"""
import sys, os, numpy as np

def psnr(a, b):
    mse = np.mean((a - b) ** 2)
    if mse == 0:
        return float('inf')
    peak = max(np.max(np.abs(a)), np.max(np.abs(b)))
    if peak == 0:
        return float('inf')
    return 10 * np.log10(peak**2 / mse)

def snr(a, b):
    diff = a - b
    signal = np.mean(b**2)
    noise = np.mean(diff**2)
    if noise == 0:
        return float('inf')
    return 10 * np.log10(signal / noise)

def compare(label, a_path, b_path):
    if not os.path.exists(a_path):
        print(f"  {label}: MISSING {a_path}")
        return None
    if not os.path.exists(b_path):
        print(f"  {label}: MISSING {b_path}")
        return None
    a = np.fromfile(a_path, dtype=np.float32)
    b = np.fromfile(b_path, dtype=np.float32)
    if len(a) != len(b):
        print(f"  {label}: SIZE MISMATCH cuda={len(a)} sdcpp={len(b)}")
        return None
    d = a - b
    p = psnr(a, b)
    s = snr(a, b)
    max_diff = np.max(np.abs(d))
    mean_diff = np.mean(np.abs(d))
    print(f"  {label:12s}: PSNR={p:7.2f} dB  SNR={s:7.2f} dB  max_diff={max_diff:.6e}  mean_diff={mean_diff:.6e}")
    print(f"    cuda  first5: {a[:5]}")
    print(f"    sdcpp first5: {b[:5]}")
    return p

dump_dir = sys.argv[1] if len(sys.argv) > 1 else "/tmp/step_dumps"
num_steps = int(sys.argv[2]) if len(sys.argv) > 2 else 20

print(f"Comparing step latents in {dump_dir} ({num_steps} steps)\n")

for step in range(num_steps):
    print(f"=== Step {step} ===")
    for kind in ["input", "denoised", "output"]:
        cuda_path = os.path.join(dump_dir, f"local_step_{step:02d}_{kind}.bin")
        sdcpp_path = os.path.join(dump_dir, f"sdcpp_step_{step:02d}_{kind}.bin")
        compare(kind, cuda_path, sdcpp_path)
    print()
