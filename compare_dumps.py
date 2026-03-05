#!/usr/bin/env python3
"""Compare binary FP32 dumps between C++ (cuda_*) and Python (py_*) implementations."""

import numpy as np
from pathlib import Path

def load_f32(path):
    return np.fromfile(path, dtype=np.float32)

def compare(name, cuda_path, py_path):
    if not Path(cuda_path).exists():
        print(f"  SKIP {name}: {cuda_path} not found")
        return
    if not Path(py_path).exists():
        print(f"  SKIP {name}: {py_path} not found")
        return

    a = load_f32(cuda_path)
    b = load_f32(py_path)

    if len(a) != len(b):
        print(f"  SIZE MISMATCH {name}: cuda={len(a)} vs py={len(b)}")
        n = min(len(a), len(b))
        a, b = a[:n], b[:n]
    else:
        n = len(a)

    diff = a - b
    abs_diff = np.abs(diff)
    mse = np.mean(diff**2)
    max_val = max(np.abs(a).max(), np.abs(b).max(), 1e-10)
    psnr = 10 * np.log10(max_val**2 / mse) if mse > 1e-20 else float('inf')

    print(f"  {name}:")
    print(f"    size={n}, MSE={mse:.2e}, PSNR={psnr:.1f} dB")
    print(f"    max_abs_diff={abs_diff.max():.6f}, mean_abs_diff={abs_diff.mean():.6f}")
    print(f"    cuda: mean={a.mean():.6f} std={a.std():.4f} first5: {a[:5]}")
    print(f"    py:   mean={b.mean():.6f} std={b.std():.4f} first5: {b[:5]}")

def main():
    print("Comparing C++ vs Python intermediate dumps:\n")
    comparisons = [
        ("timestep_embedding", "cuda_t_emb.bin", "py_time_embed.bin"),
        ("img_after_proj", "cuda_img_after_proj.bin", "py_img_in.bin"),
        ("txt_after_rms", "cuda_txt_after_rms.bin", "py_txt_norm.bin"),
        ("txt_after_proj", "cuda_txt_after_proj.bin", "py_txt_in.bin"),
        ("velocity_cond", "pipe_velocity_cond.bin", "py_velocity_cond.bin"),
        ("velocity_uncond", "pipe_velocity_uncond.bin", "py_velocity_uncond.bin"),
        ("velocity_cfg", "pipe_velocity_cfg.bin", "py_velocity_cfg.bin"),
        ("cond_embed", "cond_embed.bin", "py_cond_embed.bin"),
        ("uncond_embed", "uncond_embed.bin", "py_uncond_embed.bin"),
    ]
    for name, cuda_path, py_path in comparisons:
        compare(name, cuda_path, py_path)

if __name__ == "__main__":
    main()
