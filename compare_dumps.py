#!/usr/bin/env python3
import numpy as np

def compare(name, a_path, b_path, label_a='CUDA', label_b='SDCPP'):
    a = np.fromfile(a_path, dtype=np.float32)
    b = np.fromfile(b_path, dtype=np.float32)
    if len(a) != len(b):
        print(f'{name}: SIZE MISMATCH {label_a}={len(a)} vs {label_b}={len(b)}')
        return
    diff = a - b
    signal = np.mean(b**2)
    noise = np.mean(diff**2)
    snr = 10*np.log10(signal/noise) if noise > 0 else float('inf')
    print(f'{name}: SNR={snr:.2f} dB | {label_a} mean={a.mean():.6f} | {label_b} mean={b.mean():.6f} | max_diff={np.max(np.abs(diff)):.6f}')
    print(f'  first5: {label_a}={a[:5]}  {label_b}={b[:5]}')

# Text encoder context comparison
compare('cond_context', 'cuda_cond_context.bin', 'sdcpp_cond_context.bin')
compare('uncond_context', 'cuda_uncond_context.bin', 'sdcpp_uncond_context.bin')

# Velocity comparison
compare('velocity_cond', 'pipe_velocity_cond.bin', 'sdcpp_velocity_cond.bin')
compare('velocity_uncond', 'pipe_velocity_uncond.bin', 'sdcpp_velocity_uncond.bin')

# Denoised comparison
compare('denoised', 'pipe_denoised.bin', 'sdcpp_denoised.bin')

# Initial latent comparison
compare('latent', 'pipe_latent_fp32.bin', 'sdcpp_noised_latent.bin')
