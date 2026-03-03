#!/usr/bin/env python3
"""Compare per-block img outputs between CUDA and sd.cpp."""
import sys, os, numpy as np

def psnr(a, b):
    mse = np.mean((a - b) ** 2)
    if mse == 0: return float('inf')
    peak = max(np.max(np.abs(a)), np.max(np.abs(b)))
    if peak == 0: return float('inf')
    return 10 * np.log10(peak**2 / mse)

suffix = "_step0_ntxt8"

# Block 0 internal comparison
print("=== Block 0 internals ===")
internals = [
    ("t_emb", "cuda_t_emb{}.bin", "sdcpp_t_emb{}.bin"),
    ("b0_img_mods", "cuda_b0_img_mods.bin{}.bin", "sdcpp_b0_img_mods{}.bin"),
    ("b0_img_q_raw", "cuda_b0_img_q_raw.bin{}.bin", "sdcpp_b0_img_q_raw{}.bin"),
    ("b0_img_k_raw", "cuda_b0_img_k_raw.bin{}.bin", "sdcpp_b0_img_k_raw{}.bin"),
    ("b0_img_v_raw", "cuda_b0_img_v_raw.bin{}.bin", "sdcpp_b0_img_v_raw{}.bin"),
    ("b0_img_q_normed", "cuda_b0_img_q_normed.bin{}.bin", "sdcpp_b0_img_q_normed{}.bin"),
    ("b0_img_k_normed", "cuda_b0_img_k_normed.bin{}.bin", "sdcpp_b0_img_k_normed{}.bin"),
    ("b0_q_joint_hsd", "cuda_b0_q_joint.bin{}.bin", "sdcpp_b0_q_joint_hsd{}.bin"),
    ("b0_k_joint_hsd", "cuda_b0_k_joint.bin{}.bin", "sdcpp_b0_k_joint_hsd{}.bin"),
    ("b0_q_after_rope", "cuda_b0_q_after_rope.bin{}.bin", "sdcpp_b0_q_after_rope{}.bin"),
    ("b0_k_after_rope", "cuda_b0_k_after_rope.bin{}.bin", "sdcpp_b0_k_after_rope{}.bin"),
    ("b0_img_attn_raw", "cuda_b0_attn_out_img.bin{}.bin", "sdcpp_b0_img_attn_raw{}.bin"),
    ("b0_img_proj", "cuda_b0_img_proj.bin{}.bin", "sdcpp_b0_img_proj{}.bin"),
    ("b0_img_after_attn", "cuda_b0_img_after_attn.bin{}.bin", "sdcpp_b0_img_after_attn{}.bin"),
    ("b0_img_out", "cuda_b0_img_after_mlp.bin{}.bin", "sdcpp_b0_img_out{}.bin"),
]

for name, cuda_fmt, sdcpp_fmt in internals:
    cuda_path = cuda_fmt.format(suffix)
    sdcpp_path = sdcpp_fmt.format(suffix)
    if not os.path.exists(cuda_path) or not os.path.exists(sdcpp_path):
        print(f"  {name:25s}: MISSING ({cuda_path} / {sdcpp_path})")
        continue
    a = np.fromfile(cuda_path, dtype=np.float32)
    b = np.fromfile(sdcpp_path, dtype=np.float32)
    if len(a) != len(b):
        print(f"  {name:25s}: SIZE MISMATCH {len(a)} vs {len(b)}")
        continue
    d = a - b
    p = psnr(a, b)
    max_d = np.max(np.abs(d))
    mean_d = np.mean(np.abs(d))
    print(f"  {name:25s}: PSNR={p:8.2f} dB  max_diff={max_d:.6e}  mean_diff={mean_d:.6e}")

print()
print("=== Per-block img output (after block i) ===")
for bi in range(60):
    cuda_path = f"cuda_img_after_b{bi}{suffix}.bin"
    sdcpp_path = f"sdcpp_img_after_b{bi}{suffix}.bin"
    if not os.path.exists(cuda_path) or not os.path.exists(sdcpp_path):
        continue
    a = np.fromfile(cuda_path, dtype=np.float32)
    b = np.fromfile(sdcpp_path, dtype=np.float32)
    if len(a) != len(b):
        print(f"  block {bi:2d}: SIZE MISMATCH {len(a)} vs {len(b)}")
        continue
    d = a - b
    p = psnr(a, b)
    max_d = np.max(np.abs(d))
    mean_d = np.mean(np.abs(d))
    print(f"  block {bi:2d}: PSNR={p:8.2f} dB  max_diff={max_d:.6e}  mean_diff={mean_d:.6e}  std_cuda={np.std(a):.1f}  std_sdcpp={np.std(b):.1f}")
