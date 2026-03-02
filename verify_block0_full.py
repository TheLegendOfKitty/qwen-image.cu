#!/usr/bin/env python3
"""Verify transformer block 0 forward pass against CUDA dumps.
Loads CUDA-dumped intermediate values and compares against Python-computed reference."""

import numpy as np
import struct, json, os

MODEL_DIR = "/home/parsa/qwen-image.cu/Qwen-Image-2512/transformer"

def load_bf16_tensor(name, files_map, shard_data):
    shard = files_map[name]
    header, data = shard_data[shard]
    info = header[name]
    shape = info['shape']
    offsets = info['data_offsets']
    raw = data[offsets[0]:offsets[1]]
    n = len(raw) // 2
    bf16_vals = np.frombuffer(raw, dtype=np.uint16)
    fp32_vals = np.zeros(n, dtype=np.float32)
    # Vectorized BF16 to FP32
    fp32_vals = np.frombuffer(
        (bf16_vals.astype(np.uint32) << 16).tobytes(), dtype=np.float32
    )
    return fp32_vals.reshape(shape)

def load_sharded_tensors(model_dir):
    index_path = os.path.join(model_dir, "diffusion_pytorch_model.safetensors.index.json")
    with open(index_path) as f:
        index = json.load(f)
    files_map = index['weight_map']
    shard_data = {}
    for shard_name in set(files_map.values()):
        path = os.path.join(model_dir, shard_name)
        with open(path, 'rb') as f:
            header_size = struct.unpack('<Q', f.read(8))[0]
            header = json.loads(f.read(header_size))
            data = f.read()
        shard_data[shard_name] = (header, data)
    return files_map, shard_data

def linear(x, weight, bias=None):
    y = x @ weight.T
    if bias is not None:
        y += bias
    return y

def rms_norm(x, weight, eps=1e-6):
    rms = np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)
    return (x / rms) * weight

def silu(x):
    return x * (1.0 / (1.0 + np.exp(-x.clip(-80, 80))))

def gelu_tanh(x):
    c = 0.7978845608028654
    return 0.5 * x * (1.0 + np.tanh(c * (x + 0.044715 * x * x * x)))

def layer_norm_no_affine(x, eps=1e-6):
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps)

def modulate(x, shift, scale):
    return x * (1.0 + scale) + shift

def compare(name, cuda_data, ref_data, limit=5):
    """Compare CUDA dump against reference, report SNR and differences."""
    diff = cuda_data.flatten() - ref_data.flatten()
    signal_power = np.mean(ref_data.flatten() ** 2)
    noise_power = np.mean(diff ** 2)
    if noise_power == 0:
        snr = float('inf')
    elif signal_power == 0:
        snr = -float('inf')
    else:
        snr = 10 * np.log10(signal_power / noise_power)

    print(f"\n  {name}:")
    print(f"    SNR: {snr:.2f} dB")
    print(f"    Ref:  mean={ref_data.mean():.6f} std={ref_data.std():.4f} first{limit}: {ref_data.flatten()[:limit]}")
    print(f"    CUDA: mean={cuda_data.mean():.6f} std={cuda_data.std():.4f} first{limit}: {cuda_data.flatten()[:limit]}")
    max_abs_diff = np.max(np.abs(diff))
    print(f"    Max abs diff: {max_abs_diff:.6f}")
    return snr

def main():
    print("Loading weights...")
    files_map, shard_data = load_sharded_tensors(MODEL_DIR)
    load = lambda name: load_bf16_tensor(name, files_map, shard_data)

    inner_dim = 3072
    n_heads = 24
    head_dim = 128

    # --- Load CUDA dumps ---
    def load_dump(name):
        return np.fromfile(name, dtype=np.float32)

    cuda_t_emb = load_dump("cuda_t_emb.bin")
    cuda_img_proj = load_dump("cuda_img_after_proj.bin")
    cuda_b0_mods = load_dump("cuda_b0_img_mods.bin")
    cuda_b0_ln = load_dump("cuda_b0_img_ln.bin")
    cuda_b0_modulated = load_dump("cuda_b0_img_modulated.bin")
    cuda_b0_q_raw = load_dump("cuda_b0_img_q_raw.bin")
    cuda_b0_q_normed = load_dump("cuda_b0_img_q_normed.bin")
    cuda_b0_mlp_input = load_dump("cuda_b0_img_mlp_input.bin")
    cuda_b0_mlp_fc1 = load_dump("cuda_b0_img_mlp_fc1.bin")
    cuda_b0_mlp_gelu = load_dump("cuda_b0_img_mlp_gelu.bin")
    cuda_b0_mlp_out = load_dump("cuda_b0_img_mlp_out.bin")
    cuda_b0_after_attn = load_dump("cuda_b0_img_after_attn.bin")
    cuda_b0_after_mlp = load_dump("cuda_b0_img_after_mlp.bin")

    n_img = len(cuda_img_proj) // inner_dim
    print(f"n_img = {n_img}")

    # Reshape CUDA dumps
    img_fp32 = cuda_img_proj.reshape(n_img, inner_dim)

    # --- Verify block 0 modulation ---
    print("\n=== Block 0 Modulation ===")
    img_mod_w = load("transformer_blocks.0.img_mod.1.weight")
    img_mod_b = load("transformer_blocks.0.img_mod.1.bias")

    t_emb_silu = silu(cuda_t_emb)  # Use CUDA t_emb as input
    ref_mods = linear(t_emb_silu, img_mod_w, img_mod_b)
    compare("img_mods", cuda_b0_mods, ref_mods)

    # --- Verify LayerNorm ---
    print("\n=== Block 0 LayerNorm (no affine) ===")
    ref_ln = layer_norm_no_affine(img_fp32)
    cuda_ln_2d = cuda_b0_ln.reshape(n_img, inner_dim)
    compare("img_ln", cuda_ln_2d, ref_ln)

    # --- Verify Modulation ---
    print("\n=== Block 0 Modulate (shift1, scale1) ===")
    shift1 = ref_mods[:inner_dim]
    scale1 = ref_mods[inner_dim:2*inner_dim]
    ref_modulated = modulate(ref_ln, shift1, scale1)
    cuda_mod_2d = cuda_b0_modulated.reshape(n_img, inner_dim)
    compare("img_modulated", cuda_mod_2d, ref_modulated)

    # --- Verify Q projection ---
    print("\n=== Block 0 Q Projection ===")
    to_q_w = load("transformer_blocks.0.attn.to_q.weight")
    to_q_b = load("transformer_blocks.0.attn.to_q.bias")
    ref_q_raw = linear(ref_modulated, to_q_w, to_q_b)
    cuda_q_raw_2d = cuda_b0_q_raw.reshape(n_img, inner_dim)
    compare("img_q_raw", cuda_q_raw_2d, ref_q_raw)

    # --- Verify Q RMSNorm ---
    print("\n=== Block 0 Q RMSNorm ===")
    norm_q_w = load("transformer_blocks.0.attn.norm_q.weight")
    # Reshape to [n_img * n_heads, head_dim] for per-head norm
    ref_q_heads = ref_q_raw.reshape(n_img * n_heads, head_dim)
    ref_q_normed = rms_norm(ref_q_heads, norm_q_w)
    ref_q_normed_flat = ref_q_normed.reshape(n_img, inner_dim)
    cuda_q_normed_2d = cuda_b0_q_normed.reshape(n_img, inner_dim)
    compare("img_q_normed", cuda_q_normed_2d, ref_q_normed_flat)

    # --- Verify MLP input (LayerNorm + modulate with shift2/scale2) ---
    print("\n=== Block 0 MLP Input (after 2nd modulate) ===")
    # We need img_after_attn values for this
    img_after_attn = cuda_b0_after_attn.reshape(n_img, inner_dim)
    ref_ln2 = layer_norm_no_affine(img_after_attn)
    shift2 = ref_mods[3*inner_dim:4*inner_dim]
    scale2 = ref_mods[4*inner_dim:5*inner_dim]
    ref_mlp_input = modulate(ref_ln2, shift2, scale2)
    cuda_mlp_input_2d = cuda_b0_mlp_input.reshape(n_img, inner_dim)
    compare("mlp_input", cuda_mlp_input_2d, ref_mlp_input)

    # --- Verify MLP fc1 ---
    print("\n=== Block 0 MLP FC1 ===")
    fc1_w = load("transformer_blocks.0.img_mlp.net.0.proj.weight")
    fc1_b = load("transformer_blocks.0.img_mlp.net.0.proj.bias")
    ref_fc1 = linear(ref_mlp_input, fc1_w, fc1_b)
    cuda_fc1_2d = cuda_b0_mlp_fc1.reshape(n_img, -1)
    compare("mlp_fc1", cuda_fc1_2d, ref_fc1)

    # --- Verify GELU ---
    print("\n=== Block 0 MLP GELU ===")
    ref_gelu = gelu_tanh(ref_fc1)
    cuda_gelu_2d = cuda_b0_mlp_gelu.reshape(n_img, -1)
    compare("mlp_gelu", cuda_gelu_2d, ref_gelu)

    # --- Verify MLP fc2 ---
    print("\n=== Block 0 MLP FC2 ===")
    fc2_w = load("transformer_blocks.0.img_mlp.net.2.weight")
    fc2_b = load("transformer_blocks.0.img_mlp.net.2.bias")
    ref_fc2 = linear(ref_gelu, fc2_w, fc2_b)
    cuda_fc2_2d = cuda_b0_mlp_out.reshape(n_img, inner_dim)
    compare("mlp_fc2 (mlp_out)", cuda_fc2_2d, ref_fc2)

    # --- Verify gate2 residual ---
    print("\n=== Block 0 After MLP Residual (gate2 * mlp_out + img) ===")
    gate2 = ref_mods[5*inner_dim:6*inner_dim]
    ref_after_mlp = img_after_attn + gate2 * ref_fc2
    cuda_after_mlp_2d = cuda_b0_after_mlp.reshape(n_img, inner_dim)
    compare("img_after_mlp", cuda_after_mlp_2d, ref_after_mlp)

    print("\nDone!")

if __name__ == '__main__':
    main()
