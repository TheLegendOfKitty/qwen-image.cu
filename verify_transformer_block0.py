#!/usr/bin/env python3
"""Verify transformer block 0 against our CUDA implementation.
Generates reference intermediate values using raw PyTorch computation."""

import numpy as np
import struct, json, os, sys, math

MODEL_DIR = "/home/parsa/qwen-image.cu/Qwen-Image-2512/transformer"

def load_bf16_tensor(name, files_map, shard_data):
    """Load a BF16 tensor from safetensors files."""
    shard = files_map[name]
    header, data = shard_data[shard]
    info = header[name]
    shape = info['shape']
    offsets = info['data_offsets']
    raw = data[offsets[0]:offsets[1]]
    # BF16 -> FP32
    n = len(raw) // 2
    bf16_vals = np.frombuffer(raw, dtype=np.uint16)
    fp32_vals = np.zeros(n, dtype=np.float32)
    for i in range(n):
        fp32_vals[i] = struct.unpack('f', struct.pack('I', int(bf16_vals[i]) << 16))[0]
    return fp32_vals.reshape(shape)

def load_sharded_tensors(model_dir):
    """Load all sharded safetensors metadata and data."""
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
    """Linear layer: y = x @ W^T + b"""
    y = x @ weight.T
    if bias is not None:
        y += bias
    return y

def rms_norm(x, weight, eps=1e-6):
    """RMS normalization."""
    rms = np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)
    return (x / rms) * weight

def silu(x):
    return x * (1.0 / (1.0 + np.exp(-x)))

def gelu_tanh(x):
    c = 0.7978845608
    return 0.5 * x * (1.0 + np.tanh(c * (x + 0.044715 * x * x * x)))

def layer_norm(x, eps=1e-6):
    """LayerNorm without affine."""
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps)

def modulate(x, shift, scale):
    return x * (1.0 + scale) + shift

def timestep_embedding(timesteps, dim=256, max_period=10000.0):
    """Sinusoidal timestep embedding matching ggml."""
    half = dim // 2
    freqs = np.exp(-np.log(max_period) * np.arange(half, dtype=np.float32) / half)
    args = timesteps * freqs
    result = np.zeros(dim, dtype=np.float32)
    result[:half] = np.cos(args)
    result[half:] = np.sin(args)
    return result

def main():
    print("Loading transformer weights...")
    files_map, shard_data = load_sharded_tensors(MODEL_DIR)

    def load(name):
        return load_bf16_tensor(name, files_map, shard_data)

    # Timestep embedding for sigma=0.9965 (first step, 20 steps)
    sigma = 0.9965  # approximate first sigma
    timestep = sigma * 1000.0
    print(f"Timestep: {timestep:.1f}")

    t_emb_sin = timestep_embedding(timestep, 256, 10000.0)
    print(f"t_emb_sin: mean={t_emb_sin.mean():.6f}, std={t_emb_sin.std():.4f}")
    print(f"  First 5: {t_emb_sin[:5]}")

    # Time embedding MLP
    w1 = load("time_text_embed.timestep_embedder.linear_1.weight")
    b1 = load("time_text_embed.timestep_embedder.linear_1.bias")
    w2 = load("time_text_embed.timestep_embedder.linear_2.weight")
    b2 = load("time_text_embed.timestep_embedder.linear_2.bias")

    t_after_l1 = linear(t_emb_sin, w1, b1)
    t_after_silu = silu(t_after_l1)
    t_emb = linear(t_after_silu, w2, b2)

    print(f"\nt_emb (timestep embedding): mean={t_emb.mean():.6f}, std={t_emb.std():.4f}")
    print(f"  First 5: {t_emb[:5]}")
    t_emb.tofile("ref_t_emb.bin")

    # Modulation for block 0
    img_mod_w = load("transformer_blocks.0.img_mod.1.weight")
    img_mod_b = load("transformer_blocks.0.img_mod.1.bias")

    mod_input = silu(t_emb)
    img_mods = linear(mod_input, img_mod_w, img_mod_b)

    print(f"\nimg_mods (block 0): mean={img_mods.mean():.6f}, std={img_mods.std():.4f}")
    print(f"  Shape: {img_mods.shape}")
    # Chunk into 6 parts
    inner_dim = 3072
    for i, name in enumerate(['shift1', 'scale1', 'gate1', 'shift2', 'scale2', 'gate2']):
        chunk = img_mods[i*inner_dim:(i+1)*inner_dim]
        print(f"  {name}: mean={chunk.mean():.6f}, std={chunk.std():.4f}")

    img_mods.tofile("ref_img_mods_b0.bin")
    print("\nSaved ref_t_emb.bin and ref_img_mods_b0.bin")

if __name__ == '__main__':
    main()
