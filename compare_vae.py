#!/usr/bin/env python3
"""Compare our VAE output with PyTorch/diffusers reference."""
import numpy as np
import struct
import sys
import os

MODEL_DIR = "/home/parsa/qwen-image.cu/Qwen-Image-2512"

def load_latent(path, shape):
    """Load raw FP32 binary latent."""
    data = np.fromfile(path, dtype=np.float32)
    return data.reshape(shape)

def load_ppm(path):
    """Load PPM image as numpy array [H, W, 3]."""
    with open(path, 'rb') as f:
        magic = f.readline().strip()
        assert magic == b'P6', f"Expected P6, got {magic}"
        line = f.readline().strip()
        while line.startswith(b'#'):
            line = f.readline().strip()
        w, h = map(int, line.decode().split())
        maxval = int(f.readline().strip())
        data = f.read()
    return np.frombuffer(data, dtype=np.uint8).reshape(h, w, 3)

def psnr(a, b):
    """Compute PSNR between two images."""
    mse = np.mean((a.astype(float) - b.astype(float)) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10(255**2 / mse)

def run_vae_reference(latent_np):
    """Run VAE decoder using diffusers WanVAE."""
    import torch
    from diffusers.models import AutoencoderKLWan

    print("Loading WAN VAE from diffusers...")
    vae = AutoencoderKLWan.from_pretrained(
        MODEL_DIR, subfolder="vae", torch_dtype=torch.bfloat16
    )
    vae.eval()

    # latent_np shape: [16, 1, 64, 64] (C, T, H, W)
    # diffusers expects [B, C, T, H, W]
    latent_tensor = torch.from_numpy(latent_np).unsqueeze(0).to(torch.bfloat16)  # [1, 16, 1, 64, 64]
    print(f"Latent tensor: {latent_tensor.shape}, dtype={latent_tensor.dtype}")
    print(f"  min={latent_tensor.min():.4f}, max={latent_tensor.max():.4f}, mean={latent_tensor.mean():.4f}")

    with torch.no_grad():
        decoded = vae.decode(latent_tensor).sample  # [B, C, T, H, W]

    decoded = decoded.float().squeeze(0)  # [C, T, H, W]
    print(f"VAE output: {decoded.shape}")
    print(f"  min={decoded.min():.4f}, max={decoded.max():.4f}, mean={decoded.mean():.4f}")

    # Convert to image: clamp to [-1, 1], then to [0, 255]
    rgb = decoded[:, 0, :, :]  # [3, H, W] - take T=0
    rgb = torch.clamp((rgb + 1.0) / 2.0, 0.0, 1.0)  # [-1,1] -> [0,1]
    rgb = (rgb * 255.0).round().byte()
    rgb = rgb.permute(1, 2, 0).numpy()  # [H, W, 3]
    return rgb

def main():
    latent_path = "/home/parsa/qwen-image.cu/stable-diffusion.cpp/build/latent_denorm.bin"
    our_ppm = "/home/parsa/qwen-image.cu/output.ppm"

    print("Loading denormalized latent...")
    latent = load_latent(latent_path, (16, 1, 64, 64))
    print(f"Latent shape: {latent.shape}")
    print(f"  min={latent.min():.4f}, max={latent.max():.4f}, mean={latent.mean():.4f}, std={latent.std():.4f}")

    print("\nLoading our output image...")
    our_img = load_ppm(our_ppm)
    print(f"Our image: {our_img.shape}")
    print(f"  R mean={our_img[:,:,0].mean():.1f}, G mean={our_img[:,:,1].mean():.1f}, B mean={our_img[:,:,2].mean():.1f}")

    print("\nRunning reference VAE (CPU, may take a few minutes)...")
    ref_img = run_vae_reference(latent)
    print(f"Reference image: {ref_img.shape}")
    print(f"  R mean={ref_img[:,:,0].mean():.1f}, G mean={ref_img[:,:,1].mean():.1f}, B mean={ref_img[:,:,2].mean():.1f}")

    # Compute PSNR
    p = psnr(our_img, ref_img)
    print(f"\n=== VAE PSNR: {p:.2f} dB ===")

    # Save reference for visual comparison
    try:
        from PIL import Image
        Image.fromarray(ref_img).save("/home/parsa/qwen-image.cu/reference_vae.png")
        print("Saved reference_vae.png")
    except ImportError:
        # Save as PPM
        with open("/home/parsa/qwen-image.cu/reference_vae.ppm", 'wb') as f:
            h, w = ref_img.shape[:2]
            f.write(f"P6\n{w} {h}\n255\n".encode())
            f.write(ref_img.tobytes())
        print("Saved reference_vae.ppm")

    # Per-pixel error stats
    diff = np.abs(our_img.astype(float) - ref_img.astype(float))
    print(f"\nPer-pixel absolute error:")
    print(f"  Mean: {diff.mean():.2f}")
    print(f"  Max:  {diff.max():.0f}")
    print(f"  >5:   {(diff > 5).sum()} pixels ({100*(diff > 5).mean():.2f}%)")
    print(f"  >10:  {(diff > 10).sum()} pixels ({100*(diff > 10).mean():.2f}%)")

if __name__ == '__main__':
    main()
