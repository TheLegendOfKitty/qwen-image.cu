#!/usr/bin/env python3
"""
Compare our CUDA implementation's full pipeline output with the diffusers
QwenImagePipeline reference.

Usage:
    python3 compare_full_pipeline.py [--mode full|stepwise|sigmas_only]

Modes:
    full        - Run the full diffusers pipeline end-to-end and compare output.ppm
    stepwise    - Run individual components step by step with detailed intermediate
                  comparisons (text encoder, scheduler, transformer steps, VAE).
    sigmas_only - Just compare the sigma schedules (fast, no model loading).

IMPORTANT NOTES ON DIFFERENCES BETWEEN OUR CUDA IMPL AND DIFFUSERS:
    1. CFG: Our CUDA impl uses standard CFG:
           out = uncond + cfg_scale * (cond - uncond)
       Diffusers uses "true CFG" with norm rescaling:
           comb = neg + true_cfg_scale * (cond - neg)
           out = comb * (cond_norm / comb_norm)
       These produce DIFFERENT results for the same cfg_scale value.

    2. Scheduler shifting: Our CUDA impl uses fixed shift=3.0.
       Diffusers uses dynamic shifting (exponential time shift) based on image
       sequence length, with shift_terminal=0.02 stretching.

    3. Timestep input: Diffusers passes timestep/1000 to transformer.
       Our CUDA passes sigma * 1000. These are the same when sigma = t/1000.

    These differences mean diffusers and our CUDA will produce different images.
    To get an exact match, we either need to:
    (a) Modify this script to replicate our CUDA's exact behavior, OR
    (b) Modify our CUDA to match diffusers exactly.

    This script supports both approaches via --match-cuda flag.
"""

import argparse
import math
import os
import sys
import time

import numpy as np

MODEL_DIR = "/home/parsa/qwen-image.cu/Qwen-Image-2512"
OUR_OUTPUT = "/home/parsa/qwen-image.cu/output.ppm"
REFERENCE_OUTPUT = "/home/parsa/qwen-image.cu/reference_full.png"

# Parameters matching our CUDA main.cu defaults
PROMPT = "a red rose"
WIDTH = 512
HEIGHT = 512
STEPS = 20
CFG_SCALE = 7.0
SEED = 42
FLOW_SHIFT = 3.0  # our CUDA uses fixed shift


def load_ppm(path):
    """Load PPM image as numpy array [H, W, 3] uint8."""
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
    """Compute PSNR between two uint8 images."""
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10(255.0 ** 2 / mse)


def save_image(img_np, path):
    """Save numpy [H, W, 3] uint8 image."""
    try:
        from PIL import Image
        Image.fromarray(img_np).save(path)
        print(f"  Saved {path}")
    except ImportError:
        # Fallback to PPM
        ppm_path = path.rsplit('.', 1)[0] + '.ppm'
        with open(ppm_path, 'wb') as f:
            h, w = img_np.shape[:2]
            f.write(f"P6\n{w} {h}\n255\n".encode())
            f.write(img_np.tobytes())
        print(f"  Saved {ppm_path} (PIL not available)")


def compare_images(our_img, ref_img):
    """Print detailed comparison stats between two images."""
    p = psnr(our_img, ref_img)
    diff = np.abs(our_img.astype(np.float64) - ref_img.astype(np.float64))

    print(f"\n{'='*60}")
    print(f"  PSNR: {p:.2f} dB")
    print(f"{'='*60}")
    print(f"  Per-pixel absolute error:")
    print(f"    Mean: {diff.mean():.2f}")
    print(f"    Max:  {diff.max():.0f}")
    print(f"    Median: {np.median(diff):.1f}")
    print(f"    Std: {diff.std():.2f}")
    print(f"    >1:   {(diff > 1).sum()} pixels ({100*(diff > 1).mean():.2f}%)")
    print(f"    >5:   {(diff > 5).sum()} pixels ({100*(diff > 5).mean():.2f}%)")
    print(f"    >10:  {(diff > 10).sum()} pixels ({100*(diff > 10).mean():.2f}%)")
    print(f"    >50:  {(diff > 50).sum()} pixels ({100*(diff > 50).mean():.2f}%)")

    for c, name in enumerate(['R', 'G', 'B']):
        print(f"  Channel {name}:")
        print(f"    Our mean={our_img[:,:,c].mean():.1f}, Ref mean={ref_img[:,:,c].mean():.1f}")
        cdiff = diff[:,:,c]
        print(f"    Error mean={cdiff.mean():.2f}, max={cdiff.max():.0f}")

    return p


def compare_sigmas(match_cuda=False):
    """Compare sigma schedules between our CUDA impl and diffusers."""
    import torch
    from diffusers import FlowMatchEulerDiscreteScheduler

    print("\n=== Sigma Schedule Comparison ===")

    # Load the scheduler from config
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        MODEL_DIR, subfolder="scheduler"
    )
    print(f"  Scheduler config:")
    print(f"    use_dynamic_shifting: {scheduler.config.use_dynamic_shifting}")
    print(f"    time_shift_type: {scheduler.config.time_shift_type}")
    print(f"    shift: {scheduler.config.shift}")
    print(f"    shift_terminal: {scheduler.config.shift_terminal}")
    print(f"    base_shift: {scheduler.config.base_shift}")
    print(f"    max_shift: {scheduler.config.max_shift}")

    latent_h = HEIGHT // 8
    latent_w = WIDTH // 8
    n_img = (latent_h // 2) * (latent_w // 2)  # patch packing
    print(f"  n_img (image seq len): {n_img}")

    # Diffusers pipeline computes sigmas as:
    raw_sigmas = np.linspace(1.0, 1.0 / STEPS, STEPS)
    print(f"  Raw sigmas (linspace): {raw_sigmas[:5]}...{raw_sigmas[-3:]}")

    if match_cuda:
        # Replicate our CUDA scheduler: fixed shift, no dynamic shifting
        print(f"\n  [match-cuda mode] Using fixed shift={FLOW_SHIFT}")
        shifted = FLOW_SHIFT * raw_sigmas / (1 + (FLOW_SHIFT - 1) * raw_sigmas)
        # No stretch_shift_to_terminal in our CUDA impl
        sigmas_np = shifted
    else:
        # Dynamic shifting (what diffusers does)
        mu = _calculate_shift(n_img,
                              scheduler.config.base_image_seq_len,
                              scheduler.config.max_image_seq_len,
                              scheduler.config.base_shift,
                              scheduler.config.max_shift)
        print(f"  Computed mu (dynamic shift): {mu:.6f}")

        # Exponential time shift
        sigmas_np = np.exp(mu) / (np.exp(mu) + (1.0 / raw_sigmas - 1.0) ** 1.0)
        print(f"  After dynamic shift: {sigmas_np[:5]}...")

        # stretch_shift_to_terminal
        if scheduler.config.shift_terminal:
            one_minus_z = 1 - sigmas_np
            scale_factor = one_minus_z[-1] / (1 - scheduler.config.shift_terminal)
            sigmas_np = 1 - (one_minus_z / scale_factor)
            print(f"  After terminal stretch: {sigmas_np[:5]}...")

    # Append 0 at end
    sigmas_full = np.append(sigmas_np, 0.0)

    # Also verify via diffusers scheduler.set_timesteps
    if not match_cuda:
        mu = _calculate_shift(n_img,
                              scheduler.config.base_image_seq_len,
                              scheduler.config.max_image_seq_len,
                              scheduler.config.base_shift,
                              scheduler.config.max_shift)
        scheduler.set_timesteps(STEPS, device="cpu", sigmas=list(raw_sigmas), mu=mu)
        diffusers_sigmas = scheduler.sigmas.numpy()
        print(f"\n  Diffusers scheduler sigmas (via set_timesteps):")
        for i in range(min(STEPS + 1, len(diffusers_sigmas))):
            ours = sigmas_full[i] if i < len(sigmas_full) else float('nan')
            theirs = diffusers_sigmas[i]
            match = "OK" if abs(ours - theirs) < 1e-6 else f"DIFF ({ours - theirs:.2e})"
            print(f"    [{i:2d}] manual={ours:.6f}  diffusers={theirs:.6f}  {match}")

    print(f"\n  Our CUDA sigmas (fixed shift={FLOW_SHIFT}):")
    cuda_sigmas = FLOW_SHIFT * raw_sigmas / (1 + (FLOW_SHIFT - 1) * raw_sigmas)
    cuda_sigmas_full = np.append(cuda_sigmas, 0.0)
    for i in range(min(6, len(cuda_sigmas_full))):
        print(f"    [{i:2d}] sigma={cuda_sigmas_full[i]:.6f}  timestep={cuda_sigmas_full[i]*1000:.1f}")
    print(f"    ...")
    for i in range(max(0, len(cuda_sigmas_full) - 3), len(cuda_sigmas_full)):
        print(f"    [{i:2d}] sigma={cuda_sigmas_full[i]:.6f}  timestep={cuda_sigmas_full[i]*1000:.1f}")

    if not match_cuda:
        print(f"\n  Comparison (CUDA fixed shift vs diffusers dynamic shift):")
        for i in range(min(STEPS + 1, len(sigmas_full))):
            c = cuda_sigmas_full[i]
            d = sigmas_full[i]
            print(f"    [{i:2d}] cuda={c:.6f}  diffusers={d:.6f}  diff={c-d:.6f}")

    return sigmas_full, cuda_sigmas_full


def _calculate_shift(image_seq_len, base_seq_len=256, max_seq_len=4096,
                     base_shift=0.5, max_shift=1.15):
    """Replicate diffusers' calculate_shift function."""
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


def run_full_pipeline(match_cuda=False):
    """Run the complete diffusers pipeline and compare with our output."""
    import torch
    from diffusers import QwenImagePipeline

    print("\n=== Full Pipeline Comparison ===")
    print(f"  Prompt: '{PROMPT}'")
    print(f"  Size: {WIDTH}x{HEIGHT}")
    print(f"  Steps: {STEPS}")
    print(f"  CFG scale: {CFG_SCALE}")
    print(f"  Seed: {SEED}")
    print(f"  Match CUDA mode: {match_cuda}")

    # Load pipeline on CPU with BF16
    print("\n  Loading QwenImagePipeline (CPU, BF16)...")
    print("  WARNING: This loads ~57GB of weights. Ensure sufficient RAM.")
    t0 = time.time()

    pipe = QwenImagePipeline.from_pretrained(
        MODEL_DIR,
        torch_dtype=torch.bfloat16,
    )
    pipe.to("cpu")

    t1 = time.time()
    print(f"  Pipeline loaded in {t1-t0:.1f}s")

    # Set up generator for reproducibility
    generator = torch.Generator(device="cpu").manual_seed(SEED)

    if match_cuda:
        # Override scheduler to use fixed shift (matching our CUDA)
        pipe.scheduler.config["use_dynamic_shifting"] = False
        pipe.scheduler.config["shift"] = FLOW_SHIFT
        pipe.scheduler.config["shift_terminal"] = 0  # disable terminal stretch

        print(f"  [match-cuda] Overrode scheduler: fixed shift={FLOW_SHIFT}, no dynamic shifting")

        # For match_cuda mode, we replicate our CUDA's standard CFG (no renorm)
        # by using the pipeline in a custom way
        print(f"  [match-cuda] Running with standard CFG (no renormalization)...")
        image = _run_pipeline_standard_cfg(pipe, generator)
    else:
        # Run diffusers pipeline as-is (with its own CFG/scheduling)
        print(f"\n  Running pipeline (this will be SLOW on CPU)...")
        print(f"  Estimated time: ~30-60 minutes for 20 steps on CPU")
        t2 = time.time()

        result = pipe(
            prompt=PROMPT,
            negative_prompt="",
            true_cfg_scale=CFG_SCALE,
            height=HEIGHT,
            width=WIDTH,
            num_inference_steps=STEPS,
            generator=generator,
        )
        image = result.images[0]

        t3 = time.time()
        print(f"  Pipeline completed in {t3-t2:.1f}s")

    # Convert PIL to numpy
    ref_img = np.array(image)
    print(f"  Reference image shape: {ref_img.shape}")

    # Save reference
    save_image(ref_img, REFERENCE_OUTPUT)

    # Load our output
    if os.path.exists(OUR_OUTPUT):
        print(f"\n  Loading our output: {OUR_OUTPUT}")
        our_img = load_ppm(OUR_OUTPUT)
        compare_images(our_img, ref_img)
    else:
        print(f"\n  Our output {OUR_OUTPUT} not found. Run the CUDA pipeline first.")
        print(f"  Reference image saved to {REFERENCE_OUTPUT}")


def _run_pipeline_standard_cfg(pipe, generator):
    """
    Run the pipeline manually with standard CFG (no renormalization) to match
    our CUDA implementation's behavior.
    """
    import torch
    from diffusers.utils.torch_utils import randn_tensor

    device = "cpu"
    dtype = torch.bfloat16

    # 1. Encode prompts
    print("  [1] Encoding prompts...")
    t0 = time.time()
    prompt_embeds, prompt_mask = pipe.encode_prompt(
        prompt=PROMPT, device=device, max_sequence_length=512
    )
    neg_prompt_embeds, neg_prompt_mask = pipe.encode_prompt(
        prompt="", device=device, max_sequence_length=512
    )
    print(f"      Cond embeds: {prompt_embeds.shape}, mask sum: {prompt_mask.sum().item()}")
    print(f"      Uncond embeds: {neg_prompt_embeds.shape}, mask sum: {neg_prompt_mask.sum().item()}")
    t1 = time.time()
    print(f"      Encoding took {t1-t0:.1f}s")

    # Free text encoder to save memory
    del pipe.text_encoder
    import gc; gc.collect()
    print("      Freed text encoder")

    # 2. Prepare latents
    print("  [2] Preparing latents...")
    latent_h = HEIGHT // 8  # 64
    latent_w = WIDTH // 8   # 64
    num_channels = pipe.transformer.config.in_channels // 4  # 64//4 = 16

    # Shape: [1, 1, 16, 64, 64] then packed to [1, n_patches, 64]
    latents = pipe.prepare_latents(
        1, num_channels, HEIGHT, WIDTH, dtype, device, generator
    )
    print(f"      Latents shape (packed): {latents.shape}")

    n_img = latents.shape[1]  # number of image patches
    img_shapes = [[(1, latent_h // 2, latent_w // 2)]]

    # 3. Prepare timesteps (using fixed shift to match CUDA)
    print("  [3] Preparing timesteps...")
    import numpy as np
    raw_sigmas = np.linspace(1.0, 1.0 / STEPS, STEPS)

    # Apply fixed shift (matching our CUDA)
    shifted_sigmas = FLOW_SHIFT * raw_sigmas / (1 + (FLOW_SHIFT - 1) * raw_sigmas)
    sigmas_list = list(shifted_sigmas.astype(np.float32))

    # Manually set scheduler timesteps
    pipe.scheduler.config["use_dynamic_shifting"] = False
    pipe.scheduler.config["shift_terminal"] = 0

    sigmas_np = np.array(sigmas_list, dtype=np.float32)
    sigmas_tensor = torch.from_numpy(np.append(sigmas_np, 0.0)).to(torch.float32)
    timesteps = sigmas_tensor[:-1] * 1000.0

    pipe.scheduler.sigmas = sigmas_tensor
    pipe.scheduler.timesteps = timesteps
    pipe.scheduler.num_inference_steps = STEPS
    pipe.scheduler._step_index = None
    pipe.scheduler._begin_index = None
    pipe.scheduler.set_begin_index(0)

    for i in range(min(5, len(sigmas_tensor))):
        print(f"      [{i}] sigma={sigmas_tensor[i]:.6f}  timestep={timesteps[i] if i < len(timesteps) else 0:.1f}")

    txt_seq_lens = prompt_mask.sum(dim=1).tolist()
    neg_txt_seq_lens = neg_prompt_mask.sum(dim=1).tolist()

    # 4. Denoising loop with standard CFG
    print(f"  [4] Denoising ({STEPS} steps, standard CFG, cfg_scale={CFG_SCALE})...")
    for i, t in enumerate(timesteps):
        t_step = time.time()

        timestep = t.expand(latents.shape[0]).to(dtype)

        # Conditional prediction
        with pipe.transformer.cache_context("cond"):
            cond_output = pipe.transformer(
                hidden_states=latents,
                timestep=timestep / 1000,
                guidance=None,
                encoder_hidden_states_mask=prompt_mask,
                encoder_hidden_states=prompt_embeds,
                img_shapes=img_shapes,
                txt_seq_lens=txt_seq_lens,
                return_dict=False,
            )[0]

        # Unconditional prediction
        with pipe.transformer.cache_context("uncond"):
            uncond_output = pipe.transformer(
                hidden_states=latents,
                timestep=timestep / 1000,
                guidance=None,
                encoder_hidden_states_mask=neg_prompt_mask,
                encoder_hidden_states=neg_prompt_embeds,
                img_shapes=img_shapes,
                txt_seq_lens=neg_txt_seq_lens,
                return_dict=False,
            )[0]

        # Standard CFG (NO renormalization, matching our CUDA)
        noise_pred = uncond_output + CFG_SCALE * (cond_output - uncond_output)

        # Euler step (same as scheduler.step but we do it explicitly)
        sigma = sigmas_tensor[i]
        sigma_next = sigmas_tensor[i + 1]
        dt = sigma_next - sigma

        latents_f32 = latents.to(torch.float32)
        latents = (latents_f32 + dt * noise_pred.to(torch.float32)).to(dtype)

        elapsed = time.time() - t_step
        print(f"      Step {i+1}/{STEPS}: sigma={sigma:.4f} -> {sigma_next:.4f}, "
              f"dt={dt:.4f}, time={elapsed:.1f}s")

    # 5. Unpack latents and decode with VAE
    print("  [5] Unpacking latents and running VAE decode...")
    latents = pipe._unpack_latents(latents, HEIGHT, WIDTH, pipe.vae_scale_factor)
    latents = latents.to(pipe.vae.dtype)
    print(f"      Unpacked latents: {latents.shape}")

    # Denormalize: latents = latents / (1/std) + mean = latents * std + mean
    latents_mean = (
        torch.tensor(pipe.vae.config.latents_mean)
        .view(1, pipe.vae.config.z_dim, 1, 1, 1)
        .to(latents.device, latents.dtype)
    )
    latents_std = 1.0 / torch.tensor(pipe.vae.config.latents_std).view(
        1, pipe.vae.config.z_dim, 1, 1, 1
    ).to(latents.device, latents.dtype)
    latents = latents / latents_std + latents_mean
    print(f"      After denorm: min={latents.min():.4f}, max={latents.max():.4f}")

    t0 = time.time()
    decoded = pipe.vae.decode(latents, return_dict=False)[0][:, :, 0]
    t1 = time.time()
    print(f"      VAE decode took {t1-t0:.1f}s")
    print(f"      Decoded shape: {decoded.shape}")

    # Post-process to PIL
    image = pipe.image_processor.postprocess(decoded, output_type="pil")[0]
    return image


def run_stepwise(match_cuda=False):
    """Run components step by step with detailed comparisons."""
    import torch

    print("\n=== Stepwise Component Comparison ===")

    # Step 1: Compare sigma schedules
    compare_sigmas(match_cuda)

    # Step 2: Text encoder comparison
    print("\n--- Text Encoder ---")
    print("  Loading text encoder and tokenizer...")
    t0 = time.time()

    from transformers import Qwen2Tokenizer, Qwen2_5_VLForConditionalGeneration

    tokenizer = Qwen2Tokenizer.from_pretrained(MODEL_DIR, subfolder="tokenizer")
    text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        os.path.join(MODEL_DIR, "text_encoder"),
        torch_dtype=torch.bfloat16,
    )
    text_encoder.eval()
    t1 = time.time()
    print(f"  Loaded in {t1-t0:.1f}s")

    # Encode prompt using same template as pipeline
    template = "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
    drop_idx = 34

    for prompt_text in [PROMPT, ""]:
        txt = template.format(prompt_text)
        tokens = tokenizer(
            [txt], max_length=1024 + drop_idx, padding=True,
            truncation=True, return_tensors="pt"
        )
        print(f"\n  Prompt: '{prompt_text}'")
        print(f"    Token IDs ({len(tokens.input_ids[0])}): {tokens.input_ids[0][:20].tolist()}...")

        with torch.no_grad():
            out = text_encoder(
                input_ids=tokens.input_ids,
                attention_mask=tokens.attention_mask,
                output_hidden_states=True,
            )
        hidden = out.hidden_states[-1]
        print(f"    Hidden states shape: {hidden.shape}")

        # Extract masked hidden and drop system prompt
        bool_mask = tokens.attention_mask.bool()
        selected = hidden[bool_mask]
        selected = selected[drop_idx:]
        print(f"    After drop (drop_idx={drop_idx}): {selected.shape}")
        print(f"    min={selected.min():.4f}, max={selected.max():.4f}, "
              f"mean={selected.mean():.4f}, std={selected.std():.4f}")

        # Compare with our CUDA output if available
        suffix = "cond" if prompt_text == PROMPT else "uncond"
        dump_path = f"/home/parsa/qwen-image.cu/text_encoder_{suffix}.bin"
        if os.path.exists(dump_path):
            our_data = np.fromfile(dump_path, dtype=np.float32)
            ref_data = selected.float().numpy().flatten()
            if len(our_data) == len(ref_data):
                mse = np.mean((our_data - ref_data) ** 2)
                print(f"    vs our CUDA: MSE={mse:.6e}, "
                      f"max_diff={np.max(np.abs(our_data - ref_data)):.6e}")
            else:
                print(f"    Size mismatch: ours={len(our_data)}, ref={len(ref_data)}")
        else:
            print(f"    (No CUDA dump at {dump_path} for comparison)")

    # Free text encoder
    del text_encoder, tokenizer
    import gc; gc.collect()
    print("\n  Freed text encoder")

    print("\n--- Remaining steps (transformer + VAE) require full pipeline ---")
    print("  Use --mode full to run the complete pipeline comparison.")


def main():
    parser = argparse.ArgumentParser(description="Compare CUDA pipeline with diffusers reference")
    parser.add_argument("--mode", choices=["full", "stepwise", "sigmas_only"],
                        default="sigmas_only",
                        help="Comparison mode (default: sigmas_only)")
    parser.add_argument("--match-cuda", action="store_true",
                        help="Modify diffusers behavior to match our CUDA impl "
                             "(fixed shift, standard CFG without renormalization)")
    args = parser.parse_args()

    print("=" * 60)
    print("  Qwen Image 2512 - Full Pipeline Comparison")
    print("=" * 60)
    print(f"  Mode: {args.mode}")
    print(f"  Match CUDA: {args.match_cuda}")
    print(f"  Model dir: {MODEL_DIR}")
    print(f"  Our output: {OUR_OUTPUT}")
    print(f"  Reference output: {REFERENCE_OUTPUT}")

    if args.mode == "sigmas_only":
        compare_sigmas(args.match_cuda)

    elif args.mode == "stepwise":
        run_stepwise(args.match_cuda)

    elif args.mode == "full":
        run_full_pipeline(args.match_cuda)

    print("\nDone.")


if __name__ == "__main__":
    main()
