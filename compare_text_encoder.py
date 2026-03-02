#!/usr/bin/env python3
"""Compare our text encoder output against PyTorch reference.

Uses the Qwen2.5-VL model's built-in forward pass with output_hidden_states.
"""
import numpy as np
import torch
import os

MODEL_DIR = "/home/parsa/qwen-image.cu/Qwen-Image-2512"

def main():
    from transformers import AutoTokenizer, Qwen2_5_VLForConditionalGeneration

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(MODEL_DIR, "tokenizer"), trust_remote_code=True
    )

    # Build prompt tokens (matching our C++ code)
    prompt = "a red rose"
    im_start = 151644
    im_end = 151645

    user_tokens = tokenizer.encode("user\n" + prompt, add_special_tokens=False)
    newline_tokens = tokenizer.encode("\n", add_special_tokens=False)
    assistant_tokens = tokenizer.encode("assistant\n", add_special_tokens=False)

    cond_ids = [im_start] + user_tokens + [im_end] + newline_tokens + [im_start] + assistant_tokens
    uncond_user = tokenizer.encode("user\n", add_special_tokens=False)
    uncond_ids = [im_start] + uncond_user + [im_end] + newline_tokens + [im_start] + assistant_tokens

    print(f"Cond tokens ({len(cond_ids)}): {cond_ids}")
    print(f"Uncond tokens ({len(uncond_ids)}): {uncond_ids}")

    # Load model
    print("\nLoading Qwen2.5-VL (BF16, CPU)...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        os.path.join(MODEL_DIR, "text_encoder"),
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )
    model.eval()

    def encode_tokens(token_ids, name):
        """Run text encoder forward pass using the model's own method."""
        input_ids = torch.tensor([token_ids], dtype=torch.long)
        seq_len = len(token_ids)

        # For text-only, position_ids should be [batch, seq_len] sequential
        # The Qwen2.5-VL model handles M-RoPE internally
        position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)

        # For M-RoPE, need [batch, 3, seq_len] — all 3 axes use same positions for text-only
        # But the model's forward handles this via rope_deltas if we pass input_ids
        # Actually, let's use the model's language_model directly with proper args

        lm = model.model.language_model if hasattr(model.model, 'language_model') else model.model

        with torch.no_grad():
            # Use the Qwen2Model (language model) forward directly
            # It needs input_ids and will compute position_ids and rope internally
            # But Qwen2_5_VLModel wraps it - let's use the language model's forward

            # The language model is a Qwen2Model from transformers
            # Its forward signature: input_ids, attention_mask, position_ids, ...
            # For M-RoPE in Qwen2.5-VL, position_ids needs special handling

            # Try using the full Qwen2_5_VLForConditionalGeneration forward
            # This handles rope_deltas and M-RoPE properly
            outputs = model(
                input_ids=input_ids,
                output_hidden_states=True,
                return_dict=True,
            )

            # The last hidden state from the language model
            # outputs.hidden_states is a tuple of hidden states from all layers
            hidden = outputs.hidden_states[-1]  # [batch, seq_len, hidden_dim]

        result = hidden.float().numpy()[0]
        print(f"  {name} output: shape={result.shape}")
        print(f"    min={result.min():.4f}, max={result.max():.4f}, "
              f"mean={result.mean():.4f}, std={result.std():.4f}")
        return result

    # Run both
    print("\nRunning cond encoding...")
    cond_out = encode_tokens(cond_ids, "cond")

    print("\nRunning uncond encoding...")
    uncond_out = encode_tokens(uncond_ids, "uncond")

    # Save reference
    cond_out.tofile("/home/parsa/qwen-image.cu/ref_cond_context.bin")
    uncond_out.tofile("/home/parsa/qwen-image.cu/ref_uncond_context.bin")
    print(f"\nSaved ref_cond_context.bin ({cond_out.shape})")
    print(f"Saved ref_uncond_context.bin ({uncond_out.shape})")

    # Compare with CUDA output if available
    cuda_cond_path = "/home/parsa/qwen-image.cu/stable-diffusion.cpp/build/cuda_cond_context.bin"
    cuda_uncond_path = "/home/parsa/qwen-image.cu/stable-diffusion.cpp/build/cuda_uncond_context.bin"

    if os.path.exists(cuda_cond_path):
        print("\n=== Comparing with CUDA output ===")
        cuda_cond = np.fromfile(cuda_cond_path, dtype=np.float32).reshape(cond_out.shape)
        cuda_uncond = np.fromfile(cuda_uncond_path, dtype=np.float32).reshape(uncond_out.shape)

        for name, ref, cuda in [("cond", cond_out, cuda_cond), ("uncond", uncond_out, cuda_uncond)]:
            diff = np.abs(ref - cuda)
            rel_diff = diff / (np.abs(ref) + 1e-8)
            mse = np.mean((ref - cuda) ** 2)
            signal_power = np.mean(ref ** 2)
            snr = 10 * np.log10(signal_power / mse) if mse > 0 else float('inf')
            print(f"\n  {name}:")
            print(f"    Abs diff: mean={diff.mean():.6f}, max={diff.max():.4f}")
            print(f"    Rel diff: mean={rel_diff.mean():.6f}, max={rel_diff.max():.4f}")
            print(f"    SNR: {snr:.2f} dB")
            print(f"    First 5 ref:  {ref.flat[:5]}")
            print(f"    First 5 cuda: {cuda.flat[:5]}")
    else:
        print(f"\nCUDA output not found at {cuda_cond_path}")
        print("Run: DUMP_CONTEXT=1 ./qwen-image --model-dir ./Qwen-Image-2512/ -p 'a red rose' --steps 1 -o /dev/null")

if __name__ == '__main__':
    main()
