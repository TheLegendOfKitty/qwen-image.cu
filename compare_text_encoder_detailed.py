#!/usr/bin/env python3
"""Detailed layer-by-layer comparison of text encoder intermediates.

Uses PyTorch hooks to capture internal activations at every stage of layer 0
of the Qwen2.5-VL text encoder, and saves them as binary files for comparison
against the CUDA implementation.

Captured intermediates (all for layer 0, COND tokens):
  1. post_embedding      - After embedding lookup, before any layers
  2. post_input_norm     - After input_layernorm (RMSNorm)
  3. post_q_proj         - After Q linear projection (before reshape)
  4. post_k_proj         - After K linear projection (before reshape)
  5. post_v_proj         - After V linear projection (before reshape)
  6. post_q_rope         - After RoPE applied to Q (shape: [batch, heads, seq, head_dim])
  7. post_k_rope         - After RoPE applied to K (shape: [batch, kv_heads, seq, head_dim])
  8. post_attn_out       - After attention (before o_proj), shape [batch, seq, hidden]
  9. post_o_proj         - After o_proj linear
 10. post_attn_residual  - After attention residual add
 11. post_post_norm      - After post_attention_layernorm
 12. post_mlp            - After MLP output
 13. post_mlp_residual   - After MLP residual add (= layer 0 output)
 14. rope_cos            - RoPE cosine values [3, batch, seq, head_dim]
 15. rope_sin            - RoPE sine values [3, batch, seq, head_dim]

All saved as float32 binary files under debug_dumps/.
"""
import numpy as np
import torch
import os
import sys

MODEL_DIR = "/home/parsa/qwen-image.cu/Qwen-Image-2512"
DUMP_DIR = "/home/parsa/qwen-image.cu/debug_dumps"


def main():
    from transformers import AutoTokenizer, Qwen2_5_VLForConditionalGeneration

    os.makedirs(DUMP_DIR, exist_ok=True)

    # ---- Tokenize ----
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(MODEL_DIR, "tokenizer"), trust_remote_code=True
    )

    prompt = "a red rose"
    im_start = 151644
    im_end = 151645

    user_tokens = tokenizer.encode("user\n" + prompt, add_special_tokens=False)
    newline_tokens = tokenizer.encode("\n", add_special_tokens=False)
    assistant_tokens = tokenizer.encode("assistant\n", add_special_tokens=False)

    cond_ids = (
        [im_start] + user_tokens + [im_end] + newline_tokens
        + [im_start] + assistant_tokens
    )

    print(f"Cond tokens ({len(cond_ids)}): {cond_ids}")

    # ---- Load model ----
    print("\nLoading Qwen2.5-VL (BF16, CPU)...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        os.path.join(MODEL_DIR, "text_encoder"),
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )
    model.eval()

    # Shorthand references
    lm = model.model.language_model   # Qwen2_5_VLTextModel
    layer0 = lm.layers[0]             # Qwen2_5_VLDecoderLayer
    attn0 = layer0.self_attn          # Qwen2_5_VLAttention

    # ---- Storage for hooked values ----
    captures = {}

    def save(name, tensor):
        """Store a float32 copy of a tensor."""
        captures[name] = tensor.detach().float().cpu()

    # ---- Register hooks ----
    hooks = []

    # 1. Post-embedding: hook the embed_tokens output
    def hook_embed(module, input, output):
        save("post_embedding", output)
    hooks.append(lm.embed_tokens.register_forward_hook(hook_embed))

    # 2. Post-input-layernorm (layer 0)
    def hook_input_norm(module, input, output):
        save("post_input_norm", output)
    hooks.append(layer0.input_layernorm.register_forward_hook(hook_input_norm))

    # 3-5. Q/K/V projections (layer 0)
    def hook_q_proj(module, input, output):
        save("post_q_proj", output)
    hooks.append(attn0.q_proj.register_forward_hook(hook_q_proj))

    def hook_k_proj(module, input, output):
        save("post_k_proj", output)
    hooks.append(attn0.k_proj.register_forward_hook(hook_k_proj))

    def hook_v_proj(module, input, output):
        save("post_v_proj", output)
    hooks.append(attn0.v_proj.register_forward_hook(hook_v_proj))

    # 6-9. We need to monkey-patch the attention forward to capture post-RoPE
    # and post-attention intermediate values, since those happen inside the method.
    _orig_attn_forward = attn0.forward

    def patched_attn_forward(hidden_states, attention_mask=None, position_ids=None,
                             past_key_values=None, output_attentions=False,
                             use_cache=False, cache_position=None,
                             position_embeddings=None, **kwargs):
        bsz, q_len, _ = hidden_states.size()

        query_states = attn0.q_proj(hidden_states)
        key_states = attn0.k_proj(hidden_states)
        value_states = attn0.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, -1, attn0.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, -1, attn0.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, -1, attn0.head_dim).transpose(1, 2)

        # Save pre-RoPE Q/K in [batch, heads, seq, head_dim] layout
        save("pre_rope_q", query_states)
        save("pre_rope_k", key_states)
        save("pre_rope_v", value_states)

        cos, sin = position_embeddings
        save("rope_cos", cos)
        save("rope_sin", sin)

        from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import apply_multimodal_rotary_pos_emb
        query_states, key_states = apply_multimodal_rotary_pos_emb(
            query_states, key_states, cos, sin,
            attn0.config.rope_parameters["mrope_section"]
        )
        save("post_q_rope", query_states)
        save("post_k_rope", key_states)

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(
                key_states, value_states, attn0.layer_idx, cache_kwargs
            )

        from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
            ALL_ATTENTION_FUNCTIONS, eager_attention_forward
        )
        attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
            attn0.config._attn_implementation, eager_attention_forward
        )

        attn_output, attn_weights = attention_interface(
            attn0,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0,
            scaling=attn0.scaling,
            sliding_window=attn0.sliding_window,
            position_ids=position_ids,
            **kwargs,
        )

        # attn_output: [batch, heads, seq, head_dim] -> [batch, seq, hidden]
        save("post_attn_raw", attn_output)  # before reshape
        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
        save("post_attn_out", attn_output)  # after reshape, before o_proj

        attn_output = attn0.o_proj(attn_output)
        save("post_o_proj", attn_output)

        return attn_output, attn_weights

    attn0.forward = patched_attn_forward

    # 10. Post-attention residual: hook the full decoder layer to capture after residual add
    # We also need post_attention_layernorm output and MLP output
    _orig_layer0_forward = layer0.forward

    def patched_layer0_forward(hidden_states, attention_mask=None, position_ids=None,
                               past_key_values=None, output_attentions=False,
                               use_cache=False, cache_position=None,
                               position_embeddings=None, **kwargs):
        residual = hidden_states

        hidden_states = layer0.input_layernorm(hidden_states)
        # input_norm hook already captures this

        hidden_states, self_attn_weights = layer0.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states
        save("post_attn_residual", hidden_states)

        residual = hidden_states
        hidden_states = layer0.post_attention_layernorm(hidden_states)
        save("post_post_norm", hidden_states)

        hidden_states = layer0.mlp(hidden_states)
        save("post_mlp", hidden_states)

        hidden_states = residual + hidden_states
        save("post_mlp_residual", hidden_states)

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        return outputs

    layer0.forward = patched_layer0_forward

    # 14. Final norm output
    def hook_final_norm(module, input, output):
        save("post_final_norm", output)
    hooks.append(lm.norm.register_forward_hook(hook_final_norm))

    # ---- Run forward pass ----
    print("\nRunning forward pass with hooks...")
    input_ids = torch.tensor([cond_ids], dtype=torch.long)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            output_hidden_states=True,
            return_dict=True,
        )

    # ---- Remove hooks ----
    for h in hooks:
        h.remove()

    # ---- Save all captures ----
    print(f"\nSaving {len(captures)} intermediate tensors to {DUMP_DIR}/")
    for name, tensor in captures.items():
        arr = tensor.numpy()
        path = os.path.join(DUMP_DIR, f"ref_{name}.bin")
        arr.tofile(path)
        # Print shape and first values
        flat = arr.flatten()
        first_vals = ", ".join(f"{v:.6f}" for v in flat[:8])
        shape_str = str(list(arr.shape))
        print(f"  {name:25s}  shape={shape_str:30s}  "
              f"min={flat.min():10.4f}  max={flat.max():10.4f}  "
              f"mean={flat.mean():10.4f}  first8=[{first_vals}]")

    # ---- Also save shape metadata for easy loading ----
    meta_path = os.path.join(DUMP_DIR, "shapes.txt")
    with open(meta_path, "w") as f:
        for name, tensor in captures.items():
            shape_str = ",".join(str(s) for s in tensor.shape)
            f.write(f"{name} {shape_str} float32\n")
    print(f"\nShape metadata saved to {meta_path}")

    # ---- Print detailed analysis ----
    print("\n" + "=" * 80)
    print("DETAILED ANALYSIS OF LAYER 0 INTERMEDIATES")
    print("=" * 80)

    # RoPE analysis
    if "rope_cos" in captures and "rope_sin" in captures:
        cos = captures["rope_cos"].numpy()
        sin = captures["rope_sin"].numpy()
        print(f"\nRoPE cos shape: {cos.shape}")
        print(f"RoPE sin shape: {sin.shape}")
        # For text-only, all 3 axes should be identical
        # cos shape is [3, batch, seq, head_dim] or [batch, seq, head_dim]
        if cos.ndim == 4:
            print(f"  Axis 0 vs 1 max diff: {np.abs(cos[0] - cos[1]).max():.8f}")
            print(f"  Axis 0 vs 2 max diff: {np.abs(cos[0] - cos[2]).max():.8f}")
            print(f"  cos[0, 0, 0, :8] = {cos[0, 0, 0, :8]}")
            print(f"  cos[0, 0, 1, :8] = {cos[0, 0, 1, :8]}")
            print(f"  sin[0, 0, 0, :8] = {sin[0, 0, 0, :8]}")
            print(f"  sin[0, 0, 1, :8] = {sin[0, 0, 1, :8]}")
        elif cos.ndim == 3:
            print(f"  cos[0, 0, :8] = {cos[0, 0, :8]}")
            print(f"  cos[0, 1, :8] = {cos[0, 1, :8]}")
            print(f"  sin[0, 0, :8] = {sin[0, 0, :8]}")
            print(f"  sin[0, 1, :8] = {sin[0, 1, :8]}")

    # Post-RoPE Q analysis -- compare interleaved vs paired layout
    if "post_q_rope" in captures:
        q = captures["post_q_rope"].numpy()
        print(f"\nPost-RoPE Q shape: {q.shape}")
        print(f"  Q[0, 0, 0, :16] (head 0, pos 0) = {q[0, 0, 0, :16]}")
        print(f"  Q[0, 0, 1, :16] (head 0, pos 1) = {q[0, 0, 1, :16]}")

    # Compare Q before/after RoPE
    if "pre_rope_q" in captures and "post_q_rope" in captures:
        pre = captures["pre_rope_q"].numpy()
        post = captures["post_q_rope"].numpy()
        diff = np.abs(pre - post)
        print(f"\n  Q RoPE diff: mean={diff.mean():.6f}, max={diff.max():.4f}")
        print(f"  Pre-RoPE  Q[0,0,0,:8] = {pre[0, 0, 0, :8]}")
        print(f"  Post-RoPE Q[0,0,0,:8] = {post[0, 0, 0, :8]}")

    # Attention output analysis
    if "post_attn_out" in captures:
        attn = captures["post_attn_out"].numpy()
        print(f"\nPost-attention output shape: {attn.shape}")
        print(f"  attn[0, 0, :8] = {attn[0, 0, :8]}")

    # Residual analysis
    if "post_embedding" in captures and "post_attn_residual" in captures:
        emb = captures["post_embedding"].numpy()
        res = captures["post_attn_residual"].numpy()
        diff = np.abs(emb - res)
        print(f"\nEmbedding vs post-attn-residual:")
        print(f"  Diff: mean={diff.mean():.6f}, max={diff.max():.4f}")

    # ---- Compare with CUDA output if debug dumps exist ----
    print("\n" + "=" * 80)
    print("COMPARISON WITH CUDA DEBUG DUMPS (if available)")
    print("=" * 80)

    cuda_dump_dir = "/home/parsa/qwen-image.cu/debug_dumps"
    for name in captures:
        cuda_path = os.path.join(cuda_dump_dir, f"cuda_{name}.bin")
        if os.path.exists(cuda_path):
            ref = captures[name].numpy()
            cuda = np.fromfile(cuda_path, dtype=np.float32)
            if cuda.size == ref.size:
                cuda = cuda.reshape(ref.shape)
                diff = np.abs(ref - cuda)
                rel_diff = diff / (np.abs(ref) + 1e-8)
                mse = np.mean((ref - cuda) ** 2)
                signal_power = np.mean(ref ** 2)
                snr = 10 * np.log10(signal_power / mse) if mse > 0 else float('inf')
                print(f"\n  {name}:")
                print(f"    Abs diff: mean={diff.mean():.8f}, max={diff.max():.6f}")
                print(f"    Rel diff: mean={rel_diff.mean():.8f}, max={rel_diff.max():.6f}")
                print(f"    SNR: {snr:.2f} dB")
                ref_flat = ref.flatten()
                cuda_flat = cuda.flatten()
                print(f"    First 5 ref:  {ref_flat[:5]}")
                print(f"    First 5 cuda: {cuda_flat[:5]}")
            else:
                print(f"\n  {name}: SIZE MISMATCH ref={ref.size} vs cuda={cuda.size}")
        # else: silently skip missing files

    # ---- Also save the final output for legacy comparison ----
    final = outputs.hidden_states[-1].float().numpy()[0]
    final.tofile(os.path.join(DUMP_DIR, "ref_final_output.bin"))
    print(f"\nFinal output shape: {final.shape}")
    print(f"  First 8 values: {final.flatten()[:8]}")
    print(f"\nAll reference dumps saved to {DUMP_DIR}/")
    print("To compare, add debug dumps in your CUDA code saving tensors as:")
    print(f"  {DUMP_DIR}/cuda_<name>.bin")
    print("where <name> is one of:", ", ".join(captures.keys()))


if __name__ == "__main__":
    main()
