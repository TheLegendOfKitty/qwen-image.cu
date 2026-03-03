#include "text_encoder.h"
#include "cuda_kernels.cuh"
#include "rope.h"
#include <vector>
#include <cstdio>
#include <cmath>

// Debug dump helper: download BF16 GPU tensor to FP32 file
static void dump_bf16_as_f32(const void* gpu_data, int64_t n_elems, const char* path) {
    std::vector<__nv_bfloat16> bf16(n_elems);
    cudaMemcpy(bf16.data(), gpu_data, n_elems * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
    std::vector<float> f32(n_elems);
    for (int64_t i = 0; i < n_elems; i++) f32[i] = __bfloat162float(bf16[i]);
    FILE* f = fopen(path, "wb");
    fwrite(f32.data(), sizeof(float), n_elems, f);
    fclose(f);
    fprintf(stderr, "  Dumped %s (%lld floats)\n", path, (long long)n_elems);
}

static void dump_f32(const void* gpu_data, int64_t n_elems, const char* path) {
    std::vector<float> f32(n_elems);
    cudaMemcpy(f32.data(), gpu_data, n_elems * sizeof(float), cudaMemcpyDeviceToHost);
    FILE* f = fopen(path, "wb");
    fwrite(f32.data(), sizeof(float), n_elems, f);
    fclose(f);
    fprintf(stderr, "  Dumped %s (%lld floats)\n", path, (long long)n_elems);
}

__global__ void apply_mrope_kernel_fp32(
    float* q, float* k,
    const float* cos_vals, const float* sin_vals,
    int seq_len, int n_heads, int n_kv_heads, int head_dim, int total_pairs) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int n_total_heads = n_heads + n_kv_heads;
    int64_t total = (int64_t)seq_len * n_total_heads * (head_dim / 2);
    if (idx >= total) return;

    int pair = idx % (head_dim / 2);
    int64_t rest = idx / (head_dim / 2);
    int head = rest % n_total_heads;
    int pos = rest / n_total_heads;

    float* tensor;
    int tensor_head;
    int tensor_n_heads;
    if (head < n_heads) {
        tensor = q;
        tensor_head = head;
        tensor_n_heads = n_heads;
    } else {
        tensor = k;
        tensor_head = head - n_heads;
        tensor_n_heads = n_kv_heads;
    }

    int half_dim = head_dim / 2;
    int64_t base = ((int64_t)pos * tensor_n_heads + tensor_head) * head_dim;
    float x0 = tensor[base + pair];
    float x1 = tensor[base + pair + half_dim];

    float c = cos_vals[pos * total_pairs + pair];
    float s = sin_vals[pos * total_pairs + pair];

    tensor[base + pair] = x0 * c - x1 * s;
    tensor[base + pair + half_dim] = x1 * c + x0 * s;
}

__global__ void repeat_kv_fp32_kernel(const float* in, float* out,
                                      int S, int H_kv, int D, int repeats) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = (int64_t)S * H_kv * repeats * D;
    if (idx >= total) return;

    int d = idx % D;
    int64_t rest = idx / D;
    int h = rest % (H_kv * repeats);
    int s = rest / (H_kv * repeats);
    int h_kv = h / repeats;

    out[idx] = in[((int64_t)s * H_kv + h_kv) * D + d];
}

static void repeat_kv_fp32(const float* in, float* out,
                           int S, int H_kv, int D, int repeats) {
    int64_t total = (int64_t)S * H_kv * repeats * D;
    int block = 256;
    int grid = (int)((total + block - 1) / block);
    repeat_kv_fp32_kernel<<<grid, block>>>(in, out, S, H_kv, D, repeats);
}

__global__ void textenc_transpose_shd_to_hsd_fp32(const float* src, float* dst,
                                                  int S, int H, int D) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = (int64_t)S * H * D;
    if (idx >= total) return;
    int d = idx % D;
    int64_t rest = idx / D;
    int h = rest % H;
    int s = rest / H;
    dst[((int64_t)h * S + s) * D + d] = src[idx];
}

__global__ void textenc_transpose_hsd_to_shd_flat_fp32(const float* src, float* dst,
                                                       int H, int S, int D) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = (int64_t)H * S * D;
    if (idx >= total) return;
    int d = idx % D;
    int64_t rest = idx / D;
    int s = rest % S;
    int h = rest / S;
    dst[(int64_t)s * H * D + h * D + d] = src[idx];
}

__global__ void add_inplace_fp32_kernel(float* a, const float* b, int64_t n) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) a[idx] += b[idx];
}

static void add_inplace_fp32(float* a, const float* b, int64_t n) {
    int block = 256;
    int grid = (int)((n + block - 1) / block);
    add_inplace_fp32_kernel<<<grid, block>>>(a, b, n);
}

__global__ void mul_tensors_fp32_kernel(const float* a, const float* b, float* out, int64_t n) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = a[idx] * b[idx];
}

static void mul_tensors_fp32(const float* a, const float* b, float* out, int64_t n) {
    int block = 256;
    int grid = (int)((n + block - 1) / block);
    mul_tensors_fp32_kernel<<<grid, block>>>(a, b, out, n);
}

Tensor text_encoder_forward(const TextEncoderWeights& w,
                            const std::vector<int32_t>& token_ids) {
    const int seq_len = (int)token_ids.size();
    const int hidden_size = 3584;
    const int n_heads = 28;
    const int n_kv_heads = 4;
    const int head_dim = 128;
    const int intermediate_size = 18944;
    const int kv_repeats = n_heads / n_kv_heads; // 7

    fprintf(stderr, "Text encoder forward: seq_len=%d\n", seq_len);

    // Upload token IDs to GPU
    Tensor token_ids_gpu = Tensor::alloc({seq_len}, DType::FP32); // using FP32 as int32 container
    token_ids_gpu.from_host(token_ids.data(), seq_len * sizeof(int32_t));

    // Embedding lookup: [seq_len, 3584]
    Tensor hidden_bf16 = Tensor::alloc({seq_len, hidden_size}, DType::BF16);
    embedding_lookup(
        (__nv_bfloat16*)w.embed_tokens.data,
        (int32_t*)token_ids_gpu.data,
        (__nv_bfloat16*)hidden_bf16.data,
        seq_len, hidden_size);
    token_ids_gpu.free_data();

    Tensor hidden = Tensor::alloc({seq_len, hidden_size}, DType::FP32);
    bf16_to_fp32((__nv_bfloat16*)hidden_bf16.data, (float*)hidden.data, (int64_t)seq_len * hidden_size);
    hidden_bf16.free_data();

    // Precompute M-RoPE cos/sin on CPU
    auto rope_data = RoPE::gen_text_encoder_rope(seq_len, 1e6f, {16, 24, 24});
    int total_pairs = 64; // 16 + 24 + 24
    // rope_data: first seq_len*64 are cos, next seq_len*64 are sin
    Tensor cos_gpu = Tensor::alloc({seq_len * total_pairs}, DType::FP32);
    Tensor sin_gpu = Tensor::alloc({seq_len * total_pairs}, DType::FP32);
    cos_gpu.from_host(rope_data.data(), seq_len * total_pairs * sizeof(float));
    sin_gpu.from_host(rope_data.data() + seq_len * total_pairs, seq_len * total_pairs * sizeof(float));

    int64_t max_weight_numel = 0;
    auto grow_weight_scratch = [&](const Tensor& weight) {
        if (weight.numel() > max_weight_numel) max_weight_numel = weight.numel();
    };
    for (const auto& l : w.layers) {
        grow_weight_scratch(l.q_proj_weight);
        grow_weight_scratch(l.k_proj_weight);
        grow_weight_scratch(l.v_proj_weight);
        grow_weight_scratch(l.o_proj_weight);
        grow_weight_scratch(l.gate_proj_weight);
        grow_weight_scratch(l.up_proj_weight);
        grow_weight_scratch(l.down_proj_weight);
    }

    Tensor gemm_scratch = Tensor::alloc({max_weight_numel}, DType::FP32);

    // Allocate working buffers
    Tensor normed = Tensor::alloc({seq_len, hidden_size}, DType::FP32);
    Tensor q_buf = Tensor::alloc({seq_len, n_heads * head_dim}, DType::FP32);
    Tensor k_buf = Tensor::alloc({seq_len, n_kv_heads * head_dim}, DType::FP32);
    Tensor v_buf = Tensor::alloc({seq_len, n_kv_heads * head_dim}, DType::FP32);
    Tensor k_rep = Tensor::alloc({seq_len, n_heads * head_dim}, DType::FP32);
    Tensor v_rep = Tensor::alloc({seq_len, n_heads * head_dim}, DType::FP32);
    Tensor q_t = Tensor::alloc({n_heads, seq_len, head_dim}, DType::FP32);
    Tensor k_t = Tensor::alloc({n_heads, seq_len, head_dim}, DType::FP32);
    Tensor v_t = Tensor::alloc({n_heads, seq_len, head_dim}, DType::FP32);
    Tensor attn_t = Tensor::alloc({n_heads, seq_len, head_dim}, DType::FP32);
    Tensor attn_out = Tensor::alloc({seq_len, n_heads * head_dim}, DType::FP32);
    Tensor proj_out = Tensor::alloc({seq_len, hidden_size}, DType::FP32);
    Tensor gate_buf = Tensor::alloc({seq_len, intermediate_size}, DType::FP32);
    Tensor up_buf = Tensor::alloc({seq_len, intermediate_size}, DType::FP32);
    Tensor mlp_out = Tensor::alloc({seq_len, hidden_size}, DType::FP32);

    static bool dumped_once = false;
    bool dump_debug = (getenv("DUMP_TEXT_DEBUG") != nullptr) && !dumped_once;
    if (dump_debug) {
        dumped_once = true;
    }

    for (int layer = 0; layer < 28; layer++) {
        auto& l = w.layers[layer];

        if (dump_debug && layer == 0) {
            cudaDeviceSynchronize();
            dump_f32(hidden.data, (int64_t)seq_len * hidden_size, "debug_dumps/cuda_post_embedding.bin");
        }

        // 1. RMSNorm
        rms_norm_fp32((float*)hidden.data, (__nv_bfloat16*)l.input_layernorm_weight.data,
                      (float*)normed.data, seq_len, hidden_size, 1e-6f);

        if (dump_debug && layer == 0) {
            cudaDeviceSynchronize();
            dump_f32(normed.data, (int64_t)seq_len * hidden_size, "debug_dumps/cuda_post_input_norm.bin");
        }

        // 2. Q, K, V projections
        {
            Tensor q_2d = q_buf.view({seq_len, n_heads * head_dim});
            linear_forward_fp32in_bf16w_fp32out(normed, l.q_proj_weight, &l.q_proj_bias,
                                                q_2d, gemm_scratch);

            Tensor k_2d = k_buf.view({seq_len, n_kv_heads * head_dim});
            linear_forward_fp32in_bf16w_fp32out(normed, l.k_proj_weight, &l.k_proj_bias,
                                                k_2d, gemm_scratch);

            Tensor v_2d = v_buf.view({seq_len, n_kv_heads * head_dim});
            linear_forward_fp32in_bf16w_fp32out(normed, l.v_proj_weight, &l.v_proj_bias,
                                                v_2d, gemm_scratch);
        }

        if (dump_debug && layer == 0) {
            cudaDeviceSynchronize();
            dump_f32(q_buf.data, (int64_t)seq_len * n_heads * head_dim, "debug_dumps/cuda_post_q_proj.bin");
            dump_f32(k_buf.data, (int64_t)seq_len * n_kv_heads * head_dim, "debug_dumps/cuda_post_k_proj.bin");
            dump_f32(v_buf.data, (int64_t)seq_len * n_kv_heads * head_dim, "debug_dumps/cuda_post_v_proj.bin");
            dump_f32(cos_gpu.data, seq_len * total_pairs, "debug_dumps/cuda_rope_cos.bin");
            dump_f32(sin_gpu.data, seq_len * total_pairs, "debug_dumps/cuda_rope_sin.bin");
        }

        // 3. Apply M-RoPE to Q and K
        {
            int n_total_heads = n_heads + n_kv_heads;
            int64_t total = (int64_t)seq_len * n_total_heads * (head_dim / 2);
            int block = 256;
            int grid = (int)((total + block - 1) / block);
            apply_mrope_kernel_fp32<<<grid, block>>>(
                (float*)q_buf.data, (float*)k_buf.data,
                (float*)cos_gpu.data, (float*)sin_gpu.data,
                seq_len, n_heads, n_kv_heads, head_dim, total_pairs);
        }

        if (dump_debug && layer == 0) {
            cudaDeviceSynchronize();
            dump_f32(q_buf.data, (int64_t)seq_len * n_heads * head_dim, "debug_dumps/cuda_post_q_rope.bin");
            dump_f32(k_buf.data, (int64_t)seq_len * n_kv_heads * head_dim, "debug_dumps/cuda_post_k_rope.bin");
        }

        // 4. Repeat KV heads: 4 -> 28
        repeat_kv_fp32((float*)k_buf.data, (float*)k_rep.data, seq_len, n_kv_heads, head_dim, kv_repeats);
        repeat_kv_fp32((float*)v_buf.data, (float*)v_rep.data, seq_len, n_kv_heads, head_dim, kv_repeats);

        int64_t total_elems = (int64_t)seq_len * n_heads * head_dim;
        int block = 256;
        int grid = (int)((total_elems + block - 1) / block);
        textenc_transpose_shd_to_hsd_fp32<<<grid, block>>>((float*)q_buf.data, (float*)q_t.data, seq_len, n_heads, head_dim);
        textenc_transpose_shd_to_hsd_fp32<<<grid, block>>>((float*)k_rep.data, (float*)k_t.data, seq_len, n_heads, head_dim);
        textenc_transpose_shd_to_hsd_fp32<<<grid, block>>>((float*)v_rep.data, (float*)v_t.data, seq_len, n_heads, head_dim);

        float attn_scale = 1.0f / sqrtf((float)head_dim);
        attention_forward_fp32((float*)q_t.data, (float*)k_t.data, (float*)v_t.data,
                               (float*)attn_t.data, attn_scale, n_heads, seq_len, head_dim,
                               true /* causal */);

        textenc_transpose_hsd_to_shd_flat_fp32<<<grid, block>>>((float*)attn_t.data, (float*)attn_out.data,
                                                                n_heads, seq_len, head_dim);

        // 6. Output projection
        {
            if (dump_debug && layer == 0) {
                cudaDeviceSynchronize();
                dump_f32(attn_out.data, (int64_t)seq_len * hidden_size, "debug_dumps/cuda_post_attn_out.bin");
            }
            linear_forward_fp32in_bf16w_fp32out(attn_out, l.o_proj_weight, nullptr,
                                                proj_out, gemm_scratch);
        }

        if (dump_debug && layer == 0) {
            cudaDeviceSynchronize();
            dump_f32(proj_out.data, (int64_t)seq_len * hidden_size, "debug_dumps/cuda_post_o_proj.bin");
        }

        // 7. Residual add
        add_inplace_fp32((float*)hidden.data, (float*)proj_out.data, (int64_t)seq_len * hidden_size);

        if (dump_debug && layer == 0) {
            cudaDeviceSynchronize();
            dump_f32(hidden.data, (int64_t)seq_len * hidden_size, "debug_dumps/cuda_post_attn_residual.bin");
        }

        // 8. Post-attention RMSNorm
        rms_norm_fp32((float*)hidden.data, (__nv_bfloat16*)l.post_attention_layernorm_weight.data,
                      (float*)normed.data, seq_len, hidden_size, 1e-6f);

        // 9. SwiGLU MLP: gate = silu(gate_proj(x)) * up_proj(x), out = down_proj(gate)
        {
            linear_forward_fp32in_bf16w_fp32out(normed, l.gate_proj_weight, nullptr,
                                                gate_buf, gemm_scratch);
            linear_forward_fp32in_bf16w_fp32out(normed, l.up_proj_weight, nullptr,
                                                up_buf, gemm_scratch);

            silu_fp32((float*)gate_buf.data, (float*)gate_buf.data,
                      (int64_t)seq_len * intermediate_size);
            mul_tensors_fp32((float*)gate_buf.data, (float*)up_buf.data,
                             (float*)gate_buf.data, (int64_t)seq_len * intermediate_size);

            linear_forward_fp32in_bf16w_fp32out(gate_buf, l.down_proj_weight, nullptr,
                                                mlp_out, gemm_scratch);
        }

        // 10. Residual add
        add_inplace_fp32((float*)hidden.data, (float*)mlp_out.data, (int64_t)seq_len * hidden_size);

        if (dump_debug && layer == 0) {
            cudaDeviceSynchronize();
            dump_f32(hidden.data, (int64_t)seq_len * hidden_size, "debug_dumps/cuda_post_mlp_residual.bin");
        }

        if ((layer + 1) % 7 == 0)
            fprintf(stderr, "  Text encoder layer %d/28 done\n", layer + 1);
    }

    // Final RMSNorm
    rms_norm_fp32((float*)hidden.data, (__nv_bfloat16*)w.norm_weight.data,
                  (float*)normed.data, seq_len, hidden_size, 1e-6f);

    Tensor output = Tensor::alloc({1, seq_len, hidden_size}, DType::BF16);
    fp32_to_bf16((float*)normed.data, (__nv_bfloat16*)output.data, (int64_t)seq_len * hidden_size);

    // Free temp buffers
    hidden.free_data();
    cos_gpu.free_data();
    sin_gpu.free_data();
    gemm_scratch.free_data();
    normed.free_data();
    q_buf.free_data();
    k_buf.free_data();
    v_buf.free_data();
    k_rep.free_data();
    v_rep.free_data();
    q_t.free_data();
    k_t.free_data();
    v_t.free_data();
    attn_t.free_data();
    attn_out.free_data();
    proj_out.free_data();
    gate_buf.free_data();
    up_buf.free_data();
    mlp_out.free_data();

    fprintf(stderr, "Text encoder done: output %s\n", output.shape_str().c_str());
    CUDA_CHECK(cudaDeviceSynchronize());
    return output;
}
