#include "transformer.h"
#include "cuda_kernels.cuh"
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <string>
#include <vector>

static bool block_index_in_env_range(const char* env_name, int block_index) {
    const char* range = getenv(env_name);
    if (range == nullptr || range[0] == '\0') {
        return false;
    }
    int start = 0;
    int end = 0;
    if (sscanf(range, "%d:%d", &start, &end) != 2) {
        return false;
    }
    return block_index >= start && block_index <= end;
}

static int get_transformer_internal_dump_block() {
    const char* block = getenv("DUMP_TRANSFORMER_INTERNAL_BLOCK");
    if (block == nullptr || block[0] == '\0') {
        return 0;
    }
    return atoi(block);
}

// Helper: transpose [S, H, D] -> [H, S, D] on GPU (BF16)
__global__ void transpose_shd_to_hsd(const __nv_bfloat16* src, __nv_bfloat16* dst,
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

// Helper: transpose [H, S, D] -> [S, H*D] on GPU (BF16, concat heads)
__global__ void transpose_hsd_to_shd_flat(const __nv_bfloat16* src, __nv_bfloat16* dst,
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

// FP32 transpose: [S, H, D] -> [H, S, D]
__global__ void transpose_shd_to_hsd_fp32(const float* src, float* dst,
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

// FP32 transpose: [H, S, D] -> [S, H*D] (concat heads)
__global__ void transpose_hsd_to_shd_flat_fp32(const float* src, float* dst,
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

// Fused transpose [H, S1+S2, D] → split to txt[S1, H*D], img[S2, H*D]
__global__ void transpose_split_fused_fp32(
        const float* __restrict__ in,  // [H, S1+S2, D]
        float* __restrict__ txt,       // [S1, H*D]
        float* __restrict__ img,       // [S2, H*D]
        int S1, int S2, int H, int D) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int S = S1 + S2;
    int64_t total = (int64_t)S * H * D;
    if (idx >= total) return;
    // idx maps to input[h][s][d] = in[h * S * D + s * D + d]
    int d = idx % D;
    int64_t rest = idx / D;
    int s = rest % S;
    int h = rest / S;
    float val = in[idx];
    if (s < S1) {
        txt[(int64_t)s * H * D + h * D + d] = val;
    } else {
        img[(int64_t)(s - S1) * H * D + h * D + d] = val;
    }
}

// Fused concat [txt(S1,H*D), img(S2,H*D)] + transpose to [H, S1+S2, D]
// Output order: txt first, then img within each head
// Optionally also writes FP16 output (for V → direct FP16 in attention)
__global__ void concat_transpose_fused_fp32(
        const float* __restrict__ txt, const float* __restrict__ img,
        float* __restrict__ out,
        int S1, int S2, int H, int D,
        __half* __restrict__ out_fp16 = nullptr) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int S = S1 + S2;
    int64_t total = (int64_t)S * H * D;
    if (idx >= total) return;
    int d = idx % D;
    int64_t rest = idx / D;
    int s = rest % S;
    int h = rest / S;
    float val;
    if (s < S1) {
        val = txt[(int64_t)s * H * D + h * D + d];
    } else {
        val = img[(int64_t)(s - S1) * H * D + h * D + d];
    }
    out[idx] = val;
    if (out_fp16) out_fp16[idx] = __float2half(val);
}

Tensor transformer_forward(const TransformerWeights& w,
                           const Tensor& x_in,
                           float timestep_val,
                           const Tensor& context,
                           const Tensor& pe,
                           int H, int W,
                           CalibrationWriter* cal) {
    const int inner_dim = 3072; // 24 * 128
    const int n_heads = 24;
    const int head_dim = 128;
    const int patch_size = 2;
    const int in_channels = 64; // patchified: 16 * 2 * 2
    const int out_channels = 16;
    const int mlp_dim = 12288; // 4 * 3072

    // Pad input to patch size
    int H_pad = H + (patch_size - H % patch_size) % patch_size;
    int W_pad = W + (patch_size - W % patch_size) % patch_size;

    int h_patches = H_pad / patch_size;
    int w_patches = W_pad / patch_size;
    int n_img = h_patches * w_patches;

    int n_txt = (int)context.shape[1];
    int total_seq = n_txt + n_img;

    fprintf(stderr, "Transformer: n_img=%d, n_txt=%d, total=%d\n", n_img, n_txt, total_seq);

    // Patchify the input
    Tensor img_patches = Tensor::alloc({n_img, in_channels}, DType::BF16);

    if (H_pad != H || W_pad != W) {
        Tensor x_padded = Tensor::alloc({1, (int64_t)(x_in.shape[1]), H_pad, W_pad}, DType::BF16);
        x_padded.zero();
        for (int c = 0; c < (int)x_in.shape[1]; c++) {
            for (int h = 0; h < H; h++) {
                CUDA_CHECK(cudaMemcpy(
                    (__nv_bfloat16*)x_padded.data + (int64_t)c * H_pad * W_pad + h * W_pad,
                    (__nv_bfloat16*)x_in.data + (int64_t)c * H * W + h * W,
                    W * sizeof(__nv_bfloat16),
                    cudaMemcpyDeviceToDevice));
            }
        }
        patchify((__nv_bfloat16*)x_padded.data, (__nv_bfloat16*)img_patches.data,
                 1, (int)x_in.shape[1], H_pad, W_pad, patch_size, patch_size);
        x_padded.free_data();
    } else {
        patchify((__nv_bfloat16*)x_in.data, (__nv_bfloat16*)img_patches.data,
                 1, (int)x_in.shape[1], H, W, patch_size, patch_size);
    }

    // Scratch for quantized GEMM. Mode-aware: INT8 or INT4+SVD.
    // Use a block-level weight to detect mode (boundary layers may be BF16 in mixed-precision).
    QuantMode qmode = w.blocks[0].to_q_weight.mode;
    int svd_rank = (qmode == QuantMode::INT4_SVD) ? w.blocks[0].to_q_weight.svd_rank : 0;

    int64_t max_scratch_bytes = 0;
    auto grow_scratch = [&](int M, int K, int N) {
        int64_t bytes;
        if (qmode == QuantMode::INT4_SVD)
            bytes = int4_scratch_bytes(M, K, N, svd_rank);
        else
            bytes = int8_scratch_bytes(M, K, N);
        if (bytes > max_scratch_bytes) max_scratch_bytes = bytes;
    };

    // Timestep linears (M=1)
    grow_scratch(1, 256, inner_dim);
    grow_scratch(1, inner_dim, inner_dim);
    // Input projections
    grow_scratch(n_img, in_channels, inner_dim);
    grow_scratch(n_txt, 3584, inner_dim);
    // Output projections
    grow_scratch(n_img, inner_dim, 2 * inner_dim);  // norm_out_linear [6144, 3072]
    grow_scratch(n_img, inner_dim, in_channels);     // proj_out [64, 3072]
    // Block linears
    grow_scratch(1, inner_dim, 6 * inner_dim);       // modulation [18432, 3072]
    grow_scratch(n_img, inner_dim, inner_dim);       // attention q/k/v/out [3072, 3072]
    grow_scratch(n_txt, inner_dim, inner_dim);       // txt attention q/k/v/out
    grow_scratch(n_img, inner_dim, mlp_dim);         // img mlp fc1 [12288, 3072]
    grow_scratch(n_img, mlp_dim, inner_dim);         // img mlp fc2 [3072, 12288]
    grow_scratch(n_txt, inner_dim, mlp_dim);         // txt mlp fc1
    grow_scratch(n_txt, mlp_dim, inner_dim);         // txt mlp fc2

    // Allocate as raw bytes (use FP32 dtype for sizing: ceil(bytes/4) elements)
    Tensor gemm_scratch = Tensor::alloc({(max_scratch_bytes + 3) / 4}, DType::FP32);

    // 1. Timestep embedding: scalar -> [1, 256] sinusoidal -> linear -> silu -> linear -> [1, 3072]
    Tensor t_buf = Tensor::alloc({1}, DType::FP32);
    t_buf.from_host(&timestep_val);
    Tensor t_emb_sin = Tensor::alloc({1, 256}, DType::FP32);
    timestep_embedding_fp32((float*)t_buf.data, (float*)t_emb_sin.data, 1, 256, 10000.0f);
    t_buf.free_data();

    Tensor t_after_l1 = Tensor::alloc({1, inner_dim}, DType::FP32);
    {
        linear_forward_quantized(t_emb_sin, w.time_linear1_weight, &w.time_linear1_bias,
                            t_after_l1, gemm_scratch);
    }
    silu_fp32((float*)t_after_l1.data, (float*)t_after_l1.data, inner_dim);
    t_emb_sin.free_data();

    Tensor t_emb = Tensor::alloc({1, inner_dim}, DType::FP32);
    {
        linear_forward_quantized(t_after_l1, w.time_linear2_weight, &w.time_linear2_bias,
                            t_emb, gemm_scratch);
    }
    t_after_l1.free_data();

    // Debug dump infrastructure. The active step/branch is selected by main.cu.
    auto dump = [](const char* path, const void* gpu_data, int64_t n, bool is_bf16) {
        if (!getenv("DUMP_TRANSFORMER") || !getenv("DUMP_TRANSFORMER_ACTIVE")) return;
        std::vector<float> f32(n);
        if (is_bf16) {
            std::vector<__nv_bfloat16> bf16(n);
            cudaMemcpy(bf16.data(), gpu_data, n * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
            for (int64_t i = 0; i < n; i++) f32[i] = __bfloat162float(bf16[i]);
        } else {
            cudaMemcpy(f32.data(), gpu_data, n * sizeof(float), cudaMemcpyDeviceToHost);
        }
        std::string out_path = path;
        const char* step_env = getenv("DUMP_TRANSFORMER_ACTIVE_STEP");
        const char* ntxt_env = getenv("DUMP_TRANSFORMER_ACTIVE_NTXT");
        if ((step_env && step_env[0] != '\0') || (ntxt_env && ntxt_env[0] != '\0')) {
            std::string suffix;
            if (step_env && step_env[0] != '\0') {
                suffix += "_step";
                suffix += step_env;
            }
            if (ntxt_env && ntxt_env[0] != '\0') {
                suffix += "_ntxt";
                suffix += ntxt_env;
            }
            size_t pos = out_path.rfind(".bin");
            if (pos != std::string::npos) {
                out_path.insert(pos, suffix);
            } else {
                out_path += suffix;
            }
        }
        double sum = 0, sum2 = 0;
        for (int64_t i = 0; i < n; i++) { sum += f32[i]; sum2 += (double)f32[i] * f32[i]; }
        float mean = (float)(sum / n);
        float std_val = (float)sqrt(sum2 / n - (sum / n) * (sum / n));
        FILE* f = fopen(out_path.c_str(), "wb");
        fwrite(f32.data(), sizeof(float), n, f);
        fclose(f);
        fprintf(stderr, "  DUMP: %s [%ld] mean=%.6f std=%.4f first5: %.6f %.6f %.6f %.6f %.6f\n",
                out_path.c_str(), (long)n, mean, std_val, f32[0], f32[1], f32[2], f32[3], f32[4]);
    };
    dump("cuda_t_emb.bin", t_emb.data, inner_dim, false);

    // 2. Project inputs
    // Input projections can already stay FP32 because the source activations are BF16.
    Tensor img = Tensor::alloc({n_img, inner_dim}, DType::FP32);
    {
        Tensor patches_2d = img_patches.view({n_img, in_channels});
        Tensor img_2d = img.view({n_img, inner_dim});
        linear_forward_quantized_bf16in(patches_2d, w.img_in_weight, &w.img_in_bias, img_2d, gemm_scratch);
    }
    dump("cuda_img_patches.bin", img_patches.data, (int64_t)n_img * in_channels, true);
    img_patches.free_data();

    dump("cuda_img_after_proj.bin", img.data, (int64_t)n_img * inner_dim, false);

    // txt: RMSNorm(context) -> Linear
    Tensor txt_normed = Tensor::alloc({n_txt, 3584}, DType::BF16);
    rms_norm((__nv_bfloat16*)context.data, (__nv_bfloat16*)w.txt_norm_weight.data,
             (__nv_bfloat16*)txt_normed.data, n_txt, 3584, 1e-6f);

    dump("cuda_txt_after_rms.bin", txt_normed.data, (int64_t)n_txt * 3584, true);

    Tensor txt = Tensor::alloc({n_txt, inner_dim}, DType::FP32);
    {
        Tensor tn_2d = txt_normed.view({n_txt, 3584});
        Tensor txt_2d = txt.view({n_txt, inner_dim});
        linear_forward_quantized_bf16in(tn_2d, w.txt_in_weight, &w.txt_in_bias, txt_2d, gemm_scratch);
    }
    txt_normed.free_data();

    dump("cuda_txt_after_proj.bin", txt.data, (int64_t)n_txt * inner_dim, false);
    dump("cuda_context_input.bin", context.data, (int64_t)n_txt * 3584, true);
    dump("cuda_pe.bin", pe.data, pe.numel(), false);

    // ================================================================
    // Allocate working buffers for transformer blocks
    // ALL intermediates are FP32 to match ggml precision.
    // Convert to BF16 only at GEMM input boundaries.
    // ================================================================

    // FP32 intermediate buffers
    Tensor img_normed_fp32 = Tensor::alloc({n_img, inner_dim}, DType::FP32);
    Tensor txt_normed_fp32 = Tensor::alloc({n_txt, inner_dim}, DType::FP32);

    // Modulation buffers stay in FP32 to match ggml's transformer math.
    Tensor img_mod_buf = Tensor::alloc({1, 6 * inner_dim}, DType::FP32);
    Tensor txt_mod_buf = Tensor::alloc({1, 6 * inner_dim}, DType::FP32);
    Tensor t_emb_silu = Tensor::alloc({1, inner_dim}, DType::FP32);

    // Attention Q/K/V buffers (FP32 - output of linear_forward_fp32out)
    Tensor img_q = Tensor::alloc({n_img, inner_dim}, DType::FP32);
    Tensor img_k = Tensor::alloc({n_img, inner_dim}, DType::FP32);
    Tensor img_v = Tensor::alloc({n_img, inner_dim}, DType::FP32);
    Tensor txt_q = Tensor::alloc({n_txt, inner_dim}, DType::FP32);
    Tensor txt_k = Tensor::alloc({n_txt, inner_dim}, DType::FP32);
    Tensor txt_v = Tensor::alloc({n_txt, inner_dim}, DType::FP32);

    // Transposed attention buffers (FP32): [n_heads, total_seq, head_dim]
    Tensor q_t = Tensor::alloc({n_heads, total_seq, head_dim}, DType::FP32);
    Tensor k_t = Tensor::alloc({n_heads, total_seq, head_dim}, DType::FP32);
    Tensor v_t = Tensor::alloc({n_heads, total_seq, head_dim}, DType::FP32);
    Tensor attn_out_t = Tensor::alloc({n_heads, total_seq, head_dim}, DType::FP32);

    // FP16 Q/K/V for attention (avoids separate FP32→FP16 conversions)
    size_t qkv_fp16_size = (size_t)n_heads * total_seq * head_dim * sizeof(__half);
    __half* q_t_fp16;
    __half* k_t_fp16;
    __half* v_t_fp16;
    CUDA_CHECK(cudaMalloc(&q_t_fp16, qkv_fp16_size));
    CUDA_CHECK(cudaMalloc(&k_t_fp16, qkv_fp16_size));
    CUDA_CHECK(cudaMalloc(&v_t_fp16, qkv_fp16_size));


    // Post-attention (FP32)
    Tensor img_attn_out = Tensor::alloc({n_img, inner_dim}, DType::FP32);
    Tensor txt_attn_out = Tensor::alloc({n_txt, inner_dim}, DType::FP32);
    Tensor img_proj = Tensor::alloc({n_img, inner_dim}, DType::FP32);
    Tensor txt_proj = Tensor::alloc({n_txt, inner_dim}, DType::FP32);

    // MLP buffers
    Tensor img_mlp_fp32 = Tensor::alloc({n_img, mlp_dim}, DType::FP32);  // fc1 output + gelu
    Tensor txt_mlp_fp32 = Tensor::alloc({n_txt, mlp_dim}, DType::FP32);
    Tensor img_mlp_out = Tensor::alloc({n_img, inner_dim}, DType::FP32);
    Tensor txt_mlp_out = Tensor::alloc({n_txt, inner_dim}, DType::FP32);

    // 3. Run 60 transformer blocks
    const int internal_dump_block = get_transformer_internal_dump_block();
    static bool profile_block = (getenv("PROFILE_BLOCK") != nullptr);
    float prof_modulation = 0, prof_attn_norm = 0, prof_qkv_gemm = 0, prof_qk_norm = 0;
    float prof_concat_transpose = 0, prof_rope = 0, prof_attention = 0;
    float prof_transpose_split = 0, prof_out_gemm = 0, prof_gate_attn = 0;
    float prof_mlp_norm = 0, prof_mlp_gemm = 0, prof_gate_mlp = 0;
    cudaEvent_t ev_start, ev_end;
    if (profile_block) { cudaEventCreate(&ev_start); cudaEventCreate(&ev_end); }
    #define PROF_START() do { if (profile_block) cudaEventRecord(ev_start); } while(0)
    #define PROF_END(acc) do { if (profile_block) { cudaEventRecord(ev_end); cudaEventSynchronize(ev_end); \
        float ms; cudaEventElapsedTime(&ms, ev_start, ev_end); acc += ms; } } while(0)

    for (int bi = 0; bi < 60; bi++) {
        auto& b = w.blocks[bi];
        const bool dump_internal_block = (bi == internal_dump_block);
        auto dump_internal = [&](const char* stem, const void* data, int64_t n, bool is_bf16) {
            char path[128];
            snprintf(path, sizeof(path), "cuda_b%d_%s.bin", bi, stem);
            dump(path, data, n, is_bf16);
        };

        // Compute modulation: silu(t_emb) -> Linear -> chunk(6) in FP32.
        PROF_START();
        silu_fp32((float*)t_emb.data, (float*)t_emb_silu.data, inner_dim);
        {
            linear_forward_quantized(t_emb_silu, b.img_mod_weight, &b.img_mod_bias,
                                img_mod_buf, gemm_scratch);
            linear_forward_quantized(t_emb_silu, b.txt_mod_weight, &b.txt_mod_bias,
                                txt_mod_buf, gemm_scratch);
        }
        PROF_END(prof_modulation);

        auto* img_mods = (float*)img_mod_buf.data;
        auto* txt_mods = (float*)txt_mod_buf.data;

        if (dump_internal_block) {
            dump_internal("img_mods.bin", img_mods, 6 * inner_dim, false);
            dump_internal("txt_mods.bin", txt_mods, 6 * inner_dim, false);
        }

        // === ATTENTION PATH ===

        // LayerNorm(FP32 residual) → FP32
        PROF_START();
        layer_norm_no_affine_fp32((float*)img.data, (float*)img_normed_fp32.data,
                                   n_img, inner_dim, 1e-6f);

        if (dump_internal_block) dump_internal("img_ln.bin", img_normed_fp32.data, (int64_t)n_img * inner_dim, false);

        // Modulate(FP32, BF16 shift/scale) → FP32
        modulate_fp32_f32params((float*)img_normed_fp32.data,
                      img_mods + 0 * inner_dim,  // shift1
                      img_mods + 1 * inner_dim,  // scale1
                      (float*)img_normed_fp32.data, n_img, inner_dim);

        if (dump_internal_block) dump_internal("img_modulated.bin", img_normed_fp32.data, (int64_t)n_img * inner_dim, false);

        // Same for txt
        layer_norm_no_affine_fp32((float*)txt.data, (float*)txt_normed_fp32.data,
                                   n_txt, inner_dim, 1e-6f);
        modulate_fp32_f32params((float*)txt_normed_fp32.data,
                      txt_mods + 0 * inner_dim,
                      txt_mods + 1 * inner_dim,
                      (float*)txt_normed_fp32.data, n_txt, inner_dim);

        // Calibration: capture img_normed_fp32 for to_q/to_k/to_v (shared input)
        if (cal) {
            std::string prefix = "blocks." + std::to_string(bi) + ".img_attn_qkv";
            // Copy FP32 data (write_entry applies FWHT in-place)
            float* cal_buf;
            CUDA_CHECK(cudaMalloc(&cal_buf, (int64_t)n_img * inner_dim * sizeof(float)));
            CUDA_CHECK(cudaMemcpy(cal_buf, img_normed_fp32.data,
                                  (int64_t)n_img * inner_dim * sizeof(float), cudaMemcpyDeviceToDevice));
            int hbs = hadamard_block_size(inner_dim);
            cal->write_entry(prefix, cal_buf, n_img, inner_dim, hbs);
            CUDA_CHECK(cudaFree(cal_buf));
        }

        PROF_END(prof_attn_norm);

        // Attention projections: FP32 activations → INT8 GEMM → FP32 output.
        PROF_START();
        {
            Tensor oq = img_q.view({n_img, inner_dim});
            Tensor ok = img_k.view({n_img, inner_dim});
            Tensor ov = img_v.view({n_img, inner_dim});
            linear_forward_quantized(img_normed_fp32, b.to_q_weight, &b.to_q_bias, oq, gemm_scratch);
            linear_forward_quantized(img_normed_fp32, b.to_k_weight, &b.to_k_bias, ok, gemm_scratch);
            linear_forward_quantized(img_normed_fp32, b.to_v_weight, &b.to_v_bias, ov, gemm_scratch);
        }
        {
            Tensor oq = txt_q.view({n_txt, inner_dim});
            Tensor ok = txt_k.view({n_txt, inner_dim});
            Tensor ov = txt_v.view({n_txt, inner_dim});
            linear_forward_quantized(txt_normed_fp32, b.add_q_proj_weight, &b.add_q_proj_bias, oq, gemm_scratch);
            linear_forward_quantized(txt_normed_fp32, b.add_k_proj_weight, &b.add_k_proj_bias, ok, gemm_scratch);
            linear_forward_quantized(txt_normed_fp32, b.add_v_proj_weight, &b.add_v_proj_bias, ov, gemm_scratch);
        }

        if (dump_internal_block) {
            dump_internal("img_q_raw.bin", img_q.data, (int64_t)n_img * inner_dim, false);
            dump_internal("img_k_raw.bin", img_k.data, (int64_t)n_img * inner_dim, false);
            dump_internal("img_v_raw.bin", img_v.data, (int64_t)n_img * inner_dim, false);
        }

        PROF_END(prof_qkv_gemm);

        // RMSNorm on Q/K: FP32 input, BF16 weight → FP32 output
        PROF_START();
        rms_norm_fp32((float*)img_q.data, (__nv_bfloat16*)b.norm_q_weight.data,
                      (float*)img_q.data, n_img * n_heads, head_dim, 1e-6f);
        rms_norm_fp32((float*)img_k.data, (__nv_bfloat16*)b.norm_k_weight.data,
                      (float*)img_k.data, n_img * n_heads, head_dim, 1e-6f);
        rms_norm_fp32((float*)txt_q.data, (__nv_bfloat16*)b.norm_added_q_weight.data,
                      (float*)txt_q.data, n_txt * n_heads, head_dim, 1e-6f);
        rms_norm_fp32((float*)txt_k.data, (__nv_bfloat16*)b.norm_added_k_weight.data,
                      (float*)txt_k.data, n_txt * n_heads, head_dim, 1e-6f);

        if (dump_internal_block) {
            dump_internal("img_q_normed.bin", img_q.data, (int64_t)n_img * inner_dim, false);
            dump_internal("img_k_normed.bin", img_k.data, (int64_t)n_img * inner_dim, false);
            dump_internal("txt_q_normed.bin", txt_q.data, (int64_t)n_txt * inner_dim, false);
        }

        PROF_END(prof_qk_norm);

        // Fused concat [txt, img] + transpose [S, H, D] -> [H, S, D]
        PROF_START();
        {
            int64_t total_elems = (int64_t)total_seq * n_heads * head_dim;
            int block = 256;
            int grid = (int)((total_elems + block - 1) / block);
            concat_transpose_fused_fp32<<<grid, block>>>(
                (float*)txt_q.data, (float*)img_q.data, (float*)q_t.data,
                n_txt, n_img, n_heads, head_dim);
            concat_transpose_fused_fp32<<<grid, block>>>(
                (float*)txt_k.data, (float*)img_k.data, (float*)k_t.data,
                n_txt, n_img, n_heads, head_dim);
            concat_transpose_fused_fp32<<<grid, block>>>(
                (float*)txt_v.data, (float*)img_v.data, (float*)v_t.data,
                n_txt, n_img, n_heads, head_dim, v_t_fp16);
        }

        if (dump_internal_block) {
            dump_internal("q_joint.bin", q_t.data, (int64_t)n_heads * total_seq * head_dim, false);
            dump_internal("k_joint.bin", k_t.data, (int64_t)n_heads * total_seq * head_dim, false);
        }

        PROF_END(prof_concat_transpose);

        // Apply RoPE and output FP16 directly (saves separate FP32→FP16 conversion)
        PROF_START();
        rope_apply_fp32_to_fp16((float*)q_t.data, (float*)pe.data, q_t_fp16,
                                 n_heads, total_seq, head_dim);
        rope_apply_fp32_to_fp16((float*)k_t.data, (float*)pe.data, k_t_fp16,
                                 n_heads, total_seq, head_dim);

        if (dump_internal_block) {
            dump_internal("q_after_rope.bin", q_t.data, (int64_t)n_heads * total_seq * head_dim, false);
            dump_internal("k_after_rope.bin", k_t.data, (int64_t)n_heads * total_seq * head_dim, false);
            dump_internal("v_joint.bin", v_t.data, (int64_t)n_heads * total_seq * head_dim, false);
        }

        PROF_END(prof_rope);

        // Scaled dot-product attention in full FP32 to match the ggml fallback path.
        PROF_START();
        float attn_scale = 1.0f / sqrtf(128.0f);
        attention_forward_fp32io(
            (float*)q_t.data, (float*)k_t.data, (float*)v_t.data,
            (float*)attn_out_t.data, attn_scale, n_heads, total_seq, head_dim,
            false, 0, q_t_fp16, k_t_fp16, v_t_fp16);

        PROF_END(prof_attention);

        // Fused transpose [H, S, D] -> split to txt[S1, H*D], img[S2, H*D]
        PROF_START();
        {
            int64_t total_elems = (int64_t)n_heads * total_seq * head_dim;
            int block = 256;
            int grid = (int)((total_elems + block - 1) / block);
            transpose_split_fused_fp32<<<grid, block>>>(
                (float*)attn_out_t.data,
                (float*)txt_attn_out.data, (float*)img_attn_out.data,
                n_txt, n_img, n_heads, head_dim);
        }

        if (dump_internal_block) {
            dump_internal("attn_out_img.bin", img_attn_out.data, (int64_t)n_img * inner_dim, false);
            dump_internal("attn_out_txt.bin", txt_attn_out.data, (int64_t)n_txt * inner_dim, false);
        }

        // Calibration: capture img_attn_out for to_out
        if (cal) {
            std::string prefix = "blocks." + std::to_string(bi) + ".img_attn_out";
            float* cal_buf;
            CUDA_CHECK(cudaMalloc(&cal_buf, (int64_t)n_img * inner_dim * sizeof(float)));
            CUDA_CHECK(cudaMemcpy(cal_buf, img_attn_out.data,
                                  (int64_t)n_img * inner_dim * sizeof(float), cudaMemcpyDeviceToDevice));
            int hbs = hadamard_block_size(inner_dim);
            cal->write_entry(prefix, cal_buf, n_img, inner_dim, hbs);
            CUDA_CHECK(cudaFree(cal_buf));
        }

        PROF_END(prof_transpose_split);

        // Output projections: FP32 activations → INT8 GEMM → FP32 output.
        PROF_START();
        {
            Tensor ip_2d = img_proj.view({n_img, inner_dim});
            linear_forward_quantized(img_attn_out, b.to_out_weight, &b.to_out_bias, ip_2d, gemm_scratch);

            Tensor tp_2d = txt_proj.view({n_txt, inner_dim});
            linear_forward_quantized(txt_attn_out, b.to_add_out_weight, &b.to_add_out_bias, tp_2d, gemm_scratch);
        }

        if (dump_internal_block) {
            dump_internal("img_proj.bin", img_proj.data, (int64_t)n_img * inner_dim, false);
        }

        PROF_END(prof_out_gemm);

        // Residual: img_fp32 += gate1 * FP32_proj
        PROF_START();
        gate_add_fp32f((float*)img.data, img_mods + 2 * inner_dim,
                        (float*)img_proj.data, (float*)img.data, n_img, inner_dim);
        gate_add_fp32f((float*)txt.data, txt_mods + 2 * inner_dim,
                        (float*)txt_proj.data, (float*)txt.data, n_txt, inner_dim);
        PROF_END(prof_gate_attn);

        if (dump_internal_block) {
            dump_internal("img_after_attn.bin", img.data, (int64_t)n_img * inner_dim, false);
        }

        // === MLP PATH ===

        // LayerNorm(FP32) → Modulate(FP32) → FP32 GEMMs
        PROF_START();
        layer_norm_no_affine_fp32((float*)img.data, (float*)img_normed_fp32.data,
                                   n_img, inner_dim, 1e-6f);
        modulate_fp32_f32params((float*)img_normed_fp32.data,
                      img_mods + 3 * inner_dim,  // shift2
                      img_mods + 4 * inner_dim,  // scale2
                      (float*)img_normed_fp32.data, n_img, inner_dim);

        layer_norm_no_affine_fp32((float*)txt.data, (float*)txt_normed_fp32.data,
                                   n_txt, inner_dim, 1e-6f);
        modulate_fp32_f32params((float*)txt_normed_fp32.data,
                      txt_mods + 3 * inner_dim,
                      txt_mods + 4 * inner_dim,
                      (float*)txt_normed_fp32.data, n_txt, inner_dim);

        // Calibration: capture img_normed_fp32 for img_mlp_fc1
        if (cal) {
            std::string prefix = "blocks." + std::to_string(bi) + ".img_mlp_in";
            float* cal_buf;
            CUDA_CHECK(cudaMalloc(&cal_buf, (int64_t)n_img * inner_dim * sizeof(float)));
            CUDA_CHECK(cudaMemcpy(cal_buf, img_normed_fp32.data,
                                  (int64_t)n_img * inner_dim * sizeof(float), cudaMemcpyDeviceToDevice));
            int hbs = hadamard_block_size(inner_dim);
            cal->write_entry(prefix, cal_buf, n_img, inner_dim, hbs);
            CUDA_CHECK(cudaFree(cal_buf));
        }

        PROF_END(prof_mlp_norm);

        // GELU MLP (INT8 GEMMs)
        PROF_START();
        if (dump_internal_block) dump_internal("img_mlp_input.bin", img_normed_fp32.data, (int64_t)n_img * inner_dim, false);
        {
            Tensor mlp_2d = img_mlp_fp32.view({n_img, mlp_dim});
            linear_forward_quantized(img_normed_fp32, b.img_mlp_fc1_weight, &b.img_mlp_fc1_bias, mlp_2d, gemm_scratch);
            if (dump_internal_block) dump_internal("img_mlp_fc1.bin", img_mlp_fp32.data, (int64_t)n_img * mlp_dim, false);
            // GELU in FP32
            gelu_fp32((float*)img_mlp_fp32.data, (float*)img_mlp_fp32.data,
                      (int64_t)n_img * mlp_dim);
            if (dump_internal_block) dump_internal("img_mlp_gelu.bin", img_mlp_fp32.data, (int64_t)n_img * mlp_dim, false);
            // Calibration: capture img_mlp_fp32 (post-GELU) for img_mlp_fc2
            if (cal) {
                std::string prefix = "blocks." + std::to_string(bi) + ".img_mlp_mid";
                float* cal_buf;
                CUDA_CHECK(cudaMalloc(&cal_buf, (int64_t)n_img * mlp_dim * sizeof(float)));
                CUDA_CHECK(cudaMemcpy(cal_buf, img_mlp_fp32.data,
                                      (int64_t)n_img * mlp_dim * sizeof(float), cudaMemcpyDeviceToDevice));
                int hbs = hadamard_block_size(mlp_dim);
                cal->write_entry(prefix, cal_buf, n_img, mlp_dim, hbs);
                CUDA_CHECK(cudaFree(cal_buf));
            }
            Tensor out_2d = img_mlp_out.view({n_img, inner_dim});
            linear_forward_quantized(img_mlp_fp32, b.img_mlp_fc2_weight, &b.img_mlp_fc2_bias, out_2d, gemm_scratch);
            if (dump_internal_block) dump_internal("img_mlp_out.bin", img_mlp_out.data, (int64_t)n_img * inner_dim, false);
        }
        {
            Tensor mlp_2d = txt_mlp_fp32.view({n_txt, mlp_dim});
            linear_forward_quantized(txt_normed_fp32, b.txt_mlp_fc1_weight, &b.txt_mlp_fc1_bias, mlp_2d, gemm_scratch);
            gelu_fp32((float*)txt_mlp_fp32.data, (float*)txt_mlp_fp32.data,
                      (int64_t)n_txt * mlp_dim);
            Tensor out_2d = txt_mlp_out.view({n_txt, inner_dim});
            linear_forward_quantized(txt_mlp_fp32, b.txt_mlp_fc2_weight, &b.txt_mlp_fc2_bias, out_2d, gemm_scratch);
        }

        PROF_END(prof_mlp_gemm);

        // Residual: img_fp32 += gate2 * FP32_mlp_out
        PROF_START();
        gate_add_fp32f((float*)img.data, img_mods + 5 * inner_dim,
                        (float*)img_mlp_out.data, (float*)img.data, n_img, inner_dim);
        gate_add_fp32f((float*)txt.data, txt_mods + 5 * inner_dim,
                        (float*)txt_mlp_out.data, (float*)txt.data, n_txt, inner_dim);
        PROF_END(prof_gate_mlp);

        if (dump_internal_block) {
            dump_internal("img_after_mlp.bin", img.data, (int64_t)n_img * inner_dim, false);
            dump_internal("txt_after_mlp.bin", txt.data, (int64_t)n_txt * inner_dim, false);
        }

        if ((bi + 1) % 10 == 0) {
            fprintf(stderr, "  Transformer block %d/60\n", bi + 1);
            char path[128];
            snprintf(path, sizeof(path), "cuda_img_after_b%d.bin", bi);
            dump(path, img.data, (int64_t)n_img * inner_dim, false);
        }
        if (block_index_in_env_range("DUMP_TRANSFORMER_BLOCK_RANGE", bi)) {
            char path[128];
            snprintf(path, sizeof(path), "cuda_img_after_b%d.bin", bi);
            dump(path, img.data, (int64_t)n_img * inner_dim, false);
        }
    }

    if (profile_block) {
        float total = prof_modulation + prof_attn_norm + prof_qkv_gemm + prof_qk_norm +
                      prof_concat_transpose + prof_rope + prof_attention +
                      prof_transpose_split + prof_out_gemm + prof_gate_attn +
                      prof_mlp_norm + prof_mlp_gemm + prof_gate_mlp;
        fprintf(stderr, "\n=== Block Loop Profile (60 blocks, total %.1f ms) ===\n", total);
        fprintf(stderr, "  modulation:        %7.1f ms  (%4.1f%%)\n", prof_modulation, 100*prof_modulation/total);
        fprintf(stderr, "  attn_norm:         %7.1f ms  (%4.1f%%)\n", prof_attn_norm, 100*prof_attn_norm/total);
        fprintf(stderr, "  qkv_gemm:          %7.1f ms  (%4.1f%%)\n", prof_qkv_gemm, 100*prof_qkv_gemm/total);
        fprintf(stderr, "  qk_norm:           %7.1f ms  (%4.1f%%)\n", prof_qk_norm, 100*prof_qk_norm/total);
        fprintf(stderr, "  concat_transpose:  %7.1f ms  (%4.1f%%)\n", prof_concat_transpose, 100*prof_concat_transpose/total);
        fprintf(stderr, "  rope:              %7.1f ms  (%4.1f%%)\n", prof_rope, 100*prof_rope/total);
        fprintf(stderr, "  attention:         %7.1f ms  (%4.1f%%)\n", prof_attention, 100*prof_attention/total);
        fprintf(stderr, "  transpose_split:   %7.1f ms  (%4.1f%%)\n", prof_transpose_split, 100*prof_transpose_split/total);
        fprintf(stderr, "  out_gemm:          %7.1f ms  (%4.1f%%)\n", prof_out_gemm, 100*prof_out_gemm/total);
        fprintf(stderr, "  gate_attn:         %7.1f ms  (%4.1f%%)\n", prof_gate_attn, 100*prof_gate_attn/total);
        fprintf(stderr, "  mlp_norm:          %7.1f ms  (%4.1f%%)\n", prof_mlp_norm, 100*prof_mlp_norm/total);
        fprintf(stderr, "  mlp_gemm:          %7.1f ms  (%4.1f%%)\n", prof_mlp_gemm, 100*prof_mlp_gemm/total);
        fprintf(stderr, "  gate_mlp:          %7.1f ms  (%4.1f%%)\n", prof_gate_mlp, 100*prof_gate_mlp/total);
        fprintf(stderr, "===============================================\n\n");
        cudaEventDestroy(ev_start);
        cudaEventDestroy(ev_end);
    }

    dump("cuda_img_final_fp32.bin", img.data, (int64_t)n_img * inner_dim, false);

    // 4. AdaLayerNormContinuous output
    Tensor ada_emb = Tensor::alloc({1, 2 * inner_dim}, DType::FP32);
    silu_fp32((float*)t_emb.data, (float*)t_emb_silu.data, inner_dim);
    linear_forward_quantized(t_emb_silu, w.norm_out_linear_weight, &w.norm_out_linear_bias,
                        ada_emb, gemm_scratch);
    auto* ada_mods = (float*)ada_emb.data;

    // Final AdaLayerNormContinuous in FP32, convert to BF16 only after proj_out.
    layer_norm_no_affine_fp32((float*)img.data, (float*)img_normed_fp32.data, n_img, inner_dim, 1e-6f);
    modulate_fp32_f32params((float*)img_normed_fp32.data,
                            ada_mods + inner_dim,   // shift (second half)
                            ada_mods + 0,           // scale (first half)
                            (float*)img_normed_fp32.data, n_img, inner_dim);

    dump("cuda_img_after_adanorm.bin", img_normed_fp32.data, (int64_t)n_img * inner_dim, false);

    // 5. Project out: [n_img, 3072] -> [n_img, 64]
    Tensor img_proj_out_fp32 = Tensor::alloc({n_img, (int64_t)(patch_size * patch_size * out_channels)}, DType::FP32);
    linear_forward_quantized(img_normed_fp32, w.proj_out_weight, &w.proj_out_bias,
                        img_proj_out_fp32, gemm_scratch);

    Tensor img_proj_out = Tensor::alloc({n_img, (int64_t)(patch_size * patch_size * out_channels)}, DType::BF16);
    fp32_to_bf16((float*)img_proj_out_fp32.data, (__nv_bfloat16*)img_proj_out.data,
                 (int64_t)n_img * patch_size * patch_size * out_channels);

    dump("cuda_proj_out.bin", img_proj_out.data, (int64_t)n_img * patch_size * patch_size * out_channels, true);

    // 6. Unpatchify: [1, n_img, 64] -> [1, 16, H_pad, W_pad]
    Tensor output = Tensor::alloc({1, (int64_t)out_channels, H_pad, W_pad}, DType::BF16);
    unpatchify((__nv_bfloat16*)img_proj_out.data, (__nv_bfloat16*)output.data,
               1, out_channels, H_pad, W_pad, patch_size, patch_size);

    // Crop back to original size if padded
    if (H_pad != H || W_pad != W) {
        Tensor cropped = Tensor::alloc({1, (int64_t)out_channels, (int64_t)H, (int64_t)W}, DType::BF16);
        for (int c = 0; c < out_channels; c++) {
            for (int h = 0; h < H; h++) {
                CUDA_CHECK(cudaMemcpy(
                    (__nv_bfloat16*)cropped.data + (int64_t)c * H * W + h * W,
                    (__nv_bfloat16*)output.data + (int64_t)c * H_pad * W_pad + h * W_pad,
                    W * sizeof(__nv_bfloat16),
                    cudaMemcpyDeviceToDevice));
            }
        }
        output.free_data();
        output = std::move(cropped);
    }

    // Free all temp buffers
    img.free_data(); txt.free_data();
    t_emb.free_data(); t_emb_silu.free_data();
    img_normed_fp32.free_data(); txt_normed_fp32.free_data();
    img_mod_buf.free_data(); txt_mod_buf.free_data();
    gemm_scratch.free_data();
    img_q.free_data(); img_k.free_data(); img_v.free_data();
    txt_q.free_data(); txt_k.free_data(); txt_v.free_data();
    q_t.free_data(); k_t.free_data(); v_t.free_data();
    cudaFree(q_t_fp16); cudaFree(k_t_fp16); cudaFree(v_t_fp16);
    attn_out_t.free_data();
    img_attn_out.free_data(); txt_attn_out.free_data();
    img_proj.free_data(); txt_proj.free_data();
    img_mlp_fp32.free_data(); txt_mlp_fp32.free_data();
    img_mlp_out.free_data(); txt_mlp_out.free_data();
    ada_emb.free_data();
    img_proj_out_fp32.free_data();
    img_proj_out.free_data();

    CUDA_CHECK(cudaDeviceSynchronize());
    fprintf(stderr, "Transformer done: output %s\n", output.shape_str().c_str());
    return output;
}
