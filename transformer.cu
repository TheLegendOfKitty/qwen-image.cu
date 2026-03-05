#include "transformer.h"
#include "cuda_kernels.cuh"
#include "logging.h"
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

namespace {

struct TransformerForwardWorkspace {
    // Persistent scratch storages (all 1D); views are shaped per call.
    Tensor img_patches, x_padded, gemm_scratch;
    Tensor t_buf, t_emb_sin, t_after_l1, t_emb;
    Tensor img, txt_normed, txt;
    Tensor img_normed_fp32, txt_normed_fp32;
    Tensor img_mod_buf, txt_mod_buf, t_emb_silu;
    Tensor img_q, img_k, img_v, txt_q, txt_k, txt_v;
    Tensor q_t, k_t, v_t, attn_out_t;
    Tensor img_attn_out, txt_attn_out, img_proj, txt_proj;
    Tensor img_mlp_fp32, txt_mlp_fp32, img_mlp_out, txt_mlp_out;
    Tensor ada_emb, img_proj_out_fp32, img_proj_out;

    // Shared activation quantization caches for Q/K/V projections.
    Tensor img_qkv_act_int8, img_qkv_act_scales;
    Tensor txt_qkv_act_int8, txt_qkv_act_scales;

    // FP16 Q/K/V buffers for attention.
    __half* q_t_fp16 = nullptr;
    __half* k_t_fp16 = nullptr;
    __half* v_t_fp16 = nullptr;
    size_t qkv_fp16_capacity_elems = 0;

    ~TransformerForwardWorkspace() {
        free_all();
    }

    void free_all() {
        img_patches.free_data(); x_padded.free_data(); gemm_scratch.free_data();
        t_buf.free_data(); t_emb_sin.free_data(); t_after_l1.free_data(); t_emb.free_data();
        img.free_data(); txt_normed.free_data(); txt.free_data();
        img_normed_fp32.free_data(); txt_normed_fp32.free_data();
        img_mod_buf.free_data(); txt_mod_buf.free_data(); t_emb_silu.free_data();
        img_q.free_data(); img_k.free_data(); img_v.free_data();
        txt_q.free_data(); txt_k.free_data(); txt_v.free_data();
        q_t.free_data(); k_t.free_data(); v_t.free_data(); attn_out_t.free_data();
        img_attn_out.free_data(); txt_attn_out.free_data(); img_proj.free_data(); txt_proj.free_data();
        img_mlp_fp32.free_data(); txt_mlp_fp32.free_data(); img_mlp_out.free_data(); txt_mlp_out.free_data();
        ada_emb.free_data(); img_proj_out_fp32.free_data(); img_proj_out.free_data();
        img_qkv_act_int8.free_data(); img_qkv_act_scales.free_data();
        txt_qkv_act_int8.free_data(); txt_qkv_act_scales.free_data();
        if (q_t_fp16) CUDA_CHECK(cudaFree(q_t_fp16));
        if (k_t_fp16) CUDA_CHECK(cudaFree(k_t_fp16));
        if (v_t_fp16) CUDA_CHECK(cudaFree(v_t_fp16));
        q_t_fp16 = nullptr;
        k_t_fp16 = nullptr;
        v_t_fp16 = nullptr;
        qkv_fp16_capacity_elems = 0;
    }

    Tensor view(Tensor& storage, int64_t elems, DType dtype, std::initializer_list<int64_t> dims) {
        assert(elems >= 0);
        int64_t capacity = (storage.data && storage.ndim > 0) ? storage.shape[0] : 0;
        if (!storage.data || storage.dtype != dtype || capacity < elems) {
            int64_t new_capacity = elems;
            if (capacity > 0 && storage.dtype == dtype) {
                int64_t doubled = capacity * 2;
                if (doubled > new_capacity) new_capacity = doubled;
            }
            storage.free_data();
            storage = Tensor::alloc({new_capacity}, dtype);
        }
        Tensor flat = storage.slice(0, elems);
        return flat.view(dims);
    }

    void ensure_qkv_fp16(size_t elems) {
        if (qkv_fp16_capacity_elems >= elems) return;
        if (q_t_fp16) CUDA_CHECK(cudaFree(q_t_fp16));
        if (k_t_fp16) CUDA_CHECK(cudaFree(k_t_fp16));
        if (v_t_fp16) CUDA_CHECK(cudaFree(v_t_fp16));
        CUDA_CHECK(cudaMalloc(&q_t_fp16, elems * sizeof(__half)));
        CUDA_CHECK(cudaMalloc(&k_t_fp16, elems * sizeof(__half)));
        CUDA_CHECK(cudaMalloc(&v_t_fp16, elems * sizeof(__half)));
        qkv_fp16_capacity_elems = elems;
    }
};

static TransformerForwardWorkspace& transformer_workspace() {
    static TransformerForwardWorkspace ws;
    return ws;
}

static inline bool can_share_qkv_quant(const QuantizedWeight& wq,
                                       const QuantizedWeight& wk,
                                       const QuantizedWeight& wv,
                                       int K) {
    if (wq.mode != QuantMode::INT8_HADAMARD ||
        wk.mode != QuantMode::INT8_HADAMARD ||
        wv.mode != QuantMode::INT8_HADAMARD) {
        return false;
    }
    if (wq.had_block_size != wk.had_block_size || wq.had_block_size != wv.had_block_size) {
        return false;
    }
    return (int)wq.data.shape[1] == K && (int)wk.data.shape[1] == K && (int)wv.data.shape[1] == K;
}

} // namespace

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

    LOGV("Transformer: n_img=%d, n_txt=%d, total=%d\n", n_img, n_txt, total_seq);
    TransformerForwardWorkspace& ws = transformer_workspace();

    // Patchify the input
    Tensor img_patches = ws.view(
        ws.img_patches, (int64_t)n_img * in_channels, DType::BF16, {n_img, in_channels});

    if (H_pad != H || W_pad != W) {
        Tensor x_padded = ws.view(
            ws.x_padded,
            (int64_t)x_in.shape[1] * H_pad * W_pad,
            DType::BF16,
            {1, (int64_t)x_in.shape[1], H_pad, W_pad});
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

    // Allocate as raw bytes (use FP32 dtype for sizing: ceil(bytes/4) elements).
    Tensor gemm_scratch = ws.view(
        ws.gemm_scratch, (max_scratch_bytes + 3) / 4, DType::FP32, {(max_scratch_bytes + 3) / 4});

    // 1. Timestep embedding: scalar -> [1, 256] sinusoidal -> linear -> silu -> linear -> [1, 3072]
    Tensor t_buf = ws.view(ws.t_buf, 1, DType::FP32, {1});
    t_buf.from_host(&timestep_val);
    Tensor t_emb_sin = ws.view(ws.t_emb_sin, 256, DType::FP32, {1, 256});
    timestep_embedding_fp32((float*)t_buf.data, (float*)t_emb_sin.data, 1, 256, 10000.0f);
    t_buf.free_data();

    Tensor t_after_l1 = ws.view(ws.t_after_l1, inner_dim, DType::FP32, {1, inner_dim});
    {
        linear_forward_quantized(t_emb_sin, w.time_linear1_weight, &w.time_linear1_bias,
                            t_after_l1, gemm_scratch);
    }
    silu_fp32((float*)t_after_l1.data, (float*)t_after_l1.data, inner_dim);
    t_emb_sin.free_data();

    Tensor t_emb = ws.view(ws.t_emb, inner_dim, DType::FP32, {1, inner_dim});
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
    Tensor img = ws.view(ws.img, (int64_t)n_img * inner_dim, DType::FP32, {n_img, inner_dim});
    {
        Tensor patches_2d = img_patches.view({n_img, in_channels});
        Tensor img_2d = img.view({n_img, inner_dim});
        linear_forward_quantized_bf16in(patches_2d, w.img_in_weight, &w.img_in_bias, img_2d, gemm_scratch);
    }
    dump("cuda_img_patches.bin", img_patches.data, (int64_t)n_img * in_channels, true);
    img_patches.free_data();

    dump("cuda_img_after_proj.bin", img.data, (int64_t)n_img * inner_dim, false);

    // txt: RMSNorm(context) -> Linear
    Tensor txt_normed = ws.view(ws.txt_normed, (int64_t)n_txt * 3584, DType::BF16, {n_txt, 3584});
    rms_norm((__nv_bfloat16*)context.data, (__nv_bfloat16*)w.txt_norm_weight.data,
             (__nv_bfloat16*)txt_normed.data, n_txt, 3584, 1e-6f);

    dump("cuda_txt_after_rms.bin", txt_normed.data, (int64_t)n_txt * 3584, true);

    Tensor txt = ws.view(ws.txt, (int64_t)n_txt * inner_dim, DType::FP32, {n_txt, inner_dim});
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
    Tensor img_normed_fp32 = ws.view(ws.img_normed_fp32, (int64_t)n_img * inner_dim, DType::FP32, {n_img, inner_dim});
    Tensor txt_normed_fp32 = ws.view(ws.txt_normed_fp32, (int64_t)n_txt * inner_dim, DType::FP32, {n_txt, inner_dim});

    // Modulation buffers stay in FP32 to match ggml's transformer math.
    Tensor img_mod_buf = ws.view(ws.img_mod_buf, 6 * inner_dim, DType::FP32, {1, 6 * inner_dim});
    Tensor txt_mod_buf = ws.view(ws.txt_mod_buf, 6 * inner_dim, DType::FP32, {1, 6 * inner_dim});
    Tensor t_emb_silu = ws.view(ws.t_emb_silu, inner_dim, DType::FP32, {1, inner_dim});

    // Attention Q/K/V buffers (FP32 - output of linear_forward_fp32out)
    Tensor img_q = ws.view(ws.img_q, (int64_t)n_img * inner_dim, DType::FP32, {n_img, inner_dim});
    Tensor img_k = ws.view(ws.img_k, (int64_t)n_img * inner_dim, DType::FP32, {n_img, inner_dim});
    Tensor img_v = ws.view(ws.img_v, (int64_t)n_img * inner_dim, DType::FP32, {n_img, inner_dim});
    Tensor txt_q = ws.view(ws.txt_q, (int64_t)n_txt * inner_dim, DType::FP32, {n_txt, inner_dim});
    Tensor txt_k = ws.view(ws.txt_k, (int64_t)n_txt * inner_dim, DType::FP32, {n_txt, inner_dim});
    Tensor txt_v = ws.view(ws.txt_v, (int64_t)n_txt * inner_dim, DType::FP32, {n_txt, inner_dim});

    // Transposed attention buffers (FP32): [n_heads, total_seq, head_dim]
    Tensor q_t = ws.view(ws.q_t, (int64_t)n_heads * total_seq * head_dim, DType::FP32, {n_heads, total_seq, head_dim});
    Tensor k_t = ws.view(ws.k_t, (int64_t)n_heads * total_seq * head_dim, DType::FP32, {n_heads, total_seq, head_dim});
    Tensor v_t = ws.view(ws.v_t, (int64_t)n_heads * total_seq * head_dim, DType::FP32, {n_heads, total_seq, head_dim});
    Tensor attn_out_t = ws.view(ws.attn_out_t, (int64_t)n_heads * total_seq * head_dim, DType::FP32, {n_heads, total_seq, head_dim});

    // FP16 Q/K/V for attention (avoids separate FP32→FP16 conversions)
    size_t qkv_fp16_elems = (size_t)n_heads * total_seq * head_dim;
    ws.ensure_qkv_fp16(qkv_fp16_elems);
    __half* q_t_fp16 = ws.q_t_fp16;
    __half* k_t_fp16 = ws.k_t_fp16;
    __half* v_t_fp16 = ws.v_t_fp16;


    // Post-attention (FP32)
    Tensor img_attn_out = ws.view(ws.img_attn_out, (int64_t)n_img * inner_dim, DType::FP32, {n_img, inner_dim});
    Tensor txt_attn_out = ws.view(ws.txt_attn_out, (int64_t)n_txt * inner_dim, DType::FP32, {n_txt, inner_dim});
    Tensor img_proj = ws.view(ws.img_proj, (int64_t)n_img * inner_dim, DType::FP32, {n_img, inner_dim});
    Tensor txt_proj = ws.view(ws.txt_proj, (int64_t)n_txt * inner_dim, DType::FP32, {n_txt, inner_dim});

    // MLP buffers
    Tensor img_mlp_fp32 = ws.view(ws.img_mlp_fp32, (int64_t)n_img * mlp_dim, DType::FP32, {n_img, mlp_dim});  // fc1 output + gelu
    Tensor txt_mlp_fp32 = ws.view(ws.txt_mlp_fp32, (int64_t)n_txt * mlp_dim, DType::FP32, {n_txt, mlp_dim});
    Tensor img_mlp_out = ws.view(ws.img_mlp_out, (int64_t)n_img * inner_dim, DType::FP32, {n_img, inner_dim});
    Tensor txt_mlp_out = ws.view(ws.txt_mlp_out, (int64_t)n_txt * inner_dim, DType::FP32, {n_txt, inner_dim});

    Tensor img_qkv_act_int8 = ws.view(
        ws.img_qkv_act_int8, (int64_t)n_img * inner_dim, DType::INT8, {(int64_t)n_img * inner_dim});
    Tensor img_qkv_act_scales = ws.view(
        ws.img_qkv_act_scales, n_img, DType::FP32, {n_img});
    Tensor txt_qkv_act_int8 = ws.view(
        ws.txt_qkv_act_int8, (int64_t)n_txt * inner_dim, DType::INT8, {(int64_t)n_txt * inner_dim});
    Tensor txt_qkv_act_scales = ws.view(
        ws.txt_qkv_act_scales, n_txt, DType::FP32, {n_txt});

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

        // Fused LayerNorm + Modulate (FP32 residual → FP32)
        PROF_START();
        layer_norm_modulate_fp32((float*)img.data,
                      img_mods + 0 * inner_dim,  // shift1
                      img_mods + 1 * inner_dim,  // scale1
                      (float*)img_normed_fp32.data, n_img, inner_dim, 1e-6f);

        if (dump_internal_block) dump_internal("img_modulated.bin", img_normed_fp32.data, (int64_t)n_img * inner_dim, false);

        // Same for txt
        layer_norm_modulate_fp32((float*)txt.data,
                      txt_mods + 0 * inner_dim,
                      txt_mods + 1 * inner_dim,
                      (float*)txt_normed_fp32.data, n_txt, inner_dim, 1e-6f);

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
            if (can_share_qkv_quant(b.to_q_weight, b.to_k_weight, b.to_v_weight, inner_dim)) {
                quantize_activation_int8_hadamard_fp32(
                    (const float*)img_normed_fp32.data,
                    (int8_t*)img_qkv_act_int8.data,
                    (float*)img_qkv_act_scales.data,
                    n_img, inner_dim, b.to_q_weight.had_block_size);
                linear_forward_int8_prequantized(
                    (const int8_t*)img_qkv_act_int8.data, (const float*)img_qkv_act_scales.data,
                    n_img, inner_dim, b.to_q_weight, &b.to_q_bias, oq, gemm_scratch);
                linear_forward_int8_prequantized(
                    (const int8_t*)img_qkv_act_int8.data, (const float*)img_qkv_act_scales.data,
                    n_img, inner_dim, b.to_k_weight, &b.to_k_bias, ok, gemm_scratch);
                linear_forward_int8_prequantized(
                    (const int8_t*)img_qkv_act_int8.data, (const float*)img_qkv_act_scales.data,
                    n_img, inner_dim, b.to_v_weight, &b.to_v_bias, ov, gemm_scratch);
            } else {
                linear_forward_quantized(img_normed_fp32, b.to_q_weight, &b.to_q_bias, oq, gemm_scratch);
                linear_forward_quantized(img_normed_fp32, b.to_k_weight, &b.to_k_bias, ok, gemm_scratch);
                linear_forward_quantized(img_normed_fp32, b.to_v_weight, &b.to_v_bias, ov, gemm_scratch);
            }
        }
        {
            Tensor oq = txt_q.view({n_txt, inner_dim});
            Tensor ok = txt_k.view({n_txt, inner_dim});
            Tensor ov = txt_v.view({n_txt, inner_dim});
            if (can_share_qkv_quant(b.add_q_proj_weight, b.add_k_proj_weight, b.add_v_proj_weight, inner_dim)) {
                quantize_activation_int8_hadamard_fp32(
                    (const float*)txt_normed_fp32.data,
                    (int8_t*)txt_qkv_act_int8.data,
                    (float*)txt_qkv_act_scales.data,
                    n_txt, inner_dim, b.add_q_proj_weight.had_block_size);
                linear_forward_int8_prequantized(
                    (const int8_t*)txt_qkv_act_int8.data, (const float*)txt_qkv_act_scales.data,
                    n_txt, inner_dim, b.add_q_proj_weight, &b.add_q_proj_bias, oq, gemm_scratch);
                linear_forward_int8_prequantized(
                    (const int8_t*)txt_qkv_act_int8.data, (const float*)txt_qkv_act_scales.data,
                    n_txt, inner_dim, b.add_k_proj_weight, &b.add_k_proj_bias, ok, gemm_scratch);
                linear_forward_int8_prequantized(
                    (const int8_t*)txt_qkv_act_int8.data, (const float*)txt_qkv_act_scales.data,
                    n_txt, inner_dim, b.add_v_proj_weight, &b.add_v_proj_bias, ov, gemm_scratch);
            } else {
                linear_forward_quantized(txt_normed_fp32, b.add_q_proj_weight, &b.add_q_proj_bias, oq, gemm_scratch);
                linear_forward_quantized(txt_normed_fp32, b.add_k_proj_weight, &b.add_k_proj_bias, ok, gemm_scratch);
                linear_forward_quantized(txt_normed_fp32, b.add_v_proj_weight, &b.add_v_proj_bias, ov, gemm_scratch);
            }
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

        // Fused LayerNorm + Modulate → FP32 GEMMs
        PROF_START();
        layer_norm_modulate_fp32((float*)img.data,
                      img_mods + 3 * inner_dim,  // shift2
                      img_mods + 4 * inner_dim,  // scale2
                      (float*)img_normed_fp32.data, n_img, inner_dim, 1e-6f);

        layer_norm_modulate_fp32((float*)txt.data,
                      txt_mods + 3 * inner_dim,
                      txt_mods + 4 * inner_dim,
                      (float*)txt_normed_fp32.data, n_txt, inner_dim, 1e-6f);

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
            linear_forward_quantized(img_normed_fp32, b.img_mlp_fc1_weight, &b.img_mlp_fc1_bias, mlp_2d, gemm_scratch, true);
            if (dump_internal_block) dump_internal("img_mlp_fc1.bin", img_mlp_fp32.data, (int64_t)n_img * mlp_dim, false);
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
            linear_forward_quantized(txt_normed_fp32, b.txt_mlp_fc1_weight, &b.txt_mlp_fc1_bias, mlp_2d, gemm_scratch, true);
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
            LOGV("  Transformer block %d/60\n", bi + 1);
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
        LOGV("\n=== Block Loop Profile (60 blocks, total %.1f ms) ===\n", total);
        LOGV("  modulation:        %7.1f ms  (%4.1f%%)\n", prof_modulation, 100*prof_modulation/total);
        LOGV("  attn_norm:         %7.1f ms  (%4.1f%%)\n", prof_attn_norm, 100*prof_attn_norm/total);
        LOGV("  qkv_gemm:          %7.1f ms  (%4.1f%%)\n", prof_qkv_gemm, 100*prof_qkv_gemm/total);
        LOGV("  qk_norm:           %7.1f ms  (%4.1f%%)\n", prof_qk_norm, 100*prof_qk_norm/total);
        LOGV("  concat_transpose:  %7.1f ms  (%4.1f%%)\n", prof_concat_transpose, 100*prof_concat_transpose/total);
        LOGV("  rope:              %7.1f ms  (%4.1f%%)\n", prof_rope, 100*prof_rope/total);
        LOGV("  attention:         %7.1f ms  (%4.1f%%)\n", prof_attention, 100*prof_attention/total);
        LOGV("  transpose_split:   %7.1f ms  (%4.1f%%)\n", prof_transpose_split, 100*prof_transpose_split/total);
        LOGV("  out_gemm:          %7.1f ms  (%4.1f%%)\n", prof_out_gemm, 100*prof_out_gemm/total);
        LOGV("  gate_attn:         %7.1f ms  (%4.1f%%)\n", prof_gate_attn, 100*prof_gate_attn/total);
        LOGV("  mlp_norm:          %7.1f ms  (%4.1f%%)\n", prof_mlp_norm, 100*prof_mlp_norm/total);
        LOGV("  mlp_gemm:          %7.1f ms  (%4.1f%%)\n", prof_mlp_gemm, 100*prof_mlp_gemm/total);
        LOGV("  gate_mlp:          %7.1f ms  (%4.1f%%)\n", prof_gate_mlp, 100*prof_gate_mlp/total);
        LOGV("===============================================\n\n");
        cudaEventDestroy(ev_start);
        cudaEventDestroy(ev_end);
    }

    dump("cuda_img_final_fp32.bin", img.data, (int64_t)n_img * inner_dim, false);

    // 4. AdaLayerNormContinuous output
    Tensor ada_emb = ws.view(ws.ada_emb, 2 * inner_dim, DType::FP32, {1, 2 * inner_dim});
    silu_fp32((float*)t_emb.data, (float*)t_emb_silu.data, inner_dim);
    linear_forward_quantized(t_emb_silu, w.norm_out_linear_weight, &w.norm_out_linear_bias,
                        ada_emb, gemm_scratch);
    auto* ada_mods = (float*)ada_emb.data;

    // Final AdaLayerNormContinuous in FP32, convert to BF16 only after proj_out.
    layer_norm_modulate_fp32((float*)img.data,
                            ada_mods + inner_dim,   // shift (second half)
                            ada_mods + 0,           // scale (first half)
                            (float*)img_normed_fp32.data, n_img, inner_dim, 1e-6f);

    dump("cuda_img_after_adanorm.bin", img_normed_fp32.data, (int64_t)n_img * inner_dim, false);

    // 5. Project out: [n_img, 3072] -> [n_img, 64]
    Tensor img_proj_out_fp32 = ws.view(
        ws.img_proj_out_fp32,
        (int64_t)n_img * patch_size * patch_size * out_channels,
        DType::FP32,
        {n_img, (int64_t)(patch_size * patch_size * out_channels)});
    linear_forward_quantized(img_normed_fp32, w.proj_out_weight, &w.proj_out_bias,
                        img_proj_out_fp32, gemm_scratch);

    Tensor img_proj_out = ws.view(
        ws.img_proj_out,
        (int64_t)n_img * patch_size * patch_size * out_channels,
        DType::BF16,
        {n_img, (int64_t)(patch_size * patch_size * out_channels)});
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

    CUDA_CHECK(cudaDeviceSynchronize());
    LOGV("Transformer done: output %s\n", output.shape_str().c_str());
    return output;
}
