#pragma once
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <algorithm>
#include "tensor.h"

// ===== Type conversion =====
void bf16_to_fp32(const __nv_bfloat16* in, float* out, int64_t n, cudaStream_t stream = 0);
void fp32_to_bf16(const float* in, __nv_bfloat16* out, int64_t n, cudaStream_t stream = 0);

// ===== Normalization =====
// RMSNorm: out = (x / rms(x)) * weight, eps = 1e-6
void rms_norm(const __nv_bfloat16* x, const __nv_bfloat16* weight, __nv_bfloat16* out,
              int rows, int dim, float eps = 1e-6f, cudaStream_t stream = 0);

// RMSNorm with FP32 weight
void rms_norm_f32w(const __nv_bfloat16* x, const float* weight, __nv_bfloat16* out,
                   int rows, int dim, float eps = 1e-6f, cudaStream_t stream = 0);

// LayerNorm without affine (elementwise_affine=false)
void layer_norm_no_affine(const __nv_bfloat16* x, __nv_bfloat16* out,
                          int rows, int dim, float eps = 1e-6f, cudaStream_t stream = 0);

// LayerNorm without affine - FP32 input, BF16 output (for mixed-precision residual)
void layer_norm_no_affine_fp32_in(const float* x, __nv_bfloat16* out,
                                    int rows, int dim, float eps = 1e-6f, cudaStream_t stream = 0);

// LayerNorm with affine: out = (x - mean) / std * weight + bias
void layer_norm_affine(const __nv_bfloat16* x, const __nv_bfloat16* weight, const __nv_bfloat16* bias,
                       __nv_bfloat16* out, int rows, int dim, float eps = 1e-5f, cudaStream_t stream = 0);

// GroupNorm with 32 groups (for VAE)
// x: [N, C, ...], weight: [C], bias: [C], out: [N, C, ...]
void group_norm_32(const __nv_bfloat16* x, const float* weight, const float* bias,
                   __nv_bfloat16* out, int N, int C, int spatial, float eps = 1e-6f, cudaStream_t stream = 0);

// WAN-style RMS norm: operates on channel dim of [C, T, H, W] tensor
// weight(gamma) shape: [C], applied after permute-normalize-permute
void rms_norm_channel(const __nv_bfloat16* x, const __nv_bfloat16* gamma, __nv_bfloat16* out,
                      int C, int spatial, float eps = 1e-12f, cudaStream_t stream = 0);

void rms_norm_channel_f32w(const __nv_bfloat16* x, const float* gamma, __nv_bfloat16* out,
                           int C, int spatial, float eps = 1e-12f, cudaStream_t stream = 0);

// ===== Activations =====
void silu_inplace(__nv_bfloat16* x, int64_t n, cudaStream_t stream = 0);
void silu(const __nv_bfloat16* x, __nv_bfloat16* out, int64_t n, cudaStream_t stream = 0);
void gelu(const __nv_bfloat16* x, __nv_bfloat16* out, int64_t n, cudaStream_t stream = 0);

// ===== Element-wise ops =====
void add_tensors(const __nv_bfloat16* a, const __nv_bfloat16* b, __nv_bfloat16* out, int64_t n, cudaStream_t stream = 0);
void add_inplace(__nv_bfloat16* a, const __nv_bfloat16* b, int64_t n, cudaStream_t stream = 0);
void mul_tensors(const __nv_bfloat16* a, const __nv_bfloat16* b, __nv_bfloat16* out, int64_t n, cudaStream_t stream = 0);
void scale_tensor(const __nv_bfloat16* x, __nv_bfloat16* out, float s, int64_t n, cudaStream_t stream = 0);
void scale_inplace(__nv_bfloat16* x, float s, int64_t n, cudaStream_t stream = 0);
// fma: out = a * b + c
void fma_tensors(const __nv_bfloat16* a, const __nv_bfloat16* b, const __nv_bfloat16* c,
                 __nv_bfloat16* out, int64_t n, cudaStream_t stream = 0);

// ===== Modulate =====
// out[row][col] = x[row][col] * (1 + scale[col]) + shift[col]
// x: [rows, dim], shift: [dim], scale: [dim]
void modulate(const __nv_bfloat16* x, const __nv_bfloat16* shift, const __nv_bfloat16* scale,
              __nv_bfloat16* out, int rows, int dim, cudaStream_t stream = 0);

// Gate: out = input + gate * x, where gate: [dim], x,input: [rows, dim]
void gate_add(const __nv_bfloat16* input, const __nv_bfloat16* gate, const __nv_bfloat16* x,
              __nv_bfloat16* out, int rows, int dim, cudaStream_t stream = 0);

// FP32 accumulator gate_add: out_fp32 = input_fp32 + bf16(gate) * bf16(x)
void gate_add_fp32(const float* input, const __nv_bfloat16* gate, const __nv_bfloat16* x,
                   float* out, int rows, int dim, cudaStream_t stream = 0);

// ===== Softmax =====
// Softmax along last dimension
void softmax(const float* x, float* out, int rows, int cols, cudaStream_t stream = 0);

// ===== Embedding =====
// out = table[indices], table: [vocab, dim], indices: [seq_len], out: [seq_len, dim]
void embedding_lookup(const __nv_bfloat16* table, const int32_t* indices, __nv_bfloat16* out,
                      int seq_len, int dim, cudaStream_t stream = 0);

// ===== Timestep embedding =====
// Sinusoidal timestep embedding: t -> [dim] vector
void timestep_embedding(const float* timesteps, __nv_bfloat16* out, int batch, int dim,
                        float max_period = 10000.0f, cudaStream_t stream = 0);
void timestep_embedding_fp32(const float* timesteps, float* out, int batch, int dim,
                             float max_period = 10000.0f, cudaStream_t stream = 0);

// ===== Concat =====
// Concatenate along sequence dim: [B, S1, D] + [B, S2, D] -> [B, S1+S2, D]
void concat_seq(const __nv_bfloat16* a, const __nv_bfloat16* b, __nv_bfloat16* out,
                int B, int S1, int S2, int D, cudaStream_t stream = 0);

// ===== Split =====
// Split along sequence dim: [B, S1+S2, D] -> [B, S1, D] + [B, S2, D]
void split_seq(const __nv_bfloat16* in, __nv_bfloat16* a, __nv_bfloat16* b,
               int B, int S1, int S2, int D, cudaStream_t stream = 0);

// ===== Repeat KV =====
// Repeat KV heads: [B, S, H_kv, D] -> [B, S, H_kv * repeats, D]
void repeat_kv(const __nv_bfloat16* in, __nv_bfloat16* out, int B, int S, int H_kv, int D, int repeats,
               cudaStream_t stream = 0);

// ===== Bias add =====
// Add bias to last dimension: x[..., dim] += bias[dim]
void bias_add(const __nv_bfloat16* x, const __nv_bfloat16* bias, __nv_bfloat16* out,
              int64_t outer, int dim, cudaStream_t stream = 0);

// ===== Upsample =====
// 2x nearest neighbor upsample of spatial dims
// in: [C, H, W], out: [C, 2H, 2W]
void upsample_nearest_2d(const __nv_bfloat16* in, __nv_bfloat16* out, int C, int H, int W, cudaStream_t stream = 0);

// ===== Conv2d via im2col + GEMM =====
// Standard Conv2d: in: [N, C_in, H, W], weight: [C_out, C_in, kH, kW], bias: [C_out]
// out: [N, C_out, H_out, W_out]
void conv2d_forward(const __nv_bfloat16* input, const __nv_bfloat16* weight, const __nv_bfloat16* bias,
                    __nv_bfloat16* output, __nv_bfloat16* workspace,
                    int N, int C_in, int H, int W, int C_out, int kH, int kW,
                    int padH, int padW, int strideH, int strideW,
                    cudaStream_t stream = 0);

// Compute workspace size for conv2d
size_t conv2d_workspace_size(int N, int C_in, int H, int W, int kH, int kW,
                             int padH, int padW, int strideH, int strideW);

// ===== CausalConv3d =====
// 3D convolution with causal temporal padding
// in: [C_in, T, H, W], weight: [C_out*C_in, kT, kH, kW], bias: [C_out]
// out: [C_out, T, H_out, W_out]
void causal_conv3d_forward(const __nv_bfloat16* input, const __nv_bfloat16* weight, const __nv_bfloat16* bias,
                           __nv_bfloat16* output, __nv_bfloat16* workspace,
                           int C_in, int T, int H, int W,
                           int C_out, int kT, int kH, int kW,
                           int padH, int padW,
                           int strideT, int strideH, int strideW,
                           cudaStream_t stream = 0);

size_t causal_conv3d_workspace_size(int C_in, int T, int H, int W, int kT, int kH, int kW,
                                    int padH, int padW, int strideT, int strideH, int strideW);

// ===== RoPE =====
// Apply precomputed RoPE: q/k shape [N*nhead, L, d_head], pe: [L, d_head/2, 2, 2]
void rope_apply(const __nv_bfloat16* x, const float* pe, __nv_bfloat16* out,
                int N_nhead, int L, int d_head, cudaStream_t stream = 0);

// ===== Attention =====
// Scaled dot-product attention (FP32 accumulation)
// q: [B*H, S, D], k: [B*H, S, D], v: [B*H, S, D], out: [B*H, S, D]
void attention_forward(const __nv_bfloat16* q, const __nv_bfloat16* k, const __nv_bfloat16* v,
                       __nv_bfloat16* out, float scale, int BH, int S, int D,
                       bool causal = false, cudaStream_t stream = 0);

// ===== Patchify =====
// [N, C, H, W] -> [N, (H/p)*(W/p), C*p*p]
void patchify(const __nv_bfloat16* in, __nv_bfloat16* out,
              int N, int C, int H, int W, int pH, int pW, cudaStream_t stream = 0);

// Unpatchify: [N, (H/p)*(W/p), C*p*p] -> [N, C, H, W]
void unpatchify(const __nv_bfloat16* in, __nv_bfloat16* out,
                int N, int C, int H, int W, int pH, int pW, cudaStream_t stream = 0);

// ===== Random number generation =====
void randn_fill(__nv_bfloat16* data, int64_t n, unsigned long long seed, cudaStream_t stream = 0);
// Philox 4x32 RNG matching sd.cpp/PyTorch CUDA RNG
void randn_fill_philox(__nv_bfloat16* data, int64_t n, uint64_t seed);

// ===== Image output helpers =====
// Clamp, scale, convert: BF16 [3, H, W] in [0,1] -> uint8 [H, W, 3] in [0,255]
void bf16_to_rgb8(const __nv_bfloat16* in, uint8_t* out, int H, int W, cudaStream_t stream = 0);

// ===== Linear layer implementation =====
void linear_forward(const Tensor& x, const Tensor& weight, const Tensor* bias, Tensor& out);
void linear_forward_batched(const Tensor& x, const Tensor& weight, const Tensor* bias, Tensor& out);

// ================================================================
// QUANTIZATION SUPPORT
// INT8: Per-channel symmetric + Hadamard rotation + INT8 tensor core GEMMs
// INT4: SVDQuant — per-group INT4 residual + BF16 low-rank correction
// ================================================================

enum class QuantMode { INT8_HADAMARD, INT4_SVD, BF16 };

// Quantized weight supporting both INT8 and INT4+SVD modes.
// Unused fields stay null (no GPU memory allocated).
struct QuantizedWeight {
    QuantMode mode = QuantMode::INT8_HADAMARD;

    // --- INT8 fields ---
    Tensor data;            // [N, K] INT8
    Tensor scales;          // [N] FP32
    int had_block_size;     // Hadamard block size (power of 2)

    // --- INT4+SVD fields ---
    Tensor qweight;         // [N, K/2] UINT8 (packed INT4, 2 per byte)
    Tensor scales4;         // [N, K/group_size] FP32
    Tensor svd_up;          // [N, r] BF16 (L_up = U_r * S_r)
    Tensor svd_down;        // [K, r] BF16 (L_down = V_r)
    Tensor smooth;          // [K] FP32 — SmoothQuant per-channel factors (optional)
    int group_size;         // INT4 quantization group size
    int svd_rank;           // SVD rank r
    bool nf4_grid;          // true = NF4 lookup, false = symmetric linear INT4 [-7,7]

    // --- Nunchaku swizzled INT4 fields ---
    bool nunchaku_swizzle;  // qweight/scales in MMA-swizzled layout
    Tensor wscales_bf16;    // [num_groups, N] BF16 swizzled scales

    // --- W4A4 row-major (unswizzled) fields for INT4×INT4 GEMM ---
    Tensor qweight_rowmajor;    // [N, K/2] UINT8 — row-major packed INT4 (two's complement nibbles)
    Tensor wscales_rowmajor;    // [num_groups, N] FP32 — row-major weight scales

    // --- W4A4 MMA fragment-ordered weights for register-direct GEMM ---
    Tensor qweight_mma;         // [num_groups * N_tiles8 * 256] UINT8 — MMA-fragment-ordered packed INT4

    QuantizedWeight() : had_block_size(0), group_size(128), svd_rank(0), nf4_grid(true), nunchaku_swizzle(false) {}
    QuantizedWeight(QuantizedWeight&&) = default;
    QuantizedWeight& operator=(QuantizedWeight&&) = default;

    void free_data() {
        data.free_data();
        scales.free_data();
        qweight.free_data();
        scales4.free_data();
        svd_up.free_data();
        svd_down.free_data();
        smooth.free_data();
        wscales_bf16.free_data();
        qweight_rowmajor.free_data();
        wscales_rowmajor.free_data();
        qweight_mma.free_data();
    }
};

// Fast Walsh-Hadamard Transform (in-place, block-diagonal)
// Applies FWHT to each row of data[M, K], using blocks of had_block_size.
// K must be divisible by had_block_size, and had_block_size must be power of 2.
void fwht_inplace(float* data, int M, int K, int had_block_size, cudaStream_t stream = 0);

// Compute the Hadamard block size for a given dimension (largest power of 2 dividing K)
inline int hadamard_block_size(int K) { return K & (-K); }

// Quantize BF16 weight to INT8 per-output-channel symmetric: scale[i] = max(|row_i|) / 127
void quantize_weight_per_channel(const __nv_bfloat16* weight, int8_t* out, float* scales,
                                  int N, int K, cudaStream_t stream = 0);

// Quantize FP32 data to INT8 per-row symmetric: scale[i] = max(|row_i|) / 127
void quantize_per_row_fp32(const float* x, int8_t* out, float* scales,
                            int M, int K, cudaStream_t stream = 0);

// Quantize FP32 activation to INT8 per-token: scale[i] = max(|row_i|) / 127
void quantize_activation_per_token_fp32(const float* x, int8_t* out, float* scales,
                                         int M, int K, cudaStream_t stream = 0);

// Quantize BF16 activation to INT8 per-token
void quantize_activation_per_token_bf16(const __nv_bfloat16* x, int8_t* out, float* scales,
                                         int M, int K, cudaStream_t stream = 0);

// Dequantize INT32 GEMM output: out_fp32[i,j] = gemm_out[i,j] * act_scale[i] * w_scale[j] + bias[j]
void dequantize_and_bias(const int32_t* gemm_out, const float* act_scales, const float* w_scales,
                          const __nv_bfloat16* bias, float* out,
                          int M, int N, cudaStream_t stream = 0);

// INT8 linear with Hadamard rotation: FP32 input × INT8 weight → FP32 output
// Applies FWHT to activation before quantization to match weight rotation.
void linear_forward_int8(const Tensor& x, const QuantizedWeight& weight,
                          const Tensor* bias, Tensor& out, Tensor& scratch);

// INT8 linear with Hadamard rotation: BF16 input × INT8 weight → FP32 output
void linear_forward_int8_bf16in(const Tensor& x, const QuantizedWeight& weight,
                                 const Tensor* bias, Tensor& out, Tensor& scratch);

// Helper: quantize a BF16 weight tensor at load time (with Hadamard rotation)
QuantizedWeight quantize_weight_tensor(const Tensor& bf16_weight);

// Helper: compute required scratch bytes for INT8 GEMM with Hadamard rotation
inline int64_t int8_scratch_bytes(int M, int K, int N) {
    // Region A: max(M*K*4, M*N*4) for FP32 temp (Hadamard) then reused for INT32 output
    int64_t regionA = (((int64_t)std::max((int64_t)M * K, (int64_t)M * N) * 4 + 255) & ~255LL);
    // INT8 quantized activations
    int64_t act = (((int64_t)M * K + 255) & ~255LL);
    // FP32 per-token scales
    int64_t scales = (((int64_t)M * 4 + 255) & ~255LL);
    return regionA + act + scales;
}

// ================================================================
// INT4 SVDQuant SUPPORT
// Per-group symmetric INT4 weight quantization with SVD low-rank
// correction. W4A16: dequant INT4→BF16, BF16 GEMM + low-rank.
// ================================================================

// Dequantize packed INT4 → BF16
// nf4_grid: true = NF4 lookup, false = symmetric linear INT4
void dequantize_int4_to_bf16(const uint8_t* qweight, const float* scales,
                              __nv_bfloat16* out, int N, int K, int group_size,
                              cudaStream_t stream = 0, bool nf4_grid = true,
                              bool scales_gn = false, bool signed_int4 = false);

// Dequantize nunchaku MMA-swizzled INT4 → BF16
// qweight: [N, K/2] UINT8 in MMA-swizzled layout
// wscales: [num_groups, N] BF16 in MMA-swizzled layout
void dequantize_nunchaku_to_bf16(const uint8_t* qweight, const __nv_bfloat16* wscales,
                                  __nv_bfloat16* out, int N, int K, int num_groups,
                                  cudaStream_t stream = 0);

// Quantize FP32 residual → packed INT4 + per-group scales
void quantize_int4_per_group(const float* residual, uint8_t* qweight, float* scales,
                              int N, int K, int group_size, cudaStream_t stream = 0);

// Dequantize nunchaku AWQ INT4 modulation weights to BF16 with output de-interleaving
// qweight: [OC/4, IC/2] INT32 (kInterleave=4), wscales/wzeros: [IC/group_size, OC] BF16
// out: [OC, IC] BF16 in standard (de-interleaved) order
// num_components: 6 for modulation (shift1,scale1,gate1,shift2,scale2,gate2)
void dequantize_awq_to_bf16(const int32_t* qweight, const __nv_bfloat16* wscales,
                              const __nv_bfloat16* wzeros, __nv_bfloat16* out,
                              int OC, int IC, int group_size, int num_components,
                              cudaStream_t stream = 0);

// Unswizzle nunchaku MMA-swizzled INT4 weights to row-major layout
// qweight_swizzled: [N, K/2] UINT8 swizzled, wscales_swizzled: [num_groups, N] BF16 swizzled
// qweight_rowmajor: [N, K/2] UINT8 row-major, wscales_rowmajor: [num_groups, N] FP32 row-major
void unswizzle_nunchaku_weights(const uint8_t* qweight_swizzled, const __nv_bfloat16* wscales_swizzled,
                                 uint8_t* qweight_rowmajor, float* wscales_rowmajor,
                                 int N, int K, int num_groups, cudaStream_t stream = 0);

// Unswizzle nunchaku MMA-packed LoRA weights to standard row-major BF16
// packed: [rows, cols] BF16 in MMA fragment-tiled layout
// standard: [rows, cols] BF16 in standard row-major layout
// is_down: true for proj_down, false for proj_up
void unswizzle_lora_weights(const __nv_bfloat16* packed, __nv_bfloat16* standard,
                             int rows, int cols, bool is_down, cudaStream_t stream = 0);

// Quantize FP32 activations to packed INT4 per-group-of-group_size
// x: [M, K] FP32 input
// act_packed: [M, K/2] UINT8 output (two's complement nibbles)
// act_scales: [num_groups, M] FP32 output (scale per group per token)
void quantize_activation_int4(const float* x, uint8_t* act_packed, float* act_scales,
                               int M, int K, int group_size, cudaStream_t stream = 0);

// Swizzle row-major INT4 weights into MMA fragment order for register-direct GEMM
// wgt_rowmajor: [N, K/2] UINT8, wgt_mma: [num_groups * N_tiles8 * 256] bytes
void swizzle_w4a4_weights_mma(const uint8_t* wgt_rowmajor, uint32_t* wgt_mma,
                               int N, int K, int group_size, cudaStream_t stream = 0);

// INT4×INT4 GEMM with per-group dequantization
// act_packed: [M, K/2] UINT8, wgt_packed: [N, K/2] UINT8 (row-major)
// act_scales: [num_groups, M] FP32, wgt_scales: [num_groups, N] FP32
// output: [M, N] FP32
// wgt_mma: if non-null, use register-direct kernel (MMA fragment-ordered weights)
void w4a4_gemm(const uint8_t* act_packed, const uint8_t* wgt_packed,
               const float* act_scales, const float* wgt_scales,
               float* output, int M, int N, int K, int group_size,
               const uint32_t* wgt_mma = nullptr, bool accumulate = false,
               const __nv_bfloat16* bias = nullptr,
               cudaStream_t stream = 0);

// Fused: smooth_div + fp32_to_bf16 + quantize_activation_int4 in one pass
void fused_smooth_bf16_quant(const float* x, const float* smooth,
                              __nv_bfloat16* bf16_out,
                              uint8_t* act_packed, float* act_scales,
                              int M, int K, int group_size, cudaStream_t stream = 0);

// Element-wise FP32 add: out[i] = a[i] + b[i]
void add_fp32(const float* a, const float* b, float* out, int64_t n, cudaStream_t stream = 0);

// INT4+SVD linear: FP32 input × INT4 weight → FP32 output
// When g_w4a4_mode is true, activations are also quantized to INT4 (simulated W4A4)
extern bool g_w4a4_mode;
void linear_forward_int4(const Tensor& x, const QuantizedWeight& weight,
                          const Tensor* bias, Tensor& out, Tensor& scratch);

// Unified dispatchers: pick INT8 or INT4 based on weight.mode
void linear_forward_quantized(const Tensor& x, const QuantizedWeight& weight,
                               const Tensor* bias, Tensor& out, Tensor& scratch);
void linear_forward_quantized_bf16in(const Tensor& x, const QuantizedWeight& weight,
                                      const Tensor* bias, Tensor& out, Tensor& scratch);

// Quantize a BF16 weight to INT4+SVD (offline tool)
QuantizedWeight quantize_weight_tensor_int4(const Tensor& bf16_weight, int rank, int group_size);

// GPTQ-guided INT4+SVD quantization (offline tool)
// Same as above but uses calibration activations to minimize reconstruction error.
// x_rot: Hadamard-rotated calibration activations [M_total, K] in FP32 (GPU)
// no_hadamard: skip FWHT rotation on residual (SVDQuant-style)
// nf4_grid: true = NF4 lookup table, false = symmetric INT4 [-7,7]
// error_svd: if true, SVD the quantization error instead of the weight (SVDQuant approach)
QuantizedWeight quantize_weight_tensor_int4_gptq(const Tensor& bf16_weight, int rank, int group_size,
                                                   const float* x_rot_gpu, int M_total, int K,
                                                   bool no_hadamard = false, bool nf4_grid = true,
                                                   bool error_svd = false);

// GPTQ+Smooth INT4+SVD quantization (offline tool)
// Takes RAW (non-rotated) calibration activations, computes smooth factors,
// applies smooth to weight, then runs SVD + GPTQ on the smoothed weight.
// x_raw_gpu: raw calibration activations [M_total, K] in FP32 (GPU)
QuantizedWeight quantize_weight_tensor_int4_gptq_smooth(const Tensor& bf16_weight, int rank, int group_size,
                                                          const float* x_raw_gpu, int M_total, int K);

// Apply smooth division: out[i] = x[i] / smooth[i % K], for x of shape [M, K]
void apply_smooth_div(const float* x, const float* smooth, float* out, int M, int K, cudaStream_t stream = 0);

// Fused smooth division + FWHT: divides each row by smooth, then applies block-diagonal FWHT in-place
void smooth_hadamard_inplace(float* data, const float* smooth, int M, int K, int had_block_size, cudaStream_t stream = 0);

// ================================================================
// GPTQ CALIBRATION DATA READER
// ================================================================

struct CalibrationReader {
    struct Entry {
        std::string name;
        int M, K, had_block_size;
        std::vector<__nv_bfloat16> data; // BF16 on CPU
    };
    std::vector<Entry> entries;
    int version = 1; // 1 = FWHT-rotated activations, 2 = raw activations

    bool load(const char* path);

    // Collect all entries with the given name prefix, concatenate and convert to FP32 on GPU.
    // Returns nullptr if no entries found, caller must cudaFree.
    // Sets out_M to total M (sum of all matching entries).
    float* get_activation_gpu(const std::string& name, int& out_M, int& out_K) const;
};

// Scratch bytes for INT4 forward
inline int64_t int4_scratch_bytes(int M, int K, int N, int svd_rank) {
    // Region A: max of W4A16 (BF16 dequant weight) or W4A4 (act_packed + act_scales + w4a4_out)
    int64_t regionA_w4a16 = (int64_t)N * K * 2;
    int num_groups = K / 64; // conservative: smallest group_size = 64
    int64_t regionA_w4a4 = (((int64_t)M * (K / 2) + 255) & ~255LL)
                         + (((int64_t)num_groups * M * 4 + 255) & ~255LL)
                         + (int64_t)M * N * 4;
    int64_t regionA = (((std::max(regionA_w4a16, regionA_w4a4)) + 255) & ~255LL);
    int64_t fp32_temp  = (((int64_t)M * K * 4 + 255) & ~255LL);  // FP32 temp for Hadamard [M,K]
    int64_t bf16_x     = (((int64_t)M * K * 2 + 255) & ~255LL);  // BF16 input [M,K]
    int64_t lr_inter   = (((int64_t)M * svd_rank * 2 + 255) & ~255LL); // BF16 low-rank tmp [M,r]
    return regionA + fp32_temp + bf16_x + lr_inter;
}

// ================================================================
// FP32 INTERMEDIATE PRECISION KERNELS
// Keep activations in FP32 between GEMMs to match ggml precision.
// ================================================================

// Linear with FP32 output: BF16 input × BF16 weight → BF16 GEMM → FP32 output
// If gemm_scratch is provided, matches ggml's BF16 output + convert pattern.
void linear_forward_fp32out(const Tensor& x, const Tensor& weight, const Tensor* bias, Tensor& out,
                            Tensor* gemm_scratch = nullptr);
void linear_forward_batched_fp32out(const Tensor& x, const Tensor& weight, const Tensor* bias, Tensor& out);

// Linear with FP32 input/output, BF16 weights. Matches ggml's BF16 GEMM path:
// FP32 input → BF16, BF16×BF16→BF16 GEMM, BF16 → FP32 output.
void linear_forward_fp32in_bf16w_fp32out(const Tensor& x, const Tensor& weight, const Tensor* bias,
                                         Tensor& out, Tensor& gemm_scratch);

// Bias add: FP32 x + BF16 bias → FP32 out
void bias_add_fp32(const float* x, const __nv_bfloat16* bias, float* out,
                   int64_t outer, int dim, cudaStream_t stream = 0);

// RMSNorm: FP32 input, BF16 weight, FP32 output
void rms_norm_fp32(const float* x, const __nv_bfloat16* weight, float* out,
                   int rows, int dim, float eps = 1e-6f, cudaStream_t stream = 0);

// LayerNorm (no affine): FP32 input, FP32 output
void layer_norm_no_affine_fp32(const float* x, float* out,
                                int rows, int dim, float eps = 1e-6f, cudaStream_t stream = 0);

// Modulate: FP32 x, BF16 shift/scale, FP32 output
void modulate_fp32(const float* x, const __nv_bfloat16* shift, const __nv_bfloat16* scale,
                   float* out, int rows, int dim, cudaStream_t stream = 0);
void modulate_fp32_f32params(const float* x, const float* shift, const float* scale,
                             float* out, int rows, int dim, cudaStream_t stream = 0);

// GELU: FP32 in/out
void gelu_fp32(const float* x, float* out, int64_t n, cudaStream_t stream = 0);

// SiLU: FP32 in/out
void silu_fp32(const float* x, float* out, int64_t n, cudaStream_t stream = 0);

// RoPE: FP32 x, FP32 pe, FP32 output
void rope_apply_fp32(const float* x, const float* pe, float* out,
                     int N_nhead, int L, int d_head, cudaStream_t stream = 0);

// Concat/Split: FP32 versions
void concat_seq_fp32(const float* a, const float* b, float* out,
                     int B, int S1, int S2, int D, cudaStream_t stream = 0);
void split_seq_fp32(const float* in, float* a, float* b,
                    int B, int S1, int S2, int D, cudaStream_t stream = 0);

// Attention: FP32 Q/K/V → FP32 output (converts to BF16 internally for GEMM)
void attention_forward_fp32io(const float* q, const float* k, const float* v,
                               float* out, float scale, int BH, int S, int D,
                               bool causal = false, cudaStream_t stream = 0);

// Attention: FP32 Q/K/V → FP32 output, with full-FP32 GEMMs.
void attention_forward_fp32(const float* q, const float* k, const float* v,
                            float* out, float scale, int BH, int S, int D,
                            bool causal = false, cudaStream_t stream = 0);

// Gate add: FP32 input + BF16 gate × FP32 x → FP32 output
void gate_add_fp32v2(const float* input, const __nv_bfloat16* gate, const float* x,
                     float* out, int rows, int dim, cudaStream_t stream = 0);
void gate_add_fp32f(const float* input, const float* gate, const float* x,
                    float* out, int rows, int dim, cudaStream_t stream = 0);
