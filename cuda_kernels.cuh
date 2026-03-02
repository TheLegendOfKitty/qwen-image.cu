#pragma once
#include <cuda_runtime.h>
#include <cuda_bf16.h>
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
// FP32 INTERMEDIATE PRECISION KERNELS
// Keep activations in FP32 between GEMMs to match ggml precision.
// ================================================================

// Linear with FP32 output: BF16 input × BF16 weight → FP32 output
void linear_forward_fp32out(const Tensor& x, const Tensor& weight, const Tensor* bias, Tensor& out);
void linear_forward_batched_fp32out(const Tensor& x, const Tensor& weight, const Tensor* bias, Tensor& out);

// Linear with FP32 activations and FP32 output. BF16 weights are expanded into
// the provided FP32 scratch buffer before GEMM.
void linear_forward_fp32in_bf16w_fp32out(const Tensor& x, const Tensor& weight, const Tensor* bias,
                                         Tensor& out, Tensor& weight_scratch);

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
