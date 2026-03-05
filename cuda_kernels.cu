#include "cuda_kernels.cuh"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <curand_kernel.h>
#include <cusolverDn.h>
#include <cmath>
#include <cstdio>
#include <cfloat>

// Helper: convert bf16 to float
__device__ __forceinline__ float bf16_to_float(__nv_bfloat16 x) {
    return __bfloat162float(x);
}

__device__ __forceinline__ __nv_bfloat16 float_to_bf16(float x) {
    return __float2bfloat16(x);
}

// ==================== Type Conversion ====================

__global__ void bf16_to_fp32_kernel(const __nv_bfloat16* in, float* out, int64_t n) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = bf16_to_float(in[i]);
}

void bf16_to_fp32(const __nv_bfloat16* in, float* out, int64_t n, cudaStream_t stream) {
    int block = 256;
    int grid = (int)((n + block - 1) / block);
    bf16_to_fp32_kernel<<<grid, block, 0, stream>>>(in, out, n);
}

__global__ void fp32_to_bf16_kernel(const float* in, __nv_bfloat16* out, int64_t n) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = float_to_bf16(in[i]);
}

void fp32_to_bf16(const float* in, __nv_bfloat16* out, int64_t n, cudaStream_t stream) {
    int block = 256;
    int grid = (int)((n + block - 1) / block);
    fp32_to_bf16_kernel<<<grid, block, 0, stream>>>(in, out, n);
}

__global__ void fp32_to_fp16_kernel(const float* in, __half* out, int64_t n) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2half(in[i]);
}

void fp32_to_fp16(const float* in, __half* out, int64_t n, cudaStream_t stream) {
    int block = 256;
    int grid = (int)((n + block - 1) / block);
    fp32_to_fp16_kernel<<<grid, block, 0, stream>>>(in, out, n);
}

__global__ void bf16_to_fp16_kernel(const __nv_bfloat16* in, __half* out, int64_t n) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2half(__bfloat162float(in[i]));
}

void bf16_to_fp16(const __nv_bfloat16* in, __half* out, int64_t n, cudaStream_t stream) {
    int block = 256;
    int grid = (int)((n + block - 1) / block);
    bf16_to_fp16_kernel<<<grid, block, 0, stream>>>(in, out, n);
}

// ==================== Element-wise Subtract ====================

__global__ void elementwise_sub_kernel(float* a, const float* b, int64_t n) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) a[idx] -= b[idx];
}

void elementwise_sub(float* a, const float* b, int64_t n, cudaStream_t stream = 0) {
    int block = 256;
    int grid = (int)((n + block - 1) / block);
    elementwise_sub_kernel<<<grid, block, 0, stream>>>(a, b, n);
}

// ==================== RMS Norm ====================

__global__ void rms_norm_kernel(const __nv_bfloat16* x, const __nv_bfloat16* weight,
                                 __nv_bfloat16* out, int dim, float eps) {
    int row = blockIdx.x;
    const __nv_bfloat16* x_row = x + (int64_t)row * dim;
    __nv_bfloat16* o_row = out + (int64_t)row * dim;

    // Compute sum of squares
    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float v = bf16_to_float(x_row[i]);
        sum_sq += v * v;
    }

    // Warp reduction
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);

    // Cross-warp reduction via shared memory
    __shared__ float shared[32];
    int lane = threadIdx.x % warpSize;
    int warp = threadIdx.x / warpSize;
    if (lane == 0) shared[warp] = sum_sq;
    __syncthreads();

    if (warp == 0) {
        sum_sq = (lane < (blockDim.x + warpSize - 1) / warpSize) ? shared[lane] : 0.0f;
        for (int offset = warpSize / 2; offset > 0; offset >>= 1)
            sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
    }
    __syncthreads();

    if (threadIdx.x == 0) shared[0] = sum_sq;
    __syncthreads();
    sum_sq = shared[0];

    float rms = rsqrtf(sum_sq / dim + eps);

    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float v = bf16_to_float(x_row[i]) * rms;
        float w = bf16_to_float(weight[i]);
        o_row[i] = float_to_bf16(v * w);
    }
}

void rms_norm(const __nv_bfloat16* x, const __nv_bfloat16* weight, __nv_bfloat16* out,
              int rows, int dim, float eps, cudaStream_t stream) {
    int threads = min(1024, ((dim + 31) / 32) * 32);
    rms_norm_kernel<<<rows, threads, 0, stream>>>(x, weight, out, dim, eps);
}

__global__ void rms_norm_f32w_kernel(const __nv_bfloat16* x, const float* weight,
                                      __nv_bfloat16* out, int dim, float eps) {
    int row = blockIdx.x;
    const __nv_bfloat16* x_row = x + (int64_t)row * dim;
    __nv_bfloat16* o_row = out + (int64_t)row * dim;

    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float v = bf16_to_float(x_row[i]);
        sum_sq += v * v;
    }

    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);

    __shared__ float shared[32];
    int lane = threadIdx.x % warpSize;
    int warp = threadIdx.x / warpSize;
    if (lane == 0) shared[warp] = sum_sq;
    __syncthreads();

    if (warp == 0) {
        sum_sq = (lane < (blockDim.x + warpSize - 1) / warpSize) ? shared[lane] : 0.0f;
        for (int offset = warpSize / 2; offset > 0; offset >>= 1)
            sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
    }
    __syncthreads();
    if (threadIdx.x == 0) shared[0] = sum_sq;
    __syncthreads();
    sum_sq = shared[0];

    float rms = rsqrtf(sum_sq / dim + eps);

    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float v = bf16_to_float(x_row[i]) * rms;
        o_row[i] = float_to_bf16(v * weight[i]);
    }
}

void rms_norm_f32w(const __nv_bfloat16* x, const float* weight, __nv_bfloat16* out,
                   int rows, int dim, float eps, cudaStream_t stream) {
    int threads = min(1024, ((dim + 31) / 32) * 32);
    rms_norm_f32w_kernel<<<rows, threads, 0, stream>>>(x, weight, out, dim, eps);
}

// ==================== Layer Norm ====================

__global__ void layer_norm_no_affine_kernel(const __nv_bfloat16* x, __nv_bfloat16* out,
                                             int dim, float eps) {
    int row = blockIdx.x;
    const __nv_bfloat16* x_row = x + (int64_t)row * dim;
    __nv_bfloat16* o_row = out + (int64_t)row * dim;

    // Compute mean
    float sum = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x)
        sum += bf16_to_float(x_row[i]);

    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffff, sum, offset);

    __shared__ float shared[32];
    int lane = threadIdx.x % warpSize;
    int warp = threadIdx.x / warpSize;
    if (lane == 0) shared[warp] = sum;
    __syncthreads();

    if (warp == 0) {
        sum = (lane < (blockDim.x + warpSize - 1) / warpSize) ? shared[lane] : 0.0f;
        for (int offset = warpSize / 2; offset > 0; offset >>= 1)
            sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    __syncthreads();
    if (threadIdx.x == 0) shared[0] = sum;
    __syncthreads();
    float mean = shared[0] / dim;

    // Compute variance
    float var_sum = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float v = bf16_to_float(x_row[i]) - mean;
        var_sum += v * v;
    }

    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        var_sum += __shfl_down_sync(0xffffffff, var_sum, offset);

    if (lane == 0) shared[warp] = var_sum;
    __syncthreads();

    if (warp == 0) {
        var_sum = (lane < (blockDim.x + warpSize - 1) / warpSize) ? shared[lane] : 0.0f;
        for (int offset = warpSize / 2; offset > 0; offset >>= 1)
            var_sum += __shfl_down_sync(0xffffffff, var_sum, offset);
    }
    __syncthreads();
    if (threadIdx.x == 0) shared[0] = var_sum;
    __syncthreads();
    float inv_std = rsqrtf(shared[0] / dim + eps);

    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float v = (bf16_to_float(x_row[i]) - mean) * inv_std;
        o_row[i] = float_to_bf16(v);
    }
}

// LayerNorm with FP32 input and BF16 output (for mixed-precision residual stream)
__global__ void layer_norm_no_affine_fp32_in_kernel(const float* x, __nv_bfloat16* out,
                                                     int dim, float eps) {
    int row = blockIdx.x;
    const float* x_row = x + (int64_t)row * dim;
    __nv_bfloat16* o_row = out + (int64_t)row * dim;

    // Compute mean
    float sum = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x)
        sum += x_row[i];

    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffff, sum, offset);

    __shared__ float shared[32];
    int lane = threadIdx.x % warpSize;
    int warp = threadIdx.x / warpSize;
    if (lane == 0) shared[warp] = sum;
    __syncthreads();

    if (warp == 0) {
        sum = (lane < (blockDim.x + warpSize - 1) / warpSize) ? shared[lane] : 0.0f;
        for (int offset = warpSize / 2; offset > 0; offset >>= 1)
            sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    __syncthreads();
    if (threadIdx.x == 0) shared[0] = sum;
    __syncthreads();
    float mean = shared[0] / dim;

    // Compute variance
    float var_sum = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float v = x_row[i] - mean;
        var_sum += v * v;
    }

    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        var_sum += __shfl_down_sync(0xffffffff, var_sum, offset);

    if (lane == 0) shared[warp] = var_sum;
    __syncthreads();

    if (warp == 0) {
        var_sum = (lane < (blockDim.x + warpSize - 1) / warpSize) ? shared[lane] : 0.0f;
        for (int offset = warpSize / 2; offset > 0; offset >>= 1)
            var_sum += __shfl_down_sync(0xffffffff, var_sum, offset);
    }
    __syncthreads();
    if (threadIdx.x == 0) shared[0] = var_sum;
    __syncthreads();
    float inv_std = rsqrtf(shared[0] / dim + eps);

    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float v = (x_row[i] - mean) * inv_std;
        o_row[i] = float_to_bf16(v);
    }
}

void layer_norm_no_affine_fp32_in(const float* x, __nv_bfloat16* out,
                                    int rows, int dim, float eps, cudaStream_t stream) {
    int block = min(256, dim);
    layer_norm_no_affine_fp32_in_kernel<<<rows, block, 0, stream>>>(x, out, dim, eps);
}

void layer_norm_no_affine(const __nv_bfloat16* x, __nv_bfloat16* out,
                          int rows, int dim, float eps, cudaStream_t stream) {
    int threads = min(1024, ((dim + 31) / 32) * 32);
    layer_norm_no_affine_kernel<<<rows, threads, 0, stream>>>(x, out, dim, eps);
}

__global__ void layer_norm_affine_kernel(const __nv_bfloat16* x, const __nv_bfloat16* weight,
                                          const __nv_bfloat16* bias, __nv_bfloat16* out,
                                          int dim, float eps) {
    int row = blockIdx.x;
    const __nv_bfloat16* x_row = x + (int64_t)row * dim;
    __nv_bfloat16* o_row = out + (int64_t)row * dim;

    float sum = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x)
        sum += bf16_to_float(x_row[i]);

    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffff, sum, offset);

    __shared__ float shared[32];
    int lane = threadIdx.x % warpSize;
    int warp = threadIdx.x / warpSize;
    if (lane == 0) shared[warp] = sum;
    __syncthreads();
    if (warp == 0) {
        sum = (lane < (blockDim.x + warpSize - 1) / warpSize) ? shared[lane] : 0.0f;
        for (int offset = warpSize / 2; offset > 0; offset >>= 1)
            sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    __syncthreads();
    if (threadIdx.x == 0) shared[0] = sum;
    __syncthreads();
    float mean = shared[0] / dim;

    float var_sum = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float v = bf16_to_float(x_row[i]) - mean;
        var_sum += v * v;
    }
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        var_sum += __shfl_down_sync(0xffffffff, var_sum, offset);
    if (lane == 0) shared[warp] = var_sum;
    __syncthreads();
    if (warp == 0) {
        var_sum = (lane < (blockDim.x + warpSize - 1) / warpSize) ? shared[lane] : 0.0f;
        for (int offset = warpSize / 2; offset > 0; offset >>= 1)
            var_sum += __shfl_down_sync(0xffffffff, var_sum, offset);
    }
    __syncthreads();
    if (threadIdx.x == 0) shared[0] = var_sum;
    __syncthreads();
    float inv_std = rsqrtf(shared[0] / dim + eps);

    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float v = (bf16_to_float(x_row[i]) - mean) * inv_std;
        float w = bf16_to_float(weight[i]);
        float b = bias ? bf16_to_float(bias[i]) : 0.0f;
        o_row[i] = float_to_bf16(v * w + b);
    }
}

void layer_norm_affine(const __nv_bfloat16* x, const __nv_bfloat16* weight, const __nv_bfloat16* bias,
                       __nv_bfloat16* out, int rows, int dim, float eps, cudaStream_t stream) {
    int threads = min(1024, ((dim + 31) / 32) * 32);
    layer_norm_affine_kernel<<<rows, threads, 0, stream>>>(x, weight, bias, out, dim, eps);
}

// ==================== Channel RMS Norm (WAN style) ====================
// Input: [C, T, H, W] laid out as [C, spatial] where spatial = T*H*W
// For each spatial position, normalize across C channels, then multiply by gamma[c]

__global__ void rms_norm_channel_kernel(const __nv_bfloat16* x, const __nv_bfloat16* gamma,
                                         __nv_bfloat16* out, int C, int spatial, float eps) {
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= spatial) return;

    // Compute RMS across channels for this spatial position
    float sum_sq = 0.0f;
    for (int c = 0; c < C; c++) {
        float v = bf16_to_float(x[(int64_t)c * spatial + s]);
        sum_sq += v * v;
    }
    float rms = rsqrtf(sum_sq / C + eps);

    for (int c = 0; c < C; c++) {
        float v = bf16_to_float(x[(int64_t)c * spatial + s]) * rms;
        float g = bf16_to_float(gamma[c]);
        out[(int64_t)c * spatial + s] = float_to_bf16(v * g);
    }
}

void rms_norm_channel(const __nv_bfloat16* x, const __nv_bfloat16* gamma, __nv_bfloat16* out,
                      int C, int spatial, float eps, cudaStream_t stream) {
    int block = 256;
    int grid = (spatial + block - 1) / block;
    rms_norm_channel_kernel<<<grid, block, 0, stream>>>(x, gamma, out, C, spatial, eps);
}

__global__ void rms_norm_channel_f32w_kernel(const __nv_bfloat16* x, const float* gamma,
                                              __nv_bfloat16* out, int C, int spatial, float eps) {
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= spatial) return;

    float sum_sq = 0.0f;
    for (int c = 0; c < C; c++) {
        float v = bf16_to_float(x[(int64_t)c * spatial + s]);
        sum_sq += v * v;
    }
    float rms = rsqrtf(sum_sq / C + eps);

    for (int c = 0; c < C; c++) {
        float v = bf16_to_float(x[(int64_t)c * spatial + s]) * rms;
        out[(int64_t)c * spatial + s] = float_to_bf16(v * gamma[c]);
    }
}

void rms_norm_channel_f32w(const __nv_bfloat16* x, const float* gamma, __nv_bfloat16* out,
                           int C, int spatial, float eps, cudaStream_t stream) {
    int block = 256;
    int grid = (spatial + block - 1) / block;
    rms_norm_channel_f32w_kernel<<<grid, block, 0, stream>>>(x, gamma, out, C, spatial, eps);
}

// ==================== Activations ====================

__global__ void silu_kernel(__nv_bfloat16* x, int64_t n) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = bf16_to_float(x[i]);
        v = v / (1.0f + expf(-v));
        x[i] = float_to_bf16(v);
    }
}

void silu_inplace(__nv_bfloat16* x, int64_t n, cudaStream_t stream) {
    int block = 256;
    int grid = (int)((n + block - 1) / block);
    silu_kernel<<<grid, block, 0, stream>>>(x, n);
}

__global__ void silu_out_kernel(const __nv_bfloat16* x, __nv_bfloat16* out, int64_t n) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = bf16_to_float(x[i]);
        out[i] = float_to_bf16(v / (1.0f + expf(-v)));
    }
}

void silu(const __nv_bfloat16* x, __nv_bfloat16* out, int64_t n, cudaStream_t stream) {
    int block = 256;
    int grid = (int)((n + block - 1) / block);
    silu_out_kernel<<<grid, block, 0, stream>>>(x, out, n);
}

__global__ void gelu_kernel(const __nv_bfloat16* x, __nv_bfloat16* out, int64_t n) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = bf16_to_float(x[i]);
        // GELU approximation (tanh version, matching PyTorch default)
        const float c = 0.7978845608f; // sqrt(2/pi)
        float inner = c * (v + 0.044715f * v * v * v);
        out[i] = float_to_bf16(0.5f * v * (1.0f + tanhf(inner)));
    }
}

void gelu(const __nv_bfloat16* x, __nv_bfloat16* out, int64_t n, cudaStream_t stream) {
    int block = 256;
    int grid = (int)((n + block - 1) / block);
    gelu_kernel<<<grid, block, 0, stream>>>(x, out, n);
}

// ==================== Element-wise ops ====================

__global__ void add_kernel(const __nv_bfloat16* a, const __nv_bfloat16* b, __nv_bfloat16* out, int64_t n) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = float_to_bf16(bf16_to_float(a[i]) + bf16_to_float(b[i]));
}

void add_tensors(const __nv_bfloat16* a, const __nv_bfloat16* b, __nv_bfloat16* out, int64_t n, cudaStream_t stream) {
    int block = 256;
    int grid = (int)((n + block - 1) / block);
    add_kernel<<<grid, block, 0, stream>>>(a, b, out, n);
}

void add_inplace(__nv_bfloat16* a, const __nv_bfloat16* b, int64_t n, cudaStream_t stream) {
    add_tensors(a, b, a, n, stream);
}

__global__ void mul_kernel(const __nv_bfloat16* a, const __nv_bfloat16* b, __nv_bfloat16* out, int64_t n) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = float_to_bf16(bf16_to_float(a[i]) * bf16_to_float(b[i]));
}

void mul_tensors(const __nv_bfloat16* a, const __nv_bfloat16* b, __nv_bfloat16* out, int64_t n, cudaStream_t stream) {
    int block = 256;
    int grid = (int)((n + block - 1) / block);
    mul_kernel<<<grid, block, 0, stream>>>(a, b, out, n);
}

__global__ void scale_kernel(const __nv_bfloat16* x, __nv_bfloat16* out, float s, int64_t n) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = float_to_bf16(bf16_to_float(x[i]) * s);
}

void scale_tensor(const __nv_bfloat16* x, __nv_bfloat16* out, float s, int64_t n, cudaStream_t stream) {
    int block = 256;
    int grid = (int)((n + block - 1) / block);
    scale_kernel<<<grid, block, 0, stream>>>(x, out, s, n);
}

void scale_inplace(__nv_bfloat16* x, float s, int64_t n, cudaStream_t stream) {
    scale_tensor(x, x, s, n, stream);
}

__global__ void fma_kernel(const __nv_bfloat16* a, const __nv_bfloat16* b,
                           const __nv_bfloat16* c, __nv_bfloat16* out, int64_t n) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = float_to_bf16(bf16_to_float(a[i]) * bf16_to_float(b[i]) + bf16_to_float(c[i]));
    }
}

void fma_tensors(const __nv_bfloat16* a, const __nv_bfloat16* b, const __nv_bfloat16* c,
                 __nv_bfloat16* out, int64_t n, cudaStream_t stream) {
    int block = 256;
    int grid = (int)((n + block - 1) / block);
    fma_kernel<<<grid, block, 0, stream>>>(a, b, c, out, n);
}

// ==================== Modulate ====================

__global__ void modulate_kernel(const __nv_bfloat16* x, const __nv_bfloat16* shift,
                                 const __nv_bfloat16* scale, __nv_bfloat16* out,
                                 int rows, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * dim;
    if (idx >= total) return;
    int col = idx % dim;
    float xv = bf16_to_float(x[idx]);
    float sv = bf16_to_float(shift[col]);
    float scv = bf16_to_float(scale[col]);
    out[idx] = float_to_bf16(xv * (1.0f + scv) + sv);
}

void modulate(const __nv_bfloat16* x, const __nv_bfloat16* shift, const __nv_bfloat16* scale,
              __nv_bfloat16* out, int rows, int dim, cudaStream_t stream) {
    int total = rows * dim;
    int block = 256;
    int grid = (total + block - 1) / block;
    modulate_kernel<<<grid, block, 0, stream>>>(x, shift, scale, out, rows, dim);
}

__global__ void gate_add_kernel(const __nv_bfloat16* input, const __nv_bfloat16* gate,
                                 const __nv_bfloat16* x, __nv_bfloat16* out,
                                 int rows, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * dim;
    if (idx >= total) return;
    int col = idx % dim;
    float inv = bf16_to_float(input[idx]);
    float gv = bf16_to_float(gate[col]);
    float xv = bf16_to_float(x[idx]);
    out[idx] = float_to_bf16(inv + gv * xv);
}

void gate_add(const __nv_bfloat16* input, const __nv_bfloat16* gate, const __nv_bfloat16* x,
              __nv_bfloat16* out, int rows, int dim, cudaStream_t stream) {
    int total = rows * dim;
    int block = 256;
    int grid = (total + block - 1) / block;
    gate_add_kernel<<<grid, block, 0, stream>>>(input, gate, x, out, rows, dim);
}

// FP32 accumulator version: out[idx] = input_fp32[idx] + bf16(gate[col]) * bf16(x[idx])
__global__ void gate_add_fp32_kernel(const float* input, const __nv_bfloat16* gate,
                                      const __nv_bfloat16* x, float* out,
                                      int rows, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * dim;
    if (idx >= total) return;
    int col = idx % dim;
    float inv = input[idx];
    float gv = bf16_to_float(gate[col]);
    float xv = bf16_to_float(x[idx]);
    out[idx] = inv + gv * xv;
}

void gate_add_fp32(const float* input, const __nv_bfloat16* gate, const __nv_bfloat16* x,
                   float* out, int rows, int dim, cudaStream_t stream) {
    int total = rows * dim;
    int block = 256;
    int grid = (total + block - 1) / block;
    gate_add_fp32_kernel<<<grid, block, 0, stream>>>(input, gate, x, out, rows, dim);
}

// ==================== Softmax ====================

__global__ void softmax_kernel(const float* x, float* out, int rows, int cols) {
    int row = blockIdx.x;
    const float* x_row = x + (int64_t)row * cols;
    float* o_row = out + (int64_t)row * cols;

    // Find max
    float max_val = -FLT_MAX;
    for (int i = threadIdx.x; i < cols; i += blockDim.x)
        max_val = fmaxf(max_val, x_row[i]);

    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        max_val = fmaxf(max_val, __shfl_down_sync(0xffffffff, max_val, offset));

    __shared__ float shared[32];
    int lane = threadIdx.x % warpSize;
    int warp = threadIdx.x / warpSize;
    if (lane == 0) shared[warp] = max_val;
    __syncthreads();
    if (warp == 0) {
        max_val = (lane < (blockDim.x + warpSize - 1) / warpSize) ? shared[lane] : -FLT_MAX;
        for (int offset = warpSize / 2; offset > 0; offset >>= 1)
            max_val = fmaxf(max_val, __shfl_down_sync(0xffffffff, max_val, offset));
    }
    __syncthreads();
    if (threadIdx.x == 0) shared[0] = max_val;
    __syncthreads();
    max_val = shared[0];

    // Compute exp and sum
    float sum = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float v = expf(x_row[i] - max_val);
        o_row[i] = v;
        sum += v;
    }

    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    if (lane == 0) shared[warp] = sum;
    __syncthreads();
    if (warp == 0) {
        sum = (lane < (blockDim.x + warpSize - 1) / warpSize) ? shared[lane] : 0.0f;
        for (int offset = warpSize / 2; offset > 0; offset >>= 1)
            sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    __syncthreads();
    if (threadIdx.x == 0) shared[0] = sum;
    __syncthreads();
    float inv_sum = 1.0f / shared[0];

    for (int i = threadIdx.x; i < cols; i += blockDim.x)
        o_row[i] *= inv_sum;
}

void softmax(const float* x, float* out, int rows, int cols, cudaStream_t stream) {
    int threads = min(1024, ((cols + 31) / 32) * 32);
    softmax_kernel<<<rows, threads, 0, stream>>>(x, out, rows, cols);
}

// ==================== Embedding ====================

__global__ void embedding_kernel(const __nv_bfloat16* table, const int32_t* indices,
                                  __nv_bfloat16* out, int seq_len, int dim) {
    int s = blockIdx.x;
    int d = threadIdx.x;
    if (s >= seq_len) return;
    int idx = indices[s];
    for (int i = d; i < dim; i += blockDim.x) {
        out[(int64_t)s * dim + i] = table[(int64_t)idx * dim + i];
    }
}

void embedding_lookup(const __nv_bfloat16* table, const int32_t* indices, __nv_bfloat16* out,
                      int seq_len, int dim, cudaStream_t stream) {
    int threads = min(1024, dim);
    embedding_kernel<<<seq_len, threads, 0, stream>>>(table, indices, out, seq_len, dim);
}

// ==================== Timestep Embedding ====================

__global__ void timestep_embedding_kernel(const float* timesteps, __nv_bfloat16* out,
                                           int batch, int dim, float max_period) {
    int b = blockIdx.x;
    int d = threadIdx.x;
    int half_dim = dim / 2;
    if (d >= half_dim) return;

    float t = timesteps[b];
    float freq = expf(-logf(max_period) * (float)d / (float)half_dim);
    float angle = t * freq;

    out[(int64_t)b * dim + d] = float_to_bf16(cosf(angle));
    out[(int64_t)b * dim + half_dim + d] = float_to_bf16(sinf(angle));
}

void timestep_embedding(const float* timesteps, __nv_bfloat16* out, int batch, int dim,
                        float max_period, cudaStream_t stream) {
    timestep_embedding_kernel<<<batch, dim / 2, 0, stream>>>(timesteps, out, batch, dim, max_period);
}

__global__ void timestep_embedding_fp32_kernel(const float* timesteps, float* out,
                                               int batch, int dim, float max_period) {
    int b = blockIdx.x;
    int d = threadIdx.x;
    int half_dim = dim / 2;
    if (d >= half_dim) return;

    float t = timesteps[b];
    float freq = expf(-logf(max_period) * (float)d / (float)half_dim);
    float angle = t * freq;

    out[(int64_t)b * dim + d] = cosf(angle);
    out[(int64_t)b * dim + half_dim + d] = sinf(angle);
}

void timestep_embedding_fp32(const float* timesteps, float* out, int batch, int dim,
                             float max_period, cudaStream_t stream) {
    timestep_embedding_fp32_kernel<<<batch, dim / 2, 0, stream>>>(timesteps, out, batch, dim, max_period);
}

// ==================== Concat / Split ====================

__global__ void concat_seq_kernel(const __nv_bfloat16* a, const __nv_bfloat16* b, __nv_bfloat16* out,
                                   int B, int S1, int S2, int D) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = (int64_t)B * (S1 + S2) * D;
    if (idx >= total) return;

    int d = idx % D;
    int64_t rest = idx / D;
    int s = rest % (S1 + S2);
    int batch = rest / (S1 + S2);

    if (s < S1) {
        out[idx] = a[((int64_t)batch * S1 + s) * D + d];
    } else {
        out[idx] = b[((int64_t)batch * S2 + (s - S1)) * D + d];
    }
}

void concat_seq(const __nv_bfloat16* a, const __nv_bfloat16* b, __nv_bfloat16* out,
                int B, int S1, int S2, int D, cudaStream_t stream) {
    int64_t total = (int64_t)B * (S1 + S2) * D;
    int block = 256;
    int grid = (int)((total + block - 1) / block);
    concat_seq_kernel<<<grid, block, 0, stream>>>(a, b, out, B, S1, S2, D);
}

void split_seq(const __nv_bfloat16* in, __nv_bfloat16* a, __nv_bfloat16* b,
               int B, int S1, int S2, int D, cudaStream_t stream) {
    // Copy first S1 rows per batch to a, next S2 to b
    for (int batch = 0; batch < B; batch++) {
        CUDA_CHECK(cudaMemcpyAsync(
            a + (int64_t)batch * S1 * D,
            in + (int64_t)batch * (S1 + S2) * D,
            (size_t)S1 * D * sizeof(__nv_bfloat16),
            cudaMemcpyDeviceToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(
            b + (int64_t)batch * S2 * D,
            in + ((int64_t)batch * (S1 + S2) + S1) * D,
            (size_t)S2 * D * sizeof(__nv_bfloat16),
            cudaMemcpyDeviceToDevice, stream));
    }
}

// ==================== Repeat KV ====================

__global__ void repeat_kv_kernel(const __nv_bfloat16* in, __nv_bfloat16* out,
                                  int B, int S, int H_kv, int D, int repeats) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = (int64_t)B * S * H_kv * repeats * D;
    if (idx >= total) return;

    int d = idx % D;
    int64_t rest = idx / D;
    int h = rest % (H_kv * repeats);
    rest = rest / (H_kv * repeats);
    int s = rest % S;
    int b = rest / S;
    int h_kv = h / repeats;

    out[idx] = in[((int64_t)b * S * H_kv + (int64_t)s * H_kv + h_kv) * D + d];
}

void repeat_kv(const __nv_bfloat16* in, __nv_bfloat16* out, int B, int S, int H_kv, int D, int repeats,
               cudaStream_t stream) {
    int64_t total = (int64_t)B * S * H_kv * repeats * D;
    int block = 256;
    int grid = (int)((total + block - 1) / block);
    repeat_kv_kernel<<<grid, block, 0, stream>>>(in, out, B, S, H_kv, D, repeats);
}

// ==================== Bias Add ====================

__global__ void bias_add_kernel(const __nv_bfloat16* x, const __nv_bfloat16* bias,
                                 __nv_bfloat16* out, int64_t outer, int dim) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = outer * dim;
    if (idx >= total) return;
    int d = idx % dim;
    out[idx] = float_to_bf16(bf16_to_float(x[idx]) + bf16_to_float(bias[d]));
}

void bias_add(const __nv_bfloat16* x, const __nv_bfloat16* bias, __nv_bfloat16* out,
              int64_t outer, int dim, cudaStream_t stream) {
    int64_t total = outer * dim;
    int block = 256;
    int grid = (int)((total + block - 1) / block);
    bias_add_kernel<<<grid, block, 0, stream>>>(x, bias, out, outer, dim);
}

// ==================== Upsample ====================

__global__ void upsample_nearest_2d_kernel(const __nv_bfloat16* in, __nv_bfloat16* out,
                                            int C, int H, int W) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = (int64_t)C * (2 * H) * (2 * W);
    if (idx >= total) return;

    int ow = idx % (2 * W);
    int64_t rest = idx / (2 * W);
    int oh = rest % (2 * H);
    int c = rest / (2 * H);

    int ih = oh / 2;
    int iw = ow / 2;
    out[idx] = in[(int64_t)c * H * W + ih * W + iw];
}

void upsample_nearest_2d(const __nv_bfloat16* in, __nv_bfloat16* out, int C, int H, int W,
                         cudaStream_t stream) {
    int64_t total = (int64_t)C * (2 * H) * (2 * W);
    int block = 256;
    int grid = (int)((total + block - 1) / block);
    upsample_nearest_2d_kernel<<<grid, block, 0, stream>>>(in, out, C, H, W);
}

// ==================== Conv2d via im2col + cuBLAS GEMM ====================

__global__ void im2col_kernel(const __nv_bfloat16* input, __nv_bfloat16* col,
                               int C_in, int H, int W, int kH, int kW,
                               int padH, int padW, int strideH, int strideW,
                               int H_out, int W_out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = C_in * kH * kW * H_out * W_out;
    if (idx >= total) return;

    int w_out = idx % W_out;
    int h_out = (idx / W_out) % H_out;
    int kw = (idx / (W_out * H_out)) % kW;
    int kh = (idx / (W_out * H_out * kW)) % kH;
    int c = idx / (W_out * H_out * kW * kH);

    int h_in = h_out * strideH - padH + kh;
    int w_in = w_out * strideW - padW + kw;

    __nv_bfloat16 val;
    if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
        val = input[(int64_t)c * H * W + h_in * W + w_in];
    } else {
        val = float_to_bf16(0.0f);
    }
    // col layout: [C_in*kH*kW, H_out*W_out]
    col[(int64_t)(c * kH * kW + kh * kW + kw) * (H_out * W_out) + h_out * W_out + w_out] = val;
}

size_t conv2d_workspace_size(int N, int C_in, int H, int W, int kH, int kW,
                             int padH, int padW, int strideH, int strideW) {
    int H_out = (H + 2 * padH - kH) / strideH + 1;
    int W_out = (W + 2 * padW - kW) / strideW + 1;
    return (size_t)C_in * kH * kW * H_out * W_out * sizeof(__nv_bfloat16);
}

// Forward declaration
__global__ void conv_bias_add_kernel(const __nv_bfloat16* bias, __nv_bfloat16* out, int C, int spatial);

void conv2d_forward(const __nv_bfloat16* input, const __nv_bfloat16* weight, const __nv_bfloat16* bias,
                    __nv_bfloat16* output, __nv_bfloat16* workspace,
                    int N, int C_in, int H, int W, int C_out, int kH, int kW,
                    int padH, int padW, int strideH, int strideW,
                    cudaStream_t stream) {
    int H_out = (H + 2 * padH - kH) / strideH + 1;
    int W_out = (W + 2 * padW - kW) / strideW + 1;

    for (int n = 0; n < N; n++) {
        const __nv_bfloat16* in_n = input + (int64_t)n * C_in * H * W;
        __nv_bfloat16* out_n = output + (int64_t)n * C_out * H_out * W_out;

        // im2col
        int total_col = C_in * kH * kW * H_out * W_out;
        int block = 256;
        int grid = (total_col + block - 1) / block;
        im2col_kernel<<<grid, block, 0, stream>>>(in_n, workspace,
            C_in, H, W, kH, kW, padH, padW, strideH, strideW, H_out, W_out);

        // GEMM: out = weight * col
        // weight: [C_out, C_in*kH*kW], col: [C_in*kH*kW, H_out*W_out]
        // out: [C_out, H_out*W_out]
        float alpha = 1.0f, beta = 0.0f;
        int M = C_out;
        int K_gemm = C_in * kH * kW;
        int N_gemm = H_out * W_out;

        CUBLAS_CHECK(cublasGemmEx(
            cublas(), CUBLAS_OP_N, CUBLAS_OP_N,
            N_gemm, M, K_gemm,
            &alpha,
            workspace, CUDA_R_16BF, N_gemm,
            weight, CUDA_R_16BF, K_gemm,
            &beta,
            out_n, CUDA_R_16BF, N_gemm,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));

        // Add bias: out[c, h, w] += bias[c]
        if (bias) {
            int spatial = H_out * W_out;
            int total_bias = C_out * spatial;
            int bk = 256;
            int gd = (total_bias + bk - 1) / bk;
            conv_bias_add_kernel<<<gd, bk, 0, stream>>>(bias, out_n, C_out, spatial);
        }
    }
}

// Proper bias add for conv output: out[c, h, w] += bias[c]
__global__ void conv_bias_add_kernel(const __nv_bfloat16* bias, __nv_bfloat16* out,
                                      int C, int spatial) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = (int64_t)C * spatial;
    if (idx >= total) return;
    int c = idx / spatial;
    float v = bf16_to_float(out[idx]) + bf16_to_float(bias[c]);
    out[idx] = float_to_bf16(v);
}

// ==================== CausalConv3d ====================

size_t causal_conv3d_workspace_size(int C_in, int T, int H, int W, int kT, int kH, int kW,
                                    int padH, int padW, int strideT, int strideH, int strideW) {
    int T_padded = T + (kT - 1); // causal padding
    int H_out = (H + 2 * padH - kH) / strideH + 1;
    int W_out = (W + 2 * padW - kW) / strideW + 1;
    int T_out = (T_padded - kT) / strideT + 1;
    return (size_t)C_in * kT * kH * kW * T_out * H_out * W_out * sizeof(__nv_bfloat16);
}

// im2col for 3D convolution
__global__ void im2col_3d_kernel(const __nv_bfloat16* input, __nv_bfloat16* col,
                                  int C_in, int T, int H, int W,
                                  int kT, int kH, int kW,
                                  int padH, int padW, int causal_pad,
                                  int strideT, int strideH, int strideW,
                                  int T_out, int H_out, int W_out) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = (int64_t)C_in * kT * kH * kW * T_out * H_out * W_out;
    if (idx >= total) return;

    int w_out = idx % W_out;
    int64_t rest = idx / W_out;
    int h_out = rest % H_out;
    rest = rest / H_out;
    int t_out = rest % T_out;
    rest = rest / T_out;
    int kw = rest % kW;
    rest = rest / kW;
    int kh = rest % kH;
    rest = rest / kH;
    int kt = rest % kT;
    int c = rest / kT;

    int t_in = t_out * strideT - causal_pad + kt;
    int h_in = h_out * strideH - padH + kh;
    int w_in = w_out * strideW - padW + kw;

    __nv_bfloat16 val;
    if (t_in >= 0 && t_in < T && h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
        val = input[((int64_t)c * T + t_in) * H * W + h_in * W + w_in];
    } else {
        val = float_to_bf16(0.0f);
    }

    int col_row = c * kT * kH * kW + kt * kH * kW + kh * kW + kw;
    int col_col = t_out * H_out * W_out + h_out * W_out + w_out;
    col[(int64_t)col_row * (T_out * H_out * W_out) + col_col] = val;
}

void causal_conv3d_forward(const __nv_bfloat16* input, const __nv_bfloat16* weight, const __nv_bfloat16* bias,
                           __nv_bfloat16* output, __nv_bfloat16* workspace,
                           int C_in, int T, int H, int W,
                           int C_out, int kT, int kH, int kW,
                           int padH, int padW,
                           int strideT, int strideH, int strideW,
                           cudaStream_t stream) {
    int causal_pad = kT - 1; // left temporal padding only
    int T_padded = T + causal_pad;
    int H_out = (H + 2 * padH - kH) / strideH + 1;
    int W_out = (W + 2 * padW - kW) / strideW + 1;
    int T_out = (T_padded - kT) / strideT + 1;

    // im2col 3D
    int64_t total_col = (int64_t)C_in * kT * kH * kW * T_out * H_out * W_out;
    int block = 256;
    int grid = (int)((total_col + block - 1) / block);
    im2col_3d_kernel<<<grid, block, 0, stream>>>(input, workspace,
        C_in, T, H, W, kT, kH, kW, padH, padW, causal_pad,
        strideT, strideH, strideW, T_out, H_out, W_out);

    // GEMM: weight[C_out, C_in*kT*kH*kW] @ col[C_in*kT*kH*kW, T_out*H_out*W_out]
    float alpha = 1.0f, beta = 0.0f;
    int M = C_out;
    int K_gemm = C_in * kT * kH * kW;
    int N_gemm = T_out * H_out * W_out;

    CUBLAS_CHECK(cublasGemmEx(
        cublas(), CUBLAS_OP_N, CUBLAS_OP_N,
        N_gemm, M, K_gemm,
        &alpha,
        workspace, CUDA_R_16BF, N_gemm,
        weight, CUDA_R_16BF, K_gemm,
        &beta,
        output, CUDA_R_16BF, N_gemm,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));

    // Add bias
    if (bias) {
        int spatial = T_out * H_out * W_out;
        int total_bias = C_out * spatial;
        int bk = 256;
        int gd = (total_bias + bk - 1) / bk;
        conv_bias_add_kernel<<<gd, bk, 0, stream>>>(bias, output, C_out, spatial);
    }
}

// ==================== RoPE ====================

// Apply RoPE rotation: x shape [N_nhead, L, d_head], pe shape [L, d_head/2, 2, 2]
// pe stores [[cos, -sin], [sin, cos]] rotation matrices
// RoPE is interleaved: pairs of (x[2i], x[2i+1]) are rotated
__global__ void rope_apply_kernel(const __nv_bfloat16* x, const float* pe, __nv_bfloat16* out,
                                   int N_nhead, int L, int d_head) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int half_d = d_head / 2;
    int64_t total = (int64_t)N_nhead * L * half_d;
    if (idx >= total) return;

    int pair = idx % half_d; // which pair of elements
    int64_t rest = idx / half_d;
    int l = rest % L; // sequence position
    int nh = rest / L; // batch*head

    // Get the two elements
    float x0 = bf16_to_float(x[((int64_t)nh * L + l) * d_head + pair * 2]);
    float x1 = bf16_to_float(x[((int64_t)nh * L + l) * d_head + pair * 2 + 1]);

    // Get rotation matrix elements from pe[l, pair, :, :]
    // pe layout: [L, half_d, 2, 2] -> pe[l][pair] = [[cos, -sin], [sin, cos]]
    // Reference applies the stored matrix directly: [x0, x1] @ [[cos, -sin], [sin, cos]]
    int pe_base = (l * half_d + pair) * 4;
    float cos_val = pe[pe_base + 0]; // M[0][0] = cos
    float neg_sin = pe[pe_base + 1]; // M[0][1] = -sin
    float sin_val = pe[pe_base + 2]; // M[1][0] = sin
    float cos_val2 = pe[pe_base + 3]; // M[1][1] = cos

    float out0 = x0 * cos_val + x1 * neg_sin;   // x0*cos + x1*(-sin)
    float out1 = x0 * sin_val + x1 * cos_val2;  // x0*sin + x1*cos

    out[((int64_t)nh * L + l) * d_head + pair * 2] = float_to_bf16(out0);
    out[((int64_t)nh * L + l) * d_head + pair * 2 + 1] = float_to_bf16(out1);
}

void rope_apply(const __nv_bfloat16* x, const float* pe, __nv_bfloat16* out,
                int N_nhead, int L, int d_head, cudaStream_t stream) {
    int half_d = d_head / 2;
    int64_t total = (int64_t)N_nhead * L * half_d;
    int block = 256;
    int grid = (int)((total + block - 1) / block);
    rope_apply_kernel<<<grid, block, 0, stream>>>(x, pe, out, N_nhead, L, d_head);
}

// ==================== Attention ====================

// Causal mask kernel: set scores[bh][q][k] = -FLT_MAX when k > q
__global__ void causal_mask_kernel(float* scores, int64_t total, int S) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int k_pos = idx % S;
    int q_pos = (idx / S) % S;
    if (k_pos > q_pos)
        scores[idx] = -FLT_MAX;
}

void apply_causal_mask(float* scores, int BH, int S, cudaStream_t stream) {
    int64_t total = (int64_t)BH * S * S;
    int block = 256;
    int grid = (int)((total + block - 1) / block);
    causal_mask_kernel<<<grid, block, 0, stream>>>(scores, total, S);
}

// Tiled attention using shared memory for BF16 inputs with FP32 accumulation
// For now, a straightforward implementation
void attention_forward(const __nv_bfloat16* q, const __nv_bfloat16* k, const __nv_bfloat16* v,
                       __nv_bfloat16* out, float scale, int BH, int S, int D,
                       bool causal, cudaStream_t stream) {
    // Allocate temporary FP32 buffers for attention scores
    float* scores;
    __half* q_fp16;
    __half* k_fp16;
    CUDA_CHECK(cudaMalloc(&scores, (size_t)BH * S * S * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&q_fp16, (size_t)BH * S * D * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&k_fp16, (size_t)BH * S * D * sizeof(__half)));
    bf16_to_fp16(q, q_fp16, (int64_t)BH * S * D, stream);
    bf16_to_fp16(k, k_fp16, (int64_t)BH * S * D, stream);

    // Q @ K^T: [BH, S, D] x [BH, D, S] = [BH, S, S]
    float alpha = scale;
    float beta_val = 0.0f;

    // Use cuBLAS strided batched GEMM
    // Q: [BH, S, D] row-major -> in cuBLAS column-major: [D, S] per batch
    // K^T: we want K transposed, K is [BH, S, D] -> K^T is [BH, D, S]
    // scores = Q @ K^T = [BH, S, S]

    // cuBLAS expects column-major. For row-major [S, D]:
    // Treat as column-major [D, S], so M=S, N=S, K=D
    // C = A * B^T in row-major = B * A^T in col-major
    // Actually: scores[bh] = Q[bh] @ K[bh]^T with scale
    // In cuBLAS col-major: C^T = K * Q^T, so C = Q * K^T
    // Let's just use: op(A)=N means A as-is in col-major

    CUBLAS_CHECK(cublasGemmStridedBatchedEx(
        cublas(),
        CUBLAS_OP_T,   // K transposed
        CUBLAS_OP_N,   // Q not transposed
        S, S, D,       // M, N, K
        &alpha,
        k_fp16, CUDA_R_16F, D, (long long)S * D,  // A = K, lda=D, stride
        q_fp16, CUDA_R_16F, D, (long long)S * D,  // B = Q, ldb=D, stride
        &beta_val,
        scores, CUDA_R_32F, S, (long long)S * S, // C = scores, ldc=S, stride
        BH,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP));

    // Apply causal mask if requested: set scores[bh][q][k] = -inf when k > q
    if (causal) {
        apply_causal_mask(scores, BH, S, stream);
    }

    // Softmax over last dim
    softmax(scores, scores, BH * S, S, stream);

    // scores @ V: [BH, S, S] x [BH, S, D] = [BH, S, D]
    // Convert scores to BF16 for the second GEMM
    __nv_bfloat16* scores_bf16;
    CUDA_CHECK(cudaMalloc(&scores_bf16, (size_t)BH * S * S * sizeof(__nv_bfloat16)));
    fp32_to_bf16(scores, scores_bf16, (int64_t)BH * S * S, stream);

    alpha = 1.0f;
    beta_val = 0.0f;

    CUBLAS_CHECK(cublasGemmStridedBatchedEx(
        cublas(),
        CUBLAS_OP_N,   // V not transposed
        CUBLAS_OP_N,   // scores not transposed
        D, S, S,       // M, N, K
        &alpha,
        v, CUDA_R_16BF, D, (long long)S * D,           // A = V
        scores_bf16, CUDA_R_16BF, S, (long long)S * S, // B = scores
        &beta_val,
        out, CUDA_R_16BF, D, (long long)S * D,         // C = out
        BH,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP));

    CUDA_CHECK(cudaFree(scores));
    CUDA_CHECK(cudaFree(scores_bf16));
    CUDA_CHECK(cudaFree(q_fp16));
    CUDA_CHECK(cudaFree(k_fp16));
}

// ==================== Patchify / Unpatchify ====================

__global__ void patchify_kernel(const __nv_bfloat16* in, __nv_bfloat16* out,
                                 int N, int C, int H, int W, int pH, int pW) {
    int h_patches = H / pH;
    int w_patches = W / pW;
    int patch_dim = C * pH * pW;
    int n_patches = h_patches * w_patches;

    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = (int64_t)N * n_patches * patch_dim;
    if (idx >= total) return;

    int d = idx % patch_dim;
    int64_t rest = idx / patch_dim;
    int p = rest % n_patches;
    int n = rest / n_patches;

    int hp = p / w_patches;
    int wp = p % w_patches;

    int c = d / (pH * pW);
    int dp = d % (pH * pW);
    int dh = dp / pW;
    int dw = dp % pW;

    int ih = hp * pH + dh;
    int iw = wp * pW + dw;

    out[idx] = in[((int64_t)n * C + c) * H * W + ih * W + iw];
}

void patchify(const __nv_bfloat16* in, __nv_bfloat16* out,
              int N, int C, int H, int W, int pH, int pW, cudaStream_t stream) {
    int h_patches = H / pH;
    int w_patches = W / pW;
    int patch_dim = C * pH * pW;
    int n_patches = h_patches * w_patches;
    int64_t total = (int64_t)N * n_patches * patch_dim;
    int block = 256;
    int grid = (int)((total + block - 1) / block);
    patchify_kernel<<<grid, block, 0, stream>>>(in, out, N, C, H, W, pH, pW);
}

__global__ void unpatchify_kernel(const __nv_bfloat16* in, __nv_bfloat16* out,
                                   int N, int C, int H, int W, int pH, int pW) {
    int h_patches = H / pH;
    int w_patches = W / pW;
    int patch_dim = C * pH * pW;
    int n_patches = h_patches * w_patches;

    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = (int64_t)N * C * H * W;
    if (idx >= total) return;

    int w = idx % W;
    int64_t rest2 = idx / W;
    int h = rest2 % H;
    rest2 = rest2 / H;
    int c = rest2 % C;
    int n = rest2 / C;

    int hp = h / pH;
    int wp = w / pW;
    int dh = h % pH;
    int dw = w % pW;

    int p = hp * w_patches + wp;
    int d = c * pH * pW + dh * pW + dw;

    out[idx] = in[((int64_t)n * n_patches + p) * patch_dim + d];
}

void unpatchify(const __nv_bfloat16* in, __nv_bfloat16* out,
                int N, int C, int H, int W, int pH, int pW, cudaStream_t stream) {
    int64_t total = (int64_t)N * C * H * W;
    int block = 256;
    int grid = (int)((total + block - 1) / block);
    unpatchify_kernel<<<grid, block, 0, stream>>>(in, out, N, C, H, W, pH, pW);
}

// ==================== Random Number Generation ====================

__global__ void randn_kernel(__nv_bfloat16* data, int64_t n, unsigned long long seed) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n / 2) return; // generate pairs

    curandState state;
    curand_init(seed, idx, 0, &state);
    float2 r = curand_normal2(&state);

    data[idx * 2] = float_to_bf16(r.x);
    if (idx * 2 + 1 < n) {
        data[idx * 2 + 1] = float_to_bf16(r.y);
    }
}

void randn_fill(__nv_bfloat16* data, int64_t n, unsigned long long seed, cudaStream_t stream) {
    int block = 256;
    int64_t pairs = (n + 1) / 2;
    int grid = (int)((pairs + block - 1) / block);
    randn_kernel<<<grid, block, 0, stream>>>(data, n, seed);
}

// Philox 4x32 RNG implementation matching sd.cpp/PyTorch CUDA
void randn_fill_philox(__nv_bfloat16* data, int64_t n, uint64_t seed) {
    const uint32_t M0 = 0xD2511F53u;
    const uint32_t M1 = 0xCD9E8D57u;
    const uint32_t W0 = 0x9E3779B9u;
    const uint32_t W1 = 0xBB67AE85u;
    const float two_pow32_inv = 2.3283064e-10f;
    const float two_pow32_inv_2pi = two_pow32_inv * 6.2831855f;

    uint32_t key0 = (uint32_t)(seed & 0xFFFFFFFF);
    uint32_t key1 = (uint32_t)(seed >> 32);
    uint32_t offset = 0; // first randn() call

    std::vector<float> result(n);

    for (int64_t i = 0; i < n; i++) {
        uint32_t counter[4] = {offset, 0, (uint32_t)i, 0};
        uint32_t k0 = key0, k1 = key1;

        // 10 rounds of Philox
        for (int round = 0; round < 10; round++) {
            uint64_t v1 = (uint64_t)counter[0] * M0;
            uint64_t v2 = (uint64_t)counter[2] * M1;
            uint32_t v1_hi = (uint32_t)(v1 >> 32);
            uint32_t v1_lo = (uint32_t)v1;
            uint32_t v2_hi = (uint32_t)(v2 >> 32);
            uint32_t v2_lo = (uint32_t)v2;

            uint32_t new0 = v2_hi ^ counter[1] ^ k0;
            uint32_t new1 = v2_lo;
            uint32_t new2 = v1_hi ^ counter[3] ^ k1;
            uint32_t new3 = v1_lo;

            counter[0] = new0;
            counter[1] = new1;
            counter[2] = new2;
            counter[3] = new3;

            k0 += W0;
            k1 += W1;
        }

        // Box-Muller: use counter[0] and counter[1]
        float u = (float)counter[0] * two_pow32_inv + two_pow32_inv / 2.0f;
        float v = (float)counter[1] * two_pow32_inv_2pi + two_pow32_inv_2pi / 2.0f;
        float s = sqrtf(-2.0f * logf(u));
        result[i] = s * sinf(v);
    }

    // Convert to BF16 and upload
    std::vector<__nv_bfloat16> bf16_data(n);
    for (int64_t i = 0; i < n; i++) {
        bf16_data[i] = __float2bfloat16(result[i]);
    }
    CUDA_CHECK(cudaMemcpy(data, bf16_data.data(), n * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
}

// ==================== Image Output ====================

__global__ void bf16_to_rgb8_kernel(const __nv_bfloat16* in, uint8_t* out, int H, int W) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= H * W) return;

    int h = idx / W;
    int w = idx % W;

    for (int c = 0; c < 3; c++) {
        float v = bf16_to_float(in[(int64_t)c * H * W + h * W + w]);
        // VAE output is in [-1, 1], rescale to [0, 1]
        v = (v + 1.0f) * 0.5f;
        v = fminf(fmaxf(v, 0.0f), 1.0f) * 255.0f + 0.5f;
        out[(h * W + w) * 3 + c] = (uint8_t)v;
    }
}

void bf16_to_rgb8(const __nv_bfloat16* in, uint8_t* out, int H, int W, cudaStream_t stream) {
    int total = H * W;
    int block = 256;
    int grid = (total + block - 1) / block;
    bf16_to_rgb8_kernel<<<grid, block, 0, stream>>>(in, out, H, W);
}

// ==================== Linear Layer ====================

void linear_forward(const Tensor& x, const Tensor& weight, const Tensor* bias, Tensor& out) {
    // x: [M, K], weight: [N, K], out: [M, N]
    int M = (int)x.shape[0];
    int K = (int)x.shape[1];
    int N = (int)weight.shape[0];

    assert(x.dtype == DType::BF16);
    assert(weight.dtype == DType::BF16);
    assert(out.dtype == DType::BF16);
    assert(weight.shape[1] == K);
    assert(out.shape[0] == M && out.shape[1] == N);

    // cuBLAS: C = alpha * op(A) * op(B) + beta * C
    // Row-major [M,K] x [N,K]^T = [M,N]
    // In column-major: [K,M] x [K,N]^T = ... use CUBLAS_OP_T on weight
    // Actually for row-major: C = A * B^T
    // cuBLAS col-major: C^T = B * A^T
    // We want: out[M,N] = x[M,K] @ weight[N,K]^T
    // cuBLAS: out^T[N,M] = weight[N,K] @ x^T[K,M]
    // So: A=weight, B=x, C=out, M_cublas=N, N_cublas=M, K_cublas=K
    // lda=K (weight row stride), ldb=K (x row stride), ldc=N (out row stride)

    float alpha = 1.0f, beta_val = 0.0f;

    CUBLAS_CHECK(cublasGemmEx(
        cublas(),
        CUBLAS_OP_T,   // op(A) = weight^T
        CUBLAS_OP_N,   // op(B) = x
        N, M, K,       // M, N, K in cublas notation
        &alpha,
        weight.data, CUDA_R_16BF, K,  // A = weight, lda = K
        x.data, CUDA_R_16BF, K,       // B = x, ldb = K
        &beta_val,
        out.data, CUDA_R_16BF, N,     // C = out, ldc = N
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP));

    if (bias) {
        bias_add((__nv_bfloat16*)out.data, (__nv_bfloat16*)bias->data,
                 (__nv_bfloat16*)out.data, M, N);
    }
}

void linear_forward_batched(const Tensor& x, const Tensor& weight, const Tensor* bias, Tensor& out) {
    // x: [B, M, K], weight: [N, K], out: [B, M, N]
    int B = (int)x.shape[0];
    int M = (int)x.shape[1];
    int K = (int)x.shape[2];
    int N = (int)weight.shape[0];

    float alpha = 1.0f, beta_val = 0.0f;

    CUBLAS_CHECK(cublasGemmStridedBatchedEx(
        cublas(),
        CUBLAS_OP_T, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        weight.data, CUDA_R_16BF, K, 0,           // weight shared across batches
        x.data, CUDA_R_16BF, K, (long long)M * K, // x batch stride
        &beta_val,
        out.data, CUDA_R_16BF, N, (long long)M * N, // out batch stride
        B,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP));

    if (bias) {
        bias_add((__nv_bfloat16*)out.data, (__nv_bfloat16*)bias->data,
                 (__nv_bfloat16*)out.data, (int64_t)B * M, N);
    }
}

// ================================================================
// FP32 INTERMEDIATE PRECISION KERNELS
// All activations kept in FP32 between GEMMs to match ggml behavior.
// Weights remain BF16. Convert to BF16 only at GEMM input boundaries.
// ================================================================

// ==================== FP32 Linear (BF16 input, FP32 output) ====================

// Bias add: FP32 x + BF16 bias -> FP32 out
__global__ void bias_add_fp32_kernel(const float* x, const __nv_bfloat16* bias,
                                      float* out, int64_t outer, int dim) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = outer * dim;
    if (idx >= total) return;
    int d = idx % dim;
    out[idx] = x[idx] + bf16_to_float(bias[d]);
}

void bias_add_fp32(const float* x, const __nv_bfloat16* bias, float* out,
                   int64_t outer, int dim, cudaStream_t stream) {
    int64_t total = outer * dim;
    int block = 256;
    int grid = (int)((total + block - 1) / block);
    bias_add_fp32_kernel<<<grid, block, 0, stream>>>(x, bias, out, outer, dim);
}

void linear_forward_fp32out(const Tensor& x, const Tensor& weight, const Tensor* bias, Tensor& out,
                            Tensor* gemm_scratch) {
    // Match ggml: BF16 x BF16 -> BF16 GEMM, then BF16 -> FP32 conversion
    // x: [M, K] BF16, weight: [N, K] BF16, out: [M, N] FP32
    int M = (int)x.shape[0];
    int K = (int)x.shape[1];
    int N = (int)weight.shape[0];

    assert(x.dtype == DType::BF16);
    assert(weight.dtype == DType::BF16);
    assert(out.dtype == DType::FP32);

    float alpha = 1.0f, beta_val = 0.0f;

    if (gemm_scratch) {
        // Match ggml: BF16 output then convert to FP32
        __nv_bfloat16* output_bf16 = (__nv_bfloat16*)gemm_scratch->data;

        CUBLAS_CHECK(cublasGemmEx(
            cublas(),
            CUBLAS_OP_T, CUBLAS_OP_N,
            N, M, K,
            &alpha,
            weight.data, CUDA_R_16BF, K,
            x.data, CUDA_R_16BF, K,
            &beta_val,
            output_bf16, CUDA_R_16BF, N,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP));

        bf16_to_fp32(output_bf16, (float*)out.data, (int64_t)M * N);
    } else {
        // Fallback: direct FP32 output
        CUBLAS_CHECK(cublasGemmEx(
            cublas(),
            CUBLAS_OP_T, CUBLAS_OP_N,
            N, M, K,
            &alpha,
            weight.data, CUDA_R_16BF, K,
            x.data, CUDA_R_16BF, K,
            &beta_val,
            out.data, CUDA_R_32F, N,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }

    if (bias) {
        bias_add_fp32((float*)out.data, (__nv_bfloat16*)bias->data,
                      (float*)out.data, M, N);
    }
}

void linear_forward_batched_fp32out(const Tensor& x, const Tensor& weight, const Tensor* bias, Tensor& out) {
    // x: [B, M, K] BF16, weight: [N, K] BF16, out: [B, M, N] FP32
    int B = (int)x.shape[0];
    int M = (int)x.shape[1];
    int K = (int)x.shape[2];
    int N = (int)weight.shape[0];

    float alpha = 1.0f, beta_val = 0.0f;

    CUBLAS_CHECK(cublasGemmStridedBatchedEx(
        cublas(),
        CUBLAS_OP_T, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        weight.data, CUDA_R_16BF, K, 0,
        x.data, CUDA_R_16BF, K, (long long)M * K,
        &beta_val,
        out.data, CUDA_R_32F, N, (long long)M * N,  // FP32 output
        B,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP));

    if (bias) {
        bias_add_fp32((float*)out.data, (__nv_bfloat16*)bias->data,
                      (float*)out.data, (int64_t)B * M, N);
    }
}

void linear_forward_fp32in_bf16w_fp32out(const Tensor& x, const Tensor& weight, const Tensor* bias,
                                         Tensor& out, Tensor& gemm_scratch) {
    // Match ggml's BF16 GEMM path exactly:
    // 1. Convert FP32 input -> BF16
    // 2. BF16 x BF16 -> BF16 GEMM (CUBLAS_COMPUTE_32F + TENSOR_OP)
    // 3. Convert BF16 output -> FP32
    // 4. Add bias in FP32 (separate op, matching ggml)
    int M = (int)x.shape[0];
    int K = (int)x.shape[1];
    int N = (int)weight.shape[0];

    assert(x.dtype == DType::FP32);
    assert(weight.dtype == DType::BF16);
    assert(out.dtype == DType::FP32);
    assert(weight.shape[1] == K);
    assert(out.shape[0] == M && out.shape[1] == N);

    // Partition scratch into BF16 input and BF16 output regions.
    // Need (M*K + M*N) * 2 bytes; scratch has max_weight_numel * 4 bytes which is always larger.
    __nv_bfloat16* input_bf16  = (__nv_bfloat16*)gemm_scratch.data;
    __nv_bfloat16* output_bf16 = input_bf16 + (int64_t)M * K;

    fp32_to_bf16((const float*)x.data, input_bf16, (int64_t)M * K);

    float alpha = 1.0f, beta_val = 0.0f;

    CUBLAS_CHECK(cublasGemmEx(
        cublas(),
        CUBLAS_OP_T, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        weight.data, CUDA_R_16BF, K,
        input_bf16,  CUDA_R_16BF, K,
        &beta_val,
        output_bf16, CUDA_R_16BF, N,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP));

    bf16_to_fp32(output_bf16, (float*)out.data, (int64_t)M * N);

    if (bias) {
        bias_add_fp32((float*)out.data, (__nv_bfloat16*)bias->data,
                      (float*)out.data, M, N);
    }
}

// ==================== FP32 RMSNorm ====================
// FP32 input, BF16 weight, FP32 output

__global__ void rms_norm_fp32_kernel(const float* x, const __nv_bfloat16* weight,
                                      float* out, int dim, float eps) {
    int row = blockIdx.x;
    const float* x_row = x + (int64_t)row * dim;
    float* o_row = out + (int64_t)row * dim;

    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x)
        sum_sq += x_row[i] * x_row[i];

    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);

    __shared__ float shared[32];
    int lane = threadIdx.x % warpSize;
    int warp = threadIdx.x / warpSize;
    if (lane == 0) shared[warp] = sum_sq;
    __syncthreads();
    if (warp == 0) {
        sum_sq = (lane < (blockDim.x + warpSize - 1) / warpSize) ? shared[lane] : 0.0f;
        for (int offset = warpSize / 2; offset > 0; offset >>= 1)
            sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
    }
    __syncthreads();
    if (threadIdx.x == 0) shared[0] = sum_sq;
    __syncthreads();
    sum_sq = shared[0];

    float rms = rsqrtf(sum_sq / dim + eps);
    for (int i = threadIdx.x; i < dim; i += blockDim.x)
        o_row[i] = x_row[i] * rms * bf16_to_float(weight[i]);
}

void rms_norm_fp32(const float* x, const __nv_bfloat16* weight, float* out,
                   int rows, int dim, float eps, cudaStream_t stream) {
    int threads = min(1024, ((dim + 31) / 32) * 32);
    rms_norm_fp32_kernel<<<rows, threads, 0, stream>>>(x, weight, out, dim, eps);
}

// ==================== FP32 LayerNorm (no affine) ====================
// FP32 input, FP32 output

__global__ void layer_norm_no_affine_fp32_kernel(const float* x, float* out,
                                                   int dim, float eps) {
    int row = blockIdx.x;
    const float* x_row = x + (int64_t)row * dim;
    float* o_row = out + (int64_t)row * dim;

    float sum = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x)
        sum += x_row[i];

    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffff, sum, offset);

    __shared__ float shared[32];
    int lane = threadIdx.x % warpSize;
    int warp = threadIdx.x / warpSize;
    if (lane == 0) shared[warp] = sum;
    __syncthreads();
    if (warp == 0) {
        sum = (lane < (blockDim.x + warpSize - 1) / warpSize) ? shared[lane] : 0.0f;
        for (int offset = warpSize / 2; offset > 0; offset >>= 1)
            sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    __syncthreads();
    if (threadIdx.x == 0) shared[0] = sum;
    __syncthreads();
    float mean = shared[0] / dim;

    float var_sum = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float v = x_row[i] - mean;
        var_sum += v * v;
    }

    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        var_sum += __shfl_down_sync(0xffffffff, var_sum, offset);
    if (lane == 0) shared[warp] = var_sum;
    __syncthreads();
    if (warp == 0) {
        var_sum = (lane < (blockDim.x + warpSize - 1) / warpSize) ? shared[lane] : 0.0f;
        for (int offset = warpSize / 2; offset > 0; offset >>= 1)
            var_sum += __shfl_down_sync(0xffffffff, var_sum, offset);
    }
    __syncthreads();
    if (threadIdx.x == 0) shared[0] = var_sum;
    __syncthreads();
    float inv_std = rsqrtf(shared[0] / dim + eps);

    for (int i = threadIdx.x; i < dim; i += blockDim.x)
        o_row[i] = (x_row[i] - mean) * inv_std;
}

void layer_norm_no_affine_fp32(const float* x, float* out,
                                int rows, int dim, float eps, cudaStream_t stream) {
    int threads = min(1024, ((dim + 31) / 32) * 32);
    layer_norm_no_affine_fp32_kernel<<<rows, threads, 0, stream>>>(x, out, dim, eps);
}

// ==================== FP32 Modulate ====================
// FP32 input, BF16 shift/scale, FP32 output

__global__ void modulate_fp32_kernel(const float* x, const __nv_bfloat16* shift,
                                      const __nv_bfloat16* scale, float* out,
                                      int rows, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * dim;
    if (idx >= total) return;
    int col = idx % dim;
    float sv = bf16_to_float(shift[col]);
    float scv = bf16_to_float(scale[col]);
    out[idx] = x[idx] * (1.0f + scv) + sv;
}

void modulate_fp32(const float* x, const __nv_bfloat16* shift, const __nv_bfloat16* scale,
                   float* out, int rows, int dim, cudaStream_t stream) {
    int total = rows * dim;
    int block = 256;
    int grid = (total + block - 1) / block;
    modulate_fp32_kernel<<<grid, block, 0, stream>>>(x, shift, scale, out, rows, dim);
}

__global__ void modulate_fp32_f32params_kernel(const float* x, const float* shift,
                                               const float* scale, float* out,
                                               int rows, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * dim;
    if (idx >= total) return;
    int col = idx % dim;
    out[idx] = x[idx] * (1.0f + scale[col]) + shift[col];
}

void modulate_fp32_f32params(const float* x, const float* shift, const float* scale,
                             float* out, int rows, int dim, cudaStream_t stream) {
    int total = rows * dim;
    int block = 256;
    int grid = (total + block - 1) / block;
    modulate_fp32_f32params_kernel<<<grid, block, 0, stream>>>(x, shift, scale, out, rows, dim);
}

// ==================== FP32 GELU ====================

__global__ void gelu_fp32_kernel(const float* x, float* out, int64_t n) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = x[i];
        const float c = 0.7978845608f;
        float inner = c * (v + 0.044715f * v * v * v);
        out[i] = 0.5f * v * (1.0f + tanhf(inner));
    }
}

void gelu_fp32(const float* x, float* out, int64_t n, cudaStream_t stream) {
    int block = 256;
    int grid = (int)((n + block - 1) / block);
    gelu_fp32_kernel<<<grid, block, 0, stream>>>(x, out, n);
}

// ==================== FP32 SiLU ====================

__global__ void silu_fp32_kernel(const float* x, float* out, int64_t n) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = x[i];
        out[i] = v / (1.0f + expf(-v));
    }
}

void silu_fp32(const float* x, float* out, int64_t n, cudaStream_t stream) {
    int block = 256;
    int grid = (int)((n + block - 1) / block);
    silu_fp32_kernel<<<grid, block, 0, stream>>>(x, out, n);
}

// ==================== FP32 RoPE ====================

__global__ void rope_apply_fp32_kernel(const float* x, const float* pe, float* out,
                                        int N_nhead, int L, int d_head) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int half_d = d_head / 2;
    int64_t total = (int64_t)N_nhead * L * half_d;
    if (idx >= total) return;

    int pair = idx % half_d;
    int64_t rest = idx / half_d;
    int l = rest % L;
    int nh = rest / L;

    float x0 = x[((int64_t)nh * L + l) * d_head + pair * 2];
    float x1 = x[((int64_t)nh * L + l) * d_head + pair * 2 + 1];

    int pe_base = (l * half_d + pair) * 4;
    float cos_val = pe[pe_base + 0];
    float neg_sin = pe[pe_base + 1];
    float sin_val = pe[pe_base + 2];
    float cos_val2 = pe[pe_base + 3];

    float out0 = x0 * cos_val + x1 * neg_sin;
    float out1 = x0 * sin_val + x1 * cos_val2;

    out[((int64_t)nh * L + l) * d_head + pair * 2] = out0;
    out[((int64_t)nh * L + l) * d_head + pair * 2 + 1] = out1;
}

void rope_apply_fp32(const float* x, const float* pe, float* out,
                     int N_nhead, int L, int d_head, cudaStream_t stream) {
    int half_d = d_head / 2;
    int64_t total = (int64_t)N_nhead * L * half_d;
    int block = 256;
    int grid = (int)((total + block - 1) / block);
    rope_apply_fp32_kernel<<<grid, block, 0, stream>>>(x, pe, out, N_nhead, L, d_head);
}

// ==================== FP32 Concat / Split ====================

__global__ void concat_seq_fp32_kernel(const float* a, const float* b, float* out,
                                        int B, int S1, int S2, int D) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = (int64_t)B * (S1 + S2) * D;
    if (idx >= total) return;
    int d = idx % D;
    int64_t rest = idx / D;
    int s = rest % (S1 + S2);
    int batch = rest / (S1 + S2);
    if (s < S1) {
        out[idx] = a[((int64_t)batch * S1 + s) * D + d];
    } else {
        out[idx] = b[((int64_t)batch * S2 + (s - S1)) * D + d];
    }
}

void concat_seq_fp32(const float* a, const float* b, float* out,
                     int B, int S1, int S2, int D, cudaStream_t stream) {
    int64_t total = (int64_t)B * (S1 + S2) * D;
    int block = 256;
    int grid = (int)((total + block - 1) / block);
    concat_seq_fp32_kernel<<<grid, block, 0, stream>>>(a, b, out, B, S1, S2, D);
}

void split_seq_fp32(const float* in, float* a, float* b,
                    int B, int S1, int S2, int D, cudaStream_t stream) {
    for (int batch = 0; batch < B; batch++) {
        CUDA_CHECK(cudaMemcpyAsync(
            a + (int64_t)batch * S1 * D,
            in + (int64_t)batch * (S1 + S2) * D,
            (size_t)S1 * D * sizeof(float),
            cudaMemcpyDeviceToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(
            b + (int64_t)batch * S2 * D,
            in + ((int64_t)batch * (S1 + S2) + S1) * D,
            (size_t)S2 * D * sizeof(float),
            cudaMemcpyDeviceToDevice, stream));
    }
}

// ==================== FP32 Attention ====================
// FP32 Q/K/V input, FP32 output. Internally casts Q/K to FP16 for the score
// matmul to match the reference CUDA attention numerics more closely.

void attention_forward_fp32io(const float* q, const float* k, const float* v,
                               float* out, float scale, int BH, int S, int D,
                               bool causal, cudaStream_t stream) {
    // Pre-allocated persistent buffers (avoid cudaMalloc/Free per call)
    static float* scores = nullptr;
    static __half* qkv_fp16 = nullptr;  // holds q_fp16, k_fp16 contiguously
    static size_t scores_cap = 0;
    static size_t qkv_cap = 0;

    size_t scores_need = (size_t)BH * S * S * sizeof(float);
    size_t qkv_need = (size_t)BH * S * D * sizeof(__half) * 2;  // q + k in FP16

    if (scores_need > scores_cap) {
        if (scores) cudaFree(scores);
        CUDA_CHECK(cudaMalloc(&scores, scores_need));
        scores_cap = scores_need;
    }
    if (qkv_need > qkv_cap) {
        if (qkv_fp16) cudaFree(qkv_fp16);
        CUDA_CHECK(cudaMalloc(&qkv_fp16, qkv_need));
        qkv_cap = qkv_need;
    }

    __half* q_fp16 = qkv_fp16;
    __half* k_fp16 = qkv_fp16 + (size_t)BH * S * D;

    fp32_to_fp16(q, q_fp16, (int64_t)BH * S * D, stream);
    fp32_to_fp16(k, k_fp16, (int64_t)BH * S * D, stream);

    float alpha = scale, beta_val = 0.0f;

    // Q × K^T: FP16 × FP16 → FP32 (tensor cores)
    CUBLAS_CHECK(cublasGemmStridedBatchedEx(
        cublas(),
        CUBLAS_OP_T, CUBLAS_OP_N,
        S, S, D,
        &alpha,
        k_fp16, CUDA_R_16F, D, (long long)S * D,
        q_fp16, CUDA_R_16F, D, (long long)S * D,
        &beta_val,
        scores, CUDA_R_32F, S, (long long)S * S,
        BH,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP));

    if (causal) apply_causal_mask(scores, BH, S, stream);

    // Softmax (FP32)
    softmax(scores, scores, BH * S, S, stream);

    // scores × V: FP32 × FP16 → FP32 output
    // cuBLAS doesn't support mixed FP32/FP16, so use FP32 for both
    alpha = 1.0f;
    beta_val = 0.0f;
    CUBLAS_CHECK(cublasGemmStridedBatchedEx(
        cublas(),
        CUBLAS_OP_N, CUBLAS_OP_N,
        D, S, S,
        &alpha,
        v, CUDA_R_32F, D, (long long)S * D,
        scores, CUDA_R_32F, S, (long long)S * S,
        &beta_val,
        out, CUDA_R_32F, D, (long long)S * D,
        BH,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

void attention_forward_fp32(const float* q, const float* k, const float* v,
                            float* out, float scale, int BH, int S, int D,
                            bool causal, cudaStream_t stream) {
    float* scores;
    CUDA_CHECK(cudaMalloc(&scores, (size_t)BH * S * S * sizeof(float)));

    float alpha = scale;
    float beta_val = 0.0f;

    CUBLAS_CHECK(cublasGemmStridedBatchedEx(
        cublas(),
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        S, S, D,
        &alpha,
        k, CUDA_R_32F, D, (long long)S * D,
        q, CUDA_R_32F, D, (long long)S * D,
        &beta_val,
        scores, CUDA_R_32F, S, (long long)S * S,
        BH,
        CUBLAS_COMPUTE_32F_PEDANTIC,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP));

    if (causal) apply_causal_mask(scores, BH, S, stream);

    softmax(scores, scores, BH * S, S, stream);

    alpha = 1.0f;
    beta_val = 0.0f;
    CUBLAS_CHECK(cublasGemmStridedBatchedEx(
        cublas(),
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        D, S, S,
        &alpha,
        v, CUDA_R_32F, D, (long long)S * D,
        scores, CUDA_R_32F, S, (long long)S * S,
        &beta_val,
        out, CUDA_R_32F, D, (long long)S * D,
        BH,
        CUBLAS_COMPUTE_32F_PEDANTIC,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP));

    CUDA_CHECK(cudaFree(scores));
}

// ==================== FP32 Gate Add ====================
// out = input_fp32 + bf16(gate) * fp32(x)  (x is now FP32 from GEMM output)

__global__ void gate_add_fp32v2_kernel(const float* input, const __nv_bfloat16* gate,
                                        const float* x, float* out,
                                        int rows, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * dim;
    if (idx >= total) return;
    int col = idx % dim;
    out[idx] = input[idx] + bf16_to_float(gate[col]) * x[idx];
}

void gate_add_fp32v2(const float* input, const __nv_bfloat16* gate, const float* x,
                     float* out, int rows, int dim, cudaStream_t stream) {
    int total = rows * dim;
    int block = 256;
    int grid = (total + block - 1) / block;
    gate_add_fp32v2_kernel<<<grid, block, 0, stream>>>(input, gate, x, out, rows, dim);
}

__global__ void gate_add_fp32f_kernel(const float* input, const float* gate,
                                      const float* x, float* out,
                                      int rows, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * dim;
    if (idx >= total) return;
    int col = idx % dim;
    out[idx] = input[idx] + gate[col] * x[idx];
}

void gate_add_fp32f(const float* input, const float* gate, const float* x,
                    float* out, int rows, int dim, cudaStream_t stream) {
    int total = rows * dim;
    int block = 256;
    int grid = (total + block - 1) / block;
    gate_add_fp32f_kernel<<<grid, block, 0, stream>>>(input, gate, x, out, rows, dim);
}

// ================================================================
// INT8 QUANTIZATION KERNELS
// Per-channel symmetric weight quantization + per-token dynamic
// activation quantization for INT8 tensor core GEMMs.
// ================================================================

// ==================== Fast Walsh-Hadamard Transform ====================
// In-place FWHT on each row of data[M, K], block-diagonal with had_block_size.
// The normalized Hadamard is an orthogonal involution: H @ H = I.

__global__ void fwht_inplace_kernel(float* data, int M, int K, int block_size) {
    int row = blockIdx.x;
    if (row >= M) return;
    float* row_data = data + (int64_t)row * K;
    int num_blocks = K / block_size;

    for (int blk = 0; blk < num_blocks; blk++) {
        float* bdata = row_data + blk * block_size;

        // Butterfly stages of the Walsh-Hadamard transform
        for (int half = 1; half < block_size; half <<= 1) {
            int full = half << 1;
            int num_pairs = block_size / 2;

            for (int pair = threadIdx.x; pair < num_pairs; pair += blockDim.x) {
                int group = pair / half;
                int within = pair % half;
                int idx_a = group * full + within;
                int idx_b = idx_a + half;

                float a = bdata[idx_a];
                float b = bdata[idx_b];
                bdata[idx_a] = a + b;
                bdata[idx_b] = a - b;
            }
            __syncthreads();
        }

        // Normalize by 1/sqrt(block_size) to make it orthogonal
        float norm = rsqrtf((float)block_size);
        for (int i = threadIdx.x; i < block_size; i += blockDim.x) {
            bdata[i] *= norm;
        }
        __syncthreads();
    }
}

void fwht_inplace(float* data, int M, int K, int had_block_size, cudaStream_t stream) {
    int threads = min(1024, had_block_size / 2);
    threads = max(32, ((threads + 31) / 32) * 32);
    fwht_inplace_kernel<<<M, threads, 0, stream>>>(data, M, K, had_block_size);
}

// ==================== Smooth Factor Kernels ====================

// Apply smooth division: out[m,j] = x[m,j] / smooth[j]
__global__ void apply_smooth_div_kernel(const float* x, const float* smooth, float* out, int M, int K) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < (int64_t)M * K) {
        int j = (int)(idx % K);
        out[idx] = x[idx] / smooth[j];
    }
}

void apply_smooth_div(const float* x, const float* smooth, float* out, int M, int K, cudaStream_t stream) {
    int64_t n = (int64_t)M * K;
    int block = 256;
    int grid = (int)((n + block - 1) / block);
    apply_smooth_div_kernel<<<grid, block, 0, stream>>>(x, smooth, out, M, K);
}

// Fused smooth division + FWHT: divides each element by smooth[j], then applies block-diagonal FWHT
__global__ void smooth_hadamard_kernel(float* data, const float* smooth, int M, int K, int block_size) {
    int row = blockIdx.x;
    if (row >= M) return;
    float* row_data = data + (int64_t)row * K;

    // Step 1: Divide by smooth factors
    for (int i = threadIdx.x; i < K; i += blockDim.x) {
        row_data[i] /= smooth[i];
    }
    __syncthreads();

    // Step 2: FWHT (same as fwht_inplace_kernel)
    int num_blocks = K / block_size;
    for (int blk = 0; blk < num_blocks; blk++) {
        float* bdata = row_data + blk * block_size;

        for (int half = 1; half < block_size; half <<= 1) {
            int full = half << 1;
            int num_pairs = block_size / 2;

            for (int pair = threadIdx.x; pair < num_pairs; pair += blockDim.x) {
                int group = pair / half;
                int within = pair % half;
                int idx_a = group * full + within;
                int idx_b = idx_a + half;

                float a = bdata[idx_a];
                float b = bdata[idx_b];
                bdata[idx_a] = a + b;
                bdata[idx_b] = a - b;
            }
            __syncthreads();
        }

        float norm = rsqrtf((float)block_size);
        for (int i = threadIdx.x; i < block_size; i += blockDim.x) {
            bdata[i] *= norm;
        }
        __syncthreads();
    }
}

void smooth_hadamard_inplace(float* data, const float* smooth, int M, int K, int had_block_size, cudaStream_t stream) {
    int threads = min(1024, max(had_block_size / 2, K / 4));
    threads = max(32, ((threads + 31) / 32) * 32);
    smooth_hadamard_kernel<<<M, threads, 0, stream>>>(data, smooth, M, K, had_block_size);
}

// Warp + block reduction for max value (shared across quantization kernels)
__device__ float block_reduce_max(float val) {
    // Warp reduce
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));

    __shared__ float shared[32];
    int lane = threadIdx.x % warpSize;
    int warp = threadIdx.x / warpSize;
    if (lane == 0) shared[warp] = val;
    __syncthreads();
    if (warp == 0) {
        val = (lane < (blockDim.x + warpSize - 1) / warpSize) ? shared[lane] : 0.0f;
        for (int offset = warpSize / 2; offset > 0; offset >>= 1)
            val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    __syncthreads();
    if (threadIdx.x == 0) shared[0] = val;
    __syncthreads();
    return shared[0];
}

// Quantize BF16 weight per-output-channel: one block per row
__global__ void quantize_weight_per_channel_kernel(
    const __nv_bfloat16* weight, int8_t* out, float* scales, int N, int K) {
    int row = blockIdx.x;
    if (row >= N) return;

    const __nv_bfloat16* row_ptr = weight + (int64_t)row * K;

    // Find max absolute value in this row
    float max_val = 0.0f;
    for (int i = threadIdx.x; i < K; i += blockDim.x) {
        float v = fabsf(__bfloat162float(row_ptr[i]));
        max_val = fmaxf(max_val, v);
    }
    max_val = block_reduce_max(max_val);

    float scale = max_val / 127.0f;
    if (threadIdx.x == 0) scales[row] = scale;

    float inv_scale = (max_val > 0.0f) ? 127.0f / max_val : 0.0f;

    // Quantize
    int8_t* out_row = out + (int64_t)row * K;
    for (int i = threadIdx.x; i < K; i += blockDim.x) {
        float v = __bfloat162float(row_ptr[i]) * inv_scale;
        int vi = __float2int_rn(v);
        vi = max(-128, min(127, vi));
        out_row[i] = (int8_t)vi;
    }
}

void quantize_weight_per_channel(const __nv_bfloat16* weight, int8_t* out, float* scales,
                                  int N, int K, cudaStream_t stream) {
    int threads = min(1024, ((K + 31) / 32) * 32);
    quantize_weight_per_channel_kernel<<<N, threads, 0, stream>>>(weight, out, scales, N, K);
}

// Quantize FP32 data per-row (used for Hadamard-rotated weights): one block per row
__global__ void quantize_per_row_fp32_kernel(
    const float* x, int8_t* out, float* scales, int M, int K) {
    int row = blockIdx.x;
    if (row >= M) return;

    const float* row_ptr = x + (int64_t)row * K;

    float max_val = 0.0f;
    for (int i = threadIdx.x; i < K; i += blockDim.x) {
        max_val = fmaxf(max_val, fabsf(row_ptr[i]));
    }
    max_val = block_reduce_max(max_val);

    float scale = max_val / 127.0f;
    if (threadIdx.x == 0) scales[row] = scale;

    float inv_scale = (max_val > 0.0f) ? 127.0f / max_val : 0.0f;

    int8_t* out_row = out + (int64_t)row * K;
    for (int i = threadIdx.x; i < K; i += blockDim.x) {
        float v = row_ptr[i] * inv_scale;
        int vi = __float2int_rn(v);
        vi = max(-128, min(127, vi));
        out_row[i] = (int8_t)vi;
    }
}

void quantize_per_row_fp32(const float* x, int8_t* out, float* scales,
                            int M, int K, cudaStream_t stream) {
    int threads = min(1024, ((K + 31) / 32) * 32);
    quantize_per_row_fp32_kernel<<<M, threads, 0, stream>>>(x, out, scales, M, K);
}

// Quantize FP32 activation per-token: one block per row
__global__ void quantize_activation_per_token_fp32_kernel(
    const float* x, int8_t* out, float* scales, int M, int K) {
    int row = blockIdx.x;
    if (row >= M) return;

    const float* row_ptr = x + (int64_t)row * K;

    float max_val = 0.0f;
    for (int i = threadIdx.x; i < K; i += blockDim.x) {
        max_val = fmaxf(max_val, fabsf(row_ptr[i]));
    }
    max_val = block_reduce_max(max_val);

    float scale = max_val / 127.0f;
    if (threadIdx.x == 0) scales[row] = scale;

    float inv_scale = (max_val > 0.0f) ? 127.0f / max_val : 0.0f;

    int8_t* out_row = out + (int64_t)row * K;
    for (int i = threadIdx.x; i < K; i += blockDim.x) {
        float v = row_ptr[i] * inv_scale;
        int vi = __float2int_rn(v);
        vi = max(-128, min(127, vi));
        out_row[i] = (int8_t)vi;
    }
}

void quantize_activation_per_token_fp32(const float* x, int8_t* out, float* scales,
                                         int M, int K, cudaStream_t stream) {
    int threads = min(1024, ((K + 31) / 32) * 32);
    quantize_activation_per_token_fp32_kernel<<<M, threads, 0, stream>>>(x, out, scales, M, K);
}

// Quantize BF16 activation per-token: one block per row
__global__ void quantize_activation_per_token_bf16_kernel(
    const __nv_bfloat16* x, int8_t* out, float* scales, int M, int K) {
    int row = blockIdx.x;
    if (row >= M) return;

    const __nv_bfloat16* row_ptr = x + (int64_t)row * K;

    float max_val = 0.0f;
    for (int i = threadIdx.x; i < K; i += blockDim.x) {
        max_val = fmaxf(max_val, fabsf(__bfloat162float(row_ptr[i])));
    }
    max_val = block_reduce_max(max_val);

    float scale = max_val / 127.0f;
    if (threadIdx.x == 0) scales[row] = scale;

    float inv_scale = (max_val > 0.0f) ? 127.0f / max_val : 0.0f;

    int8_t* out_row = out + (int64_t)row * K;
    for (int i = threadIdx.x; i < K; i += blockDim.x) {
        float v = __bfloat162float(row_ptr[i]) * inv_scale;
        int vi = __float2int_rn(v);
        vi = max(-128, min(127, vi));
        out_row[i] = (int8_t)vi;
    }
}

void quantize_activation_per_token_bf16(const __nv_bfloat16* x, int8_t* out, float* scales,
                                         int M, int K, cudaStream_t stream) {
    int threads = min(1024, ((K + 31) / 32) * 32);
    quantize_activation_per_token_bf16_kernel<<<M, threads, 0, stream>>>(x, out, scales, M, K);
}

// Dequantize INT32 GEMM output + add bias
// out[i,j] = gemm_out[i,j] * act_scales[i] * w_scales[j] + bias[j]
__global__ void dequantize_and_bias_kernel(
    const int32_t* gemm_out, const float* act_scales, const float* w_scales,
    const __nv_bfloat16* bias, float* out, int M, int N) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (int64_t)M * N) return;
    int row = (int)(idx / N);
    int col = (int)(idx % N);
    float val = (float)gemm_out[idx] * act_scales[row] * w_scales[col];
    if (bias) val += __bfloat162float(bias[col]);
    out[idx] = val;
}

void dequantize_and_bias(const int32_t* gemm_out, const float* act_scales, const float* w_scales,
                          const __nv_bfloat16* bias, float* out,
                          int M, int N, cudaStream_t stream) {
    int64_t total = (int64_t)M * N;
    int block = 256;
    int grid = (int)((total + block - 1) / block);
    dequantize_and_bias_kernel<<<grid, block, 0, stream>>>(
        gemm_out, act_scales, w_scales, bias, out, M, N);
}

// ==================== INT8 Linear Forward (with Hadamard Rotation) ====================

// FP32 input × INT8 weight → FP32 output
// Applies Hadamard rotation to activation before quantization.
void linear_forward_int8(const Tensor& x, const QuantizedWeight& weight,
                          const Tensor* bias, Tensor& out, Tensor& scratch) {
    int M = (int)x.shape[0];
    int K = (int)x.shape[1];
    int N = (int)weight.data.shape[0];

    assert(x.dtype == DType::FP32);
    assert(weight.data.dtype == DType::INT8);
    assert(weight.scales.dtype == DType::FP32);
    assert(out.dtype == DType::FP32);
    assert(weight.data.shape[1] == K);
    assert(out.shape[0] == M && out.shape[1] == N);

    // Scratch layout:
    // [Region A: max(M*K*4, M*N*4)] - FP32 temp for Hadamard, then reused for INT32 output
    // [INT8 act: align256(M*K)]
    // [FP32 scales: align256(M*4)]
    int64_t regionA_size = (((int64_t)std::max((int64_t)M * K, (int64_t)M * N) * 4 + 255) & ~255LL);
    int64_t int8_offset = regionA_size;
    int64_t int8_size = (((int64_t)M * K + 255) & ~255LL);
    int64_t scales_offset = int8_offset + int8_size;

    float* fp32_temp = (float*)scratch.data;
    int8_t* act_int8 = (int8_t*)((char*)scratch.data + int8_offset);
    float* act_scales = (float*)((char*)scratch.data + scales_offset);

    // Step 1: Copy activation to scratch and apply Hadamard rotation
    CUDA_CHECK(cudaMemcpy(fp32_temp, x.data, (int64_t)M * K * sizeof(float), cudaMemcpyDeviceToDevice));
    if (weight.had_block_size > 1) {
        fwht_inplace(fp32_temp, M, K, weight.had_block_size);
    }

    // Step 2: Quantize rotated activation per-token
    quantize_activation_per_token_fp32(fp32_temp, act_int8, act_scales, M, K);

    // Step 3: INT8 GEMM (reuse Region A for INT32 output)
    int32_t* gemm_out = (int32_t*)scratch.data;
    int32_t alpha_i = 1, beta_i = 0;
    CUBLAS_CHECK(cublasGemmEx(
        cublas(),
        CUBLAS_OP_T, CUBLAS_OP_N,
        N, M, K,
        &alpha_i,
        weight.data.data, CUDA_R_8I, K,
        act_int8,          CUDA_R_8I, K,
        &beta_i,
        gemm_out,          CUDA_R_32I, N,
        CUBLAS_COMPUTE_32I,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP));

    // Step 4: Dequantize and add bias
    dequantize_and_bias(gemm_out, act_scales, (float*)weight.scales.data,
                        bias ? (__nv_bfloat16*)bias->data : nullptr,
                        (float*)out.data, M, N);
}

// BF16 input × INT8 weight → FP32 output (with Hadamard rotation)
void linear_forward_int8_bf16in(const Tensor& x, const QuantizedWeight& weight,
                                 const Tensor* bias, Tensor& out, Tensor& scratch) {
    int M = (int)x.shape[0];
    int K = (int)x.shape[1];
    int N = (int)weight.data.shape[0];

    assert(x.dtype == DType::BF16);
    assert(weight.data.dtype == DType::INT8);
    assert(out.dtype == DType::FP32);

    int64_t regionA_size = (((int64_t)std::max((int64_t)M * K, (int64_t)M * N) * 4 + 255) & ~255LL);
    int64_t int8_offset = regionA_size;
    int64_t int8_size = (((int64_t)M * K + 255) & ~255LL);
    int64_t scales_offset = int8_offset + int8_size;

    float* fp32_temp = (float*)scratch.data;
    int8_t* act_int8 = (int8_t*)((char*)scratch.data + int8_offset);
    float* act_scales = (float*)((char*)scratch.data + scales_offset);

    // Step 1: Convert BF16 → FP32 and apply Hadamard rotation
    bf16_to_fp32((__nv_bfloat16*)x.data, fp32_temp, (int64_t)M * K);
    if (weight.had_block_size > 1) {
        fwht_inplace(fp32_temp, M, K, weight.had_block_size);
    }

    // Step 2: Quantize rotated activation per-token
    quantize_activation_per_token_fp32(fp32_temp, act_int8, act_scales, M, K);

    // Step 3: INT8 GEMM (reuse Region A for INT32 output)
    int32_t* gemm_out = (int32_t*)scratch.data;
    int32_t alpha_i = 1, beta_i = 0;
    CUBLAS_CHECK(cublasGemmEx(
        cublas(),
        CUBLAS_OP_T, CUBLAS_OP_N,
        N, M, K,
        &alpha_i,
        weight.data.data, CUDA_R_8I, K,
        act_int8,          CUDA_R_8I, K,
        &beta_i,
        gemm_out,          CUDA_R_32I, N,
        CUBLAS_COMPUTE_32I,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP));

    // Step 4: Dequantize and add bias
    dequantize_and_bias(gemm_out, act_scales, (float*)weight.scales.data,
                        bias ? (__nv_bfloat16*)bias->data : nullptr,
                        (float*)out.data, M, N);
}

// Helper: quantize a BF16 weight tensor at load time (with Hadamard rotation)
QuantizedWeight quantize_weight_tensor(const Tensor& bf16_weight) {
    assert(bf16_weight.dtype == DType::BF16);
    assert(bf16_weight.ndim == 2);

    int N = (int)bf16_weight.shape[0];
    int K = (int)bf16_weight.shape[1];
    int hbs = hadamard_block_size(K);

    // Convert BF16 → FP32
    Tensor fp32_weight = Tensor::alloc({(int64_t)N, (int64_t)K}, DType::FP32);
    bf16_to_fp32((__nv_bfloat16*)bf16_weight.data, (float*)fp32_weight.data, (int64_t)N * K);

    // Apply Hadamard rotation to K dimension of each row
    if (hbs > 1) {
        fwht_inplace((float*)fp32_weight.data, N, K, hbs);
    }

    // Quantize rotated FP32 weights to INT8 per-channel
    QuantizedWeight qw;
    qw.data = Tensor::alloc({(int64_t)N, (int64_t)K}, DType::INT8);
    qw.scales = Tensor::alloc({(int64_t)N}, DType::FP32);
    qw.had_block_size = hbs;

    quantize_per_row_fp32(
        (float*)fp32_weight.data,
        (int8_t*)qw.data.data,
        (float*)qw.scales.data,
        N, K);

    fp32_weight.free_data();
    CUDA_CHECK(cudaDeviceSynchronize());
    return qw;
}

// ================================================================
// INT4 SVDQuant KERNELS
// ================================================================

// NF4 (NormalFloat4) lookup table: quantiles of N(0,1) normalized to [-1, 1]
__device__ __constant__ float NF4_GRID[16] = {
    -1.0f, -0.6962f, -0.5251f, -0.3949f, -0.2844f, -0.1848f, -0.0911f, 0.0f,
     0.0796f,  0.1609f,  0.2461f,  0.3379f,  0.4407f,  0.5626f,  0.7230f, 1.0f
};

// Dequantize packed INT4 [N, K/2] + per-group scales [N, K/gs] -> BF16 [N, K]
// nf4_grid: true = NF4 lookup, false = symmetric linear INT4 (q-8)*scale
__global__ void dequantize_int4_to_bf16_kernel(
    const uint8_t* __restrict__ qweight,
    const float* __restrict__ scales,
    __nv_bfloat16* __restrict__ out,
    int N, int K, int group_size, bool nf4_grid, bool scales_gn, bool signed_int4)
{
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total_bytes = (int64_t)N * (K / 2);
    if (idx >= total_bytes) return;

    int row = (int)(idx / (K / 2));
    int byte_col = (int)(idx % (K / 2));

    uint8_t packed = qweight[idx];
    int idx0 = packed & 0xF;
    int idx1 = (packed >> 4) & 0xF;

    int col0 = byte_col * 2;
    int col1 = byte_col * 2 + 1;

    int num_groups = K / group_size;
    int g0 = col0 / group_size;
    int g1 = col1 / group_size;
    // scales_gn: scales layout is [num_groups, N] instead of [N, num_groups]
    float s0 = scales_gn ? scales[(int64_t)g0 * N + row] : scales[(int64_t)row * num_groups + g0];
    float s1 = scales_gn ? scales[(int64_t)g1 * N + row] : scales[(int64_t)row * num_groups + g1];

    if (nf4_grid) {
        out[(int64_t)row * K + col0] = __float2bfloat16(NF4_GRID[idx0] * s0);
        out[(int64_t)row * K + col1] = __float2bfloat16(NF4_GRID[idx1] * s1);
    } else if (signed_int4) {
        // Two's complement signed INT4: nibble in [0,15] → value in [-8,7]
        int v0 = idx0 >= 8 ? idx0 - 16 : idx0;
        int v1 = idx1 >= 8 ? idx1 - 16 : idx1;
        out[(int64_t)row * K + col0] = __float2bfloat16((float)v0 * s0);
        out[(int64_t)row * K + col1] = __float2bfloat16((float)v1 * s1);
    } else {
        // Unsigned INT4 with zero-point 8: nibble → value = nibble - 8
        out[(int64_t)row * K + col0] = __float2bfloat16((float)(idx0 - 8) * s0);
        out[(int64_t)row * K + col1] = __float2bfloat16((float)(idx1 - 8) * s1);
    }
}

void dequantize_int4_to_bf16(const uint8_t* qweight, const float* scales,
                              __nv_bfloat16* out, int N, int K, int group_size,
                              cudaStream_t stream, bool nf4_grid, bool scales_gn,
                              bool signed_int4) {
    int64_t total_bytes = (int64_t)N * (K / 2);
    int block = 256;
    int grid = (int)((total_bytes + block - 1) / block);
    dequantize_int4_to_bf16_kernel<<<grid, block, 0, stream>>>(
        qweight, scales, out, N, K, group_size, nf4_grid, scales_gn, signed_int4);
}

// Dequantize nunchaku MMA-swizzled INT4 → BF16
// Maps logical (row, col) to swizzled byte address + nibble position
__global__ void dequantize_nunchaku_to_bf16_kernel(
    const uint8_t* __restrict__ qweight,
    const __nv_bfloat16* __restrict__ wscales,
    __nv_bfloat16* __restrict__ out,
    int N, int K, int num_groups)
{
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = (int64_t)N * K;
    if (idx >= total) return;

    int row = (int)(idx / K);
    int col = (int)(idx % K);
    int group_size = K / num_groups;
    int group = col / group_size;

    // Weight index: (row, col) → swizzled byte address + nibble
    int k_tiles = K / 64;
    int n_tile = row / 128;
    int nw = row % 128;
    int np = nw / 16;
    int ps = (nw % 16) / 8;
    int nl = nw % 8;

    int k_tile = col / 64;
    int kps = (col % 64) / 32;
    int kl = ((col % 64) % 32) / 8;
    int rk = (col % 64) % 8;

    int int32_idx = n_tile * (k_tiles * 1024) + k_tile * 1024
                  + np * 128 + nl * 16 + kl * 4 + ps * 2 + kps;
    int byte_offset = int32_idx * 4 + rk / 2;
    int nibble = (qweight[byte_offset] >> ((rk % 2) * 4)) & 0xF;

    // Two's complement signed 4-bit
    int signed_val = nibble >= 8 ? nibble - 16 : nibble;

    // Scale index: (row, group) → swizzled BF16 offset
    int d2 = nw / 16;
    int d3 = (nw % 16) / 8;
    int d4 = (nw % 8) / 2;
    int d5 = nw % 2;
    int scale_offset = n_tile * num_groups * 128 + group * 128
                     + d2 * 16 + d4 * 4 + d3 * 2 + d5;

    float scale = __bfloat162float(wscales[scale_offset]);
    out[idx] = __float2bfloat16((float)signed_val * scale);
}

void dequantize_nunchaku_to_bf16(const uint8_t* qweight, const __nv_bfloat16* wscales,
                                  __nv_bfloat16* out, int N, int K, int num_groups,
                                  cudaStream_t stream) {
    int64_t total = (int64_t)N * K;
    int block = 256;
    int grid = (int)((total + block - 1) / block);
    dequantize_nunchaku_to_bf16_kernel<<<grid, block, 0, stream>>>(
        qweight, wscales, out, N, K, num_groups);
}

// ==================== AWQ INT4 Dequantization (for modulation layers) ====================
// Dequantize nunchaku AWQ INT4 modulation weights to BF16 with output de-interleaving.
// AWQ packing: qweight [OC/4, IC/2] INT32, kInterleave=4.
//   pack_w4 packing layout (from nunchaku tinychat_utils.py):
//     qw_row = oc / 4
//     int16_col = (ic/64)*64 + (oc%4)*16 + ((ic%64)/32)*8 + (ic%8)
//     qw_col = int16_col / 2
//     bit_offset = (int16_col % 2) * 16 + ((ic/8) % 4) * 4
//     q = (qweight[qw_row, qw_col] >> bit_offset) & 0xF
// Dequant: weight = q_uint4 * wscales[ic/group_size, oc] + wzeros[ic/group_size, oc]
// De-interleave: nk_oc → component = nk_oc % num_components, feature = nk_oc / num_components
//   standard_oc = component * (OC/num_components) + feature
__global__ void dequantize_awq_to_bf16_kernel(
    const int32_t* __restrict__ qweight,   // [OC/4, IC/2]
    const __nv_bfloat16* __restrict__ wscales, // [IC/group_size, OC]
    const __nv_bfloat16* __restrict__ wzeros,  // [IC/group_size, OC]
    __nv_bfloat16* __restrict__ out,       // [OC, IC] in standard (de-interleaved) order
    int OC, int IC, int group_size, int num_components)
{
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = (int64_t)OC * IC;
    if (idx >= total) return;

    // Map to standard (de-interleaved) output position
    int std_oc = (int)(idx / IC);
    int ic = (int)(idx % IC);

    // Convert standard_oc to nunchaku interleaved oc
    int dim = OC / num_components;  // 3072 for 6-way modulation
    int component = std_oc / dim;
    int feature = std_oc % dim;
    int nk_oc = feature * num_components + component;

    // AWQ kInterleave=4 packing: extract nibble from packed qweight
    int qw_row = nk_oc / 4;
    int int16_col = (ic / 64) * 64 + (nk_oc % 4) * 16 + ((ic % 64) / 32) * 8 + (ic % 8);
    int qw_col = int16_col / 2;
    int bit_offset = (int16_col % 2) * 16 + ((ic / 8) % 4) * 4;
    int packed_val = qweight[qw_row * (IC / 2) + qw_col];
    int q = (packed_val >> bit_offset) & 0xF;  // unsigned 0..15

    // Dequantize: weight = q * scale + zero (wzeros is pre-scaled)
    int group = ic / group_size;
    float scale = __bfloat162float(wscales[group * OC + nk_oc]);
    float zero = __bfloat162float(wzeros[group * OC + nk_oc]);
    float weight = (float)q * scale + zero;

    out[idx] = __float2bfloat16(weight);
}

void dequantize_awq_to_bf16(const int32_t* qweight, const __nv_bfloat16* wscales,
                              const __nv_bfloat16* wzeros, __nv_bfloat16* out,
                              int OC, int IC, int group_size, int num_components,
                              cudaStream_t stream) {
    int64_t total = (int64_t)OC * IC;
    int block = 256;
    int grid = (int)((total + block - 1) / block);
    dequantize_awq_to_bf16_kernel<<<grid, block, 0, stream>>>(
        qweight, wscales, wzeros, out, OC, IC, group_size, num_components);
}

// Dequantize packed INT4 [N, K/2] + per-group scales [N, K/gs] -> FP32 [N, K]
__global__ void dequantize_int4_to_fp32_kernel(
    const uint8_t* __restrict__ qweight,
    const float* __restrict__ scales,
    float* __restrict__ out,
    int N, int K, int group_size, bool nf4_grid)
{
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total_bytes = (int64_t)N * (K / 2);
    if (idx >= total_bytes) return;

    int row = (int)(idx / (K / 2));
    int byte_col = (int)(idx % (K / 2));

    uint8_t packed = qweight[idx];
    int idx0 = packed & 0xF;
    int idx1 = (packed >> 4) & 0xF;

    int col0 = byte_col * 2;
    int col1 = byte_col * 2 + 1;

    int num_groups = K / group_size;
    float s0 = scales[(int64_t)row * num_groups + col0 / group_size];
    float s1 = scales[(int64_t)row * num_groups + col1 / group_size];

    if (nf4_grid) {
        out[(int64_t)row * K + col0] = NF4_GRID[idx0] * s0;
        out[(int64_t)row * K + col1] = NF4_GRID[idx1] * s1;
    } else {
        out[(int64_t)row * K + col0] = (float)(idx0 - 8) * s0;
        out[(int64_t)row * K + col1] = (float)(idx1 - 8) * s1;
    }
}

static void dequantize_int4_to_fp32(const uint8_t* qweight, const float* scales,
                                     float* out, int N, int K, int group_size,
                                     bool nf4_grid, cudaStream_t stream = 0) {
    int64_t total_bytes = (int64_t)N * (K / 2);
    int block = 256;
    int grid = (int)((total_bytes + block - 1) / block);
    dequantize_int4_to_fp32_kernel<<<grid, block, 0, stream>>>(
        qweight, scales, out, N, K, group_size, nf4_grid);
}

// Quantize FP32 residual [N, K] to packed symmetric INT4 [N, K/2] + per-group scales [N, K/gs]
// Symmetric linear INT4: scale = max(|group|)/7, q = round(x/scale)+8, clamp [0,15]
__global__ void quantize_linear_int4_per_group_kernel(
    const float* __restrict__ residual,
    uint8_t* __restrict__ qweight,
    float* __restrict__ scales,
    int N, int K, int group_size)
{
    int row = blockIdx.x;
    int group = blockIdx.y;
    if (row >= N) return;

    int num_groups = K / group_size;
    int col_start = group * group_size;
    const float* grp = residual + (int64_t)row * K + col_start;

    // Find max abs in group (same reduction as NF4 kernel)
    float max_abs = 0.0f;
    for (int i = threadIdx.x; i < group_size; i += blockDim.x) {
        float v = fabsf(grp[i]);
        if (v > max_abs) max_abs = v;
    }
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        max_abs = fmaxf(max_abs, __shfl_down_sync(0xffffffff, max_abs, offset));
    __shared__ float s_max[32];
    int lane = threadIdx.x % warpSize;
    int warp_id = threadIdx.x / warpSize;
    if (lane == 0) s_max[warp_id] = max_abs;
    __syncthreads();
    if (warp_id == 0) {
        int nwarps = (blockDim.x + warpSize - 1) / warpSize;
        max_abs = (lane < nwarps) ? s_max[lane] : 0.0f;
        for (int offset = warpSize / 2; offset > 0; offset >>= 1)
            max_abs = fmaxf(max_abs, __shfl_down_sync(0xffffffff, max_abs, offset));
    }
    __syncthreads();
    if (threadIdx.x == 0) s_max[0] = max_abs;
    __syncthreads();
    max_abs = s_max[0];

    // Scale = max/7 (step size for [-7,7] range)
    float scale = max_abs / 7.0f;
    if (threadIdx.x == 0)
        scales[(int64_t)row * num_groups + group] = scale;

    float inv_scale = (max_abs > 0.0f) ? (7.0f / max_abs) : 0.0f;

    // Quantize pairs and pack
    int half_gs = group_size / 2;
    for (int i = threadIdx.x; i < half_gs; i += blockDim.x) {
        int c0 = col_start + i * 2;
        int c1 = c0 + 1;
        float v0 = residual[(int64_t)row * K + c0] * inv_scale;
        float v1 = residual[(int64_t)row * K + c1] * inv_scale;

        // Round and clamp to [-7, 7], then offset to unsigned [1, 15]
        int q0 = (int)roundf(v0); q0 = max(-7, min(7, q0)); q0 += 8;
        int q1 = (int)roundf(v1); q1 = max(-7, min(7, q1)); q1 += 8;

        uint8_t packed = (uint8_t)(q0 | (q1 << 4));
        qweight[(int64_t)row * (K / 2) + (c0 / 2)] = packed;
    }
}

// Quantize FP32 residual [N, K] to packed NF4 [N, K/2] + per-group scales [N, K/gs]
// NF4: scale = max(|group|), values mapped to nearest NF4 grid point
__global__ void quantize_int4_per_group_kernel(
    const float* __restrict__ residual,
    uint8_t* __restrict__ qweight,
    float* __restrict__ scales,
    int N, int K, int group_size)
{
    int row = blockIdx.x;
    int group = blockIdx.y;
    if (row >= N) return;

    int num_groups = K / group_size;
    int col_start = group * group_size;
    const float* grp = residual + (int64_t)row * K + col_start;

    // Find max abs in group
    float max_abs = 0.0f;
    for (int i = threadIdx.x; i < group_size; i += blockDim.x) {
        float v = fabsf(grp[i]);
        if (v > max_abs) max_abs = v;
    }
    // Warp reduce
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        max_abs = fmaxf(max_abs, __shfl_down_sync(0xffffffff, max_abs, offset));
    // Cross-warp reduce
    __shared__ float s_max[32];
    int lane = threadIdx.x % warpSize;
    int warp_id = threadIdx.x / warpSize;
    if (lane == 0) s_max[warp_id] = max_abs;
    __syncthreads();
    if (warp_id == 0) {
        int nwarps = (blockDim.x + warpSize - 1) / warpSize;
        max_abs = (lane < nwarps) ? s_max[lane] : 0.0f;
        for (int offset = warpSize / 2; offset > 0; offset >>= 1)
            max_abs = fmaxf(max_abs, __shfl_down_sync(0xffffffff, max_abs, offset));
    }
    __syncthreads();
    if (threadIdx.x == 0) s_max[0] = max_abs;
    __syncthreads();
    max_abs = s_max[0];

    float scale = max_abs;  // NF4 grid spans [-1, 1], no division by 7
    if (threadIdx.x == 0)
        scales[(int64_t)row * num_groups + group] = scale;

    float inv_scale = (scale > 0.0f) ? (1.0f / scale) : 0.0f;

    // Quantize pairs and pack using nearest NF4 grid point
    int half_gs = group_size / 2;
    for (int i = threadIdx.x; i < half_gs; i += blockDim.x) {
        int c0 = col_start + i * 2;
        int c1 = c0 + 1;
        float n0 = residual[(int64_t)row * K + c0] * inv_scale;
        float n1 = residual[(int64_t)row * K + c1] * inv_scale;

        // Find nearest NF4 grid entry (linear scan, 16 entries — compiler unrolls)
        int q0 = 0, q1 = 0;
        float best0 = 1e9f, best1 = 1e9f;
        for (int j = 0; j < 16; j++) {
            float d0 = fabsf(n0 - NF4_GRID[j]);
            float d1 = fabsf(n1 - NF4_GRID[j]);
            if (d0 < best0) { best0 = d0; q0 = j; }
            if (d1 < best1) { best1 = d1; q1 = j; }
        }

        uint8_t packed = (uint8_t)(q0 | (q1 << 4));
        qweight[(int64_t)row * (K / 2) + (c0 / 2)] = packed;
    }
}

void quantize_int4_per_group(const float* residual, uint8_t* qweight, float* scales,
                              int N, int K, int group_size, cudaStream_t stream) {
    int num_groups = K / group_size;
    dim3 grid(N, num_groups);
    int block = min(256, group_size / 2);
    if (block < 1) block = 1;
    quantize_int4_per_group_kernel<<<grid, block, 0, stream>>>(
        residual, qweight, scales, N, K, group_size);
}

// Simulate W4A4 activation quantization: quantize FP32 to symmetric INT4 per-token-per-group
// and immediately dequantize back to FP32 (adds quantization noise in-place).
__global__ void act_quant_noise_int4_kernel(
    float* __restrict__ data, int M, int K, int group_size)
{
    int row = blockIdx.x;
    int group = blockIdx.y;
    if (row >= M) return;

    int col_start = group * group_size;
    float* grp = data + (int64_t)row * K + col_start;

    // Find max abs in group (warp-reduce)
    float max_abs = 0.0f;
    for (int i = threadIdx.x; i < group_size; i += blockDim.x) {
        float v = fabsf(grp[i]);
        if (v > max_abs) max_abs = v;
    }
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        max_abs = fmaxf(max_abs, __shfl_down_sync(0xffffffff, max_abs, offset));
    __shared__ float s_max[32];
    int lane = threadIdx.x % warpSize;
    int warp_id = threadIdx.x / warpSize;
    if (lane == 0) s_max[warp_id] = max_abs;
    __syncthreads();
    if (warp_id == 0) {
        int nwarps = (blockDim.x + warpSize - 1) / warpSize;
        max_abs = (lane < nwarps) ? s_max[lane] : 0.0f;
        for (int offset = warpSize / 2; offset > 0; offset >>= 1)
            max_abs = fmaxf(max_abs, __shfl_down_sync(0xffffffff, max_abs, offset));
    }
    __syncthreads();
    if (threadIdx.x == 0) s_max[0] = max_abs;
    __syncthreads();
    max_abs = s_max[0];

    float scale = max_abs / 7.0f;
    float inv_scale = (max_abs > 0.0f) ? (7.0f / max_abs) : 0.0f;

    // Quantize to [-7,7] and dequantize in-place
    for (int i = threadIdx.x; i < group_size; i += blockDim.x) {
        float v = grp[i];
        float q = roundf(v * inv_scale);
        q = fminf(7.0f, fmaxf(-7.0f, q));
        grp[i] = q * scale;
    }
}

static void act_quant_noise_int4(float* data, int M, int K, int group_size, cudaStream_t stream = 0) {
    int num_groups = K / group_size;
    dim3 grid(M, num_groups);
    int block = min(256, group_size);
    act_quant_noise_int4_kernel<<<grid, block, 0, stream>>>(data, M, K, group_size);
}

// ================================================================
// W4A4 KERNELS: INT4×INT4 GEMM with per-group dequantization
// ================================================================

// Kernel A: Unswizzle nunchaku MMA-swizzled INT4 weights to row-major
// Reuses swizzle index formulas from dequantize_nunchaku_to_bf16_kernel
// Each thread handles one (row, col_pair) = two adjacent columns → one output byte
__global__ void unswizzle_nunchaku_weights_kernel(
    const uint8_t* __restrict__ qweight_swizzled,
    const __nv_bfloat16* __restrict__ wscales_swizzled,
    uint8_t* __restrict__ qweight_rowmajor,
    float* __restrict__ wscales_rowmajor,
    int N, int K, int num_groups)
{
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total_bytes = (int64_t)N * (K / 2);
    if (idx >= total_bytes) return;

    int row = (int)(idx / (K / 2));
    int byte_col = (int)(idx % (K / 2));
    int col0 = byte_col * 2;
    int col1 = col0 + 1;

    int k_tiles = K / 64;
    int n_tile = row / 128;
    int nw = row % 128;
    int np = nw / 16;
    int ps = (nw % 16) / 8;
    int nl = nw % 8;

    // Helper lambda for swizzled nibble read
    auto read_nibble = [&](int col) -> int {
        int k_tile = col / 64;
        int kps = (col % 64) / 32;
        int kl = ((col % 64) % 32) / 8;
        int rk = (col % 64) % 8;

        int int32_idx = n_tile * (k_tiles * 1024) + k_tile * 1024
                      + np * 128 + nl * 16 + kl * 4 + ps * 2 + kps;
        int byte_offset = int32_idx * 4 + rk / 2;
        return (qweight_swizzled[byte_offset] >> ((rk % 2) * 4)) & 0xF;
    };

    int nib0 = read_nibble(col0);
    int nib1 = read_nibble(col1);

    // Pack: low nibble = even col, high nibble = odd col
    qweight_rowmajor[idx] = (uint8_t)(nib0 | (nib1 << 4));

    // Unswizzle scale for the group containing col0 (once per group boundary)
    int group_size = K / num_groups;
    int group = col0 / group_size;
    if (col0 % group_size == 0) {
        int d2 = nw / 16;
        int d3 = (nw % 16) / 8;
        int d4 = (nw % 8) / 2;
        int d5 = nw % 2;
        int scale_offset = n_tile * num_groups * 128 + group * 128
                         + d2 * 16 + d4 * 4 + d3 * 2 + d5;

        float scale = __bfloat162float(wscales_swizzled[scale_offset]);
        wscales_rowmajor[group * N + row] = scale;
    }
}

void unswizzle_nunchaku_weights(const uint8_t* qweight_swizzled, const __nv_bfloat16* wscales_swizzled,
                                 uint8_t* qweight_rowmajor, float* wscales_rowmajor,
                                 int N, int K, int num_groups, cudaStream_t stream) {
    int64_t total_bytes = (int64_t)N * (K / 2);
    int block = 256;
    int grid = (int)((total_bytes + block - 1) / block);
    unswizzle_nunchaku_weights_kernel<<<grid, block, 0, stream>>>(
        qweight_swizzled, wscales_swizzled, qweight_rowmajor, wscales_rowmajor, N, K, num_groups);
}

// Kernel: Unswizzle nunchaku MMA-packed LoRA weights to standard row-major BF16
// Nunchaku stores proj_down and proj_up in MMA fragment-tiled layout (16×16 tiles with
// lane permutations). This kernel converts to standard row-major for use with cuBLAS.
// is_down: true for proj_down [K, r], false for proj_up [N, r]
__global__ void unswizzle_lora_weights_kernel(
    const __nv_bfloat16* __restrict__ packed,
    __nv_bfloat16* __restrict__ output,
    int rows, int cols, bool is_down)
{
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = (int64_t)rows * cols;
    if (idx >= total) return;

    int row = (int)(idx / cols);
    int col = (int)(idx % cols);

    // Nunchaku MMA-packed LoRA weights use 16×16 fragment tiling with lane permutations.
    // Packed tensor is viewed as 7D: [C/16, R/16, 8, 4, 2, 2, 2] where R = cols (second dim).
    //
    // For is_down=true: packed [K, rank], output [K, rank] = lora_down.T (for cuBLAS).
    //   The standard (unpacked) lora_down has shape [rank, K], so:
    //   output[row, col] = standard[col, row] → swap row/col for standard indices.
    //
    // For is_down=false: packed [N, rank], output [N, rank] = lora_up (direct).
    //   output[row, col] = standard[row, col] → no swap needed.
    int R = cols;  // second dim of packed tensor
    int s_row, s_col;
    if (is_down) {
        s_row = col;  // standard row = output col
        s_col = row;  // standard col = output row
    } else {
        s_row = row;
        s_col = col;
    }

    // Packed tensor is [C/16, R/16, ...] where C is first dim, R is second dim.
    // For is_down=true:  packed [K, rank] → s_row=rank axis (n), s_col=K axis (k)
    //   c_frag = s_col/16 (K tiles), r_frag = s_row/16 (rank tiles)
    // For is_down=false: packed [N, rank] → s_row=N axis (n), s_col=rank axis (k)
    //   c_frag = s_row/16 (N tiles), r_frag = s_col/16 (rank tiles)
    int c_frag, r_frag;
    if (is_down) {
        c_frag = s_col / 16;  // K tiles
        r_frag = s_row / 16;  // rank tiles
    } else {
        c_frag = s_row / 16;  // N tiles
        r_frag = s_col / 16;  // rank tiles
    }
    int local_n = s_row % 16;
    int local_k = s_col % 16;

    // 7D view of inner [frag_n=16, frag_k=16] block:
    //   [n_pack(2), n_lanes(8), k_pack(2), k_lanes(4), lk(2)]
    // local_n = np * 8 + nl,  local_k = kp * 8 + kl * 2 + lk
    int np = local_n / 8;
    int nl = local_n % 8;
    int kp = local_k / 8;
    int kl = (local_k % 8) / 2;
    int lk = local_k % 2;

    // After pack permute (0,1,3,5,2,4,6): strides nl(32) kl(8) np(4) kp(2) lk(1)
    int64_t flat_packed = (int64_t)c_frag * 16 * R + r_frag * 256 + nl * 32 + kl * 8 + np * 4 + kp * 2 + lk;
    output[idx] = packed[flat_packed];
}

void unswizzle_lora_weights(const __nv_bfloat16* packed, __nv_bfloat16* standard,
                             int rows, int cols, bool is_down, cudaStream_t stream) {
    int64_t total = (int64_t)rows * cols;
    int block = 256;
    int grid = (int)((total + block - 1) / block);
    unswizzle_lora_weights_kernel<<<grid, block, 0, stream>>>(packed, standard, rows, cols, is_down);
}

// Kernel B: Quantize FP32 activations to packed INT4 per-group
// Grid: (M, num_groups), Block: min(256, group_size)
// act_scales layout: [num_groups, M] — scale for group g, token m at act_scales[g * M + m]
__global__ void quantize_activation_int4_kernel(
    const float* __restrict__ x,
    uint8_t* __restrict__ act_packed,
    float* __restrict__ act_scales,
    int M, int K, int group_size)
{
    int row = blockIdx.x;
    int group = blockIdx.y;
    if (row >= M) return;

    int col_start = group * group_size;
    const float* grp = x + (int64_t)row * K + col_start;

    // Find max abs in group (warp-reduce + cross-warp reduce)
    float max_abs = 0.0f;
    for (int i = threadIdx.x; i < group_size; i += blockDim.x) {
        float v = fabsf(grp[i]);
        if (v > max_abs) max_abs = v;
    }
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        max_abs = fmaxf(max_abs, __shfl_down_sync(0xffffffff, max_abs, offset));
    __shared__ float s_max[32];
    int lane = threadIdx.x % warpSize;
    int warp_id = threadIdx.x / warpSize;
    if (lane == 0) s_max[warp_id] = max_abs;
    __syncthreads();
    if (warp_id == 0) {
        int nwarps = (blockDim.x + warpSize - 1) / warpSize;
        max_abs = (lane < nwarps) ? s_max[lane] : 0.0f;
        for (int offset = warpSize / 2; offset > 0; offset >>= 1)
            max_abs = fmaxf(max_abs, __shfl_down_sync(0xffffffff, max_abs, offset));
    }
    __syncthreads();
    if (threadIdx.x == 0) s_max[0] = max_abs;
    __syncthreads();
    max_abs = s_max[0];

    float scale = max_abs / 7.0f;
    float inv_scale = (max_abs > 0.0f) ? (7.0f / max_abs) : 0.0f;

    // Store scale: [num_groups, M] layout
    int num_groups = K / group_size;
    if (threadIdx.x == 0)
        act_scales[group * M + row] = scale;

    // Quantize pairs and pack into bytes
    // Two's complement nibble: val & 0xF (for val in [-7,7], this gives [0x9..0xF,0x0..0x7])
    int half_gs = group_size / 2;
    for (int i = threadIdx.x; i < half_gs; i += blockDim.x) {
        int c0 = col_start + i * 2;
        int c1 = c0 + 1;
        float v0 = x[(int64_t)row * K + c0] * inv_scale;
        float v1 = x[(int64_t)row * K + c1] * inv_scale;

        int q0 = (int)roundf(v0); q0 = max(-7, min(7, q0));
        int q1 = (int)roundf(v1); q1 = max(-7, min(7, q1));

        // Pack as two's complement nibbles: low nibble = even, high = odd
        uint8_t packed = (uint8_t)((q0 & 0xF) | ((q1 & 0xF) << 4));
        act_packed[(int64_t)row * (K / 2) + c0 / 2] = packed;
    }
}

void quantize_activation_int4(const float* x, uint8_t* act_packed, float* act_scales,
                               int M, int K, int group_size, cudaStream_t stream) {
    int num_groups = K / group_size;
    dim3 grid(M, num_groups);
    int block = min(256, group_size);
    quantize_activation_int4_kernel<<<grid, block, 0, stream>>>(
        x, act_packed, act_scales, M, K, group_size);
}

// Fused kernel: smooth_div + fp32_to_bf16 + quantize_activation_int4
// One block per row, each warp processes one group independently.
// No shared memory, no __syncthreads — pure warp-level parallelism.
__global__ void fused_smooth_bf16_quant_kernel(
    const float* __restrict__ x,           // [M, K] FP32 input
    const float* __restrict__ smooth,      // [K] FP32 smooth factors (or nullptr)
    __nv_bfloat16* __restrict__ bf16_out,  // [M, K] BF16 output for LoRA
    uint8_t* __restrict__ act_packed,      // [M, K/2] INT4 packed output
    float* __restrict__ act_scales,        // [num_groups, M] FP32 scales
    int M, int K, int group_size)
{
    const int row = blockIdx.x;
    if (row >= M) return;

    const int num_groups = K / group_size;
    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int num_warps = blockDim.x / 32;
    const int elems_per_thread = group_size / 32;  // gs=64 → 2 elements per thread
    const int K_half = K / 2;

    for (int g = warp_id; g < num_groups; g += num_warps) {
        const int col_start = g * group_size;
        const float* grp = x + (int64_t)row * K + col_start;

        // Phase 1: Read, smooth-divide, write BF16, compute absmax
        float vals[4];  // max elems_per_thread (up to gs=128)
        float max_abs = 0.0f;
        #pragma unroll
        for (int i = 0; i < elems_per_thread; i++) {
            int col = lane * elems_per_thread + i;
            float v = grp[col];
            if (smooth) v /= smooth[col_start + col];
            vals[i] = v;
            bf16_out[(int64_t)row * K + col_start + col] = __float2bfloat16(v);
            max_abs = fmaxf(max_abs, fabsf(v));
        }

        // Warp-only absmax reduction (no shared memory!)
        for (int offset = 16; offset > 0; offset >>= 1)
            max_abs = fmaxf(max_abs, __shfl_down_sync(0xffffffff, max_abs, offset));
        max_abs = __shfl_sync(0xffffffff, max_abs, 0);  // broadcast to all lanes

        float scale = max_abs / 7.0f;
        float inv_scale = (max_abs > 0.0f) ? (7.0f / max_abs) : 0.0f;
        if (lane == 0) act_scales[g * M + row] = scale;

        // Phase 2: Quantize and pack (each thread packs its elements)
        // elems_per_thread is always even (gs is multiple of 64)
        #pragma unroll
        for (int i = 0; i < elems_per_thread; i += 2) {
            int q0 = (int)roundf(vals[i] * inv_scale);   q0 = max(-7, min(7, q0));
            int q1 = (int)roundf(vals[i+1] * inv_scale); q1 = max(-7, min(7, q1));
            uint8_t packed = (uint8_t)((q0 & 0xF) | ((q1 & 0xF) << 4));
            int byte_idx = (col_start + lane * elems_per_thread + i) / 2;
            act_packed[(int64_t)row * K_half + byte_idx] = packed;
        }
    }
}

void fused_smooth_bf16_quant(const float* x, const float* smooth,
                              __nv_bfloat16* bf16_out,
                              uint8_t* act_packed, float* act_scales,
                              int M, int K, int group_size, cudaStream_t stream) {
    // One block per row, 8 warps per block (each warp handles one group)
    int block = 256;  // 8 warps
    fused_smooth_bf16_quant_kernel<<<M, block, 0, stream>>>(
        x, smooth, bf16_out, act_packed, act_scales, M, K, group_size);
}

// ================================================================
// W4A4 INT4×INT4 GEMM — Naive scalar kernel (fallback)
// ================================================================
// Block tile: BM×BN output tile, each thread computes TM×TN outputs
// Shared memory loads weight tiles cooperatively
// act_packed: [M, K/2] UINT8, wgt_packed: [N, K/2] UINT8
// act_scales: [num_groups, M] FP32, wgt_scales: [num_groups, N] FP32
// output: [M, N] FP32

#define W4A4_BM 32
#define W4A4_BN 64
#define W4A4_BK 64  // = group_size, process one group at a time
#define W4A4_TM 4   // each thread computes 4 rows
#define W4A4_TN 4   // each thread computes 4 cols

__global__ void w4a4_gemm_naive_kernel(
    const uint8_t* __restrict__ act_packed,
    const uint8_t* __restrict__ wgt_packed,
    const float* __restrict__ act_scales,
    const float* __restrict__ wgt_scales,
    float* __restrict__ output,
    int M, int N, int K, int group_size)
{
    // Block output tile: rows [bm, bm+BM), cols [bn, bn+BN)
    int bm = blockIdx.y * W4A4_BM;
    int bn = blockIdx.x * W4A4_BN;

    // Thread position within block
    int tid = threadIdx.x;
    int threads_per_block = (W4A4_BM / W4A4_TM) * (W4A4_BN / W4A4_TN); // 8*16=128
    int ty = tid / (W4A4_BN / W4A4_TN); // row of thread tile
    int tx = tid % (W4A4_BN / W4A4_TN); // col of thread tile

    int num_groups = K / group_size;
    int half_gs = group_size / 2; // bytes per group

    // Shared memory for weight tile: [BN, half_gs] bytes
    // and activation tile: [BM, half_gs] bytes
    __shared__ uint8_t s_wgt[W4A4_BN][W4A4_BK / 2]; // 64 × 32 = 2048 bytes
    __shared__ uint8_t s_act[W4A4_BM][W4A4_BK / 2]; // 32 × 32 = 1024 bytes

    // Accumulators
    float acc[W4A4_TM][W4A4_TN];
    for (int i = 0; i < W4A4_TM; i++)
        for (int j = 0; j < W4A4_TN; j++)
            acc[i][j] = 0.0f;

    for (int g = 0; g < num_groups; g++) {
        int k_byte_start = g * half_gs;

        // Cooperatively load weight tile [BN, half_gs] into shared mem
        // Total bytes: BN * half_gs = 64 * 32 = 2048, threads = 128, each loads 16 bytes
        for (int idx = tid; idx < W4A4_BN * half_gs; idx += threads_per_block) {
            int wn = idx / half_gs;
            int wb = idx % half_gs;
            int gn = bn + wn;
            s_wgt[wn][wb] = (gn < N) ? wgt_packed[(int64_t)gn * (K / 2) + k_byte_start + wb] : 0;
        }

        // Cooperatively load activation tile [BM, half_gs]
        for (int idx = tid; idx < W4A4_BM * half_gs; idx += threads_per_block) {
            int am = idx / half_gs;
            int ab = idx % half_gs;
            int gm = bm + am;
            s_act[am][ab] = (gm < M) ? act_packed[(int64_t)gm * (K / 2) + k_byte_start + ab] : 0;
        }

        __syncthreads();

        // Each thread computes TM×TN partial INT32 dot products
        int partial[W4A4_TM][W4A4_TN];
        for (int i = 0; i < W4A4_TM; i++)
            for (int j = 0; j < W4A4_TN; j++)
                partial[i][j] = 0;

        for (int kb = 0; kb < half_gs; kb++) {
            // Load and unpack activation bytes for TM rows
            int a_vals[W4A4_TM][2];
            for (int i = 0; i < W4A4_TM; i++) {
                uint8_t ab = s_act[ty * W4A4_TM + i][kb];
                int a0 = ab & 0xF;
                int a1 = (ab >> 4) & 0xF;
                a_vals[i][0] = (a0 >= 8) ? (a0 - 16) : a0;
                a_vals[i][1] = (a1 >= 8) ? (a1 - 16) : a1;
            }

            // Load and unpack weight bytes for TN cols
            int w_vals[W4A4_TN][2];
            for (int j = 0; j < W4A4_TN; j++) {
                uint8_t wb = s_wgt[tx * W4A4_TN + j][kb];
                int w0 = wb & 0xF;
                int w1 = (wb >> 4) & 0xF;
                w_vals[j][0] = (w0 >= 8) ? (w0 - 16) : w0;
                w_vals[j][1] = (w1 >= 8) ? (w1 - 16) : w1;
            }

            // Accumulate
            for (int i = 0; i < W4A4_TM; i++)
                for (int j = 0; j < W4A4_TN; j++)
                    partial[i][j] += a_vals[i][0] * w_vals[j][0] + a_vals[i][1] * w_vals[j][1];
        }

        // Dequantize and accumulate into FP32
        for (int i = 0; i < W4A4_TM; i++) {
            int gm = bm + ty * W4A4_TM + i;
            if (gm >= M) continue;
            float as = act_scales[g * M + gm];
            for (int j = 0; j < W4A4_TN; j++) {
                int gn = bn + tx * W4A4_TN + j;
                if (gn >= N) continue;
                float ws = wgt_scales[g * N + gn];
                acc[i][j] += (float)partial[i][j] * as * ws;
            }
        }

        __syncthreads();
    }

    // Write output
    for (int i = 0; i < W4A4_TM; i++) {
        int gm = bm + ty * W4A4_TM + i;
        if (gm >= M) continue;
        for (int j = 0; j < W4A4_TN; j++) {
            int gn = bn + tx * W4A4_TN + j;
            if (gn >= N) continue;
            output[(int64_t)gm * N + gn] = acc[i][j];
        }
    }
}

// ================================================================
// W4A4 INT4×INT4 GEMM — Tensor Core MMA kernel
// Uses mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32
// BLOCK_M=64, BLOCK_N=64, K_STEP=64 (=group_size)
// 4 warps (128 threads), 2×2 warp layout → WARP_M=32, WARP_N=32
// Per warp: 2×2 M/N-tiles of 16, each needing 2 MMA calls = 8 MMAs/K-step
// ================================================================

// Inline PTX helper for INT4×INT4 tensor core MMA
__device__ __forceinline__ void mma_s4s4_m16n8k64(
    const uint32_t a[4], const uint32_t b[2],
    const int32_t c[4], int32_t d[4])
{
    asm volatile(
        "mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%10, %11, %12, %13};\n"
        : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]),
          "r"(c[0]), "r"(c[1]), "r"(c[2]), "r"(c[3])
    );
}

__global__ void w4a4_gemm_mma_kernel(
    const uint8_t* __restrict__ act_packed,    // [M, K/2] UINT8
    const uint8_t* __restrict__ wgt_packed,    // [N, K/2] UINT8
    const float* __restrict__ act_scales,      // [num_groups, M] FP32
    const float* __restrict__ wgt_scales,      // [num_groups, N] FP32
    float* __restrict__ output,                // [M, N] FP32
    int M, int N, int K, int group_size)
{
    // Block position
    const int bn = blockIdx.x * 64;
    const int bm = blockIdx.y * 64;

    // Warp and lane info
    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int gid = lane / 4;   // groupID (0..7)
    const int tid = lane % 4;   // threadID (0..3)

    // 2×2 warp layout
    const int warp_m = warp_id / 2;  // 0 or 1
    const int warp_n = warp_id % 2;  // 0 or 1

    const int num_groups = K / group_size;
    const int half_gs = group_size / 2;  // 32 bytes for gs=64

    // Double-buffered shared memory with cp.async for software pipelining
    // cp.async copies global→shared directly (no register stall), enabling
    // overlap of next-group prefetch with current-group MMA compute
    __shared__ uint32_t shmem_a[2][64][9];
    __shared__ uint32_t shmem_w[2][64][9];

    // FP32 accumulators: [mt][nt][mma_half][d_idx] = 2×2×2×4 = 32 per thread
    float acc[2][2][2][4];
    #pragma unroll
    for (int i = 0; i < 2; i++)
        #pragma unroll
        for (int j = 0; j < 2; j++)
            #pragma unroll
            for (int k = 0; k < 2; k++)
                #pragma unroll
                for (int l = 0; l < 4; l++)
                    acc[i][j][k][l] = 0.0f;

    // Preload group 0 into buffer 0 using cp.async
    {
        #pragma unroll
        for (int i = threadIdx.x; i < 64 * 8; i += 128) {
            const int row = i / 8;
            const int col = i % 8;

            const int ma = bm + row;
            if (ma < M) {
                uint32_t dst = static_cast<uint32_t>(__cvta_generic_to_shared(&shmem_a[0][row][col]));
                const void* src = &((const uint32_t*)(act_packed + (int64_t)ma * (K/2)))[col];
                asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" :: "r"(dst), "l"(src));
            } else {
                shmem_a[0][row][col] = 0;
            }

            const int na = bn + row;
            if (na < N) {
                uint32_t dst = static_cast<uint32_t>(__cvta_generic_to_shared(&shmem_w[0][row][col]));
                const void* src = &((const uint32_t*)(wgt_packed + (int64_t)na * (K/2)))[col];
                asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" :: "r"(dst), "l"(src));
            } else {
                shmem_w[0][row][col] = 0;
            }
        }
    }
    asm volatile("cp.async.commit_group;\n" ::: "memory");
    asm volatile("cp.async.wait_group 0;\n" ::: "memory");
    __syncthreads();

    for (int g = 0; g < num_groups; g++) {
        const int cur = g & 1;
        const int nxt = 1 - cur;

        // Prefetch next group into buf[nxt] using cp.async (non-blocking)
        if (g + 1 < num_groups) {
            const int k_byte_start_nxt = (g + 1) * half_gs;
            #pragma unroll
            for (int i = threadIdx.x; i < 64 * 8; i += 128) {
                const int row = i / 8;
                const int col = i % 8;

                const int ma = bm + row;
                if (ma < M) {
                    uint32_t dst = static_cast<uint32_t>(__cvta_generic_to_shared(&shmem_a[nxt][row][col]));
                    const void* src = &((const uint32_t*)(act_packed + (int64_t)ma * (K/2) + k_byte_start_nxt))[col];
                    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" :: "r"(dst), "l"(src));
                } else {
                    shmem_a[nxt][row][col] = 0;
                }

                const int na = bn + row;
                if (na < N) {
                    uint32_t dst = static_cast<uint32_t>(__cvta_generic_to_shared(&shmem_w[nxt][row][col]));
                    const void* src = &((const uint32_t*)(wgt_packed + (int64_t)na * (K/2) + k_byte_start_nxt))[col];
                    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" :: "r"(dst), "l"(src));
                } else {
                    shmem_w[nxt][row][col] = 0;
                }
            }
            asm volatile("cp.async.commit_group;\n" ::: "memory");
        }

        // Load act scales for this thread's 4 M positions
        float as[4];
        #pragma unroll
        for (int mt = 0; mt < 2; mt++) {
            const int m0 = bm + warp_m * 32 + mt * 16 + gid;
            const int m1 = m0 + 8;
            as[mt * 2]     = (m0 < M) ? act_scales[g * M + m0] : 0.0f;
            as[mt * 2 + 1] = (m1 < M) ? act_scales[g * M + m1] : 0.0f;
        }

        // Load wgt scales for this thread's 8 N positions
        float ws[8];
        #pragma unroll
        for (int nt = 0; nt < 2; nt++) {
            #pragma unroll
            for (int mh = 0; mh < 2; mh++) {
                const int n0 = bn + warp_n * 32 + nt * 16 + mh * 8 + tid * 2;
                const int n1 = n0 + 1;
                const int idx = nt * 4 + mh * 2;
                ws[idx]     = (n0 < N) ? wgt_scales[g * N + n0] : 0.0f;
                ws[idx + 1] = (n1 < N) ? wgt_scales[g * N + n1] : 0.0f;
            }
        }

        // Compute: load MMA fragments from buf[cur], run 8 MMAs, dequant+accumulate
        #pragma unroll
        for (int mt = 0; mt < 2; mt++) {
            uint32_t a[4];
            const int m_base = warp_m * 32 + mt * 16;
            a[0] = shmem_a[cur][m_base + gid][tid];
            a[1] = shmem_a[cur][m_base + gid + 8][tid];
            a[2] = shmem_a[cur][m_base + gid][4 + tid];
            a[3] = shmem_a[cur][m_base + gid + 8][4 + tid];

            #pragma unroll
            for (int nt = 0; nt < 2; nt++) {
                #pragma unroll
                for (int mh = 0; mh < 2; mh++) {
                    uint32_t b[2];
                    const int n_start = warp_n * 32 + nt * 16 + mh * 8;
                    b[0] = shmem_w[cur][n_start + gid][tid];
                    b[1] = shmem_w[cur][n_start + gid][4 + tid];

                    int32_t zeros[4] = {0, 0, 0, 0};
                    int32_t d[4];
                    mma_s4s4_m16n8k64(a, b, zeros, d);

                    const float as_lo = as[mt * 2];
                    const float as_hi = as[mt * 2 + 1];
                    const int ws_idx = nt * 4 + mh * 2;

                    acc[mt][nt][mh][0] += (float)d[0] * as_lo * ws[ws_idx];
                    acc[mt][nt][mh][1] += (float)d[1] * as_lo * ws[ws_idx + 1];
                    acc[mt][nt][mh][2] += (float)d[2] * as_hi * ws[ws_idx];
                    acc[mt][nt][mh][3] += (float)d[3] * as_hi * ws[ws_idx + 1];
                }
            }
        }

        // Wait for cp.async prefetch to complete, then sync all threads
        asm volatile("cp.async.wait_group 0;\n" ::: "memory");
        __syncthreads();
    }

    // Write output: each thread writes 32 elements
    #pragma unroll
    for (int mt = 0; mt < 2; mt++) {
        const int m0 = bm + warp_m * 32 + mt * 16 + gid;
        const int m1 = m0 + 8;
        #pragma unroll
        for (int nt = 0; nt < 2; nt++) {
            #pragma unroll
            for (int mh = 0; mh < 2; mh++) {
                const int n0 = bn + warp_n * 32 + nt * 16 + mh * 8 + tid * 2;
                const int n1 = n0 + 1;
                if (m0 < M && n0 < N) output[(int64_t)m0 * N + n0] = acc[mt][nt][mh][0];
                if (m0 < M && n1 < N) output[(int64_t)m0 * N + n1] = acc[mt][nt][mh][1];
                if (m1 < M && n0 < N) output[(int64_t)m1 * N + n0] = acc[mt][nt][mh][2];
                if (m1 < M && n1 < N) output[(int64_t)m1 * N + n1] = acc[mt][nt][mh][3];
            }
        }
    }
}

// ================================================================
// Weight swizzle: row-major → MMA fragment order for register-direct GEMM
// Layout: wgt_mma[g, nt, lane*2..lane*2+1] where lane = gid*4+tid
//   wgt_mma[g,nt,lane*2]   = wgt_rowmajor[nt*8+gid][ g*half_gs + tid*4     : +4 ] as uint32
//   wgt_mma[g,nt,lane*2+1] = wgt_rowmajor[nt*8+gid][ g*half_gs + (4+tid)*4 : +4 ] as uint32
// Grid: (num_groups, N_tiles8), Block: 32 threads
// ================================================================
__global__ void swizzle_w4a4_weights_kernel(
    const uint8_t* __restrict__ wgt_rowmajor,  // [N, K/2]
    uint32_t* __restrict__ wgt_mma,            // [num_groups, N_tiles8, 64]
    int N, int K, int group_size)
{
    const int g = blockIdx.x;       // group index
    const int nt = blockIdx.y;      // N tile index (0..N_tiles8-1)
    const int lane = threadIdx.x;   // 0..31
    const int gid = lane / 4;      // 0..7
    const int tid = lane % 4;      // 0..3

    const int half_gs = group_size / 2;
    const int K_half = K / 2;
    const int N_tiles8 = (N + 7) / 8;

    const int row = nt * 8 + gid;
    uint32_t* out = wgt_mma + ((int64_t)g * N_tiles8 + nt) * 64;

    if (row >= N) {
        out[lane * 2]     = 0;
        out[lane * 2 + 1] = 0;
        return;
    }

    const uint32_t* row_data = (const uint32_t*)(wgt_rowmajor + (int64_t)row * K_half + g * half_gs);
    out[lane * 2]     = row_data[tid];
    out[lane * 2 + 1] = row_data[4 + tid];
}

void swizzle_w4a4_weights_mma(const uint8_t* wgt_rowmajor, uint32_t* wgt_mma,
                               int N, int K, int group_size, cudaStream_t stream) {
    int num_groups = K / group_size;
    int N_tiles8 = (N + 7) / 8;
    dim3 grid(num_groups, N_tiles8);
    dim3 block(32);
    swizzle_w4a4_weights_kernel<<<grid, block, 0, stream>>>(
        wgt_rowmajor, wgt_mma, N, K, group_size);
}

// ================================================================
// W4A4 INT4×INT4 GEMM — Register-Direct kernel (no shared memory)
// Loads data directly from global memory into MMA fragment registers.
// No shared memory, no __syncthreads — warps execute fully independently.
// BLOCK_M=64, BLOCK_N=64, 4 warps stacked along M (WARP_M=16, WARP_N=64)
// Per warp per K-step: 8 MMAs (1 m-tile × 8 n-tiles of m16×n8)
// Weights must be in MMA fragment order (from swizzle_w4a4_weights_mma).
// ================================================================
__global__ void w4a4_gemm_regdirect_kernel(
    const uint8_t* __restrict__ act_packed,    // [M, K/2] UINT8
    const uint32_t* __restrict__ wgt_mma,      // [num_groups, N_tiles8, 64] UINT32
    const float* __restrict__ act_scales,      // [num_groups, M] FP32
    const float* __restrict__ wgt_scales,      // [num_groups, N] FP32
    float* __restrict__ output,                // [M, N] FP32
    int M, int N, int K, int group_size,
    bool accumulate)                           // if true, output[i] += result instead of output[i] = result
{
    // M-tile-major ordering: blockIdx.x = M-tile, blockIdx.y = N-tile
    // This maximizes L2 cache reuse for weight data (all M-blocks for same N-tile
    // run consecutively, sharing weight data in L2 cache)
    const int bm = blockIdx.x * 64;
    const int bn = blockIdx.y * 64;

    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int gid = lane / 4;
    const int tid = lane % 4;

    // 2×2 warp layout: reduces weight load redundancy from 4× to 2×
    const int warp_m = warp_id / 2;  // 0 or 1
    const int warp_n = warp_id % 2;  // 0 or 1

    const int num_groups = K / group_size;
    const int half_gs = group_size / 2;
    const int K_half = K / 2;
    const int N_tiles8 = (N + 7) / 8;

    // Accumulators: 2 mt × 4 nt × 4 elements = 32 floats per thread
    float acc[2][4][4];
    #pragma unroll
    for (int i = 0; i < 2; i++)
        #pragma unroll
        for (int j = 0; j < 4; j++)
            #pragma unroll
            for (int k = 0; k < 4; k++)
                acc[i][j][k] = 0.0f;

    // Precompute row pointers for 2 m-tiles (32 M rows per warp)
    const int m_base = bm + warp_m * 32;
    const uint32_t* a_rows[4];  // [mt][lo/hi]
    int m_vals[4];
    #pragma unroll
    for (int mt = 0; mt < 2; mt++) {
        m_vals[mt * 2]     = m_base + mt * 16 + gid;
        m_vals[mt * 2 + 1] = m_base + mt * 16 + gid + 8;
        a_rows[mt * 2]     = (m_vals[mt * 2] < M)     ? (const uint32_t*)(act_packed + (int64_t)m_vals[mt * 2] * K_half) : nullptr;
        a_rows[mt * 2 + 1] = (m_vals[mt * 2 + 1] < M) ? (const uint32_t*)(act_packed + (int64_t)m_vals[mt * 2 + 1] * K_half) : nullptr;
    }

    const int bn_tile = bn / 8 + warp_n * 4;  // first n-tile for this warp's N half

    for (int g = 0; g < num_groups; g++) {
        const int k_u32_start = g * (half_gs / 4);

        // Preload all 4 B fragments for this warp's N half (reused across mt=0,1)
        uint32_t b_all[4][2];
        float ws_all[8];
        #pragma unroll
        for (int nt = 0; nt < 4; nt++) {
            const int nt_global = bn_tile + nt;
            if (nt_global < N_tiles8) {
                const uint32_t* b_ptr = wgt_mma + ((int64_t)g * N_tiles8 + nt_global) * 64 + lane * 2;
                b_all[nt][0] = b_ptr[0];
                b_all[nt][1] = b_ptr[1];
            } else {
                b_all[nt][0] = 0; b_all[nt][1] = 0;
            }
            const int n0 = bn + warp_n * 32 + nt * 8 + tid * 2;
            ws_all[nt * 2]     = (n0 < N)     ? wgt_scales[g * N + n0] : 0.0f;
            ws_all[nt * 2 + 1] = (n0 + 1 < N) ? wgt_scales[g * N + n0 + 1] : 0.0f;
        }

        // Process 2 m-tiles, each reusing the preloaded B fragments
        #pragma unroll
        for (int mt = 0; mt < 2; mt++) {
            // Load A fragment for this m-tile
            uint32_t a[4];
            if (a_rows[mt * 2]) {
                a[0] = a_rows[mt * 2][k_u32_start + tid];
                a[2] = a_rows[mt * 2][k_u32_start + 4 + tid];
            } else {
                a[0] = 0; a[2] = 0;
            }
            if (a_rows[mt * 2 + 1]) {
                a[1] = a_rows[mt * 2 + 1][k_u32_start + tid];
                a[3] = a_rows[mt * 2 + 1][k_u32_start + 4 + tid];
            } else {
                a[1] = 0; a[3] = 0;
            }

            const float as_lo = (m_vals[mt * 2] < M)     ? act_scales[g * M + m_vals[mt * 2]] : 0.0f;
            const float as_hi = (m_vals[mt * 2 + 1] < M) ? act_scales[g * M + m_vals[mt * 2 + 1]] : 0.0f;

            #pragma unroll
            for (int nt = 0; nt < 4; nt++) {
                int32_t zeros[4] = {0, 0, 0, 0};
                int32_t d[4];
                mma_s4s4_m16n8k64(a, b_all[nt], zeros, d);

                acc[mt][nt][0] += (float)d[0] * as_lo * ws_all[nt * 2];
                acc[mt][nt][1] += (float)d[1] * as_lo * ws_all[nt * 2 + 1];
                acc[mt][nt][2] += (float)d[2] * as_hi * ws_all[nt * 2];
                acc[mt][nt][3] += (float)d[3] * as_hi * ws_all[nt * 2 + 1];
            }
        }
    }

    // Write output (accumulate mode: add to existing values e.g. LoRA output)
    #pragma unroll
    for (int mt = 0; mt < 2; mt++) {
        const int m0 = m_vals[mt * 2];
        const int m1 = m_vals[mt * 2 + 1];
        #pragma unroll
        for (int nt = 0; nt < 4; nt++) {
            const int n0 = bn + warp_n * 32 + nt * 8 + tid * 2;
            const int n1 = n0 + 1;
            if (accumulate) {
                if (m0 < M && n0 < N) output[(int64_t)m0 * N + n0] += acc[mt][nt][0];
                if (m0 < M && n1 < N) output[(int64_t)m0 * N + n1] += acc[mt][nt][1];
                if (m1 < M && n0 < N) output[(int64_t)m1 * N + n0] += acc[mt][nt][2];
                if (m1 < M && n1 < N) output[(int64_t)m1 * N + n1] += acc[mt][nt][3];
            } else {
                if (m0 < M && n0 < N) output[(int64_t)m0 * N + n0] = acc[mt][nt][0];
                if (m0 < M && n1 < N) output[(int64_t)m0 * N + n1] = acc[mt][nt][1];
                if (m1 < M && n0 < N) output[(int64_t)m1 * N + n0] = acc[mt][nt][2];
                if (m1 < M && n1 < N) output[(int64_t)m1 * N + n1] = acc[mt][nt][3];
            }
        }
    }
}

void w4a4_gemm(const uint8_t* act_packed, const uint8_t* wgt_packed,
               const float* act_scales, const float* wgt_scales,
               float* output, int M, int N, int K, int group_size,
               const uint32_t* wgt_mma, bool accumulate, cudaStream_t stream) {
    static int force_naive = -1;
    if (force_naive < 0) force_naive = (getenv("FORCE_NAIVE_W4A4") != nullptr) ? 1 : 0;
    static int force_shmem = -1;
    if (force_shmem < 0) force_shmem = (getenv("FORCE_SHMEM_W4A4") != nullptr) ? 1 : 0;
    static int verify_mma = -1;
    if (verify_mma < 0) verify_mma = (getenv("VERIFY_MMA_W4A4") != nullptr) ? 1 : 0;

    if (force_naive) {
        dim3 block(128);
        dim3 grid((N + W4A4_BN - 1) / W4A4_BN, (M + W4A4_BM - 1) / W4A4_BM);
        w4a4_gemm_naive_kernel<<<grid, block, 0, stream>>>(
            act_packed, wgt_packed, act_scales, wgt_scales, output, M, N, K, group_size);
    } else if (wgt_mma && !force_shmem) {
        dim3 block(128);
        dim3 grid((M + 63) / 64, (N + 63) / 64);  // M-tile-major for L2 cache reuse
        w4a4_gemm_regdirect_kernel<<<grid, block, 0, stream>>>(
            act_packed, wgt_mma, act_scales, wgt_scales, output, M, N, K, group_size, accumulate);

        // Verify regdirect vs shmem kernel (spot-check first few calls)
        if (verify_mma) {
            static int verify_count = 0;
            if (verify_count < 3) {
                verify_count++;
                float* ref_output;
                cudaMalloc(&ref_output, (int64_t)M * N * sizeof(float));
                w4a4_gemm_mma_kernel<<<grid, block, 0, stream>>>(
                    act_packed, wgt_packed, act_scales, wgt_scales, ref_output, M, N, K, group_size);
                cudaDeviceSynchronize();

                // Spot-check 64 elements
                int check = std::min(64, M * N);
                std::vector<float> rd(check), sm(check);
                cudaMemcpy(rd.data(), output, check * sizeof(float), cudaMemcpyDeviceToHost);
                cudaMemcpy(sm.data(), ref_output, check * sizeof(float), cudaMemcpyDeviceToHost);
                int mismatches = 0;
                double max_diff = 0;
                for (int i = 0; i < check; i++) {
                    if (rd[i] != sm[i]) { mismatches++; double d = fabs(rd[i] - sm[i]); if (d > max_diff) max_diff = d; }
                }
                fprintf(stderr, "  [VERIFY_MMA_W4A4 #%d] M=%d N=%d K=%d: first-%d mismatches=%d max_diff=%.6f\n",
                        verify_count, M, N, K, check, mismatches, max_diff);
                cudaFree(ref_output);
            }
        }
    } else {
        dim3 block(128);
        dim3 grid((N + 63) / 64, (M + 63) / 64);
        w4a4_gemm_mma_kernel<<<grid, block, 0, stream>>>(
            act_packed, wgt_packed, act_scales, wgt_scales, output, M, N, K, group_size);
    }
}

// Simple FP32 element-wise add
__global__ void add_fp32_kernel(const float* __restrict__ a, const float* __restrict__ b,
                                 float* __restrict__ out, int64_t n) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = a[idx] + b[idx];
}

void add_fp32(const float* a, const float* b, float* out, int64_t n, cudaStream_t stream) {
    int block = 256;
    int grid = (int)((n + block - 1) / block);
    add_fp32_kernel<<<grid, block, 0, stream>>>(a, b, out, n);
}

// Global flag: simulate W4A4 by adding INT4 activation quantization noise
bool g_w4a4_mode = false;

// INT4+SVD linear forward: FP32 input, INT4 weight -> FP32 output
// out = x @ R_dequant^T + x @ svd_down @ svd_up^T + bias
void linear_forward_int4(const Tensor& x, const QuantizedWeight& weight,
                          const Tensor* bias, Tensor& out, Tensor& scratch) {
    assert(x.dtype == DType::FP32);
    assert(out.dtype == DType::FP32);
    assert(weight.mode == QuantMode::INT4_SVD);

    int M = (int)x.shape[0];
    // Use rowmajor tensors for dimensions when swizzled originals are freed
    const Tensor& qw_ref = weight.qweight_rowmajor.data ? weight.qweight_rowmajor : weight.qweight;
    int N = (int)qw_ref.shape[0];
    int K = (int)(qw_ref.shape[1] * 2);
    int r = weight.svd_rank;

    // Scratch layout (256-byte aligned regions):
    //   [A] BF16 dequantized weight [N, K]  (Hadamard-rotated residual)
    //   [B] FP32 temp [M, K]  (for Hadamard rotation of input)
    //   [C] BF16 input [M, K]  (reused: first x_orig for low-rank, then x_rot for residual)
    //   [D] BF16 low-rank intermediate [M, r]
    int64_t offB = (((int64_t)N * K * 2 + 255) & ~255LL);
    int64_t offC = offB + (((int64_t)M * K * 4 + 255) & ~255LL);
    int64_t offD = offC + (((int64_t)M * K * 2 + 255) & ~255LL);

    __nv_bfloat16* deq_w    = (__nv_bfloat16*)scratch.data;
    float*         fp32_tmp = (float*)((char*)scratch.data + offB);
    __nv_bfloat16* bf16_x   = (__nv_bfloat16*)((char*)scratch.data + offC);
    __nv_bfloat16* lr_tmp   = (__nv_bfloat16*)((char*)scratch.data + offD);

    float alpha = 1.0f, beta0 = 0.0f, beta1 = 1.0f;

    bool has_smooth = (weight.smooth.data != nullptr);
    bool use_w4a4 = weight.nunchaku_swizzle && weight.qweight_rowmajor.data && !getenv("FORCE_W4A16");

    // Internal profiling (activated with PROFILE_LINEAR env var)
    static bool profile_linear = (getenv("PROFILE_LINEAR") != nullptr);
    static float prof_prep = 0, prof_lora = 0, prof_w4a4 = 0, prof_other = 0;
    static int prof_count = 0;
    static cudaEvent_t pl_start, pl_end;
    static bool pl_inited = false;
    if (profile_linear && !pl_inited) {
        cudaEventCreate(&pl_start); cudaEventCreate(&pl_end); pl_inited = true;
    }
    #define PL_START() do { if (profile_linear) cudaEventRecord(pl_start); } while(0)
    #define PL_END(acc) do { if (profile_linear) { cudaEventRecord(pl_end); cudaEventSynchronize(pl_end); \
        float ms; cudaEventElapsedTime(&ms, pl_start, pl_end); acc += ms; } } while(0)

    // For W4A4 path: fused smooth_div + bf16 conversion + INT4 quantization in one pass
    // For non-W4A4 paths: separate smooth_div + bf16 conversion
    int num_groups = K / weight.group_size;
    uint8_t* act_packed = (uint8_t*)deq_w;
    float* act_scales_ptr = (float*)((char*)act_packed + (((int64_t)M * (K / 2) + 255) & ~255LL));

    PL_START();
    if (use_w4a4) {
        static int w4a4_count = 0;
        if (w4a4_count < 2) {
            fprintf(stderr, "  [W4A4 path] M=%d K=%d N=%d r=%d gs=%d smooth=%d\n",
                    M, K, N, r, weight.group_size, has_smooth ? 1 : 0);
            w4a4_count++;
        }
        // Fused: read x once → write bf16_x + act_packed + act_scales
        fused_smooth_bf16_quant((const float*)x.data,
                                has_smooth ? (const float*)weight.smooth.data : nullptr,
                                bf16_x, act_packed, act_scales_ptr,
                                M, K, weight.group_size);
    } else {
        // Non-W4A4: separate smooth + bf16 conversion
        if (has_smooth) {
            apply_smooth_div((const float*)x.data, (const float*)weight.smooth.data, fp32_tmp, M, K);
            fp32_to_bf16(fp32_tmp, bf16_x, (int64_t)M * K);
        } else {
            fp32_to_bf16((const float*)x.data, bf16_x, (int64_t)M * K);
        }
    }
    PL_END(prof_prep);

    // 2. Low-rank down: lr_tmp[M,r] = bf16_x[M,K] @ svd_down[K,r]
    PL_START();
    CUBLAS_CHECK(cublasGemmEx(cublas(),
        CUBLAS_OP_N, CUBLAS_OP_N,
        r, M, K, &alpha,
        weight.svd_down.data, CUDA_R_16BF, r,
        bf16_x,               CUDA_R_16BF, K,
        &beta0,
        lr_tmp, CUDA_R_16BF, r,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));

    // 3. Low-rank up: out[M,N] = lr_tmp[M,r] @ svd_up[N,r]^T  (beta=0)
    CUBLAS_CHECK(cublasGemmEx(cublas(),
        CUBLAS_OP_T, CUBLAS_OP_N,
        N, M, r, &alpha,
        weight.svd_up.data, CUDA_R_16BF, r,
        lr_tmp,             CUDA_R_16BF, r,
        &beta0,
        (float*)out.data, CUDA_R_32F, N,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    PL_END(prof_lora);

    // 4. Residual path
    PL_START();
    if (use_w4a4) {
        // W4A4 GEMM with accumulate=true: adds result directly to out (which has LoRA output)
        w4a4_gemm(act_packed, (uint8_t*)weight.qweight_rowmajor.data,
                  act_scales_ptr, (float*)weight.wscales_rowmajor.data,
                  (float*)out.data, M, N, K, weight.group_size,
                  weight.qweight_mma.data ? (const uint32_t*)weight.qweight_mma.data : nullptr,
                  true);  // accumulate into LoRA output
        PL_END(prof_w4a4);
    } else if (weight.nunchaku_swizzle && weight.qweight_rowmajor.data) {
        PL_END(prof_w4a4); // end timing for non-W4A4 paths (counts as residual)
        // W4A16 fallback for nunchaku (FORCE_W4A16): dequant unswizzled → BF16 GEMM
        static int w4a16_fb = 0;
        if (w4a16_fb < 2) { fprintf(stderr, "  [W4A16-nunchaku fallback] M=%d K=%d N=%d\n", M, K, N); w4a16_fb++; }

        if (has_smooth) {
            apply_smooth_div((const float*)x.data, (const float*)weight.smooth.data, fp32_tmp, M, K);
        } else {
            CUDA_CHECK(cudaMemcpy(fp32_tmp, x.data, (int64_t)M * K * sizeof(float), cudaMemcpyDeviceToDevice));
        }
        fp32_to_bf16(fp32_tmp, bf16_x, (int64_t)M * K);

        // Dequantize row-major INT4 → BF16 using unswizzled weights & scales
        // wscales_rowmajor is [num_groups, N] layout (scales_gn=true)
        // nunchaku uses signed INT4 two's complement (signed_int4=true)
        int num_groups = K / weight.group_size;
        dequantize_int4_to_bf16(
            (uint8_t*)weight.qweight_rowmajor.data, (float*)weight.wscales_rowmajor.data,
            deq_w, N, K, weight.group_size, 0, false, true, true);

        // Residual GEMM: out[M,N] += bf16_x[M,K] @ deq_w[N,K]^T  (beta=1, accumulate)
        CUBLAS_CHECK(cublasGemmEx(cublas(),
            CUBLAS_OP_T, CUBLAS_OP_N,
            N, M, K, &alpha,
            deq_w,  CUDA_R_16BF, K,
            bf16_x, CUDA_R_16BF, K,
            &beta1,
            (float*)out.data, CUDA_R_32F, N,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    } else {
        // W4A16 path (original non-nunchaku)
        CUDA_CHECK(cudaMemcpy(fp32_tmp, x.data, (int64_t)M * K * sizeof(float),
                               cudaMemcpyDeviceToDevice));
        if (has_smooth && weight.had_block_size > 1) {
            smooth_hadamard_inplace(fp32_tmp, (const float*)weight.smooth.data, M, K, weight.had_block_size);
        } else if (has_smooth) {
            apply_smooth_div((const float*)x.data, (const float*)weight.smooth.data, fp32_tmp, M, K);
        } else if (weight.had_block_size > 1) {
            fwht_inplace(fp32_tmp, M, K, weight.had_block_size);
        }

        // W4A4 noise simulation (legacy)
        if (g_w4a4_mode) {
            act_quant_noise_int4(fp32_tmp, M, K, weight.group_size);
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        // Convert rotated FP32 -> BF16 (reuse region C)
        fp32_to_bf16(fp32_tmp, bf16_x, (int64_t)M * K);

        // Dequantize INT4 packed [N, K/2] -> BF16 [N, K]
        dequantize_int4_to_bf16(
            (uint8_t*)weight.qweight.data, (float*)weight.scales4.data,
            deq_w, N, K, weight.group_size, 0, weight.nf4_grid);

        // Residual GEMM: out[M,N] += bf16_x_rot[M,K] @ deq_w[N,K]^T  (beta=1, accumulate)
        CUBLAS_CHECK(cublasGemmEx(cublas(),
            CUBLAS_OP_T, CUBLAS_OP_N,
            N, M, K, &alpha,
            deq_w,  CUDA_R_16BF, K,
            bf16_x, CUDA_R_16BF, K,
            &beta1,
            (float*)out.data, CUDA_R_32F, N,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        PL_END(prof_w4a4);
    }

    if (profile_linear) {
        prof_count++;
        // Print every 720 calls (one full step)
        if (prof_count % 1440 == 0) {
            float total = prof_prep + prof_lora + prof_w4a4;
            fprintf(stderr, "\n=== Linear INT4 Profile (%d calls, total %.1f ms) ===\n", prof_count, total);
            fprintf(stderr, "  prep (fused/smooth+bf16): %7.1f ms  (%4.1f%%)\n", prof_prep, 100*prof_prep/total);
            fprintf(stderr, "  lora (svd_down+svd_up):   %7.1f ms  (%4.1f%%)\n", prof_lora, 100*prof_lora/total);
            fprintf(stderr, "  w4a4 (GEMM+accumulate):   %7.1f ms  (%4.1f%%)\n", prof_w4a4, 100*prof_w4a4/total);
            fprintf(stderr, "=============================================\n\n");
            prof_prep = prof_lora = prof_w4a4 = 0;
        }
    }

    // 8. Add bias
    if (bias) {
        bias_add_fp32((float*)out.data, (__nv_bfloat16*)bias->data,
                      (float*)out.data, M, N);
    }
}

// BF16 native forward: FP32 input, BF16 weight → FP32 output
static void linear_forward_bf16(const Tensor& x, const QuantizedWeight& weight,
                                 const Tensor* bias, Tensor& out, Tensor& scratch) {
    assert(x.dtype == DType::FP32);
    assert(out.dtype == DType::FP32);
    assert(weight.data.dtype == DType::BF16);

    int M = (int)x.shape[0];
    int K = (int)x.shape[1];
    int N = (int)weight.data.shape[0];

    // Convert FP32 input → BF16 in scratch
    __nv_bfloat16* bf16_x = (__nv_bfloat16*)scratch.data;
    fp32_to_bf16((const float*)x.data, bf16_x, (int64_t)M * K);

    // BF16 × BF16 → FP32 GEMM
    float alpha = 1.0f, beta = 0.0f;
    CUBLAS_CHECK(cublasGemmEx(cublas(),
        CUBLAS_OP_T, CUBLAS_OP_N,
        N, M, K, &alpha,
        weight.data.data, CUDA_R_16BF, K,
        bf16_x,           CUDA_R_16BF, K,
        &beta,
        (float*)out.data, CUDA_R_32F, N,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));

    if (bias) {
        bias_add_fp32((float*)out.data, (__nv_bfloat16*)bias->data,
                      (float*)out.data, M, N);
    }
}

// BF16 native forward: BF16 input, BF16 weight → FP32 output
static void linear_forward_bf16_bf16in(const Tensor& x, const QuantizedWeight& weight,
                                        const Tensor* bias, Tensor& out) {
    assert(x.dtype == DType::BF16);
    assert(out.dtype == DType::FP32);
    assert(weight.data.dtype == DType::BF16);

    int M = (int)x.shape[0];
    int K = (int)x.shape[1];
    int N = (int)weight.data.shape[0];

    float alpha = 1.0f, beta = 0.0f;
    CUBLAS_CHECK(cublasGemmEx(cublas(),
        CUBLAS_OP_T, CUBLAS_OP_N,
        N, M, K, &alpha,
        weight.data.data, CUDA_R_16BF, K,
        x.data,           CUDA_R_16BF, K,
        &beta,
        (float*)out.data, CUDA_R_32F, N,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));

    if (bias) {
        bias_add_fp32((float*)out.data, (__nv_bfloat16*)bias->data,
                      (float*)out.data, M, N);
    }
}

// Unified dispatchers
void linear_forward_quantized(const Tensor& x, const QuantizedWeight& weight,
                               const Tensor* bias, Tensor& out, Tensor& scratch) {
    if (weight.mode == QuantMode::BF16) {
        linear_forward_bf16(x, weight, bias, out, scratch);
    } else if (weight.mode == QuantMode::INT4_SVD) {
        linear_forward_int4(x, weight, bias, out, scratch);
    } else {
        linear_forward_int8(x, weight, bias, out, scratch);
    }
}

void linear_forward_quantized_bf16in(const Tensor& x, const QuantizedWeight& weight,
                                      const Tensor* bias, Tensor& out, Tensor& scratch) {
    if (weight.mode == QuantMode::BF16) {
        linear_forward_bf16_bf16in(x, weight, bias, out);
    } else if (weight.mode == QuantMode::INT4_SVD) {
        int M = (int)x.shape[0];
        int K = (int)x.shape[1];
        Tensor fp32_x = Tensor::alloc({(int64_t)M, (int64_t)K}, DType::FP32);
        bf16_to_fp32((__nv_bfloat16*)x.data, (float*)fp32_x.data, (int64_t)M * K);
        linear_forward_int4(fp32_x, weight, bias, out, scratch);
        fp32_x.free_data();
    } else {
        linear_forward_int8_bf16in(x, weight, bias, out, scratch);
    }
}

// ================================================================
// Randomized SVD helpers (CPU)
// ================================================================

// Modified Gram-Schmidt QR on CPU: Q [N, rp] (in-place on input)
static void cpu_gram_schmidt(float* Q, int N, int rp) {
    for (int j = 0; j < rp; j++) {
        // Normalize column j
        double norm = 0;
        for (int i = 0; i < N; i++) {
            double v = Q[(int64_t)i * rp + j];
            norm += v * v;
        }
        norm = sqrt(norm);
        if (norm > 1e-12) {
            float inv = (float)(1.0 / norm);
            for (int i = 0; i < N; i++) Q[(int64_t)i * rp + j] *= inv;
        }
        // Orthogonalize remaining columns against j
        for (int k = j + 1; k < rp; k++) {
            double dot = 0;
            for (int i = 0; i < N; i++)
                dot += (double)Q[(int64_t)i * rp + j] * Q[(int64_t)i * rp + k];
            float d = (float)dot;
            for (int i = 0; i < N; i++)
                Q[(int64_t)i * rp + k] -= d * Q[(int64_t)i * rp + j];
        }
    }
}

// Jacobi eigendecomposition of symmetric matrix C [n, n] on CPU
// Returns eigenvalues in D[n] (descending), eigenvectors as columns of C (overwritten)
static void cpu_jacobi_eig(float* C, float* D, int n, int max_iters = 100) {
    // Initialize eigenvectors to identity
    std::vector<float> V(n * n, 0.0f);
    for (int i = 0; i < n; i++) V[i * n + i] = 1.0f;

    for (int iter = 0; iter < max_iters; iter++) {
        // Find largest off-diagonal element
        float max_off = 0;
        int p = 0, q = 1;
        for (int i = 0; i < n; i++)
            for (int j = i + 1; j < n; j++) {
                float v = fabsf(C[i * n + j]);
                if (v > max_off) { max_off = v; p = i; q = j; }
            }
        if (max_off < 1e-10f) break;

        // Compute rotation
        float app = C[p * n + p], aqq = C[q * n + q], apq = C[p * n + q];
        float theta = 0.5f * atan2f(2.0f * apq, app - aqq);
        float c = cosf(theta), s = sinf(theta);

        // Apply rotation to C
        for (int i = 0; i < n; i++) {
            float cip = C[i * n + p], ciq = C[i * n + q];
            C[i * n + p] = c * cip + s * ciq;
            C[i * n + q] = -s * cip + c * ciq;
        }
        for (int j = 0; j < n; j++) {
            float cpj = C[p * n + j], cqj = C[q * n + j];
            C[p * n + j] = c * cpj + s * cqj;
            C[q * n + j] = -s * cpj + c * cqj;
        }
        // Apply rotation to V (eigenvectors)
        for (int i = 0; i < n; i++) {
            float vip = V[i * n + p], viq = V[i * n + q];
            V[i * n + p] = c * vip + s * viq;
            V[i * n + q] = -s * vip + c * viq;
        }
    }

    // Extract eigenvalues and sort descending
    std::vector<int> idx(n);
    for (int i = 0; i < n; i++) { D[i] = C[i * n + i]; idx[i] = i; }
    std::sort(idx.begin(), idx.end(), [&](int a, int b) { return D[a] > D[b]; });

    // Reorder D and V
    std::vector<float> D_sorted(n);
    std::vector<float> V_sorted(n * n);
    for (int j = 0; j < n; j++) {
        D_sorted[j] = D[idx[j]];
        for (int i = 0; i < n; i++)
            V_sorted[i * n + j] = V[i * n + idx[j]];
    }
    memcpy(D, D_sorted.data(), n * sizeof(float));
    memcpy(C, V_sorted.data(), n * n * sizeof(float)); // C now holds sorted eigenvectors
}

// Quantize BF16 weight to INT4+SVD
QuantizedWeight quantize_weight_tensor_int4(const Tensor& bf16_weight, int rank, int group_size) {
    assert(bf16_weight.dtype == DType::BF16);
    assert(bf16_weight.ndim == 2);

    int N = (int)bf16_weight.shape[0];
    int K = (int)bf16_weight.shape[1];

    // Adjust group_size for small K
    int gs = std::min(group_size, K);
    // Ensure K is divisible by gs
    while (K % gs != 0 && gs > 2) gs /= 2;
    assert(K % gs == 0);
    assert(K % 2 == 0);

    // Clamp rank to min(N, K) - 1
    int r = std::min(rank, std::min(N, K) - 1);
    int p = 10; // oversampling
    int rp = r + p;
    if (rp > std::min(N, K)) rp = std::min(N, K);
    if (r > rp) r = rp;

    // Convert BF16 -> FP32 on GPU
    Tensor fp32_w = Tensor::alloc({(int64_t)N, (int64_t)K}, DType::FP32);
    bf16_to_fp32((__nv_bfloat16*)bf16_weight.data, (float*)fp32_w.data, (int64_t)N * K);

    // === Randomized SVD (all heavy compute on GPU) ===
    float alpha_f = 1.0f, beta_f = 0.0f, neg_one = -1.0f, one = 1.0f;

    // 1. Random Omega [K, rp] on CPU, upload
    std::vector<float> omega_host((int64_t)K * rp);
    srand(42);
    for (auto& v : omega_host) v = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
    Tensor omega = Tensor::alloc({(int64_t)K, (int64_t)rp}, DType::FP32);
    omega.from_host(omega_host.data());

    // 2. Y = W @ Omega: [N, rp] on GPU
    // Row-major: Y_rm[N,rp] = W_rm[N,K] @ Omega_rm[K,rp]
    // cuBLAS col-major: C[rp,N] = Omega_cm[rp,K] @ W_cm[K,N]
    Tensor Y = Tensor::alloc({(int64_t)N, (int64_t)rp}, DType::FP32);
    CUBLAS_CHECK(cublasGemmEx(cublas(),
        CUBLAS_OP_N, CUBLAS_OP_N, rp, N, K, &alpha_f,
        omega.data, CUDA_R_32F, rp, fp32_w.data, CUDA_R_32F, K, &beta_f,
        Y.data, CUDA_R_32F, rp, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
    omega.free_data();

    // 3. GPU Modified Gram-Schmidt QR: Y -> Q in-place, row-major [N, rp]
    //    Column j of Q_rm has stride rp in memory: elements at Q_data[j], Q_data[j+rp], ...
    //    cuBLAS Level-1/2 ops work with strided vectors natively.
    {
        float* Q_data = (float*)Y.data;
        Tensor dots_buf = Tensor::alloc({(int64_t)rp}, DType::FP32);
        float zero_f = 0.0f, neg_one_f = -1.0f, one_f = 1.0f;

        for (int j = 0; j < rp; j++) {
            // Normalize column j (stride = rp)
            float norm_sq;
            CUBLAS_CHECK(cublasSdot(cublas(), N, Q_data + j, rp, Q_data + j, rp, &norm_sq));
            float inv_norm = (norm_sq > 1e-24f) ? (1.0f / sqrtf(norm_sq)) : 0.0f;
            CUBLAS_CHECK(cublasSscal(cublas(), N, &inv_norm, Q_data + j, rp));

            if (j + 1 < rp) {
                int rem = rp - j - 1;
                // Compute dot products of column j with columns j+1..rp-1
                // In cuBLAS col-major: the row-major [N,rp] matrix is [rp,N] with lda=rp.
                // Submatrix at Q_data+(j+1) with dimensions [rem, N] col-major, lda=rp.
                // CUBLAS_OP_N: A[rem,N] @ x[N] -> y[rem]
                // x = column j (stride rp), y = dots
                CUBLAS_CHECK(cublasSgemv(cublas(), CUBLAS_OP_N,
                    rem, N, &one_f,
                    Q_data + (j + 1), rp,
                    Q_data + j, rp,
                    &zero_f, (float*)dots_buf.data, 1));

                // Rank-1 update: Q[:,j+1:] -= col_j @ dots^T
                // A[rem,N] -= x[rem] @ y[N]^T
                CUBLAS_CHECK(cublasSger(cublas(), rem, N, &neg_one_f,
                    (float*)dots_buf.data, 1,
                    Q_data + j, rp,
                    Q_data + (j + 1), rp));
            }
        }
        dots_buf.free_data();
    }
    // Y now holds Q in row-major [N, rp]

    // 4. B = Q^T @ W: [rp, K] on GPU
    // Row-major: B_rm[rp,K]. cuBLAS col-major: B_cm[K,rp] lda=K
    // C[K,rp] = W_cm[K,N] @ Q_cm^T[N,rp]  (Q_cm is [rp,N] lda=rp, OP_T gives [N,rp])
    Tensor B = Tensor::alloc({(int64_t)rp, (int64_t)K}, DType::FP32);
    CUBLAS_CHECK(cublasGemmEx(cublas(),
        CUBLAS_OP_N, CUBLAS_OP_T, K, rp, N, &alpha_f,
        fp32_w.data, CUDA_R_32F, K, Y.data, CUDA_R_32F, rp, &beta_f,
        B.data, CUDA_R_32F, K, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));

    // 5. C = B @ B^T: [rp, rp] on GPU, then download for tiny eigendecomposition
    // cuBLAS: C[rp,rp] = B_cm^T[rp,K] @ B_cm[K,rp]
    Tensor C_gpu = Tensor::alloc({(int64_t)rp, (int64_t)rp}, DType::FP32);
    CUBLAS_CHECK(cublasGemmEx(cublas(),
        CUBLAS_OP_T, CUBLAS_OP_N, rp, rp, K, &alpha_f,
        B.data, CUDA_R_32F, K, B.data, CUDA_R_32F, K, &beta_f,
        C_gpu.data, CUDA_R_32F, rp, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
    CUDA_CHECK(cudaDeviceSynchronize());

    // Download C (tiny: rp*rp*4 bytes, e.g. 138*138*4 = 76KB)
    std::vector<float> C_host(rp * rp);
    C_gpu.to_host(C_host.data());
    C_gpu.free_data();

    // CPU: Jacobi eigendecomposition of C [rp, rp]
    std::vector<float> D(rp);
    cpu_jacobi_eig(C_host.data(), D.data(), rp);

    // Compute sigma_i = sqrt(D_i) for top r
    std::vector<float> sigma(r);
    for (int i = 0; i < r; i++)
        sigma[i] = (D[i] > 0) ? sqrtf(D[i]) : 0.0f;

    // Prepare scaled eigenvector matrices on CPU (tiny: rp*r)
    // U_inv_sigma[rp, r]: column j = C_host[:, j] * (1/sigma[j])  (for V recovery)
    // U_sigma[rp, r]:     column j = C_host[:, j] * sigma[j]       (for L_up recovery)
    std::vector<float> U_inv_sigma(rp * r);
    std::vector<float> U_sigma(rp * r);
    for (int j = 0; j < r; j++) {
        float s = sigma[j];
        float inv_s = (s > 1e-12f) ? (1.0f / s) : 0.0f;
        for (int m = 0; m < rp; m++) {
            U_inv_sigma[m * r + j] = C_host[m * rp + j] * inv_s;
            U_sigma[m * r + j] = C_host[m * rp + j] * s;
        }
    }

    // Upload scaled eigenvectors to GPU (tiny: rp*r*4 bytes each)
    Tensor U_inv_sigma_gpu = Tensor::alloc({(int64_t)rp, (int64_t)r}, DType::FP32);
    Tensor U_sigma_gpu = Tensor::alloc({(int64_t)rp, (int64_t)r}, DType::FP32);
    U_inv_sigma_gpu.from_host(U_inv_sigma.data());
    U_sigma_gpu.from_host(U_sigma.data());

    // 6. V (L_down) [K, r] = B^T @ U_inv_sigma on GPU
    // Row-major result [K,r] = col-major [r,K] lda=r
    // cuBLAS: C[r,K] = U_inv_sigma_cm[r,rp] @ B_cm^T[rp,K]
    //   U_inv_sigma_rm [rp,r] = U_inv_sigma_cm [r,rp] with lda=r → OP_N = [r,rp]...
    //   Actually: U_inv_sigma_cm^T = U_inv_sigma_rm. So OP_N on U_inv_sigma_cm = U_inv_sigma_rm^T = [r,rp].
    //   B_cm [K,rp] lda=K → OP_T = [rp,K].
    //   C[r,K] = [r,rp] @ [rp,K] = OK
    Tensor L_down_gpu = Tensor::alloc({(int64_t)K, (int64_t)r}, DType::FP32);
    CUBLAS_CHECK(cublasGemmEx(cublas(),
        CUBLAS_OP_N, CUBLAS_OP_T, r, K, rp, &alpha_f,
        U_inv_sigma_gpu.data, CUDA_R_32F, r, B.data, CUDA_R_32F, K, &beta_f,
        L_down_gpu.data, CUDA_R_32F, r, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
    U_inv_sigma_gpu.free_data();
    B.free_data();

    // 7. L_up [N, r] = Q @ U_sigma on GPU
    // Row-major [N,r] = col-major [r,N] lda=r
    // cuBLAS: C[r,N] = U_sigma_cm[r,rp] @ Q_cm[rp,N]
    //   U_sigma_cm^T = U_sigma_rm → OP_N on U_sigma_cm = [r,rp]
    //   Q_cm [rp,N] lda=rp → OP_N = [rp,N]
    Tensor L_up_gpu = Tensor::alloc({(int64_t)N, (int64_t)r}, DType::FP32);
    CUBLAS_CHECK(cublasGemmEx(cublas(),
        CUBLAS_OP_N, CUBLAS_OP_N, r, N, rp, &alpha_f,
        U_sigma_gpu.data, CUDA_R_32F, r, Y.data, CUDA_R_32F, rp, &beta_f,
        L_up_gpu.data, CUDA_R_32F, r, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
    U_sigma_gpu.free_data();
    Y.free_data(); // Q no longer needed

    // 8. Residual R = W - L_up @ L_down^T on GPU
    // fp32_w[N,K] += -1 * L_up[N,r] @ L_down^T[r,K]
    // cuBLAS: C[K,N] += -1 * L_down_cm^T[K,r] @ L_up_cm[r,N]
    CUBLAS_CHECK(cublasGemmEx(cublas(),
        CUBLAS_OP_T, CUBLAS_OP_N, K, N, r, &neg_one,
        L_down_gpu.data, CUDA_R_32F, r, L_up_gpu.data, CUDA_R_32F, r, &one,
        fp32_w.data, CUDA_R_32F, K, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
    CUDA_CHECK(cudaDeviceSynchronize());

    // fp32_w now contains the residual R[N, K]

    // 7a. Apply Hadamard rotation to residual columns (smooths outliers for INT4)
    int hbs = hadamard_block_size(K);
    if (hbs > 1) {
        fwht_inplace((float*)fp32_w.data, N, K, hbs);
    }

    // 7b. INT4 per-group quantize R_rotated
    Tensor qw_data = Tensor::alloc({(int64_t)N, (int64_t)(K / 2)}, DType::UINT8);
    int num_groups = K / gs;
    Tensor qw_scales = Tensor::alloc({(int64_t)N, (int64_t)num_groups}, DType::FP32);

    quantize_int4_per_group((float*)fp32_w.data, (uint8_t*)qw_data.data,
                             (float*)qw_scales.data, N, K, gs);
    CUDA_CHECK(cudaDeviceSynchronize());
    fp32_w.free_data();

    // 8. Convert L_up, L_down to BF16
    Tensor svd_up_bf16 = Tensor::alloc({(int64_t)N, (int64_t)r}, DType::BF16);
    Tensor svd_down_bf16 = Tensor::alloc({(int64_t)K, (int64_t)r}, DType::BF16);
    fp32_to_bf16((float*)L_up_gpu.data, (__nv_bfloat16*)svd_up_bf16.data, (int64_t)N * r);
    fp32_to_bf16((float*)L_down_gpu.data, (__nv_bfloat16*)svd_down_bf16.data, (int64_t)K * r);
    CUDA_CHECK(cudaDeviceSynchronize());
    L_up_gpu.free_data();
    L_down_gpu.free_data();

    // 9. Package
    QuantizedWeight qw;
    qw.mode = QuantMode::INT4_SVD;
    qw.qweight = std::move(qw_data);
    qw.scales4 = std::move(qw_scales);
    qw.svd_up = std::move(svd_up_bf16);
    qw.svd_down = std::move(svd_down_bf16);
    qw.group_size = gs;
    qw.svd_rank = r;
    qw.had_block_size = hbs;
    return qw;
}

// ================================================================
// GPTQ CALIBRATION READER
// ================================================================

bool CalibrationReader::load(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) return false;

    uint32_t magic, file_version, num;
    if (fread(&magic, 4, 1, f) != 1 || magic != 0x51545047) {
        fprintf(stderr, "CalibrationReader: bad magic\n");
        fclose(f); return false;
    }
    if (fread(&file_version, 4, 1, f) != 1 || (file_version != 1 && file_version != 2)) {
        fprintf(stderr, "CalibrationReader: unsupported version %u\n", file_version);
        fclose(f); return false;
    }
    this->version = (int)file_version;
    if (fread(&num, 4, 1, f) != 1) { fclose(f); return false; }

    entries.resize(num);
    for (uint32_t i = 0; i < num; i++) {
        uint32_t name_len;
        if (fread(&name_len, 4, 1, f) != 1) { fclose(f); return false; }
        entries[i].name.resize(name_len);
        if (fread(&entries[i].name[0], 1, name_len, f) != name_len) { fclose(f); return false; }

        uint32_t m, k, hbs;
        if (fread(&m, 4, 1, f) != 1) { fclose(f); return false; }
        if (fread(&k, 4, 1, f) != 1) { fclose(f); return false; }
        if (fread(&hbs, 4, 1, f) != 1) { fclose(f); return false; }
        entries[i].M = (int)m;
        entries[i].K = (int)k;
        entries[i].had_block_size = (int)hbs;

        int64_t n = (int64_t)m * k;
        entries[i].data.resize(n);
        if (fread(entries[i].data.data(), sizeof(__nv_bfloat16), n, f) != (size_t)n) {
            fclose(f); return false;
        }
    }
    fclose(f);
    fprintf(stderr, "CalibrationReader: loaded %u entries (v%d%s) from %s\n", num, this->version,
            this->version == 2 ? ", raw" : ", FWHT-rotated", path);
    return true;
}

float* CalibrationReader::get_activation_gpu(const std::string& name, int& out_M, int& out_K) const {
    // Collect all matching entries
    std::vector<const Entry*> matches;
    for (auto& e : entries) {
        if (e.name == name) matches.push_back(&e);
    }
    if (matches.empty()) { out_M = 0; out_K = 0; return nullptr; }

    int K = matches[0]->K;
    int total_M = 0;
    for (auto* e : matches) {
        assert(e->K == K);
        total_M += e->M;
    }

    // Concatenate BF16 on CPU, convert to FP32
    std::vector<float> fp32((int64_t)total_M * K);
    int64_t offset = 0;
    for (auto* e : matches) {
        int64_t n = (int64_t)e->M * K;
        for (int64_t j = 0; j < n; j++)
            fp32[offset + j] = __bfloat162float(e->data[j]);
        offset += n;
    }

    // Upload to GPU
    float* gpu;
    CUDA_CHECK(cudaMalloc(&gpu, (int64_t)total_M * K * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(gpu, fp32.data(), (int64_t)total_M * K * sizeof(float), cudaMemcpyHostToDevice));

    out_M = total_M;
    out_K = K;
    return gpu;
}

// ================================================================
// GPTQ NF4 QUANTIZATION
// ================================================================

// Host-side NF4 grid (same as device __constant__)
static const float NF4_GRID_HOST[16] = {
    -1.0f, -0.6962f, -0.5251f, -0.3949f, -0.2844f, -0.1848f, -0.0911f, 0.0f,
     0.0796f,  0.1609f,  0.2461f,  0.3379f,  0.4407f,  0.5626f,  0.7230f, 1.0f
};

// GPTQ block kernel for symmetric linear INT4: process one group of columns with error feedback.
// Same structure as NF4 kernel but uses round-to-nearest instead of grid search.
// Scale = max/7, q = round(x*7/max)+8, clamp [0,15]
__global__ void gptq_linear_int4_block_kernel(
    float* __restrict__ R,
    const float* __restrict__ H_inv,
    uint8_t* __restrict__ qweight,
    float* __restrict__ scales,
    float* __restrict__ E_out,
    int N, int K, int block_start, int block_size, int gs)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N) return;

    int num_groups = K / gs;

    // Process columns sequentially with error feedback
    // Scale recomputed at each group boundary
    float scale = 0.0f, inv_scale = 0.0f;
    for (int j = 0; j < block_size; j++) {
        int col = block_start + j;

        // At group boundary: compute scale for this group
        if (j % gs == 0) {
            int group_idx = col / gs;
            int group_end = min(gs, block_size - j);
            float max_abs = 0.0f;
            for (int g = 0; g < group_end; g++) {
                float v = fabsf(R[(int64_t)row * K + col + g]);
                if (v > max_abs) max_abs = v;
            }
            scale = max_abs / 7.0f;
            inv_scale = (max_abs > 0.0f) ? (7.0f / max_abs) : 0.0f;
            scales[(int64_t)row * num_groups + group_idx] = scale;
        }

        float w = R[(int64_t)row * K + col];

        // Round to nearest integer, clamp to [-7, 7]
        int q_signed = (int)roundf(w * inv_scale);
        q_signed = max(-7, min(7, q_signed));
        int q_unsigned = q_signed + 8;

        // Dequantized value
        float w_hat = (float)q_signed * scale;
        float error = w - w_hat;

        // Pack quantized value
        int byte_idx = col / 2;
        if (col % 2 == 0) {
            qweight[(int64_t)row * (K / 2) + byte_idx] =
                (qweight[(int64_t)row * (K / 2) + byte_idx] & 0xF0) | (uint8_t)q_unsigned;
        } else {
            qweight[(int64_t)row * (K / 2) + byte_idx] =
                (qweight[(int64_t)row * (K / 2) + byte_idx] & 0x0F) | ((uint8_t)q_unsigned << 4);
        }

        // GPTQ error propagation
        float h_jj = H_inv[(int64_t)col * K + col];
        float err_scale = (h_jj > 1e-12f) ? (error / h_jj) : 0.0f;
        E_out[(int64_t)row * block_size + j] = err_scale;

        for (int k = j + 1; k < block_size; k++) {
            int col_k = block_start + k;
            R[(int64_t)row * K + col_k] -= err_scale * H_inv[(int64_t)col * K + col_k];
        }
    }
}

// GPTQ block kernel: process one block of columns with error feedback.
// One thread per row of R. Sequentially processes block_size columns.
// R: [N, K] residual matrix (row-major) — modified in place
// H_inv: [K, K] inverse Hessian (row-major)
// qweight: [N, K/2] packed output
// scales: [N, num_groups] per-group scales
// block_start: first column of this block
// block_size: number of columns in this block (GPTQ error propagation window)
// gs: group_size (scale granularity, may differ from block_size)
// E_out: [N, block_size] error matrix output for cross-block update
__global__ void gptq_nf4_block_kernel(
    float* __restrict__ R,
    const float* __restrict__ H_inv,
    uint8_t* __restrict__ qweight,
    float* __restrict__ scales,
    float* __restrict__ E_out,
    int N, int K, int block_start, int block_size, int gs)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N) return;

    int num_groups = K / gs;

    // Process columns sequentially with error feedback
    // Scale is recomputed at each group boundary within the block
    float scale = 0.0f, inv_scale = 0.0f;
    for (int j = 0; j < block_size; j++) {
        int col = block_start + j;

        // At group boundary: compute scale for this group
        if (j % gs == 0) {
            int group_idx = col / gs;
            int group_end = min(gs, block_size - j);
            float max_abs = 0.0f;
            for (int g = 0; g < group_end; g++) {
                float v = fabsf(R[(int64_t)row * K + col + g]);
                if (v > max_abs) max_abs = v;
            }
            scale = max_abs;
            inv_scale = (scale > 0.0f) ? (1.0f / scale) : 0.0f;
            scales[(int64_t)row * num_groups + group_idx] = scale;
        }

        float w = R[(int64_t)row * K + col];
        float normalized = w * inv_scale;

        // Find nearest NF4 grid point
        int best_q = 0;
        float best_dist = 1e9f;
        for (int q = 0; q < 16; q++) {
            float d = fabsf(normalized - NF4_GRID[q]);
            if (d < best_dist) { best_dist = d; best_q = q; }
        }

        // Dequantized value
        float w_hat = NF4_GRID[best_q] * scale;
        float error = w - w_hat;

        // Pack quantized value
        int byte_idx = col / 2;
        if (col % 2 == 0) {
            qweight[(int64_t)row * (K / 2) + byte_idx] =
                (qweight[(int64_t)row * (K / 2) + byte_idx] & 0xF0) | (uint8_t)best_q;
        } else {
            qweight[(int64_t)row * (K / 2) + byte_idx] =
                (qweight[(int64_t)row * (K / 2) + byte_idx] & 0x0F) | ((uint8_t)best_q << 4);
        }

        // GPTQ error propagation: E_j = error / H_inv[j,j]
        float h_jj = H_inv[(int64_t)col * K + col];
        float err_scale = (h_jj > 1e-12f) ? (error / h_jj) : 0.0f;

        // Store scaled error for cross-block GEMM update
        E_out[(int64_t)row * block_size + j] = err_scale;

        // Propagate within block: R[:, k] -= E_j * H_inv[j, k]
        for (int k = j + 1; k < block_size; k++) {
            int col_k = block_start + k;
            R[(int64_t)row * K + col_k] -= err_scale * H_inv[(int64_t)col * K + col_k];
        }
    }
}

// Host function: GPTQ quantize the residual R using calibration data.
// R: [N, K] FP32 on GPU (modified in place — consumed)
// x_rot: [M_total, K] FP32 on GPU (calibration activations, rotated or not)
// nf4_grid: true = NF4 lookup, false = symmetric linear INT4
// Returns packed qweight [N, K/2] and scales [N, K/gs] via output tensors.
static void gptq_quantize_nf4(
    float* R_gpu, int N, int K, int group_size,
    const float* x_rot_gpu, int M_total,
    Tensor& out_qweight, Tensor& out_scales,
    bool nf4_grid = true)
{
    int gs = group_size;
    int num_groups = K / gs;
    assert(K % gs == 0);
    assert(K % 2 == 0);

    // 1. Compute Hessian H = X_rot^T × X_rot [K, K] via cuBLAS SGEMM
    float* H_gpu;
    CUDA_CHECK(cudaMalloc(&H_gpu, (int64_t)K * K * sizeof(float)));
    {
        float alpha = 1.0f, beta = 0.0f;
        // H_col[K,K] = X_col^T[K,M] × X_col[M,K]
        // X row-major [M,K] = X col-major [K,M] lda=K
        // So: C[K,K] = OP_T(X_col)[K,M] × OP_N(X_col)[M,K]... no:
        // X_col has lda=K, shape [K,M]. X_col^T = [M,K].
        // H = X^T X in row-major = X_col X_col^T in col-major
        // cublasSgemm(N, T, K, K, M, ..., X, K, X, K, ..., H, K)
        CUBLAS_CHECK(cublasSgemm(cublas(), CUBLAS_OP_N, CUBLAS_OP_T,
            K, K, M_total, &alpha,
            x_rot_gpu, K, x_rot_gpu, K,
            &beta, H_gpu, K));
    }

    // 2. Dampen diagonal: H[i,i] += 0.01 * mean_nonzero_eigenvalue
    //    H has rank min(M,K), so divide trace by min(M,K) not K to get the
    //    mean non-zero eigenvalue. This prevents under-dampening when M << K.
    {
        std::vector<float> H_host((int64_t)K * K);
        CUDA_CHECK(cudaMemcpy(H_host.data(), H_gpu, (int64_t)K * K * sizeof(float), cudaMemcpyDeviceToHost));
        double diag_sum = 0.0;
        for (int i = 0; i < K; i++)
            diag_sum += H_host[(int64_t)i * K + i];
        float damp = (float)(0.01 * diag_sum / std::min(M_total, K));
        for (int i = 0; i < K; i++)
            H_host[(int64_t)i * K + i] += damp;
        CUDA_CHECK(cudaMemcpy(H_gpu, H_host.data(), (int64_t)K * K * sizeof(float), cudaMemcpyHostToDevice));
    }

    // 3. Cholesky factorization + inversion via cuSOLVER
    cusolverDnHandle_t solver;
    cusolverDnCreate(&solver);
    {
        int ws_potrf = 0, ws_potri = 0;
        cusolverDnSpotrf_bufferSize(solver, CUBLAS_FILL_MODE_UPPER, K, H_gpu, K, &ws_potrf);
        cusolverDnSpotri_bufferSize(solver, CUBLAS_FILL_MODE_UPPER, K, H_gpu, K, &ws_potri);
        int workspace_size = std::max(ws_potrf, ws_potri);
        float* d_work;
        CUDA_CHECK(cudaMalloc(&d_work, (int64_t)workspace_size * sizeof(float)));
        int* d_info;
        CUDA_CHECK(cudaMalloc(&d_info, sizeof(int)));

        // Cholesky: H = L L^T (upper triangular)
        cusolverDnSpotrf(solver, CUBLAS_FILL_MODE_UPPER, K, H_gpu, K, d_work, workspace_size, d_info);
        CUDA_CHECK(cudaDeviceSynchronize());

        int h_info;
        CUDA_CHECK(cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
        if (h_info != 0) {
            fprintf(stderr, "GPTQ: Cholesky failed (info=%d), falling back to damped identity\n", h_info);
            // Fall back: set H_inv = identity (no error redistribution)
            std::vector<float> identity((int64_t)K * K, 0.0f);
            for (int i = 0; i < K; i++) identity[(int64_t)i * K + i] = 1.0f;
            CUDA_CHECK(cudaMemcpy(H_gpu, identity.data(), (int64_t)K * K * sizeof(float), cudaMemcpyHostToDevice));
        } else {
            // Invert from Cholesky
            cusolverDnSpotri(solver, CUBLAS_FILL_MODE_UPPER, K, H_gpu, K, d_work, workspace_size, d_info);
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
            if (h_info != 0) {
                fprintf(stderr, "GPTQ: Cholesky inverse failed (info=%d)\n", h_info);
            }

            // Symmetrize: cuSOLVER only fills upper triangle (col-major: row <= col)
            // Upper triangle element (row=i, col=j) at index j*K+i where j >= i
            // Copy upper → lower: H(j,i) = H(i,j)
            std::vector<float> H_host((int64_t)K * K);
            CUDA_CHECK(cudaMemcpy(H_host.data(), H_gpu, (int64_t)K * K * sizeof(float), cudaMemcpyDeviceToHost));
            for (int i = 0; i < K; i++)
                for (int j = i + 1; j < K; j++)
                    H_host[(int64_t)i * K + j] = H_host[(int64_t)j * K + i];
            CUDA_CHECK(cudaMemcpy(H_gpu, H_host.data(), (int64_t)K * K * sizeof(float), cudaMemcpyHostToDevice));
        }

        CUDA_CHECK(cudaFree(d_work));
        CUDA_CHECK(cudaFree(d_info));
    }
    cusolverDnDestroy(solver);

    // H_gpu now contains H^{-1} [K, K]

    // 4. Allocate output tensors
    out_qweight = Tensor::alloc({(int64_t)N, (int64_t)(K / 2)}, DType::UINT8);
    out_qweight.zero();
    out_scales = Tensor::alloc({(int64_t)N, (int64_t)num_groups}, DType::FP32);

    // GPTQ block size: controls error propagation window (independent of group_size)
    // Larger blocks = better error redistribution but more compute per kernel launch
    int gptq_block_size = 128;
    // Ensure block size is multiple of group size and doesn't exceed K
    if (gptq_block_size < gs) gptq_block_size = gs;
    while (gptq_block_size % gs != 0) gptq_block_size += gs;
    if (gptq_block_size > K) gptq_block_size = K;

    // Error buffer for cross-block update
    float* E_gpu;
    CUDA_CHECK(cudaMalloc(&E_gpu, (int64_t)N * gptq_block_size * sizeof(float)));

    // 5. Process column blocks
    for (int block_start = 0; block_start < K; block_start += gptq_block_size) {
        int current_bs = std::min(gptq_block_size, K - block_start);

        // Launch GPTQ block kernel: one thread per row
        int threads = 256;
        int blocks = (N + threads - 1) / threads;
        if (nf4_grid) {
            gptq_nf4_block_kernel<<<blocks, threads>>>(
                R_gpu, H_gpu,
                (uint8_t*)out_qweight.data, (float*)out_scales.data,
                E_gpu,
                N, K, block_start, current_bs, gs);
        } else {
            gptq_linear_int4_block_kernel<<<blocks, threads>>>(
                R_gpu, H_gpu,
                (uint8_t*)out_qweight.data, (float*)out_scales.data,
                E_gpu,
                N, K, block_start, current_bs, gs);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        // Cross-block update: R[:, remaining] -= E × H_inv[block, remaining]
        int remaining = K - (block_start + current_bs);
        if (remaining > 0) {
            float alpha = -1.0f, beta = 1.0f;
            // E: [N, current_bs] row-major → col-major [current_bs, N] lda=current_bs
            // H_sub: rows [block_start..block_start+current_bs), cols [block_start+current_bs..K)
            //   shape [current_bs, remaining], row-major → col-major [remaining, current_bs] lda=K
            // R_sub: [N, remaining] starting at col block_start+current_bs, stride K
            CUBLAS_CHECK(cublasSgemm(cublas(),
                CUBLAS_OP_N, CUBLAS_OP_N,
                remaining, N, current_bs,
                &alpha,
                H_gpu + (int64_t)block_start * K + (block_start + current_bs), K,
                E_gpu, current_bs,
                &beta,
                R_gpu + (block_start + current_bs), K));
            CUDA_CHECK(cudaDeviceSynchronize());
        }
    }

    CUDA_CHECK(cudaFree(E_gpu));
    CUDA_CHECK(cudaFree(H_gpu));
}

// GPTQ-guided INT4+SVD quantization
QuantizedWeight quantize_weight_tensor_int4_gptq(const Tensor& bf16_weight, int rank, int group_size,
                                                   const float* x_rot_gpu, int M_total, int x_K,
                                                   bool no_hadamard, bool nf4_grid, bool error_svd) {
    assert(bf16_weight.dtype == DType::BF16);
    assert(bf16_weight.ndim == 2);

    int N = (int)bf16_weight.shape[0];
    int K = (int)bf16_weight.shape[1];

    // Adjust group_size for small K
    int gs = std::min(group_size, K);
    while (K % gs != 0 && gs > 2) gs /= 2;
    assert(K % gs == 0);
    assert(K % 2 == 0);

    // Clamp rank
    int r = std::min(rank, std::min(N, K) - 1);
    int p = 10;
    int rp = r + p;
    if (rp > std::min(N, K)) rp = std::min(N, K);
    if (r > rp) r = rp;

    // Convert BF16 -> FP32
    Tensor fp32_w = Tensor::alloc({(int64_t)N, (int64_t)K}, DType::FP32);
    bf16_to_fp32((__nv_bfloat16*)bf16_weight.data, (float*)fp32_w.data, (int64_t)N * K);

    // For error SVD: compute preliminary quantization error and SVD that instead of the weight.
    // This is the SVDQuant approach: SVD captures what INT4 struggles with.
    Tensor svd_target; // matrix to apply randomized SVD to
    if (error_svd) {
        // 1. Apply Hadamard rotation to weight (if using Hadamard)
        Tensor w_rot = Tensor::alloc({(int64_t)N, (int64_t)K}, DType::FP32);
        CUDA_CHECK(cudaMemcpy(w_rot.data, fp32_w.data, (int64_t)N * K * sizeof(float), cudaMemcpyDeviceToDevice));
        int hbs_prelim = no_hadamard ? 1 : hadamard_block_size(K);
        if (hbs_prelim > 1)
            fwht_inplace((float*)w_rot.data, N, K, hbs_prelim);

        // 2. Naive NF4 per-group quantization of rotated weight (no GPTQ)
        Tensor prelim_qw = Tensor::alloc({(int64_t)N, (int64_t)(K / 2)}, DType::UINT8);
        Tensor prelim_sc = Tensor::alloc({(int64_t)N, (int64_t)(K / gs)}, DType::FP32);
        quantize_int4_per_group((float*)w_rot.data, (uint8_t*)prelim_qw.data,
                                 (float*)prelim_sc.data, N, K, gs);
        CUDA_CHECK(cudaDeviceSynchronize());

        // 3. Dequantize back to FP32 (in rotated domain) via BF16 intermediate
        Tensor deq_bf16 = Tensor::alloc({(int64_t)N, (int64_t)K}, DType::BF16);
        dequantize_int4_to_bf16((uint8_t*)prelim_qw.data, (float*)prelim_sc.data,
                                 (__nv_bfloat16*)deq_bf16.data, N, K, gs, 0, true /* NF4 */);
        prelim_qw.free_data();
        prelim_sc.free_data();

        // Convert dequantized BF16 → FP32
        Tensor deq_fp32 = Tensor::alloc({(int64_t)N, (int64_t)K}, DType::FP32);
        bf16_to_fp32((__nv_bfloat16*)deq_bf16.data, (float*)deq_fp32.data, (int64_t)N * K);
        deq_bf16.free_data();

        // 4. Error in rotated domain: E_rot = W_rot - W_hat_rot
        // Then un-rotate: E = FWHT(E_rot) to get error in original domain
        // E_rot = W_rot - deq_fp32, then FWHT(E_rot) = FWHT(W_rot) - FWHT(deq_fp32)
        // Since FWHT(W_rot) = W (self-inverse), E = W - FWHT(deq_fp32)
        // Compute in-place: w_rot = w_rot - deq_fp32 (= E_rot), then FWHT(w_rot) (= E)
        {
            float neg = -1.0f, one_f = 1.0f;
            // w_rot -= deq_fp32
            CUBLAS_CHECK(cublasSaxpy(cublas(), N * K, &neg, (float*)deq_fp32.data, 1, (float*)w_rot.data, 1));
        }
        deq_fp32.free_data();

        // Un-rotate error back to original domain
        if (hbs_prelim > 1)
            fwht_inplace((float*)w_rot.data, N, K, hbs_prelim);
        CUDA_CHECK(cudaDeviceSynchronize());

        svd_target = std::move(w_rot); // SVD will operate on the quantization error
        fprintf(stderr, "    [error SVD: SVD of quantization error instead of weight]\n");
    }

    // Matrix to SVD: either the weight (standard) or the quantization error (error_svd)
    float* svd_matrix = error_svd ? (float*)svd_target.data : (float*)fp32_w.data;

    // === Randomized SVD with power iterations ===
    float alpha_f = 1.0f, beta_f = 0.0f, neg_one = -1.0f, one = 1.0f;
    const int n_power_iter = 0;  // Power iterations don't improve quality (tested: 16.06 dB with or without)

    // Modified Gram-Schmidt QR helper (orthogonalizes columns in-place)
    // Matrix is [rows × cols] row-major (cols × rows col-major), stride = cols
    auto gram_schmidt_qr = [&](float* Q_data, int rows, int cols) {
        Tensor dots_buf = Tensor::alloc({(int64_t)cols}, DType::FP32);
        float zero_f = 0.0f, neg_one_f = -1.0f, one_f = 1.0f;
        for (int j = 0; j < cols; j++) {
            float norm_sq;
            CUBLAS_CHECK(cublasSdot(cublas(), rows, Q_data + j, cols, Q_data + j, cols, &norm_sq));
            float inv_norm = (norm_sq > 1e-24f) ? (1.0f / sqrtf(norm_sq)) : 0.0f;
            CUBLAS_CHECK(cublasSscal(cublas(), rows, &inv_norm, Q_data + j, cols));
            if (j + 1 < cols) {
                int rem = cols - j - 1;
                CUBLAS_CHECK(cublasSgemv(cublas(), CUBLAS_OP_N,
                    rem, rows, &one_f, Q_data + (j + 1), cols, Q_data + j, cols,
                    &zero_f, (float*)dots_buf.data, 1));
                CUBLAS_CHECK(cublasSger(cublas(), rem, rows, &neg_one_f,
                    (float*)dots_buf.data, 1, Q_data + j, cols, Q_data + (j + 1), cols));
            }
        }
        dots_buf.free_data();
    };

    // 1. Random Omega
    std::vector<float> omega_host((int64_t)K * rp);
    srand(42);
    for (auto& v : omega_host) v = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
    Tensor omega = Tensor::alloc({(int64_t)K, (int64_t)rp}, DType::FP32);
    omega.from_host(omega_host.data());

    // 2. Y = M @ Omega  (M = weight or quantization error)
    Tensor Y = Tensor::alloc({(int64_t)N, (int64_t)rp}, DType::FP32);
    CUBLAS_CHECK(cublasGemmEx(cublas(),
        CUBLAS_OP_N, CUBLAS_OP_N, rp, N, K, &alpha_f,
        omega.data, CUDA_R_32F, rp, svd_matrix, CUDA_R_32F, K, &beta_f,
        Y.data, CUDA_R_32F, rp, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
    omega.free_data();

    // Power iterations: Y ← M @ (M^T @ QR(Y)), repeated n_power_iter times
    // This computes the range of (M M^T)^q M, amplifying singular value gaps
    for (int pi = 0; pi < n_power_iter; pi++) {
        // QR(Y) in-place
        gram_schmidt_qr((float*)Y.data, N, rp);

        // Z = M^T @ Y  [K × rp]
        Tensor Z = Tensor::alloc({(int64_t)K, (int64_t)rp}, DType::FP32);
        CUBLAS_CHECK(cublasGemmEx(cublas(),
            CUBLAS_OP_N, CUBLAS_OP_T, rp, K, N, &alpha_f,
            Y.data, CUDA_R_32F, rp, svd_matrix, CUDA_R_32F, K, &beta_f,
            Z.data, CUDA_R_32F, rp, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));

        // QR(Z) in-place
        gram_schmidt_qr((float*)Z.data, K, rp);

        // Y = M @ Z  [N × rp]
        CUBLAS_CHECK(cublasGemmEx(cublas(),
            CUBLAS_OP_N, CUBLAS_OP_N, rp, N, K, &alpha_f,
            Z.data, CUDA_R_32F, rp, svd_matrix, CUDA_R_32F, K, &beta_f,
            Y.data, CUDA_R_32F, rp, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
        Z.free_data();
    }

    // 3. Final QR(Y)
    gram_schmidt_qr((float*)Y.data, N, rp);

    // 4. B = Q^T @ M
    Tensor B = Tensor::alloc({(int64_t)rp, (int64_t)K}, DType::FP32);
    CUBLAS_CHECK(cublasGemmEx(cublas(),
        CUBLAS_OP_N, CUBLAS_OP_T, K, rp, N, &alpha_f,
        svd_matrix, CUDA_R_32F, K, Y.data, CUDA_R_32F, rp, &beta_f,
        B.data, CUDA_R_32F, K, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));

    // 5. C = B @ B^T
    Tensor C_gpu = Tensor::alloc({(int64_t)rp, (int64_t)rp}, DType::FP32);
    CUBLAS_CHECK(cublasGemmEx(cublas(),
        CUBLAS_OP_T, CUBLAS_OP_N, rp, rp, K, &alpha_f,
        B.data, CUDA_R_32F, K, B.data, CUDA_R_32F, K, &beta_f,
        C_gpu.data, CUDA_R_32F, rp, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> C_host(rp * rp);
    C_gpu.to_host(C_host.data());
    C_gpu.free_data();

    // CPU Jacobi eigendecomposition
    std::vector<float> D(rp);
    cpu_jacobi_eig(C_host.data(), D.data(), rp);

    std::vector<float> sigma(r);
    for (int i = 0; i < r; i++)
        sigma[i] = (D[i] > 0) ? sqrtf(D[i]) : 0.0f;

    std::vector<float> U_inv_sigma(rp * r);
    std::vector<float> U_sigma(rp * r);
    for (int j = 0; j < r; j++) {
        float s = sigma[j];
        float inv_s = (s > 1e-12f) ? (1.0f / s) : 0.0f;
        for (int m = 0; m < rp; m++) {
            U_inv_sigma[m * r + j] = C_host[m * rp + j] * inv_s;
            U_sigma[m * r + j] = C_host[m * rp + j] * s;
        }
    }

    Tensor U_inv_sigma_gpu = Tensor::alloc({(int64_t)rp, (int64_t)r}, DType::FP32);
    Tensor U_sigma_gpu = Tensor::alloc({(int64_t)rp, (int64_t)r}, DType::FP32);
    U_inv_sigma_gpu.from_host(U_inv_sigma.data());
    U_sigma_gpu.from_host(U_sigma.data());

    // 6. L_down [K, r]
    Tensor L_down_gpu = Tensor::alloc({(int64_t)K, (int64_t)r}, DType::FP32);
    CUBLAS_CHECK(cublasGemmEx(cublas(),
        CUBLAS_OP_N, CUBLAS_OP_T, r, K, rp, &alpha_f,
        U_inv_sigma_gpu.data, CUDA_R_32F, r, B.data, CUDA_R_32F, K, &beta_f,
        L_down_gpu.data, CUDA_R_32F, r, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
    U_inv_sigma_gpu.free_data();
    B.free_data();

    // 7. L_up [N, r]
    Tensor L_up_gpu = Tensor::alloc({(int64_t)N, (int64_t)r}, DType::FP32);
    CUBLAS_CHECK(cublasGemmEx(cublas(),
        CUBLAS_OP_N, CUBLAS_OP_N, r, N, rp, &alpha_f,
        U_sigma_gpu.data, CUDA_R_32F, r, Y.data, CUDA_R_32F, rp, &beta_f,
        L_up_gpu.data, CUDA_R_32F, r, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
    U_sigma_gpu.free_data();
    Y.free_data();

    // 8. Residual R = W - L_up @ L_down^T (always from original weight, not error)
    CUBLAS_CHECK(cublasGemmEx(cublas(),
        CUBLAS_OP_T, CUBLAS_OP_N, K, N, r, &neg_one,
        L_down_gpu.data, CUDA_R_32F, r, L_up_gpu.data, CUDA_R_32F, r, &one,
        fp32_w.data, CUDA_R_32F, K, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
    CUDA_CHECK(cudaDeviceSynchronize());

    // Free error SVD target if it was allocated
    if (error_svd) svd_target.free_data();

    // 9. Hadamard rotation of residual (skipped if no_hadamard)
    int hbs = no_hadamard ? 1 : hadamard_block_size(K);
    if (hbs > 1) {
        fwht_inplace((float*)fp32_w.data, N, K, hbs);
    }

    // 10. GPTQ quantization of residual
    // x_rot_gpu [M_total, K] already has the correct K (calibration activations)
    assert(x_K == K);

    Tensor qw_data, qw_scales;
    gptq_quantize_nf4((float*)fp32_w.data, N, K, gs,
                       x_rot_gpu, M_total,
                       qw_data, qw_scales, nf4_grid);
    fp32_w.free_data();

    // SVD refinement was tested but hurts quality (-0.85 dB).
    // The GPTQ-optimized residual becomes stale when SVD factors change.

    // 11. Convert L_up, L_down to BF16
    Tensor svd_up_bf16 = Tensor::alloc({(int64_t)N, (int64_t)r}, DType::BF16);
    Tensor svd_down_bf16 = Tensor::alloc({(int64_t)K, (int64_t)r}, DType::BF16);
    fp32_to_bf16((float*)L_up_gpu.data, (__nv_bfloat16*)svd_up_bf16.data, (int64_t)N * r);
    fp32_to_bf16((float*)L_down_gpu.data, (__nv_bfloat16*)svd_down_bf16.data, (int64_t)K * r);
    CUDA_CHECK(cudaDeviceSynchronize());
    L_up_gpu.free_data();
    L_down_gpu.free_data();

    // 12. Package
    QuantizedWeight qw;
    qw.mode = QuantMode::INT4_SVD;
    qw.qweight = std::move(qw_data);
    qw.scales4 = std::move(qw_scales);
    qw.svd_up = std::move(svd_up_bf16);
    qw.svd_down = std::move(svd_down_bf16);
    qw.group_size = gs;
    qw.svd_rank = r;
    qw.had_block_size = hbs;
    qw.nf4_grid = nf4_grid;
    return qw;
}

// ================================================================
// GPTQ+Smooth INT4+SVD quantization
// Takes RAW activations, computes smooth factors, applies to weight,
// then runs SVD + GPTQ on the smoothed weight.
// ================================================================

QuantizedWeight quantize_weight_tensor_int4_gptq_smooth(const Tensor& bf16_weight, int rank, int group_size,
                                                          const float* x_raw_gpu, int M_total, int x_K) {
    assert(bf16_weight.dtype == DType::BF16);
    assert(bf16_weight.ndim == 2);

    int N = (int)bf16_weight.shape[0];
    int K = (int)bf16_weight.shape[1];
    assert(x_K == K);

    // Adjust group_size for small K
    int gs = std::min(group_size, K);
    while (K % gs != 0 && gs > 2) gs /= 2;
    assert(K % gs == 0);
    assert(K % 2 == 0);

    // Clamp rank
    int r = std::min(rank, std::min(N, K) - 1);
    int p = 10;
    int rp = r + p;
    if (rp > std::min(N, K)) rp = std::min(N, K);
    if (r > rp) r = rp;

    // Convert BF16 weight -> FP32 on GPU
    Tensor fp32_w = Tensor::alloc({(int64_t)N, (int64_t)K}, DType::FP32);
    bf16_to_fp32((__nv_bfloat16*)bf16_weight.data, (float*)fp32_w.data, (int64_t)N * K);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Download weight and activations to CPU for smooth factor computation
    std::vector<float> w_host((int64_t)N * K);
    CUDA_CHECK(cudaMemcpy(w_host.data(), fp32_w.data, (int64_t)N * K * sizeof(float), cudaMemcpyDeviceToHost));

    std::vector<float> x_host((int64_t)M_total * K);
    CUDA_CHECK(cudaMemcpy(x_host.data(), x_raw_gpu, (int64_t)M_total * K * sizeof(float), cudaMemcpyDeviceToHost));

    // Compute smooth factors using activation-aware channel balancing.
    // Raw formula: s_raw[j] = (max|X_j|)^alpha / (max|W_j|)^(1-alpha)
    // Then normalize by geometric mean so factors are centered around 1.
    // This redistributes outlier energy between channels without changing overall scale.
    const float alpha = 0.5f;
    std::vector<float> smooth_host(K);
    {
        std::vector<float> s_raw(K);
        double log_sum = 0.0;
        for (int j = 0; j < K; j++) {
            float amax = 0.0f;
            for (int m = 0; m < M_total; m++) {
                float v = fabsf(x_host[(int64_t)m * K + j]);
                if (v > amax) amax = v;
            }
            float wmax = 0.0f;
            for (int n = 0; n < N; n++) {
                float v = fabsf(w_host[(int64_t)n * K + j]);
                if (v > wmax) wmax = v;
            }
            float s = powf(amax + 1e-12f, alpha) / powf(wmax + 1e-12f, 1.0f - alpha);
            s = fminf(fmaxf(s, 1e-5f), 1e5f);
            s_raw[j] = s;
            log_sum += logf(s);
        }
        // Normalize by geometric mean to center around 1
        float geo_mean = expf((float)(log_sum / K));
        float inv_geo = 1.0f / geo_mean;
        for (int j = 0; j < K; j++) {
            float s = s_raw[j] * inv_geo;
            // Clamp to prevent extreme factors
            s = fminf(fmaxf(s, 0.1f), 10.0f);
            smooth_host[j] = s;
        }
    }

    // Apply smooth to weight on CPU: W[n,j] *= s[j]
    for (int n = 0; n < N; n++) {
        for (int j = 0; j < K; j++) {
            w_host[(int64_t)n * K + j] *= smooth_host[j];
        }
    }

    // Re-upload smoothed weight to GPU
    CUDA_CHECK(cudaMemcpy(fp32_w.data, w_host.data(), (int64_t)N * K * sizeof(float), cudaMemcpyHostToDevice));
    w_host.clear();
    x_host.clear();

    // Upload smooth factors to GPU
    Tensor smooth_gpu = Tensor::alloc({(int64_t)K}, DType::FP32);
    smooth_gpu.from_host(smooth_host.data());

    // Apply smooth to activations on GPU: X_smooth = X / s
    float* x_smooth_gpu;
    CUDA_CHECK(cudaMalloc(&x_smooth_gpu, (int64_t)M_total * K * sizeof(float)));
    apply_smooth_div(x_raw_gpu, (const float*)smooth_gpu.data, x_smooth_gpu, M_total, K);
    CUDA_CHECK(cudaDeviceSynchronize());

    // === Randomized SVD on smoothed weight ===
    float alpha_f = 1.0f, beta_f = 0.0f, neg_one = -1.0f, one = 1.0f;

    // 1. Random Omega
    std::vector<float> omega_host((int64_t)K * rp);
    srand(42);
    for (auto& v : omega_host) v = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
    Tensor omega = Tensor::alloc({(int64_t)K, (int64_t)rp}, DType::FP32);
    omega.from_host(omega_host.data());

    // 2. Y = W_smooth @ Omega
    Tensor Y = Tensor::alloc({(int64_t)N, (int64_t)rp}, DType::FP32);
    CUBLAS_CHECK(cublasGemmEx(cublas(),
        CUBLAS_OP_N, CUBLAS_OP_N, rp, N, K, &alpha_f,
        omega.data, CUDA_R_32F, rp, fp32_w.data, CUDA_R_32F, K, &beta_f,
        Y.data, CUDA_R_32F, rp, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
    omega.free_data();

    // 3. Modified Gram-Schmidt QR
    {
        float* Q_data = (float*)Y.data;
        Tensor dots_buf = Tensor::alloc({(int64_t)rp}, DType::FP32);
        float zero_f = 0.0f, neg_one_f = -1.0f, one_f = 1.0f;
        for (int j = 0; j < rp; j++) {
            float norm_sq;
            CUBLAS_CHECK(cublasSdot(cublas(), N, Q_data + j, rp, Q_data + j, rp, &norm_sq));
            float inv_norm = (norm_sq > 1e-24f) ? (1.0f / sqrtf(norm_sq)) : 0.0f;
            CUBLAS_CHECK(cublasSscal(cublas(), N, &inv_norm, Q_data + j, rp));
            if (j + 1 < rp) {
                int rem = rp - j - 1;
                CUBLAS_CHECK(cublasSgemv(cublas(), CUBLAS_OP_N,
                    rem, N, &one_f, Q_data + (j + 1), rp, Q_data + j, rp,
                    &zero_f, (float*)dots_buf.data, 1));
                CUBLAS_CHECK(cublasSger(cublas(), rem, N, &neg_one_f,
                    (float*)dots_buf.data, 1, Q_data + j, rp, Q_data + (j + 1), rp));
            }
        }
        dots_buf.free_data();
    }

    // 4. B = Q^T @ W_smooth
    Tensor B = Tensor::alloc({(int64_t)rp, (int64_t)K}, DType::FP32);
    CUBLAS_CHECK(cublasGemmEx(cublas(),
        CUBLAS_OP_N, CUBLAS_OP_T, K, rp, N, &alpha_f,
        fp32_w.data, CUDA_R_32F, K, Y.data, CUDA_R_32F, rp, &beta_f,
        B.data, CUDA_R_32F, K, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));

    // 5. C = B @ B^T
    Tensor C_gpu = Tensor::alloc({(int64_t)rp, (int64_t)rp}, DType::FP32);
    CUBLAS_CHECK(cublasGemmEx(cublas(),
        CUBLAS_OP_T, CUBLAS_OP_N, rp, rp, K, &alpha_f,
        B.data, CUDA_R_32F, K, B.data, CUDA_R_32F, K, &beta_f,
        C_gpu.data, CUDA_R_32F, rp, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> C_host(rp * rp);
    C_gpu.to_host(C_host.data());
    C_gpu.free_data();

    std::vector<float> D(rp);
    cpu_jacobi_eig(C_host.data(), D.data(), rp);

    std::vector<float> sigma(r);
    for (int i = 0; i < r; i++)
        sigma[i] = (D[i] > 0) ? sqrtf(D[i]) : 0.0f;

    std::vector<float> U_inv_sigma(rp * r);
    std::vector<float> U_sigma(rp * r);
    for (int j = 0; j < r; j++) {
        float s = sigma[j];
        float inv_s = (s > 1e-12f) ? (1.0f / s) : 0.0f;
        for (int m = 0; m < rp; m++) {
            U_inv_sigma[m * r + j] = C_host[m * rp + j] * inv_s;
            U_sigma[m * r + j] = C_host[m * rp + j] * s;
        }
    }

    Tensor U_inv_sigma_gpu = Tensor::alloc({(int64_t)rp, (int64_t)r}, DType::FP32);
    Tensor U_sigma_gpu = Tensor::alloc({(int64_t)rp, (int64_t)r}, DType::FP32);
    U_inv_sigma_gpu.from_host(U_inv_sigma.data());
    U_sigma_gpu.from_host(U_sigma.data());

    // 6. L_down [K, r]
    Tensor L_down_gpu = Tensor::alloc({(int64_t)K, (int64_t)r}, DType::FP32);
    CUBLAS_CHECK(cublasGemmEx(cublas(),
        CUBLAS_OP_N, CUBLAS_OP_T, r, K, rp, &alpha_f,
        U_inv_sigma_gpu.data, CUDA_R_32F, r, B.data, CUDA_R_32F, K, &beta_f,
        L_down_gpu.data, CUDA_R_32F, r, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
    U_inv_sigma_gpu.free_data();
    B.free_data();

    // 7. L_up [N, r]
    Tensor L_up_gpu = Tensor::alloc({(int64_t)N, (int64_t)r}, DType::FP32);
    CUBLAS_CHECK(cublasGemmEx(cublas(),
        CUBLAS_OP_N, CUBLAS_OP_N, r, N, rp, &alpha_f,
        U_sigma_gpu.data, CUDA_R_32F, r, Y.data, CUDA_R_32F, rp, &beta_f,
        L_up_gpu.data, CUDA_R_32F, r, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
    U_sigma_gpu.free_data();
    Y.free_data();

    // 8. Residual R = W_smooth - L_up @ L_down^T
    CUBLAS_CHECK(cublasGemmEx(cublas(),
        CUBLAS_OP_T, CUBLAS_OP_N, K, N, r, &neg_one,
        L_down_gpu.data, CUDA_R_32F, r, L_up_gpu.data, CUDA_R_32F, r, &one,
        fp32_w.data, CUDA_R_32F, K, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
    CUDA_CHECK(cudaDeviceSynchronize());

    // 9. Hadamard rotation of residual (TODO: add no_hadamard support)
    int hbs = hadamard_block_size(K);
    if (hbs > 1) {
        fwht_inplace((float*)fp32_w.data, N, K, hbs);
    }

    // 10. Hadamard-rotate smoothed activations for GPTQ Hessian
    if (hbs > 1) {
        fwht_inplace(x_smooth_gpu, M_total, K, hbs);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // 11. GPTQ quantization of Hadamard-rotated residual
    Tensor qw_data, qw_scales;
    gptq_quantize_nf4((float*)fp32_w.data, N, K, gs,
                       x_smooth_gpu, M_total,
                       qw_data, qw_scales, true /* nf4 for smooth path */);
    fp32_w.free_data();
    CUDA_CHECK(cudaFree(x_smooth_gpu));

    // 12. Convert L_up, L_down to BF16
    Tensor svd_up_bf16 = Tensor::alloc({(int64_t)N, (int64_t)r}, DType::BF16);
    Tensor svd_down_bf16 = Tensor::alloc({(int64_t)K, (int64_t)r}, DType::BF16);
    fp32_to_bf16((float*)L_up_gpu.data, (__nv_bfloat16*)svd_up_bf16.data, (int64_t)N * r);
    fp32_to_bf16((float*)L_down_gpu.data, (__nv_bfloat16*)svd_down_bf16.data, (int64_t)K * r);
    CUDA_CHECK(cudaDeviceSynchronize());
    L_up_gpu.free_data();
    L_down_gpu.free_data();

    // 13. Package
    QuantizedWeight qw;
    qw.mode = QuantMode::INT4_SVD;
    qw.qweight = std::move(qw_data);
    qw.scales4 = std::move(qw_scales);
    qw.svd_up = std::move(svd_up_bf16);
    qw.svd_down = std::move(svd_down_bf16);
    qw.smooth = std::move(smooth_gpu);
    qw.group_size = gs;
    qw.svd_rank = r;
    qw.had_block_size = hbs;
    qw.nf4_grid = true;
    return qw;
}
