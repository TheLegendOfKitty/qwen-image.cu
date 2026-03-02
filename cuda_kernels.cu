#include "cuda_kernels.cuh"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <curand_kernel.h>
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
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));

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
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));

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
        CUBLAS_GEMM_DEFAULT));

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
        CUBLAS_GEMM_DEFAULT));

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
        CUBLAS_GEMM_DEFAULT));

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
        CUBLAS_GEMM_DEFAULT));

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

void linear_forward_fp32out(const Tensor& x, const Tensor& weight, const Tensor* bias, Tensor& out) {
    // x: [M, K] BF16, weight: [N, K] BF16, out: [M, N] FP32
    int M = (int)x.shape[0];
    int K = (int)x.shape[1];
    int N = (int)weight.shape[0];

    assert(x.dtype == DType::BF16);
    assert(weight.dtype == DType::BF16);
    assert(out.dtype == DType::FP32);

    float alpha = 1.0f, beta_val = 0.0f;

    CUBLAS_CHECK(cublasGemmEx(
        cublas(),
        CUBLAS_OP_T, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        weight.data, CUDA_R_16BF, K,
        x.data, CUDA_R_16BF, K,
        &beta_val,
        out.data, CUDA_R_32F, N,  // FP32 output
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT));

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
        CUBLAS_GEMM_DEFAULT));

    if (bias) {
        bias_add_fp32((float*)out.data, (__nv_bfloat16*)bias->data,
                      (float*)out.data, (int64_t)B * M, N);
    }
}

void linear_forward_fp32in_bf16w_fp32out(const Tensor& x, const Tensor& weight, const Tensor* bias,
                                         Tensor& out, Tensor& weight_scratch) {
    // x: [M, K] FP32, weight: [N, K] BF16, out: [M, N] FP32
    int M = (int)x.shape[0];
    int K = (int)x.shape[1];
    int N = (int)weight.shape[0];

    assert(x.dtype == DType::FP32);
    assert(weight.dtype == DType::BF16);
    assert(out.dtype == DType::FP32);
    assert(weight.shape[1] == K);
    assert(out.shape[0] == M && out.shape[1] == N);
    assert(weight_scratch.dtype == DType::FP32);
    assert(weight_scratch.numel() >= (int64_t)N * K);

    // Expand BF16 weights into FP32 scratch so GEMM can stay fully FP32.
    bf16_to_fp32((const __nv_bfloat16*)weight.data, (float*)weight_scratch.data, (int64_t)N * K);

    float alpha = 1.0f, beta_val = 0.0f;

    CUBLAS_CHECK(cublasGemmEx(
        cublas(),
        CUBLAS_OP_T, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        weight_scratch.data, CUDA_R_32F, K,
        x.data, CUDA_R_32F, K,
        &beta_val,
        out.data, CUDA_R_32F, N,
        CUBLAS_COMPUTE_32F_PEDANTIC,
        CUBLAS_GEMM_DEFAULT));

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
    float* scores;
    __half* q_fp16;
    __half* k_fp16;
    CUDA_CHECK(cudaMalloc(&scores, (size_t)BH * S * S * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&q_fp16, (size_t)BH * S * D * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&k_fp16, (size_t)BH * S * D * sizeof(__half)));

    fp32_to_fp16(q, q_fp16, (int64_t)BH * S * D, stream);
    fp32_to_fp16(k, k_fp16, (int64_t)BH * S * D, stream);

    float alpha = scale, beta_val = 0.0f;

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
        CUBLAS_COMPUTE_32F_PEDANTIC,
        CUBLAS_GEMM_DEFAULT));

    if (causal) apply_causal_mask(scores, BH, S, stream);

    // Softmax (FP32)
    softmax(scores, scores, BH * S, S, stream);

    // scores @ V: FP32 × FP32 -> FP32 output
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
        out, CUDA_R_32F, D, (long long)S * D,  // FP32 output
        BH,
        CUBLAS_COMPUTE_32F_PEDANTIC,
        CUBLAS_GEMM_DEFAULT));
    CUDA_CHECK(cudaFree(q_fp16));
    CUDA_CHECK(cudaFree(k_fp16));
    CUDA_CHECK(cudaFree(scores));
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
        CUBLAS_GEMM_DEFAULT));

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
        CUBLAS_GEMM_DEFAULT));

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
