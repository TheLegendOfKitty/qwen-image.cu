#pragma once
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <cassert>

// Error checking macros
#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

#define CUBLAS_CHECK(call) do { \
    cublasStatus_t status = (call); \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS error at %s:%d: %d\n", __FILE__, __LINE__, (int)status); \
        exit(1); \
    } \
} while(0)

#define CUSOLVER_CHECK(call) do { \
    cusolverStatus_t status = (call); \
    if (status != CUSOLVER_STATUS_SUCCESS) { \
        fprintf(stderr, "cuSOLVER error at %s:%d: %d\n", __FILE__, __LINE__, (int)status); \
        exit(1); \
    } \
} while(0)

enum class DType { BF16, FP32, INT8, INT32, UINT8 };

inline size_t dtype_size(DType dt) {
    switch (dt) {
        case DType::BF16: return 2;
        case DType::FP32: return 4;
        case DType::INT8: return 1;
        case DType::INT32: return 4;
        case DType::UINT8: return 1;
    }
    return 0;
}

inline const char* dtype_name(DType dt) {
    switch (dt) {
        case DType::BF16: return "BF16";
        case DType::FP32: return "FP32";
        case DType::INT8: return "INT8";
        case DType::INT32: return "INT32";
        case DType::UINT8: return "UINT8";
    }
    return "???";
}

struct Tensor {
    void* data = nullptr;
    int64_t shape[5] = {0, 0, 0, 0, 0};
    int ndim = 0;
    DType dtype = DType::BF16;
    bool owns_data = false;

    Tensor() = default;

    // Disable copy
    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;

    // Move
    Tensor(Tensor&& other) noexcept {
        data = other.data;
        memcpy(shape, other.shape, sizeof(shape));
        ndim = other.ndim;
        dtype = other.dtype;
        owns_data = other.owns_data;
        other.data = nullptr;
        other.owns_data = false;
    }

    Tensor& operator=(Tensor&& other) noexcept {
        if (this != &other) {
            free_data();
            data = other.data;
            memcpy(shape, other.shape, sizeof(shape));
            ndim = other.ndim;
            dtype = other.dtype;
            owns_data = other.owns_data;
            other.data = nullptr;
            other.owns_data = false;
        }
        return *this;
    }

    ~Tensor() { free_data(); }

    void free_data() {
        if (data && owns_data) {
            CUDA_CHECK(cudaFree(data));
        }
        data = nullptr;
        owns_data = false;
    }

    int64_t numel() const {
        if (ndim == 0) return 0;
        int64_t n = 1;
        for (int i = 0; i < ndim; i++) n *= shape[i];
        return n;
    }

    size_t nbytes() const {
        return (size_t)numel() * dtype_size(dtype);
    }

    // Allocate GPU memory
    static Tensor alloc(std::initializer_list<int64_t> dims, DType dt = DType::BF16) {
        Tensor t;
        t.ndim = (int)dims.size();
        assert(t.ndim <= 5);
        int i = 0;
        for (auto d : dims) t.shape[i++] = d;
        t.dtype = dt;
        t.owns_data = true;
        CUDA_CHECK(cudaMalloc(&t.data, t.nbytes()));
        return t;
    }

    static Tensor alloc(const std::vector<int64_t>& dims, DType dt = DType::BF16) {
        Tensor t;
        t.ndim = (int)dims.size();
        assert(t.ndim <= 5);
        for (int i = 0; i < t.ndim; i++) t.shape[i] = dims[i];
        t.dtype = dt;
        t.owns_data = true;
        CUDA_CHECK(cudaMalloc(&t.data, t.nbytes()));
        return t;
    }

    static Tensor alloc_like(const Tensor& other) {
        std::vector<int64_t> dims(other.shape, other.shape + other.ndim);
        return alloc(dims, other.dtype);
    }

    // Zero fill
    void zero() {
        CUDA_CHECK(cudaMemset(data, 0, nbytes()));
    }

    // Upload from host
    void from_host(const void* host_data, size_t bytes) {
        assert(bytes <= nbytes());
        CUDA_CHECK(cudaMemcpy(data, host_data, bytes, cudaMemcpyHostToDevice));
    }

    void from_host(const void* host_data) {
        from_host(host_data, nbytes());
    }

    // Download to host
    void to_host(void* host_data, size_t bytes) const {
        assert(bytes <= nbytes());
        CUDA_CHECK(cudaMemcpy(host_data, data, bytes, cudaMemcpyDeviceToHost));
    }

    void to_host(void* host_data) const {
        to_host(host_data, nbytes());
    }

    // Create a view (non-owning) with different shape
    Tensor view(std::initializer_list<int64_t> new_dims) const {
        Tensor t;
        t.data = data;
        t.ndim = (int)new_dims.size();
        int i = 0;
        for (auto d : new_dims) t.shape[i++] = d;
        t.dtype = dtype;
        t.owns_data = false;
        assert(t.numel() == numel());
        return t;
    }

    Tensor view(const std::vector<int64_t>& new_dims) const {
        Tensor t;
        t.data = data;
        t.ndim = (int)new_dims.size();
        for (int i = 0; i < t.ndim; i++) t.shape[i] = new_dims[i];
        t.dtype = dtype;
        t.owns_data = false;
        assert(t.numel() == numel());
        return t;
    }

    // Non-owning view of a slice along dimension 0
    Tensor slice(int64_t start, int64_t end) const {
        assert(ndim >= 1 && start >= 0 && end <= shape[0] && start < end);
        int64_t inner = 1;
        for (int i = 1; i < ndim; i++) inner *= shape[i];
        Tensor t;
        t.data = (char*)data + start * inner * dtype_size(dtype);
        t.ndim = ndim;
        t.shape[0] = end - start;
        for (int i = 1; i < ndim; i++) t.shape[i] = shape[i];
        t.dtype = dtype;
        t.owns_data = false;
        return t;
    }

    // Clone (deep copy)
    Tensor clone() const {
        std::vector<int64_t> dims(shape, shape + ndim);
        Tensor t = Tensor::alloc(dims, dtype);
        CUDA_CHECK(cudaMemcpy(t.data, data, nbytes(), cudaMemcpyDeviceToDevice));
        return t;
    }

    // Print shape
    std::string shape_str() const {
        std::string s = "[";
        for (int i = 0; i < ndim; i++) {
            if (i > 0) s += ", ";
            s += std::to_string(shape[i]);
        }
        s += "]";
        return s;
    }

    void print_info(const char* name = "") const {
        fprintf(stderr, "Tensor %s: %s %s, %zu bytes\n",
                name, shape_str().c_str(), dtype_name(dtype), nbytes());
    }
};

// Global cuBLAS handle
struct CublasHandle {
    cublasHandle_t handle = nullptr;

    void init() {
        if (!handle) {
            CUBLAS_CHECK(cublasCreate(&handle));
            // Use tensor cores
            CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));
        }
    }

    ~CublasHandle() {
        if (handle) cublasDestroy(handle);
    }

    operator cublasHandle_t() const { return handle; }
};

inline CublasHandle& cublas() {
    static CublasHandle h;
    h.init();
    return h;
}

// Linear layer: out = x @ weight^T + bias
// x: [M, K] (BF16), weight: [N, K] (BF16), bias: [N] (BF16 or nullptr), out: [M, N] (BF16)
void linear_forward(const Tensor& x, const Tensor& weight, const Tensor* bias, Tensor& out);

// Batched linear: x: [B, M, K], weight: [N, K], out: [B, M, N]
void linear_forward_batched(const Tensor& x, const Tensor& weight, const Tensor* bias, Tensor& out);
