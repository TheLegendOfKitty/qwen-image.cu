// Test program to verify our Philox RNG matches sd.cpp's PhiloxRNG exactly.
//
// Compares:
//   1. Our randn_fill_philox() from cuda_kernels.cu (runs on host, uploads to GPU, we download back)
//   2. A direct inline port of sd.cpp's PhiloxRNG::randn() for reference
//
// Build:
//   nvcc -O2 -o test_philox test_philox.cu cuda_kernels.cu -lcublas -std=c++17
// Run:
//   ./test_philox

#include <cstdio>
#include <cstdint>
#include <cmath>
#include <vector>
#include <cuda_bf16.h>
#include "cuda_kernels.cuh"

// ========================================================================
// Inline reimplementation of sd.cpp's PhiloxRNG::randn() for reference
// (copied logic directly from stable-diffusion.cpp/src/rng_philox.hpp)
// ========================================================================
static std::vector<float> sdcpp_philox_randn(uint64_t seed, uint32_t n) {
    const uint32_t M0 = 0xD2511F53u;
    const uint32_t M1 = 0xCD9E8D57u;
    const uint32_t W0 = 0x9E3779B9u;
    const uint32_t W1 = 0xBB67AE85u;
    const float two_pow32_inv = 2.3283064e-10f;
    const float two_pow32_inv_2pi = 2.3283064e-10f * 6.2831855f;

    uint32_t offset = 0; // first call

    // Build counter: 4 x N
    // counter[0][i] = offset, counter[1][i] = 0, counter[2][i] = i, counter[3][i] = 0
    std::vector<std::vector<uint32_t>> counter(4, std::vector<uint32_t>(n, 0));
    for (uint32_t i = 0; i < n; i++) {
        counter[0][i] = offset;
        counter[2][i] = i;
    }

    // Build key: 2 x N (all elements same seed)
    std::vector<std::vector<uint32_t>> key(2, std::vector<uint32_t>(n));
    for (uint32_t i = 0; i < n; i++) {
        key[0][i] = (uint32_t)(seed & 0xFFFFFFFF);
        key[1][i] = (uint32_t)(seed >> 32);
    }

    // philox4_32: 10 rounds
    int rounds = 10;
    for (int r = 0; r < rounds - 1; r++) {
        // philox4_round
        for (uint32_t i = 0; i < n; i++) {
            uint64_t v1 = (uint64_t)counter[0][i] * (uint64_t)M0;
            uint64_t v2 = (uint64_t)counter[2][i] * (uint64_t)M1;
            uint32_t v1_hi = (uint32_t)(v1 >> 32);
            uint32_t v1_lo = (uint32_t)(v1 & 0xFFFFFFFF);
            uint32_t v2_hi = (uint32_t)(v2 >> 32);
            uint32_t v2_lo = (uint32_t)(v2 & 0xFFFFFFFF);

            uint32_t new0 = v2_hi ^ counter[1][i] ^ key[0][i];
            uint32_t new1 = v2_lo;
            uint32_t new2 = v1_hi ^ counter[3][i] ^ key[1][i];
            uint32_t new3 = v1_lo;

            counter[0][i] = new0;
            counter[1][i] = new1;
            counter[2][i] = new2;
            counter[3][i] = new3;
        }
        // key bump
        for (uint32_t i = 0; i < n; i++) {
            key[0][i] += W0;
            key[1][i] += W1;
        }
    }
    // Final round (no key bump after)
    for (uint32_t i = 0; i < n; i++) {
        uint64_t v1 = (uint64_t)counter[0][i] * (uint64_t)M0;
        uint64_t v2 = (uint64_t)counter[2][i] * (uint64_t)M1;
        uint32_t v1_hi = (uint32_t)(v1 >> 32);
        uint32_t v1_lo = (uint32_t)(v1 & 0xFFFFFFFF);
        uint32_t v2_hi = (uint32_t)(v2 >> 32);
        uint32_t v2_lo = (uint32_t)(v2 & 0xFFFFFFFF);

        uint32_t new0 = v2_hi ^ counter[1][i] ^ key[0][i];
        uint32_t new1 = v2_lo;
        uint32_t new2 = v1_hi ^ counter[3][i] ^ key[1][i];
        uint32_t new3 = v1_lo;

        counter[0][i] = new0;
        counter[1][i] = new1;
        counter[2][i] = new2;
        counter[3][i] = new3;
    }

    // Box-Muller using counter[0] and counter[1] (same as sd.cpp g[0][i], g[1][i])
    std::vector<float> result(n);
    for (uint32_t i = 0; i < n; i++) {
        float u = (float)counter[0][i] * two_pow32_inv + two_pow32_inv / 2.0f;
        float v = (float)counter[1][i] * two_pow32_inv_2pi + two_pow32_inv_2pi / 2.0f;
        float s = sqrtf(-2.0f * logf(u));
        result[i] = s * sinf(v);
    }
    return result;
}

int main() {
    const uint64_t seed = 42;
    const int64_t n = 16;

    // ---- Run our CUDA implementation ----
    __nv_bfloat16* d_data;
    cudaMalloc(&d_data, n * sizeof(__nv_bfloat16));
    randn_fill_philox(d_data, n, seed);

    // Download BF16 to host, convert to float
    std::vector<__nv_bfloat16> h_bf16(n);
    cudaMemcpy(h_bf16.data(), d_data, n * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
    cudaFree(d_data);

    std::vector<float> ours_f32(n);
    for (int i = 0; i < n; i++) {
        ours_f32[i] = __bfloat162float(h_bf16[i]);
    }

    // ---- Run sd.cpp reference (inline) ----
    std::vector<float> ref = sdcpp_philox_randn(seed, (uint32_t)n);

    // ---- Print comparison ----
    printf("%-6s  %-16s  %-16s  %-10s\n", "Index", "Ours (BF16->F32)", "sd.cpp (F32)", "Match?");
    printf("------  ----------------  ----------------  ----------\n");

    bool all_match = true;
    for (int i = 0; i < n; i++) {
        // BF16 has ~3 decimal digits of precision, so check approximate match
        float diff = fabsf(ours_f32[i] - ref[i]);
        // BF16 relative error is about 2^-8 ~ 0.004; use generous threshold
        bool match = (diff < 0.02f) || (ref[i] != 0.0f && fabsf(diff / ref[i]) < 0.01f);
        if (!match) all_match = false;

        printf("%-6d  %16.8f  %16.8f  %s\n", i, ours_f32[i], ref[i],
               match ? "OK" : "MISMATCH");
    }

    printf("\n");
    if (all_match) {
        printf("SUCCESS: All values match (within BF16 precision).\n");
    } else {
        printf("FAILURE: Some values differ beyond BF16 precision.\n");
    }

    // Also print the exact FP32 values from our implementation BEFORE bf16 conversion
    // by re-running the Philox on host directly with our algorithm
    printf("\n--- Exact FP32 values from our Philox algorithm (no BF16 truncation) ---\n");
    {
        const uint32_t M0 = 0xD2511F53u;
        const uint32_t M1 = 0xCD9E8D57u;
        const uint32_t W0 = 0x9E3779B9u;
        const uint32_t W1 = 0xBB67AE85u;
        const float two_pow32_inv = 2.3283064e-10f;
        const float two_pow32_inv_2pi = two_pow32_inv * 6.2831855f;

        uint32_t key0 = (uint32_t)(seed & 0xFFFFFFFF);
        uint32_t key1 = (uint32_t)(seed >> 32);
        uint32_t offset = 0;

        for (int64_t i = 0; i < n; i++) {
            uint32_t counter[4] = {offset, 0, (uint32_t)i, 0};
            uint32_t k0 = key0, k1 = key1;

            for (int round = 0; round < 10; round++) {
                uint64_t v1 = (uint64_t)counter[0] * M0;
                uint64_t v2 = (uint64_t)counter[2] * M1;
                uint32_t v1_hi = (uint32_t)(v1 >> 32);
                uint32_t v1_lo = (uint32_t)(v1);
                uint32_t v2_hi = (uint32_t)(v2 >> 32);
                uint32_t v2_lo = (uint32_t)(v2);

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

            float u = (float)counter[0] * two_pow32_inv + two_pow32_inv / 2.0f;
            float v = (float)counter[1] * two_pow32_inv_2pi + two_pow32_inv_2pi / 2.0f;
            float s = sqrtf(-2.0f * logf(u));
            float val = s * sinf(v);

            printf("  [%2lld] ours_exact=%.8f  sdcpp=%.8f  diff=%.2e\n",
                   (long long)i, val, ref[i], fabsf(val - ref[i]));
        }
    }

    return 0;
}
