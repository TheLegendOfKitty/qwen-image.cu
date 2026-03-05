#pragma once
#include "tensor.h"
#include "safetensors.h"
#include "cuda_kernels.cuh"
#include <string>
#include <vector>
#include <cstdio>

// Calibration file writer for GPTQ quantization.
// Captures Hadamard-rotated input activations during inference.
// File format: "GPTQ" magic (4B), version=1 (4B), num_entries (4B),
//   then per entry: name_len(4B), name(chars), M(4B), K(4B), had_block_size(4B), data(BF16[M*K])
struct CalibrationWriter {
    FILE* f = nullptr;
    uint32_t num_entries = 0;
    long count_offset = 0; // file offset of num_entries field

    bool open(const char* path) {
        f = fopen(path, "wb");
        if (!f) return false;
        // Write header
        uint32_t magic = 0x51545047; // "GPTQ" little-endian
        uint32_t version = 1;
        fwrite(&magic, 4, 1, f);
        fwrite(&version, 4, 1, f);
        count_offset = ftell(f);
        fwrite(&num_entries, 4, 1, f); // placeholder
        return true;
    }

    // Write a calibration entry. fp32_data is GPU memory [M, K].
    // Applies FWHT in-place, converts to BF16, writes to file.
    void write_entry(const std::string& name, float* fp32_data_gpu, int M, int K, int had_block_size) {
        // Apply Hadamard rotation in-place
        if (had_block_size > 1) {
            fwht_inplace(fp32_data_gpu, M, K, had_block_size);
        }
        // Convert FP32 → BF16 on GPU
        int64_t n = (int64_t)M * K;
        __nv_bfloat16* bf16_gpu;
        CUDA_CHECK(cudaMalloc(&bf16_gpu, n * sizeof(__nv_bfloat16)));
        fp32_to_bf16(fp32_data_gpu, bf16_gpu, n);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Download to CPU
        std::vector<__nv_bfloat16> buf(n);
        CUDA_CHECK(cudaMemcpy(buf.data(), bf16_gpu, n * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(bf16_gpu));

        // Write entry
        uint32_t name_len = (uint32_t)name.size();
        uint32_t m = (uint32_t)M, k = (uint32_t)K, hbs = (uint32_t)had_block_size;
        fwrite(&name_len, 4, 1, f);
        fwrite(name.data(), 1, name_len, f);
        fwrite(&m, 4, 1, f);
        fwrite(&k, 4, 1, f);
        fwrite(&hbs, 4, 1, f);
        fwrite(buf.data(), sizeof(__nv_bfloat16), n, f);

        num_entries++;
    }

    void close() {
        if (!f) return;
        // Patch num_entries in header
        fseek(f, count_offset, SEEK_SET);
        fwrite(&num_entries, 4, 1, f);
        fclose(f);
        f = nullptr;
    }
};

struct TransformerWeights {
    // Timestep embedding (INT8 quantized)
    QuantizedWeight time_linear1_weight; // [3072, 256]
    Tensor time_linear1_bias;            // [3072]
    QuantizedWeight time_linear2_weight; // [3072, 3072]
    Tensor time_linear2_bias;            // [3072]

    // Input projections (INT8 quantized)
    Tensor txt_norm_weight;              // [3584] RMSNorm (NOT quantized)
    QuantizedWeight img_in_weight;       // [3072, 64]
    Tensor img_in_bias;                  // [3072]
    QuantizedWeight txt_in_weight;       // [3072, 3584]
    Tensor txt_in_bias;                  // [3072]

    // Output (INT8 quantized)
    QuantizedWeight norm_out_linear_weight; // [6144, 3072]
    Tensor norm_out_linear_bias;            // [6144]
    QuantizedWeight proj_out_weight;        // [64, 3072]
    Tensor proj_out_bias;                   // [64]

    struct Block {
        // Image modulation (INT8 quantized)
        QuantizedWeight img_mod_weight; // [18432, 3072]
        Tensor img_mod_bias;            // [18432]

        // Text modulation (INT8 quantized)
        QuantizedWeight txt_mod_weight; // [18432, 3072]
        Tensor txt_mod_bias;            // [18432]

        // Attention (INT8 quantized)
        QuantizedWeight to_q_weight; // [3072, 3072]
        Tensor to_q_bias;            // [3072]
        QuantizedWeight to_k_weight; // [3072, 3072]
        Tensor to_k_bias;            // [3072]
        QuantizedWeight to_v_weight; // [3072, 3072]
        Tensor to_v_bias;            // [3072]

        Tensor norm_q_weight; // [128] RMSNorm per head (NOT quantized)
        Tensor norm_k_weight; // [128]

        QuantizedWeight add_q_proj_weight; // [3072, 3072]
        Tensor add_q_proj_bias;            // [3072]
        QuantizedWeight add_k_proj_weight; // [3072, 3072]
        Tensor add_k_proj_bias;            // [3072]
        QuantizedWeight add_v_proj_weight; // [3072, 3072]
        Tensor add_v_proj_bias;            // [3072]

        Tensor norm_added_q_weight; // [128] (NOT quantized)
        Tensor norm_added_k_weight; // [128]

        QuantizedWeight to_out_weight;     // [3072, 3072]
        Tensor to_out_bias;                // [3072]
        QuantizedWeight to_add_out_weight; // [3072, 3072]
        Tensor to_add_out_bias;            // [3072]

        // Image MLP (INT8 quantized)
        QuantizedWeight img_mlp_fc1_weight; // [12288, 3072]
        Tensor img_mlp_fc1_bias;            // [12288]
        QuantizedWeight img_mlp_fc2_weight; // [3072, 12288]
        Tensor img_mlp_fc2_bias;            // [3072]

        // Text MLP (INT8 quantized)
        QuantizedWeight txt_mlp_fc1_weight; // [12288, 3072]
        Tensor txt_mlp_fc1_bias;            // [12288]
        QuantizedWeight txt_mlp_fc2_weight; // [3072, 12288]
        Tensor txt_mlp_fc2_bias;            // [3072]
    };

    std::vector<Block> blocks;

    void load(const SafeTensorsLoader& loader, const std::string& model_dir = "") {
        // Detect quantization mode (check a block-level weight, not boundary layers which may be BF16)
        bool nunchaku = loader.has_tensor("transformer_blocks.0.attn.to_qkv.qweight");
        bool pre_quantized_int4 = !nunchaku &&
                                   loader.has_tensor("transformer_blocks.0.attn.to_q.weight.__svd_up__");
        bool pre_quantized_int8 = !nunchaku && !pre_quantized_int4 &&
                                   loader.has_tensor("transformer_blocks.0.attn.to_q.weight") &&
                                   loader.get_info("transformer_blocks.0.attn.to_q.weight").dtype == DType::INT8;
        fprintf(stderr, "Loading transformer weights (%s)...\n",
                nunchaku ? "nunchaku INT4+SVD (swizzled)" :
                pre_quantized_int4 ? "pre-quantized INT4+SVD" :
                pre_quantized_int8 ? "pre-quantized INT8" : "quantizing BF16->INT8");

        // Helper: load weight as QuantizedWeight (auto-detects INT4/INT8/BF16)
        auto load_q = [&](const std::string& name) -> QuantizedWeight {
            // INT4+SVD path
            if (loader.has_tensor(name + ".__svd_up__")) {
                QuantizedWeight qw;
                qw.mode = QuantMode::INT4_SVD;
                qw.qweight = loader.load_tensor(name + ".__qweight__");
                qw.scales4 = loader.load_tensor(name + ".__scales4__");
                qw.svd_up = loader.load_tensor(name + ".__svd_up__");
                qw.svd_down = loader.load_tensor(name + ".__svd_down__");
                if (loader.has_tensor(name + ".__smooth__"))
                    qw.smooth = loader.load_tensor(name + ".__smooth__");
                qw.nf4_grid = !loader.has_tensor(name + ".__linear_int4__");
                qw.svd_rank = (int)qw.svd_up.shape[1];
                int K_packed = (int)qw.qweight.shape[1];
                int num_groups = (int)qw.scales4.shape[1];
                qw.group_size = (K_packed * 2) / num_groups;
                qw.had_block_size = loader.has_tensor(name + ".__no_hadamard__") ?
                    1 : hadamard_block_size(K_packed * 2);
                return qw;
            }
            // Pre-quantized INT8 path
            if (loader.has_tensor(name) && loader.get_info(name).dtype == DType::INT8) {
                QuantizedWeight qw;
                qw.mode = QuantMode::INT8_HADAMARD;
                qw.data = loader.load_tensor(name);
                qw.scales = loader.load_tensor(name + ".__scales__");
                int K = (int)qw.data.shape[1];
                qw.had_block_size = hadamard_block_size(K);
                return qw;
            }
            // BF16 native path (sensitive layers kept unquantized)
            if (loader.has_tensor(name) && loader.get_info(name).dtype == DType::BF16) {
                QuantizedWeight qw;
                qw.mode = QuantMode::BF16;
                qw.data = loader.load_tensor(name);
                return qw;
            }
            // Fallback: quantize BF16 at load time (INT8)
            Tensor bf16 = loader.load_tensor(name);
            QuantizedWeight qw = quantize_weight_tensor(bf16);
            bf16.free_data();
            return qw;
        };

        // Helper: load one SVDQuant layer from nunchaku naming
        auto load_nk = [&](const std::string& nk_prefix) -> QuantizedWeight {
            QuantizedWeight qw;
            qw.mode = QuantMode::INT4_SVD;
            qw.nunchaku_swizzle = true;
            qw.nf4_grid = false;
            qw.had_block_size = 1;

            qw.qweight = loader.load_tensor(nk_prefix + ".qweight");
            qw.wscales_bf16 = loader.load_tensor(nk_prefix + ".wscales");
            qw.svd_up = loader.load_tensor(nk_prefix + ".proj_up");
            qw.svd_down = loader.load_tensor(nk_prefix + ".proj_down");

            // Smooth factor: may be BF16, convert to FP32
            if (loader.has_tensor(nk_prefix + ".smooth_factor")) {
                Tensor sf = loader.load_tensor(nk_prefix + ".smooth_factor");
                if (sf.dtype == DType::BF16) {
                    qw.smooth = Tensor::alloc({sf.shape[0]}, DType::FP32);
                    bf16_to_fp32((__nv_bfloat16*)sf.data, (float*)qw.smooth.data, sf.shape[0]);
                    CUDA_CHECK(cudaDeviceSynchronize());
                    sf.free_data();
                } else {
                    qw.smooth = std::move(sf);
                }
            }

            qw.svd_rank = (int)qw.svd_up.shape[1];
            int K_packed = (int)qw.qweight.shape[1];
            int num_groups = (int)qw.wscales_bf16.shape[0];
            qw.group_size = (K_packed * 2) / num_groups;
            int N = (int)qw.qweight.shape[0];
            int K = K_packed * 2;
            int r = qw.svd_rank;

            // Unswizzle LoRA weights from MMA-packed to standard row-major
            {
                Tensor svd_down_std = Tensor::alloc({(int64_t)K, (int64_t)r}, DType::BF16);
                unswizzle_lora_weights((__nv_bfloat16*)qw.svd_down.data,
                                        (__nv_bfloat16*)svd_down_std.data, K, r, true);
                CUDA_CHECK(cudaDeviceSynchronize());
                qw.svd_down.free_data();
                qw.svd_down = std::move(svd_down_std);

                Tensor svd_up_std = Tensor::alloc({(int64_t)N, (int64_t)r}, DType::BF16);
                unswizzle_lora_weights((__nv_bfloat16*)qw.svd_up.data,
                                        (__nv_bfloat16*)svd_up_std.data, N, r, false);
                CUDA_CHECK(cudaDeviceSynchronize());
                qw.svd_up.free_data();
                qw.svd_up = std::move(svd_up_std);
            }

            // Unswizzle to row-major for W4A4 GEMM
            qw.qweight_rowmajor = Tensor::alloc({(int64_t)N, (int64_t)(K / 2)}, DType::UINT8);
            qw.wscales_rowmajor = Tensor::alloc({(int64_t)num_groups, (int64_t)N}, DType::FP32);
            unswizzle_nunchaku_weights(
                (uint8_t*)qw.qweight.data, (__nv_bfloat16*)qw.wscales_bf16.data,
                (uint8_t*)qw.qweight_rowmajor.data, (float*)qw.wscales_rowmajor.data,
                N, K, num_groups);
            CUDA_CHECK(cudaDeviceSynchronize());
            // Free swizzled originals (no longer needed)
            qw.qweight.free_data();
            qw.wscales_bf16.free_data();

            // Create MMA fragment-ordered weights for register-direct GEMM
            {
                int N_tiles8 = (N + 7) / 8;
                int64_t mma_bytes = (int64_t)num_groups * N_tiles8 * 256;
                qw.qweight_mma = Tensor::alloc({mma_bytes}, DType::UINT8);
                swizzle_w4a4_weights_mma(
                    (uint8_t*)qw.qweight_rowmajor.data,
                    (uint32_t*)qw.qweight_mma.data,
                    N, K, qw.group_size);
                CUDA_CHECK(cudaDeviceSynchronize());
            }

            return qw;
        };

        // Helper: split fused nunchaku QKV [3N,K] into separate Q/K/V
        auto split_nk_qkv = [&](const std::string& nk_prefix,
                                 QuantizedWeight& q_w, QuantizedWeight& k_w, QuantizedWeight& v_w,
                                 Tensor& q_bias, Tensor& k_bias, Tensor& v_bias) {
            // Load fused tensors
            Tensor fused_qw = loader.load_tensor(nk_prefix + ".qweight");    // [3N, K/2]
            Tensor fused_ws = loader.load_tensor(nk_prefix + ".wscales");    // [num_groups, 3N] BF16
            Tensor fused_up = loader.load_tensor(nk_prefix + ".proj_up");    // [3N, r]
            Tensor fused_down = loader.load_tensor(nk_prefix + ".proj_down");// [K, r]
            Tensor fused_bias = loader.load_tensor(nk_prefix + ".bias");     // [3N]

            // Smooth factor: may be BF16, convert to FP32
            Tensor smooth_fp32;
            if (loader.has_tensor(nk_prefix + ".smooth_factor")) {
                Tensor sf = loader.load_tensor(nk_prefix + ".smooth_factor");
                if (sf.dtype == DType::BF16) {
                    smooth_fp32 = Tensor::alloc({sf.shape[0]}, DType::FP32);
                    bf16_to_fp32((__nv_bfloat16*)sf.data, (float*)smooth_fp32.data, sf.shape[0]);
                    CUDA_CHECK(cudaDeviceSynchronize());
                    sf.free_data();
                } else {
                    smooth_fp32 = std::move(sf);
                }
            }

            int64_t N3 = fused_qw.shape[0];
            int64_t N = N3 / 3;
            int64_t K_packed = fused_qw.shape[1];
            int64_t K = K_packed * 2;
            int64_t num_groups = fused_ws.shape[0];
            int64_t r = fused_up.shape[1];

            QuantizedWeight* parts[3] = {&q_w, &k_w, &v_w};
            Tensor* biases[3] = {&q_bias, &k_bias, &v_bias};

            for (int i = 0; i < 3; i++) {
                QuantizedWeight& qw = *parts[i];
                qw.mode = QuantMode::INT4_SVD;
                qw.nunchaku_swizzle = true;
                qw.nf4_grid = false;
                qw.had_block_size = 1;
                qw.svd_rank = (int)r;
                qw.group_size = (int)((K_packed * 2) / num_groups);

                // qweight: [N, K/2] — n_tile outermost → contiguous slice
                qw.qweight = Tensor::alloc({N, K_packed}, DType::UINT8);
                CUDA_CHECK(cudaMemcpy(qw.qweight.data,
                    (uint8_t*)fused_qw.data + i * N * K_packed,
                    N * K_packed, cudaMemcpyDeviceToDevice));

                // wscales: n_tile outermost in packed layout → contiguous slice
                qw.wscales_bf16 = Tensor::alloc({num_groups, N}, DType::BF16);
                CUDA_CHECK(cudaMemcpy(qw.wscales_bf16.data,
                    (uint8_t*)fused_ws.data + i * N * num_groups * sizeof(__nv_bfloat16),
                    N * num_groups * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice));

                // proj_up: [N, r] from [3N, r] — unswizzle from MMA-packed
                {
                    Tensor packed_up = Tensor::alloc({N, r}, DType::BF16);
                    CUDA_CHECK(cudaMemcpy(packed_up.data,
                        (uint8_t*)fused_up.data + i * N * r * sizeof(__nv_bfloat16),
                        N * r * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice));
                    qw.svd_up = Tensor::alloc({N, r}, DType::BF16);
                    unswizzle_lora_weights((__nv_bfloat16*)packed_up.data,
                                            (__nv_bfloat16*)qw.svd_up.data, (int)N, (int)r, false);
                    CUDA_CHECK(cudaDeviceSynchronize());
                    packed_up.free_data();
                }

                // proj_down: [K, r] — shared, clone and unswizzle from MMA-packed
                {
                    qw.svd_down = Tensor::alloc({K, r}, DType::BF16);
                    unswizzle_lora_weights((__nv_bfloat16*)fused_down.data,
                                            (__nv_bfloat16*)qw.svd_down.data, (int)K, (int)r, true);
                    CUDA_CHECK(cudaDeviceSynchronize());
                }

                // Debug: dump first Q svd_down and svd_up for verification
                // smooth: [K] — shared, clone
                if (smooth_fp32.data) {
                    qw.smooth = Tensor::alloc({K}, DType::FP32);
                    CUDA_CHECK(cudaMemcpy(qw.smooth.data, smooth_fp32.data,
                        K * sizeof(float), cudaMemcpyDeviceToDevice));
                }

                // bias: [N] from [3N]
                *biases[i] = Tensor::alloc({N}, DType::BF16);
                CUDA_CHECK(cudaMemcpy(biases[i]->data,
                    (uint8_t*)fused_bias.data + i * N * sizeof(__nv_bfloat16),
                    N * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice));

                // Unswizzle to row-major for W4A4 GEMM
                qw.qweight_rowmajor = Tensor::alloc({N, K_packed}, DType::UINT8);
                qw.wscales_rowmajor = Tensor::alloc({(int64_t)num_groups, N}, DType::FP32);
                unswizzle_nunchaku_weights(
                    (uint8_t*)qw.qweight.data, (__nv_bfloat16*)qw.wscales_bf16.data,
                    (uint8_t*)qw.qweight_rowmajor.data, (float*)qw.wscales_rowmajor.data,
                    (int)N, (int)K, (int)num_groups);
                CUDA_CHECK(cudaDeviceSynchronize());
                qw.qweight.free_data();
                qw.wscales_bf16.free_data();

                // Create MMA fragment-ordered weights for register-direct GEMM
                {
                    int N_tiles8 = ((int)N + 7) / 8;
                    int64_t mma_bytes = (int64_t)num_groups * N_tiles8 * 256;
                    qw.qweight_mma = Tensor::alloc({mma_bytes}, DType::UINT8);
                    swizzle_w4a4_weights_mma(
                        (uint8_t*)qw.qweight_rowmajor.data,
                        (uint32_t*)qw.qweight_mma.data,
                        (int)N, (int)K, qw.group_size);
                    CUDA_CHECK(cudaDeviceSynchronize());
                }
            }

            fused_qw.free_data();
            fused_ws.free_data();
            fused_up.free_data();
            fused_down.free_data();
            fused_bias.free_data();
            smooth_fp32.free_data();
        };

        // Timestep
        time_linear1_weight = load_q("time_text_embed.timestep_embedder.linear_1.weight");
        time_linear1_bias = loader.load_tensor("time_text_embed.timestep_embedder.linear_1.bias");
        time_linear2_weight = load_q("time_text_embed.timestep_embedder.linear_2.weight");
        time_linear2_bias = loader.load_tensor("time_text_embed.timestep_embedder.linear_2.bias");

        // Input projections
        txt_norm_weight = loader.load_tensor("txt_norm.weight");
        img_in_weight = load_q("img_in.weight");
        img_in_bias = loader.load_tensor("img_in.bias");
        txt_in_weight = load_q("txt_in.weight");
        txt_in_bias = loader.load_tensor("txt_in.bias");

        // Output
        norm_out_linear_weight = load_q("norm_out.linear.weight");
        norm_out_linear_bias = loader.load_tensor("norm_out.linear.bias");
        proj_out_weight = load_q("proj_out.weight");
        proj_out_bias = loader.load_tensor("proj_out.bias");

        // Helper: load AWQ modulation weights from nunchaku, dequantize to BF16,
        // de-interleave output dim, and adjust bias (remove fused +1 from scale components).
        auto load_awq_modulation = [&](const std::string& key_prefix,
                                        QuantizedWeight& out_weight, Tensor& out_bias) {
            // Load raw AWQ tensors from nunchaku safetensors
            Tensor qw_tensor = loader.load_tensor(key_prefix + ".qweight");   // [OC/4, IC/2] INT32
            Tensor ws_tensor = loader.load_tensor(key_prefix + ".wscales");   // [num_groups, OC] BF16
            Tensor wz_tensor = loader.load_tensor(key_prefix + ".wzeros");    // [num_groups, OC] BF16
            Tensor bias_raw = loader.load_tensor(key_prefix + ".bias");       // [OC] BF16

            int OC = (int)(qw_tensor.shape[0] * 4);  // 4608 * 4 = 18432
            int IC = (int)(qw_tensor.shape[1] * 2);   // 1536 * 2 = 3072
            int num_groups = (int)ws_tensor.shape[0];  // 48
            int group_size = IC / num_groups;           // 64
            int num_components = 6;                     // modulation: shift1,scale1,gate1,shift2,scale2,gate2
            int dim = OC / num_components;              // 3072

            // Allocate output BF16 weight [OC, IC] and dequantize with de-interleaving
            out_weight.mode = QuantMode::BF16;
            out_weight.data = Tensor::alloc({(int64_t)OC, (int64_t)IC}, DType::BF16);
            dequantize_awq_to_bf16((const int32_t*)qw_tensor.data,
                                     (const __nv_bfloat16*)ws_tensor.data,
                                     (const __nv_bfloat16*)wz_tensor.data,
                                     (__nv_bfloat16*)out_weight.data.data,
                                     OC, IC, group_size, num_components);
            CUDA_CHECK(cudaDeviceSynchronize());

            // De-interleave bias on CPU and subtract 1 from scale components
            // Nunchaku bias is interleaved: [feat0_comp0, feat0_comp1, ..., feat0_comp5, feat1_comp0, ...]
            // Standard order: [shift1[0..dim-1], scale1[0..dim-1], gate1[0..dim-1], ...]
            // Scale components are at indices [1*dim : 2*dim] (scale1) and [4*dim : 5*dim] (scale2)
            std::vector<__nv_bfloat16> bias_cpu(OC);
            CUDA_CHECK(cudaMemcpy(bias_cpu.data(), bias_raw.data, OC * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));

            std::vector<__nv_bfloat16> bias_deinterleaved(OC);
            for (int j = 0; j < OC; j++) {
                int component = j % num_components;
                int feature = j / num_components;
                int std_idx = component * dim + feature;
                float val = __bfloat162float(bias_cpu[j]);
                // Subtract 1 from scale components (scale1=comp1, scale2=comp4)
                if (component == 1 || component == 4) {
                    val -= 1.0f;
                }
                bias_deinterleaved[std_idx] = __float2bfloat16(val);
            }

            out_bias = Tensor::alloc({(int64_t)OC}, DType::BF16);
            CUDA_CHECK(cudaMemcpy(out_bias.data, bias_deinterleaved.data(),
                                  OC * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));

            // Free temporary AWQ tensors
            qw_tensor.free_data();
            ws_tensor.free_data();
            wz_tensor.free_data();
            bias_raw.free_data();
        };

        // Transformer blocks
        blocks.resize(60);
        for (int i = 0; i < 60; i++) {
            std::string prefix = "transformer_blocks." + std::to_string(i) + ".";
            Block& b = blocks[i];

            if (nunchaku) {
                // Modulation: dequantize AWQ INT4 weights from nunchaku safetensors
                // (nunchaku uses AWQ W4A16 for modulation with interleaved output + fused +1 in bias)
                load_awq_modulation(prefix + "img_mod.1", b.img_mod_weight, b.img_mod_bias);
                load_awq_modulation(prefix + "txt_mod.1", b.txt_mod_weight, b.txt_mod_bias);

                // Fused QKV → split Q/K/V
                split_nk_qkv(prefix + "attn.to_qkv",
                    b.to_q_weight, b.to_k_weight, b.to_v_weight,
                    b.to_q_bias, b.to_k_bias, b.to_v_bias);

                b.norm_q_weight = loader.load_tensor(prefix + "attn.norm_q.weight");
                b.norm_k_weight = loader.load_tensor(prefix + "attn.norm_k.weight");

                // Fused add_QKV → split
                split_nk_qkv(prefix + "attn.add_qkv_proj",
                    b.add_q_proj_weight, b.add_k_proj_weight, b.add_v_proj_weight,
                    b.add_q_proj_bias, b.add_k_proj_bias, b.add_v_proj_bias);

                b.norm_added_q_weight = loader.load_tensor(prefix + "attn.norm_added_q.weight");
                b.norm_added_k_weight = loader.load_tensor(prefix + "attn.norm_added_k.weight");

                // Attention output
                b.to_out_weight = load_nk(prefix + "attn.to_out.0");
                b.to_out_bias = loader.load_tensor(prefix + "attn.to_out.0.bias");
                b.to_add_out_weight = load_nk(prefix + "attn.to_add_out");
                b.to_add_out_bias = loader.load_tensor(prefix + "attn.to_add_out.bias");

                // Image MLP
                b.img_mlp_fc1_weight = load_nk(prefix + "img_mlp.net.0.proj");
                b.img_mlp_fc1_bias = loader.load_tensor(prefix + "img_mlp.net.0.proj.bias");
                b.img_mlp_fc2_weight = load_nk(prefix + "img_mlp.net.2");
                b.img_mlp_fc2_bias = loader.load_tensor(prefix + "img_mlp.net.2.bias");

                // Text MLP
                b.txt_mlp_fc1_weight = load_nk(prefix + "txt_mlp.net.0.proj");
                b.txt_mlp_fc1_bias = loader.load_tensor(prefix + "txt_mlp.net.0.proj.bias");
                b.txt_mlp_fc2_weight = load_nk(prefix + "txt_mlp.net.2");
                b.txt_mlp_fc2_bias = loader.load_tensor(prefix + "txt_mlp.net.2.bias");
            } else {
                b.img_mod_weight = load_q(prefix + "img_mod.1.weight");
                b.img_mod_bias = loader.load_tensor(prefix + "img_mod.1.bias");
                b.txt_mod_weight = load_q(prefix + "txt_mod.1.weight");
                b.txt_mod_bias = loader.load_tensor(prefix + "txt_mod.1.bias");

                b.to_q_weight = load_q(prefix + "attn.to_q.weight");
                b.to_q_bias = loader.load_tensor(prefix + "attn.to_q.bias");
                b.to_k_weight = load_q(prefix + "attn.to_k.weight");
                b.to_k_bias = loader.load_tensor(prefix + "attn.to_k.bias");
                b.to_v_weight = load_q(prefix + "attn.to_v.weight");
                b.to_v_bias = loader.load_tensor(prefix + "attn.to_v.bias");

                b.norm_q_weight = loader.load_tensor(prefix + "attn.norm_q.weight");
                b.norm_k_weight = loader.load_tensor(prefix + "attn.norm_k.weight");

                b.add_q_proj_weight = load_q(prefix + "attn.add_q_proj.weight");
                b.add_q_proj_bias = loader.load_tensor(prefix + "attn.add_q_proj.bias");
                b.add_k_proj_weight = load_q(prefix + "attn.add_k_proj.weight");
                b.add_k_proj_bias = loader.load_tensor(prefix + "attn.add_k_proj.bias");
                b.add_v_proj_weight = load_q(prefix + "attn.add_v_proj.weight");
                b.add_v_proj_bias = loader.load_tensor(prefix + "attn.add_v_proj.bias");

                b.norm_added_q_weight = loader.load_tensor(prefix + "attn.norm_added_q.weight");
                b.norm_added_k_weight = loader.load_tensor(prefix + "attn.norm_added_k.weight");

                b.to_out_weight = load_q(prefix + "attn.to_out.0.weight");
                b.to_out_bias = loader.load_tensor(prefix + "attn.to_out.0.bias");
                b.to_add_out_weight = load_q(prefix + "attn.to_add_out.weight");
                b.to_add_out_bias = loader.load_tensor(prefix + "attn.to_add_out.bias");

                b.img_mlp_fc1_weight = load_q(prefix + "img_mlp.net.0.proj.weight");
                b.img_mlp_fc1_bias = loader.load_tensor(prefix + "img_mlp.net.0.proj.bias");
                b.img_mlp_fc2_weight = load_q(prefix + "img_mlp.net.2.weight");
                b.img_mlp_fc2_bias = loader.load_tensor(prefix + "img_mlp.net.2.bias");

                b.txt_mlp_fc1_weight = load_q(prefix + "txt_mlp.net.0.proj.weight");
                b.txt_mlp_fc1_bias = loader.load_tensor(prefix + "txt_mlp.net.0.proj.bias");
                b.txt_mlp_fc2_weight = load_q(prefix + "txt_mlp.net.2.weight");
                b.txt_mlp_fc2_bias = loader.load_tensor(prefix + "txt_mlp.net.2.bias");
            }

            if ((i + 1) % 10 == 0)
                fprintf(stderr, "  Loaded block %d/60\n", i + 1);
        }
        fprintf(stderr, "Transformer weights loaded.\n");
    }

    void free_all() {
        time_linear1_weight.free_data();
        time_linear1_bias.free_data();
        time_linear2_weight.free_data();
        time_linear2_bias.free_data();
        txt_norm_weight.free_data();
        img_in_weight.free_data();
        img_in_bias.free_data();
        txt_in_weight.free_data();
        txt_in_bias.free_data();
        norm_out_linear_weight.free_data();
        norm_out_linear_bias.free_data();
        proj_out_weight.free_data();
        proj_out_bias.free_data();
        for (auto& b : blocks) {
            b.img_mod_weight.free_data(); b.img_mod_bias.free_data();
            b.txt_mod_weight.free_data(); b.txt_mod_bias.free_data();
            b.to_q_weight.free_data(); b.to_q_bias.free_data();
            b.to_k_weight.free_data(); b.to_k_bias.free_data();
            b.to_v_weight.free_data(); b.to_v_bias.free_data();
            b.norm_q_weight.free_data(); b.norm_k_weight.free_data();
            b.add_q_proj_weight.free_data(); b.add_q_proj_bias.free_data();
            b.add_k_proj_weight.free_data(); b.add_k_proj_bias.free_data();
            b.add_v_proj_weight.free_data(); b.add_v_proj_bias.free_data();
            b.norm_added_q_weight.free_data(); b.norm_added_k_weight.free_data();
            b.to_out_weight.free_data(); b.to_out_bias.free_data();
            b.to_add_out_weight.free_data(); b.to_add_out_bias.free_data();
            b.img_mlp_fc1_weight.free_data(); b.img_mlp_fc1_bias.free_data();
            b.img_mlp_fc2_weight.free_data(); b.img_mlp_fc2_bias.free_data();
            b.txt_mlp_fc1_weight.free_data(); b.txt_mlp_fc1_bias.free_data();
            b.txt_mlp_fc2_weight.free_data(); b.txt_mlp_fc2_bias.free_data();
        }
        blocks.clear();
    }
};

// Forward pass: x: [1, C, H, W], timestep: scalar, context: [1, seq, 3584], pe: [pos, 64, 2, 2]
// Returns: [1, out_channels, H, W]
// If cal is non-null, captures Hadamard-rotated input activations for GPTQ calibration.
Tensor transformer_forward(const TransformerWeights& w,
                           const Tensor& x,       // [1, 64, H, W] after patchify input
                           float timestep,
                           const Tensor& context,  // [1, seq_len, 3584]
                           const Tensor& pe,       // [pos_len, 64, 2, 2] FP32
                           int H, int W,           // original latent H, W
                           CalibrationWriter* cal = nullptr);
