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

    void load(const SafeTensorsLoader& loader) {
        // Detect quantization mode (check a block-level weight, not boundary layers which may be BF16)
        bool pre_quantized_int4 = loader.has_tensor("transformer_blocks.0.attn.to_q.weight.__svd_up__");
        bool pre_quantized_int8 = !pre_quantized_int4 &&
                                   loader.has_tensor("transformer_blocks.0.attn.to_q.weight") &&
                                   loader.get_info("transformer_blocks.0.attn.to_q.weight").dtype == DType::INT8;
        fprintf(stderr, "Loading transformer weights (%s)...\n",
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
                qw.svd_rank = (int)qw.svd_up.shape[1];
                int K_packed = (int)qw.qweight.shape[1];
                int num_groups = (int)qw.scales4.shape[1];
                qw.group_size = (K_packed * 2) / num_groups;
                qw.had_block_size = hadamard_block_size(K_packed * 2);
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

        // Transformer blocks
        blocks.resize(60);
        for (int i = 0; i < 60; i++) {
            std::string prefix = "transformer_blocks." + std::to_string(i) + ".";
            Block& b = blocks[i];

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
