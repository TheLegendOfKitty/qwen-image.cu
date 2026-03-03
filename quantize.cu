// Offline quantization tool for transformer weights.
// Loads BF16 transformer safetensors, quantizes weights, and writes a single
// pre-quantized safetensors file.
//
// INT8 (default): Hadamard rotation + per-channel INT8 quantization
// INT4 (--int4):  SVDQuant — truncated SVD low-rank correction + per-group INT4
//
// Usage: ./quantize --model-dir ./Qwen-Image-2512/ -o transformer_int8.safetensors
//        ./quantize --model-dir ./Qwen-Image-2512/ --int4 --rank 32 --group-size 128 -o transformer_int4.safetensors

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <chrono>

#include "tensor.h"
#include "safetensors.h"
#include "safetensors_writer.h"
#include "cuda_kernels.cuh"

struct QuantizeConfig {
    std::string model_dir = "./Qwen-Image-2512/";
    std::string output = "";
    bool int4 = false;
    int rank = 32;
    int group_size = 128;
};

static QuantizeConfig parse_args(int argc, char** argv) {
    QuantizeConfig cfg;
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--model-dir" && i + 1 < argc) {
            cfg.model_dir = argv[++i];
        } else if ((arg == "-o" || arg == "--output") && i + 1 < argc) {
            cfg.output = argv[++i];
        } else if (arg == "--int4") {
            cfg.int4 = true;
        } else if (arg == "--rank" && i + 1 < argc) {
            cfg.rank = atoi(argv[++i]);
        } else if (arg == "--group-size" && i + 1 < argc) {
            cfg.group_size = atoi(argv[++i]);
        } else if (arg == "-h" || arg == "--help") {
            fprintf(stderr, "Usage: %s [options]\n", argv[0]);
            fprintf(stderr, "  --model-dir DIR     Model directory (default: ./Qwen-Image-2512/)\n");
            fprintf(stderr, "  -o, --output FILE   Output safetensors file\n");
            fprintf(stderr, "  --int4              Use INT4 SVDQuant quantization (default: INT8)\n");
            fprintf(stderr, "  --rank N            SVD rank for INT4 mode (default: 32)\n");
            fprintf(stderr, "  --group-size N      Group size for INT4 quantization (default: 128)\n");
            exit(0);
        }
    }
    if (!cfg.model_dir.empty() && cfg.model_dir.back() != '/')
        cfg.model_dir += '/';
    if (cfg.output.empty())
        cfg.output = cfg.int4 ? "transformer_int4.safetensors" : "transformer_int8.safetensors";
    return cfg;
}

// Download GPU tensor to CPU vector
static std::vector<uint8_t> download_tensor(const Tensor& t) {
    size_t nb = t.nbytes();
    std::vector<uint8_t> buf(nb);
    CUDA_CHECK(cudaMemcpy(buf.data(), t.data, nb, cudaMemcpyDeviceToHost));
    return buf;
}

int main(int argc, char** argv) {
    QuantizeConfig cfg = parse_args(argc, argv);

    fprintf(stderr, "=== Transformer Weight Quantizer ===\n");
    fprintf(stderr, "Model dir: %s\n", cfg.model_dir.c_str());
    fprintf(stderr, "Output:    %s\n", cfg.output.c_str());
    fprintf(stderr, "Mode:      %s\n", cfg.int4 ? "INT4 SVDQuant" : "INT8 Hadamard");
    if (cfg.int4)
        fprintf(stderr, "SVD rank:  %d, group size: %d\n", cfg.rank, cfg.group_size);

    auto t_start = std::chrono::high_resolution_clock::now();

    // Load transformer safetensors
    fprintf(stderr, "\nLoading transformer weights...\n");
    SafeTensorsLoader loader = load_model_dir(cfg.model_dir + "transformer");
    loader.print_summary();

    SafeTensorsWriter writer;

    // Quantize a weight tensor: BF16 → Hadamard-rotated INT8 + FP32 scales
    // Writes "<name>" (I8) and "<name>.__scales__" (F32) to writer
    auto quantize_and_add = [&](const std::string& name) {
        Tensor bf16 = loader.load_tensor(name);
        QuantizedWeight qw = quantize_weight_tensor(bf16);
        bf16.free_data();

        int N = (int)qw.data.shape[0];
        int K = (int)qw.data.shape[1];

        // Download INT8 data and FP32 scales
        auto int8_buf = download_tensor(qw.data);
        auto scales_buf = download_tensor(qw.scales);

        writer.add(name, "I8", {(int64_t)N, (int64_t)K}, int8_buf.data(), int8_buf.size());
        writer.add(name + ".__scales__", "F32", {(int64_t)N}, scales_buf.data(), scales_buf.size());

        qw.free_data();
    };

    // Pass through a non-quantized tensor (bias, norm weight) as BF16
    auto passthrough = [&](const std::string& name) {
        const TensorInfo& info = loader.get_info(name);
        // Read raw bytes from file
        size_t data_start = 0;
        for (auto& sf : loader.files) {
            if (sf.filepath == info.filename) {
                data_start = sf.data_start;
                break;
            }
        }
        FILE* f = fopen(info.filename.c_str(), "rb");
        if (!f) { fprintf(stderr, "Failed to open: %s\n", info.filename.c_str()); exit(1); }
        fseek(f, (long)(data_start + info.data_offset), SEEK_SET);
        std::vector<uint8_t> buf(info.nbytes);
        if (fread(buf.data(), 1, info.nbytes, f) != info.nbytes) {
            fprintf(stderr, "Failed to read tensor: %s\n", name.c_str()); exit(1);
        }
        fclose(f);

        std::string dtype_str = (info.dtype == DType::FP32) ? "F32" : "BF16";
        writer.add(name, dtype_str, info.shape, buf.data(), buf.size());
    };

    // INT4 SVDQuant quantization: BF16 -> truncated SVD + per-group INT4
    // Writes .__qweight__ (U8), .__scales4__ (F32), .__svd_up__ (BF16), .__svd_down__ (BF16)
    auto quantize_and_add_int4 = [&](const std::string& name) {
        Tensor bf16 = loader.load_tensor(name);
        QuantizedWeight qw = quantize_weight_tensor_int4(bf16, cfg.rank, cfg.group_size);
        bf16.free_data();

        int N = (int)qw.qweight.shape[0];
        int K_packed = (int)qw.qweight.shape[1];
        int num_groups = (int)qw.scales4.shape[1];
        int r = qw.svd_rank;
        int K_full = K_packed * 2;

        auto qw_buf = download_tensor(qw.qweight);
        auto sc_buf = download_tensor(qw.scales4);
        auto up_buf = download_tensor(qw.svd_up);
        auto dn_buf = download_tensor(qw.svd_down);

        writer.add(name + ".__qweight__", "U8", {(int64_t)N, (int64_t)K_packed},
                   qw_buf.data(), qw_buf.size());
        writer.add(name + ".__scales4__", "F32", {(int64_t)N, (int64_t)num_groups},
                   sc_buf.data(), sc_buf.size());
        writer.add(name + ".__svd_up__", "BF16", {(int64_t)N, (int64_t)r},
                   up_buf.data(), up_buf.size());
        writer.add(name + ".__svd_down__", "BF16", {(int64_t)K_full, (int64_t)r},
                   dn_buf.data(), dn_buf.size());

        qw.free_data();
    };

    // Dispatch to the right quantization function
    auto quantize_weight = [&](const std::string& name) {
        if (cfg.int4)
            quantize_and_add_int4(name);
        else
            quantize_and_add(name);
    };

    // --- Timestep embedding ---
    fprintf(stderr, "Quantizing timestep embeddings...\n");
    quantize_weight("time_text_embed.timestep_embedder.linear_1.weight");
    passthrough("time_text_embed.timestep_embedder.linear_1.bias");
    quantize_weight("time_text_embed.timestep_embedder.linear_2.weight");
    passthrough("time_text_embed.timestep_embedder.linear_2.bias");

    // --- Input/output projections (kept BF16 — sensitive boundary layers) ---
    fprintf(stderr, "Passing through input/output projections as BF16...\n");
    passthrough("txt_norm.weight");
    passthrough("img_in.weight");
    passthrough("img_in.bias");
    passthrough("txt_in.weight");
    passthrough("txt_in.bias");
    passthrough("norm_out.linear.weight");
    passthrough("norm_out.linear.bias");
    passthrough("proj_out.weight");
    passthrough("proj_out.bias");

    // --- 60 transformer blocks ---
    for (int i = 0; i < 60; i++) {
        std::string prefix = "transformer_blocks." + std::to_string(i) + ".";

        quantize_weight(prefix + "img_mod.1.weight");
        passthrough(prefix + "img_mod.1.bias");
        quantize_weight(prefix + "txt_mod.1.weight");
        passthrough(prefix + "txt_mod.1.bias");

        quantize_weight(prefix + "attn.to_q.weight");
        passthrough(prefix + "attn.to_q.bias");
        quantize_weight(prefix + "attn.to_k.weight");
        passthrough(prefix + "attn.to_k.bias");
        quantize_weight(prefix + "attn.to_v.weight");
        passthrough(prefix + "attn.to_v.bias");

        passthrough(prefix + "attn.norm_q.weight");
        passthrough(prefix + "attn.norm_k.weight");

        quantize_weight(prefix + "attn.add_q_proj.weight");
        passthrough(prefix + "attn.add_q_proj.bias");
        quantize_weight(prefix + "attn.add_k_proj.weight");
        passthrough(prefix + "attn.add_k_proj.bias");
        quantize_weight(prefix + "attn.add_v_proj.weight");
        passthrough(prefix + "attn.add_v_proj.bias");

        passthrough(prefix + "attn.norm_added_q.weight");
        passthrough(prefix + "attn.norm_added_k.weight");

        quantize_weight(prefix + "attn.to_out.0.weight");
        passthrough(prefix + "attn.to_out.0.bias");
        quantize_weight(prefix + "attn.to_add_out.weight");
        passthrough(prefix + "attn.to_add_out.bias");

        quantize_weight(prefix + "img_mlp.net.0.proj.weight");
        passthrough(prefix + "img_mlp.net.0.proj.bias");
        quantize_weight(prefix + "img_mlp.net.2.weight");
        passthrough(prefix + "img_mlp.net.2.bias");

        quantize_weight(prefix + "txt_mlp.net.0.proj.weight");
        passthrough(prefix + "txt_mlp.net.0.proj.bias");
        quantize_weight(prefix + "txt_mlp.net.2.weight");
        passthrough(prefix + "txt_mlp.net.2.bias");

        if ((i + 1) % 10 == 0)
            fprintf(stderr, "  Quantized block %d/60\n", i + 1);
    }

    fprintf(stderr, "\nWriting %zu tensors to %s...\n", writer.num_entries(), cfg.output.c_str());
    if (!writer.write(cfg.output)) {
        fprintf(stderr, "ERROR: Failed to write output file!\n");
        return 1;
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    float secs = std::chrono::duration<float>(t_end - t_start).count();
    fprintf(stderr, "Done! (%.1f seconds)\n", secs);

    return 0;
}
