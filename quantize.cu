// Offline quantization tool for transformer weights.
// Loads BF16 transformer safetensors, applies Hadamard rotation + INT8 quantization,
// and writes a single pre-quantized safetensors file.
//
// Usage: ./quantize --model-dir ./Qwen-Image-2512/ -o transformer_int8.safetensors

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
    std::string output = "transformer_int8.safetensors";
};

static QuantizeConfig parse_args(int argc, char** argv) {
    QuantizeConfig cfg;
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--model-dir" && i + 1 < argc) {
            cfg.model_dir = argv[++i];
        } else if ((arg == "-o" || arg == "--output") && i + 1 < argc) {
            cfg.output = argv[++i];
        } else if (arg == "-h" || arg == "--help") {
            fprintf(stderr, "Usage: %s [options]\n", argv[0]);
            fprintf(stderr, "  --model-dir DIR   Model directory (default: ./Qwen-Image-2512/)\n");
            fprintf(stderr, "  -o, --output FILE Output safetensors file (default: transformer_int8.safetensors)\n");
            exit(0);
        }
    }
    if (!cfg.model_dir.empty() && cfg.model_dir.back() != '/')
        cfg.model_dir += '/';
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

    // --- Timestep embedding ---
    fprintf(stderr, "Quantizing timestep embeddings...\n");
    quantize_and_add("time_text_embed.timestep_embedder.linear_1.weight");
    passthrough("time_text_embed.timestep_embedder.linear_1.bias");
    quantize_and_add("time_text_embed.timestep_embedder.linear_2.weight");
    passthrough("time_text_embed.timestep_embedder.linear_2.bias");

    // --- Input projections ---
    fprintf(stderr, "Quantizing input projections...\n");
    passthrough("txt_norm.weight");
    quantize_and_add("img_in.weight");
    passthrough("img_in.bias");
    quantize_and_add("txt_in.weight");
    passthrough("txt_in.bias");

    // --- Output ---
    fprintf(stderr, "Quantizing output layers...\n");
    quantize_and_add("norm_out.linear.weight");
    passthrough("norm_out.linear.bias");
    quantize_and_add("proj_out.weight");
    passthrough("proj_out.bias");

    // --- 60 transformer blocks ---
    for (int i = 0; i < 60; i++) {
        std::string prefix = "transformer_blocks." + std::to_string(i) + ".";

        quantize_and_add(prefix + "img_mod.1.weight");
        passthrough(prefix + "img_mod.1.bias");
        quantize_and_add(prefix + "txt_mod.1.weight");
        passthrough(prefix + "txt_mod.1.bias");

        quantize_and_add(prefix + "attn.to_q.weight");
        passthrough(prefix + "attn.to_q.bias");
        quantize_and_add(prefix + "attn.to_k.weight");
        passthrough(prefix + "attn.to_k.bias");
        quantize_and_add(prefix + "attn.to_v.weight");
        passthrough(prefix + "attn.to_v.bias");

        passthrough(prefix + "attn.norm_q.weight");
        passthrough(prefix + "attn.norm_k.weight");

        quantize_and_add(prefix + "attn.add_q_proj.weight");
        passthrough(prefix + "attn.add_q_proj.bias");
        quantize_and_add(prefix + "attn.add_k_proj.weight");
        passthrough(prefix + "attn.add_k_proj.bias");
        quantize_and_add(prefix + "attn.add_v_proj.weight");
        passthrough(prefix + "attn.add_v_proj.bias");

        passthrough(prefix + "attn.norm_added_q.weight");
        passthrough(prefix + "attn.norm_added_k.weight");

        quantize_and_add(prefix + "attn.to_out.0.weight");
        passthrough(prefix + "attn.to_out.0.bias");
        quantize_and_add(prefix + "attn.to_add_out.weight");
        passthrough(prefix + "attn.to_add_out.bias");

        quantize_and_add(prefix + "img_mlp.net.0.proj.weight");
        passthrough(prefix + "img_mlp.net.0.proj.bias");
        quantize_and_add(prefix + "img_mlp.net.2.weight");
        passthrough(prefix + "img_mlp.net.2.bias");

        quantize_and_add(prefix + "txt_mlp.net.0.proj.weight");
        passthrough(prefix + "txt_mlp.net.0.proj.bias");
        quantize_and_add(prefix + "txt_mlp.net.2.weight");
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
