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
    std::string gptq_cal; // optional: GPTQ calibration file
    bool int4 = false;
    bool smooth = false; // SmoothQuant-style per-channel factors
    bool no_hadamard = false; // skip Hadamard rotation on residual
    bool linear_int4 = false; // use symmetric INT4 [-7,7] instead of NF4 grid
    bool error_svd = false;   // SVD the quantization error (SVDQuant approach)
    bool no_fuse_qkv = false; // disable QKV fusion (quantize Q/K/V separately)
    int rank = 32;
    int group_size = 64;
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
        } else if (arg == "--gptq" && i + 1 < argc) {
            cfg.gptq_cal = argv[++i];
        } else if (arg == "--smooth") {
            cfg.smooth = true;
        } else if (arg == "--no-hadamard") {
            cfg.no_hadamard = true;
        } else if (arg == "--linear") {
            cfg.linear_int4 = true;
        } else if (arg == "--error-svd") {
            cfg.error_svd = true;
        } else if (arg == "--no-fuse-qkv") {
            cfg.no_fuse_qkv = true;
        } else if (arg == "-h" || arg == "--help") {
            fprintf(stderr, "Usage: %s [options]\n", argv[0]);
            fprintf(stderr, "  --model-dir DIR     Model directory (default: ./Qwen-Image-2512/)\n");
            fprintf(stderr, "  -o, --output FILE   Output safetensors file\n");
            fprintf(stderr, "  --int4              Use INT4 SVDQuant quantization (default: INT8)\n");
            fprintf(stderr, "  --rank N            SVD rank for INT4 mode (default: 32)\n");
            fprintf(stderr, "  --group-size N      Group size for INT4 quantization (default: 64)\n");
            fprintf(stderr, "  --gptq FILE         GPTQ calibration file (requires --int4)\n");
            fprintf(stderr, "  --smooth            Apply SmoothQuant factors (requires --gptq)\n");
            fprintf(stderr, "  --no-hadamard       Skip Hadamard rotation on residual (requires --gptq)\n");
            fprintf(stderr, "  --linear            Use symmetric INT4 [-7,7] instead of NF4 grid\n");
            fprintf(stderr, "  --error-svd         SVD the quantization error instead of weight\n");
            fprintf(stderr, "  --no-fuse-qkv       Disable QKV fusion (quantize Q/K/V separately)\n");
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
    fprintf(stderr, "Mode:      %s%s%s%s%s%s%s\n", cfg.int4 ? "INT4 SVDQuant" : "INT8 Hadamard",
            cfg.gptq_cal.empty() ? "" : " + GPTQ",
            cfg.smooth ? " + Smooth" : "",
            cfg.no_hadamard ? " + NoHad" : "",
            cfg.linear_int4 ? " + Linear" : " + NF4",
            cfg.error_svd ? " + ErrorSVD" : "",
            cfg.no_fuse_qkv ? " + NoFuseQKV" : " + FuseQKV");
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

    // Load GPTQ calibration data if provided
    CalibrationReader cal_reader;
    if (!cfg.gptq_cal.empty()) {
        if (!cfg.int4) {
            fprintf(stderr, "ERROR: --gptq requires --int4\n");
            return 1;
        }
        fprintf(stderr, "\nLoading GPTQ calibration from %s...\n", cfg.gptq_cal.c_str());
        if (!cal_reader.load(cfg.gptq_cal.c_str())) {
            fprintf(stderr, "ERROR: failed to load calibration file\n");
            return 1;
        }
    }

    // Name mapping: safetensors weight name → calibration entry name
    auto get_cal_name = [](const std::string& weight_name) -> std::string {
        // Extract block index from "transformer_blocks.{i}.xxx"
        auto pos = weight_name.find("transformer_blocks.");
        if (pos == std::string::npos) return "";
        int block_idx = -1;
        sscanf(weight_name.c_str() + pos + 19, "%d", &block_idx);
        if (block_idx < 0) return "";

        std::string prefix = "blocks." + std::to_string(block_idx) + ".";

        if (weight_name.find("attn.to_q.weight") != std::string::npos ||
            weight_name.find("attn.to_k.weight") != std::string::npos ||
            weight_name.find("attn.to_v.weight") != std::string::npos)
            return prefix + "img_attn_qkv";
        if (weight_name.find("attn.to_out.0.weight") != std::string::npos)
            return prefix + "img_attn_out";
        if (weight_name.find("img_mlp.net.0.proj.weight") != std::string::npos)
            return prefix + "img_mlp_in";
        if (weight_name.find("img_mlp.net.2.weight") != std::string::npos)
            return prefix + "img_mlp_mid";
        return "";
    };

    // INT4 GPTQ quantization: uses calibration data for error-optimal rounding.
    // v1 calibration: FWHT-rotated activations → existing GPTQ path
    // v2 calibration: raw activations → smooth + GPTQ path
    auto quantize_and_add_int4_gptq = [&](const std::string& name) {
        std::string cal_name = get_cal_name(name);
        int cal_M = 0, cal_K = 0;
        float* x_gpu = nullptr;

        if (!cal_name.empty()) {
            x_gpu = cal_reader.get_activation_gpu(cal_name, cal_M, cal_K);
        }

        if (!x_gpu) {
            // No calibration data → fall back to naive NF4
            quantize_and_add_int4(name);
            return;
        }

        Tensor bf16 = loader.load_tensor(name);
        QuantizedWeight qw;
        bool nf4 = !cfg.linear_int4;

        if (cfg.smooth && cal_reader.version == 1) {
            // v1 data is FWHT-rotated — apply FWHT again to unrotate (FWHT is self-inverse)
            int hbs = hadamard_block_size(cal_K);
            if (hbs > 1)
                fwht_inplace(x_gpu, cal_M, cal_K, hbs);
            CUDA_CHECK(cudaDeviceSynchronize());
            // Now x_gpu contains (approximately) raw activations → smooth + GPTQ
            qw = quantize_weight_tensor_int4_gptq_smooth(bf16, cfg.rank, cfg.group_size,
                                                           x_gpu, cal_M, cal_K);
        } else if (cfg.smooth && cal_reader.version == 2) {
            // v2: raw activations → smooth + GPTQ directly
            qw = quantize_weight_tensor_int4_gptq_smooth(bf16, cfg.rank, cfg.group_size,
                                                           x_gpu, cal_M, cal_K);
        } else if (cal_reader.version == 2) {
            // v2 without smooth: apply FWHT to raw activations for standard GPTQ
            int hbs = hadamard_block_size(cal_K);
            if (!cfg.no_hadamard && hbs > 1)
                fwht_inplace(x_gpu, cal_M, cal_K, hbs);
            CUDA_CHECK(cudaDeviceSynchronize());
            qw = quantize_weight_tensor_int4_gptq(bf16, cfg.rank, cfg.group_size,
                                                    x_gpu, cal_M, cal_K,
                                                    cfg.no_hadamard, nf4, cfg.error_svd);
        } else {
            // v1: FWHT-rotated activations → GPTQ
            // If no_hadamard: unrotate the v1 data first, then quantize without Hadamard
            if (cfg.no_hadamard) {
                int hbs = hadamard_block_size(cal_K);
                if (hbs > 1)
                    fwht_inplace(x_gpu, cal_M, cal_K, hbs);
                CUDA_CHECK(cudaDeviceSynchronize());
            }
            qw = quantize_weight_tensor_int4_gptq(bf16, cfg.rank, cfg.group_size,
                                                    x_gpu, cal_M, cal_K,
                                                    cfg.no_hadamard, nf4, cfg.error_svd);
        }
        bf16.free_data();
        CUDA_CHECK(cudaFree(x_gpu));

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

        // Write smooth factors if present
        if (qw.smooth.data != nullptr) {
            auto sm_buf = download_tensor(qw.smooth);
            writer.add(name + ".__smooth__", "F32", {(int64_t)K_full},
                       sm_buf.data(), sm_buf.size());
        }

        // Write linear INT4 marker (absence = NF4 for backward compat)
        if (!qw.nf4_grid) {
            uint8_t marker = 1;
            writer.add(name + ".__linear_int4__", "U8", {1}, &marker, 1);
        }

        // Write no-Hadamard marker when had_block_size == 1
        if (qw.had_block_size <= 1 && K_full > 1) {
            uint8_t marker = 1;
            writer.add(name + ".__no_hadamard__", "U8", {1}, &marker, 1);
        }

        qw.free_data();
    };

    // Dispatch to the right quantization function
    auto quantize_weight = [&](const std::string& name) {
        if (cfg.int4 && !cfg.gptq_cal.empty())
            quantize_and_add_int4_gptq(name);
        else if (cfg.int4)
            quantize_and_add_int4(name);
        else
            quantize_and_add(name);
    };

    // Write a single projection's INT4 tensors from a QuantizedWeight that may span
    // multiple fused projections.  proj_idx selects the [N, ...] row slice.
    auto write_int4_slice = [&](const std::string& name, const QuantizedWeight& qw,
                                int N, int proj_idx) {
        int K_packed = (int)qw.qweight.shape[1];
        int num_groups = (int)qw.scales4.shape[1];
        int r = qw.svd_rank;
        int K_full = K_packed * 2;

        auto qw_buf = download_tensor(qw.qweight);
        auto sc_buf = download_tensor(qw.scales4);
        auto up_buf = download_tensor(qw.svd_up);
        auto dn_buf = download_tensor(qw.svd_down);

        size_t qw_slice = (size_t)N * K_packed;     // U8
        size_t sc_slice = (size_t)N * num_groups * 4; // F32
        size_t up_slice = (size_t)N * r * 2;          // BF16

        writer.add(name + ".__qweight__", "U8", {(int64_t)N, (int64_t)K_packed},
                   qw_buf.data() + (size_t)proj_idx * qw_slice, qw_slice);
        writer.add(name + ".__scales4__", "F32", {(int64_t)N, (int64_t)num_groups},
                   sc_buf.data() + (size_t)proj_idx * sc_slice, sc_slice);
        writer.add(name + ".__svd_up__", "BF16", {(int64_t)N, (int64_t)r},
                   up_buf.data() + (size_t)proj_idx * up_slice, up_slice);
        // svd_down is shared (same for all projections)
        writer.add(name + ".__svd_down__", "BF16", {(int64_t)K_full, (int64_t)r},
                   dn_buf.data(), dn_buf.size());

        if (!qw.nf4_grid) {
            uint8_t marker = 1;
            writer.add(name + ".__linear_int4__", "U8", {1}, &marker, 1);
        }
        if (qw.had_block_size <= 1 && K_full > 1) {
            uint8_t marker = 1;
            writer.add(name + ".__no_hadamard__", "U8", {1}, &marker, 1);
        }
        if (qw.smooth.data != nullptr) {
            auto sm_buf = download_tensor(qw.smooth);
            writer.add(name + ".__smooth__", "F32", {(int64_t)K_full},
                       sm_buf.data(), sm_buf.size());
        }
    };

    // Fused QKV quantization: concatenate 3 weight matrices into [3N, K],
    // run joint SVD+GPTQ, then split results back into per-projection tensors.
    auto quantize_fused_qkv = [&](const std::string& name_q,
                                   const std::string& name_k,
                                   const std::string& name_v) {
        fprintf(stderr, "    [fused QKV] %s\n", name_q.c_str());
        Tensor wq = loader.load_tensor(name_q);
        Tensor wk = loader.load_tensor(name_k);
        Tensor wv = loader.load_tensor(name_v);

        int N = (int)wq.shape[0];
        int K = (int)wq.shape[1];
        int N_fused = N * 3;

        // Concatenate [N,K] x3 → [3N, K] on GPU
        Tensor fused = Tensor::alloc({(int64_t)N_fused, (int64_t)K}, DType::BF16);
        CUDA_CHECK(cudaMemcpy((char*)fused.data,
                              wq.data, wq.nbytes(), cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy((char*)fused.data + wq.nbytes(),
                              wk.data, wk.nbytes(), cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy((char*)fused.data + wq.nbytes() + wk.nbytes(),
                              wv.data, wv.nbytes(), cudaMemcpyDeviceToDevice));
        wq.free_data(); wk.free_data(); wv.free_data();

        // Get calibration data (shared by Q/K/V)
        std::string cal_name = get_cal_name(name_q);
        int cal_M = 0, cal_K = 0;
        float* x_gpu = nullptr;
        if (!cal_name.empty())
            x_gpu = cal_reader.get_activation_gpu(cal_name, cal_M, cal_K);

        QuantizedWeight qw;
        if (x_gpu) {
            bool nf4 = !cfg.linear_int4;
            // Handle calibration version / smooth / hadamard (same logic as quantize_and_add_int4_gptq)
            if (cfg.smooth && (cal_reader.version == 1 || cal_reader.version == 2)) {
                if (cal_reader.version == 1) {
                    int hbs = hadamard_block_size(cal_K);
                    if (hbs > 1) fwht_inplace(x_gpu, cal_M, cal_K, hbs);
                    CUDA_CHECK(cudaDeviceSynchronize());
                }
                qw = quantize_weight_tensor_int4_gptq_smooth(fused, cfg.rank, cfg.group_size,
                                                               x_gpu, cal_M, cal_K);
            } else if (cal_reader.version == 2) {
                int hbs = hadamard_block_size(cal_K);
                if (!cfg.no_hadamard && hbs > 1) fwht_inplace(x_gpu, cal_M, cal_K, hbs);
                CUDA_CHECK(cudaDeviceSynchronize());
                qw = quantize_weight_tensor_int4_gptq(fused, cfg.rank, cfg.group_size,
                                                        x_gpu, cal_M, cal_K,
                                                        cfg.no_hadamard, nf4, cfg.error_svd);
            } else {
                if (cfg.no_hadamard) {
                    int hbs = hadamard_block_size(cal_K);
                    if (hbs > 1) fwht_inplace(x_gpu, cal_M, cal_K, hbs);
                    CUDA_CHECK(cudaDeviceSynchronize());
                }
                qw = quantize_weight_tensor_int4_gptq(fused, cfg.rank, cfg.group_size,
                                                        x_gpu, cal_M, cal_K,
                                                        cfg.no_hadamard, nf4, cfg.error_svd);
            }
            CUDA_CHECK(cudaFree(x_gpu));
        } else {
            // No calibration → naive SVD+NF4 (still benefits from fused SVD)
            qw = quantize_weight_tensor_int4(fused, cfg.rank, cfg.group_size);
        }
        fused.free_data();

        // Split and write per-projection tensors
        write_int4_slice(name_q, qw, N, 0);
        write_int4_slice(name_k, qw, N, 1);
        write_int4_slice(name_v, qw, N, 2);
        qw.free_data();
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

        // QKV fusion: joint SVD+GPTQ on concatenated [3N, K] matrix
        if (cfg.int4 && !cfg.no_fuse_qkv) {
            quantize_fused_qkv(prefix + "attn.to_q.weight",
                               prefix + "attn.to_k.weight",
                               prefix + "attn.to_v.weight");
        } else {
            quantize_weight(prefix + "attn.to_q.weight");
            quantize_weight(prefix + "attn.to_k.weight");
            quantize_weight(prefix + "attn.to_v.weight");
        }
        passthrough(prefix + "attn.to_q.bias");
        passthrough(prefix + "attn.to_k.bias");
        passthrough(prefix + "attn.to_v.bias");

        passthrough(prefix + "attn.norm_q.weight");
        passthrough(prefix + "attn.norm_k.weight");

        // Fuse add_q/k/v projections too (text stream QKV)
        if (cfg.int4 && !cfg.no_fuse_qkv) {
            quantize_fused_qkv(prefix + "attn.add_q_proj.weight",
                               prefix + "attn.add_k_proj.weight",
                               prefix + "attn.add_v_proj.weight");
        } else {
            quantize_weight(prefix + "attn.add_q_proj.weight");
            quantize_weight(prefix + "attn.add_k_proj.weight");
            quantize_weight(prefix + "attn.add_v_proj.weight");
        }
        passthrough(prefix + "attn.add_q_proj.bias");
        passthrough(prefix + "attn.add_k_proj.bias");
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
