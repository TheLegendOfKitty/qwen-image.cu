#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cctype>
#include <algorithm>
#include <string>
#include <vector>
#include <chrono>

#include "tensor.h"
#include "safetensors.h"
#include "json_parser.h"
#include "tokenizer.h"
#include "text_encoder.h"
#include "transformer.h"
#include "vae_decoder.h"
#include "scheduler.h"
#include "rope.h"
#include "cuda_kernels.cuh"
#include "image_io.h"

struct Config {
    std::string prompt = "a red rose";
    std::string model_dir = "./Qwen-Image-2512/";
    std::string text_encoder;      // optional: single safetensors file for text encoder
    std::string transformer;       // optional: single safetensors file for transformer
    std::string vae;               // optional: single safetensors file for VAE
    std::string output = "output.ppm";
    std::string calibrate; // optional: output calibration file for GPTQ
    int width = 512;
    int height = 512;
    int steps = 20;
    float cfg_scale = 7.0f;
    unsigned long long seed = 42;
    float flow_shift = 0.0f; // 0 = dynamic shifting (diffusers default); >0 = fixed shift (3.0 for sd.cpp)
    bool legacy_cfg = false; // true = plain CFG (sd.cpp); false = norm-preserving CFG (diffusers)
};

static Config parse_args(int argc, char** argv) {
    Config cfg;
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if ((arg == "-p" || arg == "--prompt") && i + 1 < argc) {
            cfg.prompt = argv[++i];
        } else if ((arg == "-o" || arg == "--output") && i + 1 < argc) {
            cfg.output = argv[++i];
        } else if (arg == "--model-dir" && i + 1 < argc) {
            cfg.model_dir = argv[++i];
        } else if (arg == "--text-encoder" && i + 1 < argc) {
            cfg.text_encoder = argv[++i];
        } else if ((arg == "--transformer" || arg == "--quant-transformer") && i + 1 < argc) {
            cfg.transformer = argv[++i];
        } else if (arg == "--vae" && i + 1 < argc) {
            cfg.vae = argv[++i];
        } else if (arg == "--calibrate" && i + 1 < argc) {
            cfg.calibrate = argv[++i];
        } else if (arg == "--width" && i + 1 < argc) {
            cfg.width = atoi(argv[++i]);
        } else if (arg == "--height" && i + 1 < argc) {
            cfg.height = atoi(argv[++i]);
        } else if (arg == "--steps" && i + 1 < argc) {
            cfg.steps = atoi(argv[++i]);
        } else if (arg == "--cfg-scale" && i + 1 < argc) {
            cfg.cfg_scale = atof(argv[++i]);
        } else if (arg == "--seed" && i + 1 < argc) {
            cfg.seed = strtoull(argv[++i], nullptr, 10);
        } else if (arg == "--w4a4") {
            g_w4a4_mode = true;
        } else if (arg == "--legacy-cfg") {
            cfg.legacy_cfg = true;
        } else if (arg == "--flow-shift" && i + 1 < argc) {
            cfg.flow_shift = atof(argv[++i]);
        } else if (arg == "-h" || arg == "--help") {
            fprintf(stderr, "Usage: %s [options]\n", argv[0]);
            fprintf(stderr, "  -p, --prompt TEXT       Prompt text\n");
            fprintf(stderr, "  -o, --output FILE       Output file (PPM, BMP, or PNG)\n");
            fprintf(stderr, "  --model-dir DIR         Model directory\n");
            fprintf(stderr, "  --text-encoder FILE     Single safetensors for text encoder\n");
            fprintf(stderr, "  --transformer FILE      Single safetensors for transformer\n");
            fprintf(stderr, "  --vae FILE              Single safetensors for VAE\n");
            fprintf(stderr, "  --calibrate FILE        Capture calibration data for GPTQ (1 step)\n");
            fprintf(stderr, "  --width N               Image width (default: 512)\n");
            fprintf(stderr, "  --height N              Image height (default: 512)\n");
            fprintf(stderr, "  --steps N               Sampling steps (default: 20)\n");
            fprintf(stderr, "  --cfg-scale F           CFG scale (default: 7.0)\n");
            fprintf(stderr, "  --seed N                Random seed (default: 42)\n");
            fprintf(stderr, "  --flow-shift F          Fixed flow shift (0=dynamic, 3.0=sd.cpp)\n");
            exit(0);
        }
    }
    // Ensure model_dir ends with /
    if (!cfg.model_dir.empty() && cfg.model_dir.back() != '/')
        cfg.model_dir += '/';
    return cfg;
}

static bool has_suffix_ci(const std::string& s, const char* suffix) {
    size_t n = strlen(suffix);
    if (s.size() < n) return false;
    size_t off = s.size() - n;
    for (size_t i = 0; i < n; i++) {
        if (std::tolower((unsigned char)s[off + i]) != std::tolower((unsigned char)suffix[i])) {
            return false;
        }
    }
    return true;
}

// Latent normalization constants from VAE config
static const float latents_mean[16] = {
    -0.7571f, -0.7089f, -0.9113f, 0.1075f, -0.1745f, 0.9653f, -0.1517f, 1.5508f,
    0.4134f, -0.0715f, 0.5517f, -0.3632f, -0.1922f, -0.9497f, 0.2503f, -0.2921f
};

static const float latents_std[16] = {
    2.8184f, 1.4541f, 2.3275f, 2.6558f, 1.2196f, 1.7708f, 2.6052f, 2.0743f,
    3.2687f, 2.1526f, 2.8652f, 1.5579f, 1.6382f, 1.1253f, 2.8251f, 1.916f
};

// FP32 scale kernel
__global__ void scale_fp32_kernel(float* x, float s, int64_t n) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    x[idx] *= s;
}

// FP32 Euler step: x = x + (x - denoised) / sigma * dt
// All FP32
__global__ void euler_step_fp32_kernel(float* x, const float* denoised,
                                        float sigma, float dt, int64_t n) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float xv = x[idx];
    float dv = denoised[idx];
    float noise_dir = (xv - dv) / sigma;
    x[idx] = xv + noise_dir * dt;
}

// FP32 denoiser scaling: denoised_fp32 = x_fp32 - sigma * velocity_fp32
__global__ void denoiser_scaling_fp32_kernel(const float* velocity, const float* x,
                                              float* denoised, float sigma, int64_t n) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    denoised[idx] = x[idx] - sigma * velocity[idx];
}

// Legacy CFG: out = uncond + cfg_scale * (cond - uncond)
__global__ void cfg_combine_legacy_kernel(const __nv_bfloat16* cond, const __nv_bfloat16* uncond,
                                          float* out, float cfg_scale, int64_t n) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float c = __bfloat162float(cond[idx]);
    float u = __bfloat162float(uncond[idx]);
    out[idx] = u + cfg_scale * (c - u);
}

// Norm-preserving CFG combine (matches diffusers Qwen Image pipeline):
//   comb = uncond + cfg_scale * (cond - uncond)
//   cond_norm = ||cond||_dim=-1 per (b,c,h)
//   comb_norm = ||comb||_dim=-1 per (b,c,h)
//   out = comb * (cond_norm / comb_norm)
// Phase 1: compute combined CFG into out, accumulate cond/comb squared norms
// Norm-preserving CFG: norms computed in PACKED format grouping
// NCHW element at flat idx → packed group = (h/2)*(W/2) + (w/2)
// where c = idx/(H*W), h = (idx/W)%H, w = idx%W
// Each group has C*4 = 64 elements (matches diffusers dim=-1 on packed output)
__global__ void cfg_combine_and_norms_kernel(
    const __nv_bfloat16* cond, const __nv_bfloat16* uncond,
    float* out, float* cond_norm_sq, float* comb_norm_sq,
    float cfg_scale, int64_t n, int C, int H, int W)
{
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float c = __bfloat162float(cond[idx]);
    float u = __bfloat162float(uncond[idx]);
    float comb = u + cfg_scale * (c - u);
    out[idx] = comb;

    // Map NCHW flat index to packed sequence group
    int hw = H * W;
    int ch = (int)(idx / hw);    // channel (unused for group calc, just skip)
    int rem = (int)(idx % hw);
    int h = rem / W;
    int w = rem % W;
    int group = (h / 2) * (W / 2) + (w / 2);
    atomicAdd(&cond_norm_sq[group], c * c);
    atomicAdd(&comb_norm_sq[group], comb * comb);
}

// Phase 2: scale each element by cond_norm / comb_norm for its group
__global__ void cfg_norm_scale_kernel(
    float* out, const float* cond_norm_sq, const float* comb_norm_sq,
    int64_t n, int C, int H, int W)
{
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    int hw = H * W;
    int rem = (int)(idx % hw);
    int h = rem / W;
    int w = rem % W;
    int group = (h / 2) * (W / 2) + (w / 2);
    float cn = sqrtf(cond_norm_sq[group]);
    float nn = sqrtf(comb_norm_sq[group]);
    if (nn > 0.0f) {
        out[idx] *= cn / nn;
    }
}

// FP32 denormalize latent: x[c] = x[c] * std[c] + mean[c]
__global__ void denormalize_latent_fp32_kernel(float* x, const float* mean, const float* std_dev,
                                                int C, int spatial) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = (int64_t)C * spatial;
    if (idx >= total) return;
    int c = idx / spatial;
    x[idx] = x[idx] * std_dev[c] + mean[c];
}

// Generate FP32 Philox random numbers (CPU, matching sd.cpp)
static std::vector<float> randn_philox_fp32(int64_t n, uint64_t seed) {
    const uint32_t M0 = 0xD2511F53u;
    const uint32_t M1 = 0xCD9E8D57u;
    const uint32_t W0 = 0x9E3779B9u;
    const uint32_t W1 = 0xBB67AE85u;
    const float two_pow32_inv = 2.3283064e-10f;
    const float two_pow32_inv_2pi = two_pow32_inv * 6.2831855f;

    uint32_t key0 = (uint32_t)(seed & 0xFFFFFFFF);
    uint32_t key1 = (uint32_t)(seed >> 32);
    uint32_t offset = 0;

    std::vector<float> result(n);
    for (int64_t i = 0; i < n; i++) {
        uint32_t counter[4] = {offset, 0, (uint32_t)i, 0};
        uint32_t k0 = key0, k1 = key1;
        for (int round = 0; round < 10; round++) {
            uint64_t v1 = (uint64_t)counter[0] * M0;
            uint64_t v2 = (uint64_t)counter[2] * M1;
            uint32_t v1_hi = (uint32_t)(v1 >> 32), v1_lo = (uint32_t)v1;
            uint32_t v2_hi = (uint32_t)(v2 >> 32), v2_lo = (uint32_t)v2;
            counter[0] = v2_hi ^ counter[1] ^ k0;
            counter[1] = v2_lo;
            counter[2] = v1_hi ^ counter[3] ^ k1;
            counter[3] = v1_lo;
            k0 += W0; k1 += W1;
        }
        float u = (float)counter[0] * two_pow32_inv + two_pow32_inv / 2.0f;
        float v = (float)counter[1] * two_pow32_inv_2pi + two_pow32_inv_2pi / 2.0f;
        result[i] = sqrtf(-2.0f * logf(u)) * sinf(v);
    }
    return result;
}

int main(int argc, char** argv) {
    Config cfg = parse_args(argc, argv);

    fprintf(stderr, "=== Qwen Image 2512 - CUDA Implementation ===\n");
    fprintf(stderr, "Prompt: %s\n", cfg.prompt.c_str());
    fprintf(stderr, "Size: %dx%d, Steps: %d, CFG: %.1f, Seed: %llu\n",
            cfg.width, cfg.height, cfg.steps, cfg.cfg_scale, cfg.seed);

    auto t_start = std::chrono::high_resolution_clock::now();

    // ========== Step 1: Load text encoder file(s) ==========
    fprintf(stderr, "\n[1/8] Loading text encoder...\n");
    SafeTensorsLoader te_loader;
    if (!cfg.text_encoder.empty()) {
        fprintf(stderr, "  Using single file: %s\n", cfg.text_encoder.c_str());
        te_loader.load_file(cfg.text_encoder);
    } else {
        te_loader = load_model_dir(cfg.model_dir + "text_encoder");
    }
    te_loader.print_summary();

    // ========== Step 2: Load tokenizer ==========
    fprintf(stderr, "\n[2/8] Loading tokenizer...\n");
    Qwen2Tokenizer tokenizer;
    if (te_loader.has_tensor("tokenizer.vocab_json")) {
        tokenizer.load(te_loader);
    } else {
        tokenizer.load(cfg.model_dir + "tokenizer/vocab.json",
                       cfg.model_dir + "tokenizer/merges.txt");
    }

    // ========== Step 3: Tokenize & encode ==========
    fprintf(stderr, "\n[3/8] Tokenizing...\n");
    auto [cond_tokens, cond_strip_idx] = tokenizer.tokenize_prompt(cfg.prompt);
    auto [uncond_tokens, uncond_strip_idx] = tokenizer.tokenize_empty();
    fprintf(stderr, "  Cond tokens: %zu (strip first %d), Uncond tokens: %zu (strip first %d)\n",
            cond_tokens.size(), cond_strip_idx, uncond_tokens.size(), uncond_strip_idx);
    fprintf(stderr, "  Cond IDs:");
    for (auto t : cond_tokens) fprintf(stderr, " %d", t);
    fprintf(stderr, "\n  Uncond IDs:");
    for (auto t : uncond_tokens) fprintf(stderr, " %d", t);
    fprintf(stderr, "\n");

    TextEncoderWeights te_weights;
    te_weights.load(te_loader);

    fprintf(stderr, "  Encoding conditional prompt...\n");
    Tensor cond_context_full = text_encoder_forward(te_weights, cond_tokens);

    fprintf(stderr, "  Encoding unconditional prompt...\n");
    Tensor uncond_context_full = text_encoder_forward(te_weights, uncond_tokens);

    // Strip system prompt tokens from text encoder output
    // sd.cpp encodes the full prompt (system + user + assistant) but then removes
    // the first prompt_template_encode_start_idx tokens from the hidden states,
    // so only the user content + assistant tokens are passed to the diffusion model.
    const int hidden_size = 3584;
    auto strip_context = [&](Tensor& full, int strip_idx) -> Tensor {
        int full_seq = (int)full.shape[1];
        int new_seq = full_seq - strip_idx;
        fprintf(stderr, "  Stripping first %d tokens: [1, %d, %d] -> [1, %d, %d]\n",
                strip_idx, full_seq, hidden_size, new_seq, hidden_size);
        Tensor stripped = Tensor::alloc({1, (int64_t)new_seq, (int64_t)hidden_size}, DType::BF16);
        // Copy from offset strip_idx * hidden_size
        size_t offset_bytes = (size_t)strip_idx * hidden_size * sizeof(__nv_bfloat16);
        size_t copy_bytes = (size_t)new_seq * hidden_size * sizeof(__nv_bfloat16);
        CUDA_CHECK(cudaMemcpy(stripped.data, (char*)full.data + offset_bytes,
                              copy_bytes, cudaMemcpyDeviceToDevice));
        full.free_data();
        return stripped;
    };

    Tensor cond_context = strip_context(cond_context_full, cond_strip_idx);
    Tensor uncond_context = strip_context(uncond_context_full, uncond_strip_idx);

    // Override with external context dumps for debugging
    if (getenv("LOAD_SDCPP_CONTEXT")) {
        fprintf(stderr, "  LOADING EXTERNAL CONTEXT from sd.cpp dumps!\n");
        auto load_f32_as_bf16 = [&](const char* path, int64_t expected_n) -> Tensor {
            std::vector<float> fp(expected_n);
            FILE* f = fopen(path, "rb");
            if (!f) { fprintf(stderr, "ERROR: cannot open %s\n", path); exit(1); }
            fread(fp.data(), sizeof(float), expected_n, f);
            fclose(f);
            // Convert to BF16 and upload
            std::vector<__nv_bfloat16> bf(expected_n);
            for (int64_t i = 0; i < expected_n; i++) bf[i] = __float2bfloat16(fp[i]);
            int seq = (int)(expected_n / hidden_size);
            Tensor t = Tensor::alloc({1, (int64_t)seq, (int64_t)hidden_size}, DType::BF16);
            CUDA_CHECK(cudaMemcpy(t.data, bf.data(), expected_n * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
            fprintf(stderr, "    Loaded %s: [1, %d, %d]\n", path, seq, hidden_size);
            return t;
        };
        cond_context.free_data();
        uncond_context.free_data();
        cond_context = load_f32_as_bf16("sdcpp_cond_context.bin", 8 * hidden_size);
        uncond_context = load_f32_as_bf16("sdcpp_uncond_context.bin", 5 * hidden_size);
    }

    // Load diffusers text encoder embeddings for matching comparison
    if (getenv("LOAD_DIFFUSERS_EMBED")) {
        fprintf(stderr, "  LOADING DIFFUSERS EMBEDDINGS!\n");
        auto load_embed = [&](const char* path) -> Tensor {
            FILE* f = fopen(path, "rb");
            if (!f) { fprintf(stderr, "ERROR: cannot open %s\n", path); exit(1); }
            int32_t hdr[2];
            fread(hdr, sizeof(int32_t), 2, f);
            int seq = hdr[0], dim = hdr[1];
            int64_t n = (int64_t)seq * dim;
            std::vector<float> fp(n);
            fread(fp.data(), sizeof(float), n, f);
            fclose(f);
            // Convert FP32 → BF16 and upload
            std::vector<__nv_bfloat16> bf(n);
            for (int64_t i = 0; i < n; i++) bf[i] = __float2bfloat16(fp[i]);
            Tensor t = Tensor::alloc({1, (int64_t)seq, (int64_t)dim}, DType::BF16);
            CUDA_CHECK(cudaMemcpy(t.data, bf.data(), n * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
            fprintf(stderr, "    Loaded %s: [1, %d, %d]\n", path, seq, dim);
            return t;
        };
        cond_context.free_data();
        uncond_context.free_data();
        cond_context = load_embed("cond_embed.bin");
        uncond_context = load_embed("uncond_embed.bin");
    }

    // Optionally dump text encoder outputs for verification
    if (getenv("DUMP_CONTEXT")) {
        auto dump_bf16 = [](const char* path, void* data, int64_t n) {
            std::vector<__nv_bfloat16> h(n);
            std::vector<float> fp(n);
            CUDA_CHECK(cudaMemcpy(h.data(), data, n * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
            for (int64_t i = 0; i < n; i++) fp[i] = __bfloat162float(h[i]);
            FILE* f = fopen(path, "wb");
            if (f) { fwrite(fp.data(), sizeof(float), n, f); fclose(f); }
            fprintf(stderr, "    Dumped %s (%ld floats)\n", path, (long)n);
        };
        dump_bf16("cuda_cond_context.bin", cond_context.data, cond_context.numel());
        dump_bf16("cuda_uncond_context.bin", uncond_context.data, uncond_context.numel());
    }

    // Free text encoder
    fprintf(stderr, "  Freeing text encoder...\n");
    te_weights.free_all();

    // ========== Step 4: Load transformer ==========
    fprintf(stderr, "\n[4/8] Loading transformer...\n");
    SafeTensorsLoader tf_loader;
    if (!cfg.transformer.empty()) {
        fprintf(stderr, "  Using: %s\n", cfg.transformer.c_str());
        tf_loader.load_file(cfg.transformer);
    } else {
        tf_loader = load_model_dir(cfg.model_dir + "transformer");
    }
    tf_loader.print_summary();

    TransformerWeights tf_weights;
    tf_weights.load(tf_loader);

    // ========== Step 5: Compute RoPE PE ==========
    fprintf(stderr, "\n[5/8] Computing RoPE positional embeddings...\n");
    int latent_h = cfg.height / 8;
    int latent_w = cfg.width / 8;
    int patch_size = 2;
    // Use stripped context sizes (after removing system prompt tokens)
    int n_txt = (int)cond_context.shape[1];
    int h_patches = (latent_h + (patch_size / 2)) / patch_size;
    int w_patches = (latent_w + (patch_size / 2)) / patch_size;
    int n_img = h_patches * w_patches;
    int total_pos = n_txt + n_img;

    auto pe_vec = RoPE::gen_qwen_image_pe(latent_h, latent_w, patch_size, 1, n_txt,
                                            10000, {16, 56, 56});
    // pe shape: [total_pos, 64, 2, 2] = [total_pos, 256]
    Tensor pe = Tensor::alloc({(int64_t)total_pos, 64, 2, 2}, DType::FP32);
    pe.from_host(pe_vec.data(), pe_vec.size() * sizeof(float));
    fprintf(stderr, "  PE shape: [%d, 64, 2, 2] (n_txt=%d, n_img=%d)\n", total_pos, n_txt, n_img);

    // Also compute PE for uncond (different seq_len potentially)
    int n_txt_uncond = (int)uncond_context.shape[1];
    int total_pos_uncond = n_txt_uncond + n_img;
    auto pe_uncond_vec = RoPE::gen_qwen_image_pe(latent_h, latent_w, patch_size, 1, n_txt_uncond,
                                                    10000, {16, 56, 56});
    Tensor pe_uncond = Tensor::alloc({(int64_t)total_pos_uncond, 64, 2, 2}, DType::FP32);
    pe_uncond.from_host(pe_uncond_vec.data(), pe_uncond_vec.size() * sizeof(float));

    // ========== Step 6: Initialize latent and run denoising ==========
    fprintf(stderr, "\n[6/8] Denoising (%d steps)...\n", cfg.steps);

    // Initialize latent noise in FP32 (matching sd.cpp precision)
    int latent_channels = 16;
    int64_t latent_numel = (int64_t)latent_channels * latent_h * latent_w;
    Tensor latent = Tensor::alloc({1, (int64_t)latent_channels, (int64_t)latent_h, (int64_t)latent_w}, DType::FP32);
    {
        // Generate FP32 noise on CPU via Philox, upload as FP32
        std::vector<float> noise_fp32 = randn_philox_fp32(latent_numel, cfg.seed);
        CUDA_CHECK(cudaMemcpy(latent.data, noise_fp32.data(), latent_numel * sizeof(float), cudaMemcpyHostToDevice));
    }

    // Setup scheduler
    FlowMatchScheduler scheduler;
    if (cfg.flow_shift > 0.0f) {
        // Fixed shift mode (e.g., --flow-shift 3.0 for sd.cpp compatibility)
        scheduler.use_dynamic_shifting = false;
        scheduler.shift = cfg.flow_shift;
    }
    // else: default dynamic shifting (matches diffusers Qwen Image pipeline)
    auto sigmas = scheduler.get_sigmas(cfg.steps, n_img);

    // Scale initial noise by sigma_max (FP32)
    {
        int block = 256;
        int grid = (int)((latent_numel + block - 1) / block);
        scale_fp32_kernel<<<grid, block>>>((float*)latent.data, sigmas[0], latent_numel);
    }

    // Upload latent stats to GPU
    float* d_mean;
    float* d_std;
    CUDA_CHECK(cudaMalloc(&d_mean, 16 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_std, 16 * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_mean, latents_mean, 16 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_std, latents_std, 16 * sizeof(float), cudaMemcpyHostToDevice));

    // Calibration mode: run 1 denoising step capturing activations, then exit
    CalibrationWriter cal_writer;
    CalibrationWriter* cal_ptr = nullptr;
    if (!cfg.calibrate.empty()) {
        if (!cal_writer.open(cfg.calibrate.c_str())) {
            fprintf(stderr, "ERROR: cannot open calibration file: %s\n", cfg.calibrate.c_str());
            return 1;
        }
        cal_ptr = &cal_writer;
        fprintf(stderr, "\n*** CALIBRATION MODE: capturing activations to %s ***\n", cfg.calibrate.c_str());
    }

    // Denoising loop (latent stays in FP32 throughout, matching sd.cpp)
    // BF16 copy of latent for transformer input
    Tensor latent_bf16 = Tensor::alloc({1, (int64_t)latent_channels, (int64_t)latent_h, (int64_t)latent_w}, DType::BF16);
    Tensor denoised_cond;
    Tensor denoised_uncond;

    int actual_steps = cal_ptr ? 1 : cfg.steps; // calibrate only needs 1 step
    for (int step = 0; step < actual_steps; step++) {
        float sigma = sigmas[step];
        float sigma_next = sigmas[step + 1];
        float timestep = sigma * 1000.0f;

        fprintf(stderr, "  Step %d/%d: sigma=%.4f, timestep=%.1f\n",
                step + 1, actual_steps, sigma, timestep);

        auto t_step = std::chrono::high_resolution_clock::now();

        // Convert FP32 latent to BF16 for transformer
        fp32_to_bf16((float*)latent.data, (__nv_bfloat16*)latent_bf16.data, latent_numel);

        // Activate transformer dump for step 0 cond pass
        if (step == 0 && getenv("DUMP_TRANSFORMER")) {
            setenv("DUMP_TRANSFORMER_ACTIVE", "1", 1);
            setenv("DUMP_TRANSFORMER_ACTIVE_STEP", "0", 1);
        }

        // Run transformer for conditional
        denoised_cond = transformer_forward(tf_weights, latent_bf16, timestep, cond_context, pe, latent_h, latent_w, cal_ptr);

        // Deactivate transformer dump after cond pass
        if (step == 0 && getenv("DUMP_TRANSFORMER")) {
            unsetenv("DUMP_TRANSFORMER_ACTIVE");
        }

        // Run transformer for unconditional
        denoised_uncond = transformer_forward(tf_weights, latent_bf16, timestep, uncond_context, pe_uncond, latent_h, latent_w, cal_ptr);

        int64_t n = denoised_cond.numel();
        // Dump raw cond/uncond outputs for comparison
        if (step == 0 && getenv("DUMP_PIPELINE")) {
            auto dump_gpu_f32 = [](const char* path, const void* data, int64_t n) {
                std::vector<float> buf(n);
                cudaMemcpy(buf.data(), data, n * sizeof(float), cudaMemcpyDeviceToHost);
                FILE* f = fopen(path, "wb"); fwrite(buf.data(), 4, n, f); fclose(f);
                double sum = 0; for(auto v : buf) sum += v;
                fprintf(stderr, "  PIPE: %s [%ld] mean=%.6f first5=%.4f %.4f %.4f %.4f %.4f\n",
                    path, (long)n, sum/n, buf[0], buf[1], buf[2], buf[3], buf[4]);
            };
            auto dump_gpu_bf16 = [](const char* path, const void* data, int64_t n) {
                std::vector<__nv_bfloat16> bf(n);
                cudaMemcpy(bf.data(), data, n * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
                std::vector<float> buf(n);
                for(int64_t i=0;i<n;i++) buf[i]=__bfloat162float(bf[i]);
                FILE* f = fopen(path, "wb"); fwrite(buf.data(), 4, n, f); fclose(f);
                double sum = 0; for(auto v : buf) sum += v;
                fprintf(stderr, "  PIPE: %s [%ld] mean=%.6f first5=%.4f %.4f %.4f %.4f %.4f\n",
                    path, (long)n, sum/n, buf[0], buf[1], buf[2], buf[3], buf[4]);
            };
            dump_gpu_bf16("pipe_velocity_cond.bin", denoised_cond.data, n);
            dump_gpu_bf16("pipe_velocity_uncond.bin", denoised_uncond.data, n);
            dump_gpu_f32("pipe_latent_fp32.bin", latent.data, n);
        }

        // CFG combine on raw model outputs (velocities) → FP32 result
        Tensor velocity_fp32 = Tensor::alloc({1, (int64_t)latent_channels, (int64_t)latent_h, (int64_t)latent_w}, DType::FP32);
        {
            int block = 256;
            int grid = (int)((n + block - 1) / block);

            if (cfg.legacy_cfg) {
                // Plain CFG (matches sd.cpp)
                cfg_combine_legacy_kernel<<<grid, block>>>(
                    (__nv_bfloat16*)denoised_cond.data,
                    (__nv_bfloat16*)denoised_uncond.data,
                    (float*)velocity_fp32.data,
                    cfg.cfg_scale, n);
            } else {
                // Norm-preserving CFG (matches diffusers Qwen Image pipeline)
                // Norms in packed format: groups = (H/2)*(W/2) patches, each has C*4 elements
                int64_t n_groups = (int64_t)(latent_h / 2) * (latent_w / 2);
                float* d_cond_norm_sq;
                float* d_comb_norm_sq;
                CUDA_CHECK(cudaMalloc(&d_cond_norm_sq, n_groups * sizeof(float)));
                CUDA_CHECK(cudaMalloc(&d_comb_norm_sq, n_groups * sizeof(float)));
                CUDA_CHECK(cudaMemset(d_cond_norm_sq, 0, n_groups * sizeof(float)));
                CUDA_CHECK(cudaMemset(d_comb_norm_sq, 0, n_groups * sizeof(float)));

                cfg_combine_and_norms_kernel<<<grid, block>>>(
                    (__nv_bfloat16*)denoised_cond.data,
                    (__nv_bfloat16*)denoised_uncond.data,
                    (float*)velocity_fp32.data,
                    d_cond_norm_sq, d_comb_norm_sq,
                    cfg.cfg_scale, n, latent_channels, latent_h, latent_w);

                cfg_norm_scale_kernel<<<grid, block>>>(
                    (float*)velocity_fp32.data,
                    d_cond_norm_sq, d_comb_norm_sq,
                    n, latent_channels, latent_h, latent_w);

                CUDA_CHECK(cudaFree(d_cond_norm_sq));
                CUDA_CHECK(cudaFree(d_comb_norm_sq));
            }
        }
        denoised_uncond.free_data();

        // Apply denoiser scaling in FP32: denoised = x - sigma * velocity
        // velocity_fp32 holds the CFG-combined velocity, overwrite with denoised
        Tensor denoised_fp32 = Tensor::alloc({1, (int64_t)latent_channels, (int64_t)latent_h, (int64_t)latent_w}, DType::FP32);
        {
            int block = 256;
            int grid = (int)((n + block - 1) / block);
            denoiser_scaling_fp32_kernel<<<grid, block>>>(
                (float*)velocity_fp32.data,
                (float*)latent.data,
                (float*)denoised_fp32.data,
                sigma, n);
        }

        if (step == 0 && getenv("DUMP_PIPELINE")) {
            auto dump_gpu_f32 = [](const char* path, const void* data, int64_t n) {
                std::vector<float> buf(n);
                cudaMemcpy(buf.data(), data, n * sizeof(float), cudaMemcpyDeviceToHost);
                FILE* f = fopen(path, "wb"); fwrite(buf.data(), 4, n, f); fclose(f);
                double sum = 0; for(auto v : buf) sum += v;
                fprintf(stderr, "  PIPE: %s [%ld] mean=%.6f first5=%.4f %.4f %.4f %.4f %.4f\n",
                    path, (long)n, sum/n, buf[0], buf[1], buf[2], buf[3], buf[4]);
            };
            dump_gpu_f32("pipe_velocity_cfg.bin", velocity_fp32.data, n);
            dump_gpu_f32("pipe_denoised.bin", denoised_fp32.data, n);
        }

        velocity_fp32.free_data();
        denoised_cond.free_data();

        // Euler step in FP32: x = x + (x - denoised) / sigma * dt
        float dt = sigma_next - sigma;
        {
            int block = 256;
            int grid = (int)((n + block - 1) / block);
            euler_step_fp32_kernel<<<grid, block>>>(
                (float*)latent.data,
                (float*)denoised_fp32.data,
                sigma, dt, n);
        }

        denoised_fp32.free_data();

        auto t_step_end = std::chrono::high_resolution_clock::now();
        float ms = std::chrono::duration<float, std::milli>(t_step_end - t_step).count();
        fprintf(stderr, "    Step time: %.0f ms\n", ms);
    }
    latent_bf16.free_data();

    // Calibration mode: close file and exit early
    if (cal_ptr) {
        cal_writer.close();
        auto t_end = std::chrono::high_resolution_clock::now();
        float secs = std::chrono::duration<float>(t_end - t_start).count();
        fprintf(stderr, "\nCalibration done! %u entries written to %s (%.1f seconds)\n",
                cal_writer.num_entries, cfg.calibrate.c_str(), secs);
        tf_weights.free_all();
        pe.free_data(); pe_uncond.free_data();
        cond_context.free_data(); uncond_context.free_data();
        latent.free_data();
        CUDA_CHECK(cudaFree(d_mean)); CUDA_CHECK(cudaFree(d_std));
        return 0;
    }

    // Free transformer
    fprintf(stderr, "\n  Freeing transformer...\n");
    tf_weights.free_all();
    pe.free_data();
    pe_uncond.free_data();
    cond_context.free_data();
    uncond_context.free_data();

    // ========== Step 7: VAE decode ==========
    fprintf(stderr, "\n[7/8] VAE decoding...\n");

    // Denormalize latent in FP32: x[c] = x[c] * std[c] + mean[c]
    {
        int spatial = latent_h * latent_w;
        int total = latent_channels * spatial;
        int block = 256;
        int grid = (total + block - 1) / block;
        denormalize_latent_fp32_kernel<<<grid, block>>>(
            (float*)latent.data, d_mean, d_std, latent_channels, spatial);
    }
    CUDA_CHECK(cudaFree(d_mean));
    CUDA_CHECK(cudaFree(d_std));

    // Convert FP32 latent to BF16 for VAE
    Tensor latent_bf16_vae = Tensor::alloc({1, (int64_t)latent_channels, (int64_t)latent_h, (int64_t)latent_w}, DType::BF16);
    fp32_to_bf16((float*)latent.data, (__nv_bfloat16*)latent_bf16_vae.data, latent_numel);
    latent.free_data();

    // Load VAE
    SafeTensorsLoader vae_loader;
    if (!cfg.vae.empty()) {
        fprintf(stderr, "  Using: %s\n", cfg.vae.c_str());
        vae_loader.load_file(cfg.vae);
    } else {
        vae_loader = load_model_dir(cfg.model_dir + "vae");
    }
    vae_loader.print_summary();

    VAEDecoderWeights vae_weights;
    vae_weights.load(vae_loader);

    // Reshape latent to [16, 1, H/8, W/8] for WAN VAE
    Tensor latent_4d = latent_bf16_vae.view({(int64_t)latent_channels, 1, (int64_t)latent_h, (int64_t)latent_w});

    Tensor rgb = vae_decode(vae_weights, latent_4d, latent_h, latent_w);
    latent_bf16_vae.free_data();
    vae_weights.free_all();

    // ========== Step 8: Save image ==========
    fprintf(stderr, "\n[8/8] Saving image to %s...\n", cfg.output.c_str());

    // Convert BF16 [3, 1, H, W] -> uint8 [H, W, 3]
    // First reshape to [3, H, W]
    int out_H = cfg.height;
    int out_W = cfg.width;

    // Allocate output pixels on GPU and CPU
    uint8_t* d_pixels;
    CUDA_CHECK(cudaMalloc(&d_pixels, out_H * out_W * 3));
    bf16_to_rgb8((__nv_bfloat16*)rgb.data, d_pixels, out_H, out_W);
    rgb.free_data();

    std::vector<uint8_t> pixels(out_H * out_W * 3);
    CUDA_CHECK(cudaMemcpy(pixels.data(), d_pixels, out_H * out_W * 3, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_pixels));

    // Write file
    bool ok;
    if (has_suffix_ci(cfg.output, ".bmp")) {
        ok = write_bmp(cfg.output, pixels.data(), out_W, out_H);
    } else if (has_suffix_ci(cfg.output, ".png")) {
        ok = write_png(cfg.output, pixels.data(), out_W, out_H);
    } else {
        ok = write_ppm(cfg.output, pixels.data(), out_W, out_H);
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    float total_ms = std::chrono::duration<float, std::milli>(t_end - t_start).count();

    if (ok) {
        fprintf(stderr, "\nDone! Image saved to %s (%.1f seconds total)\n", cfg.output.c_str(), total_ms / 1000.0f);
    } else {
        fprintf(stderr, "\nError saving image!\n");
        return 1;
    }

    return 0;
}
