#include "vae_decoder.h"
#include "cuda_kernels.cuh"
#include <cstdio>
#include <cmath>

// Helper to load CausalConv3d weights
static CausalConv3dWeights load_causal_conv3d(const SafeTensorsLoader& loader, const std::string& prefix,
                                                int in_ch, int out_ch, int kT, int kH, int kW,
                                                int padH, int padW,
                                                int sT = 1, int sH = 1, int sW = 1) {
    CausalConv3dWeights w;
    w.weight = loader.load_tensor(prefix + "weight");
    if (loader.has_tensor(prefix + "bias"))
        w.bias = loader.load_tensor(prefix + "bias");
    w.in_channels = in_ch;
    w.out_channels = out_ch;
    w.kT = kT; w.kH = kH; w.kW = kW;
    w.padH = padH; w.padW = padW;
    w.strideT = sT; w.strideH = sH; w.strideW = sW;
    return w;
}

static ResBlockWeights load_resblock(const SafeTensorsLoader& loader, const std::string& prefix,
                                      int in_dim, int out_dim) {
    ResBlockWeights rb;
    rb.in_dim = in_dim;
    rb.out_dim = out_dim;
    rb.has_shortcut = (in_dim != out_dim);

    rb.norm1_gamma = loader.load_tensor(prefix + "norm1.gamma");
    rb.conv1 = load_causal_conv3d(loader, prefix + "conv1.", in_dim, out_dim, 3, 3, 3, 1, 1);
    rb.norm2_gamma = loader.load_tensor(prefix + "norm2.gamma");
    rb.conv2 = load_causal_conv3d(loader, prefix + "conv2.", out_dim, out_dim, 3, 3, 3, 1, 1);

    if (rb.has_shortcut) {
        rb.shortcut = load_causal_conv3d(loader, prefix + "conv_shortcut.", in_dim, out_dim, 1, 1, 1, 0, 0);
    }

    return rb;
}

void VAEDecoderWeights::load(const SafeTensorsLoader& loader) {
    fprintf(stderr, "Loading VAE decoder weights...\n");

    // post_quant_conv: CausalConv3d(16, 16, 1,1,1) - identity-like
    conv2 = load_causal_conv3d(loader, "post_quant_conv.", 16, 16, 1, 1, 1, 0, 0);

    // decoder.conv_in: CausalConv3d(16, 384, 3,3,3)
    conv1 = load_causal_conv3d(loader, "decoder.conv_in.", 16, 384, 3, 3, 3, 1, 1);

    // Middle blocks
    middle_0 = load_resblock(loader, "decoder.mid_block.resnets.0.", 384, 384);
    middle_2 = load_resblock(loader, "decoder.mid_block.resnets.1.", 384, 384);

    // Attention block
    middle_1.dim = 384;
    middle_1.norm_gamma = loader.load_tensor("decoder.mid_block.attentions.0.norm.gamma");
    middle_1.to_qkv_weight = loader.load_tensor("decoder.mid_block.attentions.0.to_qkv.weight");
    middle_1.to_qkv_bias = loader.load_tensor("decoder.mid_block.attentions.0.to_qkv.bias");
    middle_1.proj_weight = loader.load_tensor("decoder.mid_block.attentions.0.proj.weight");
    middle_1.proj_bias = loader.load_tensor("decoder.mid_block.attentions.0.proj.bias");

    // Up blocks: 4 stages with resnets + upsamplers
    // up_blocks.0: 3x ResBlock(384,384) + upsamplers.0 (upsample3d, 384->192)
    // up_blocks.1: ResBlock(192,384) + 2x ResBlock(384,384) + upsamplers.0 (upsample3d, 384->192)
    // up_blocks.2: 3x ResBlock(192,192) + upsamplers.0 (upsample2d, 192->96)
    // up_blocks.3: 3x ResBlock(96,96), no upsamplers

    // Block 0
    {
        std::string bp = "decoder.up_blocks.0.";
        for (int j = 0; j < 3; j++) {
            UpsampleBlock ub;
            ub.type = UpsampleBlock::RESBLOCK;
            ub.res = load_resblock(loader, bp + "resnets." + std::to_string(j) + ".", 384, 384);
            upsamples.push_back(std::move(ub));
        }
        UpsampleBlock ub;
        ub.type = UpsampleBlock::RESAMPLE;
        ub.resample.mode = "upsample3d";
        ub.resample.dim = 384;
        ub.resample.conv_weight = loader.load_tensor(bp + "upsamplers.0.resample.1.weight");
        ub.resample.conv_bias = loader.load_tensor(bp + "upsamplers.0.resample.1.bias");
        ub.resample.has_time_conv = loader.has_tensor(bp + "upsamplers.0.time_conv.weight");
        if (ub.resample.has_time_conv) {
            ub.resample.time_conv = load_causal_conv3d(loader, bp + "upsamplers.0.time_conv.",
                384, 384 * 2, 3, 1, 1, 0, 0);
        }
        upsamples.push_back(std::move(ub));
    }

    // Block 1
    {
        std::string bp = "decoder.up_blocks.1.";
        int in_dim = 192;
        for (int j = 0; j < 3; j++) {
            UpsampleBlock ub;
            ub.type = UpsampleBlock::RESBLOCK;
            int out_dim = 384;
            ub.res = load_resblock(loader, bp + "resnets." + std::to_string(j) + ".", in_dim, out_dim);
            upsamples.push_back(std::move(ub));
            in_dim = out_dim;
        }
        UpsampleBlock ub;
        ub.type = UpsampleBlock::RESAMPLE;
        ub.resample.mode = "upsample3d";
        ub.resample.dim = 384;
        ub.resample.conv_weight = loader.load_tensor(bp + "upsamplers.0.resample.1.weight");
        ub.resample.conv_bias = loader.load_tensor(bp + "upsamplers.0.resample.1.bias");
        ub.resample.has_time_conv = loader.has_tensor(bp + "upsamplers.0.time_conv.weight");
        if (ub.resample.has_time_conv) {
            ub.resample.time_conv = load_causal_conv3d(loader, bp + "upsamplers.0.time_conv.",
                384, 384 * 2, 3, 1, 1, 0, 0);
        }
        upsamples.push_back(std::move(ub));
    }

    // Block 2
    {
        std::string bp = "decoder.up_blocks.2.";
        for (int j = 0; j < 3; j++) {
            UpsampleBlock ub;
            ub.type = UpsampleBlock::RESBLOCK;
            ub.res = load_resblock(loader, bp + "resnets." + std::to_string(j) + ".", 192, 192);
            upsamples.push_back(std::move(ub));
        }
        UpsampleBlock ub;
        ub.type = UpsampleBlock::RESAMPLE;
        ub.resample.mode = "upsample2d";
        ub.resample.dim = 192;
        ub.resample.conv_weight = loader.load_tensor(bp + "upsamplers.0.resample.1.weight");
        ub.resample.conv_bias = loader.load_tensor(bp + "upsamplers.0.resample.1.bias");
        ub.resample.has_time_conv = false;
        upsamples.push_back(std::move(ub));
    }

    // Block 3 (no upsampler)
    {
        std::string bp = "decoder.up_blocks.3.";
        for (int j = 0; j < 3; j++) {
            UpsampleBlock ub;
            ub.type = UpsampleBlock::RESBLOCK;
            ub.res = load_resblock(loader, bp + "resnets." + std::to_string(j) + ".", 96, 96);
            upsamples.push_back(std::move(ub));
        }
    }

    // Head
    head_norm_gamma = loader.load_tensor("decoder.norm_out.gamma");
    head_conv = load_causal_conv3d(loader, "decoder.conv_out.", 96, 3, 3, 3, 3, 1, 1);

    fprintf(stderr, "VAE decoder loaded: %d upsample blocks\n", (int)upsamples.size());
}

void VAEDecoderWeights::free_all() {
    conv2.weight.free_data(); conv2.bias.free_data();
    conv1.weight.free_data(); conv1.bias.free_data();
    // Middle
    middle_0.norm1_gamma.free_data(); middle_0.norm2_gamma.free_data();
    middle_0.conv1.weight.free_data(); middle_0.conv1.bias.free_data();
    middle_0.conv2.weight.free_data(); middle_0.conv2.bias.free_data();
    if (middle_0.has_shortcut) { middle_0.shortcut.weight.free_data(); middle_0.shortcut.bias.free_data(); }
    middle_1.norm_gamma.free_data();
    middle_1.to_qkv_weight.free_data(); middle_1.to_qkv_bias.free_data();
    middle_1.proj_weight.free_data(); middle_1.proj_bias.free_data();
    middle_2.norm1_gamma.free_data(); middle_2.norm2_gamma.free_data();
    middle_2.conv1.weight.free_data(); middle_2.conv1.bias.free_data();
    middle_2.conv2.weight.free_data(); middle_2.conv2.bias.free_data();
    if (middle_2.has_shortcut) { middle_2.shortcut.weight.free_data(); middle_2.shortcut.bias.free_data(); }
    for (auto& ub : upsamples) {
        if (ub.type == VAEDecoderWeights::UpsampleBlock::RESBLOCK) {
            ub.res.norm1_gamma.free_data(); ub.res.norm2_gamma.free_data();
            ub.res.conv1.weight.free_data(); ub.res.conv1.bias.free_data();
            ub.res.conv2.weight.free_data(); ub.res.conv2.bias.free_data();
            if (ub.res.has_shortcut) { ub.res.shortcut.weight.free_data(); ub.res.shortcut.bias.free_data(); }
        } else {
            ub.resample.conv_weight.free_data(); ub.resample.conv_bias.free_data();
            if (ub.resample.has_time_conv) {
                ub.resample.time_conv.weight.free_data(); ub.resample.time_conv.bias.free_data();
            }
        }
    }
    head_norm_gamma.free_data();
    head_conv.weight.free_data(); head_conv.bias.free_data();
}

// Run a CausalConv3d layer
static Tensor run_causal_conv3d(const CausalConv3dWeights& cw, const Tensor& x,
                                 int C_in, int T, int H, int W) {
    int H_out = (H + 2 * cw.padH - cw.kH) / cw.strideH + 1;
    int W_out = (W + 2 * cw.padW - cw.kW) / cw.strideW + 1;
    int T_padded = T + (cw.kT - 1);
    int T_out = (T_padded - cw.kT) / cw.strideT + 1;

    Tensor output = Tensor::alloc({(int64_t)cw.out_channels, T_out, H_out, W_out}, DType::BF16);

    size_t ws_size = causal_conv3d_workspace_size(C_in, T, H, W, cw.kT, cw.kH, cw.kW,
                                                   cw.padH, cw.padW, cw.strideT, cw.strideH, cw.strideW);
    Tensor workspace = Tensor::alloc({(int64_t)(ws_size / 2 + 1)}, DType::BF16); // BF16 elements

    causal_conv3d_forward(
        (__nv_bfloat16*)x.data, (__nv_bfloat16*)cw.weight.data,
        cw.bias.data ? (__nv_bfloat16*)cw.bias.data : nullptr,
        (__nv_bfloat16*)output.data, (__nv_bfloat16*)workspace.data,
        C_in, T, H, W, cw.out_channels, cw.kT, cw.kH, cw.kW,
        cw.padH, cw.padW, cw.strideT, cw.strideH, cw.strideW);

    workspace.free_data();
    return output;
}

// Run a residual block
static Tensor run_resblock(const ResBlockWeights& rb, Tensor x, int C, int T, int H, int W) {
    Tensor h = x.clone();
    if (rb.has_shortcut) {
        h = run_causal_conv3d(rb.shortcut, x, rb.in_dim, T, H, W);
    }

    int spatial = T * H * W;

    // norm1 -> silu -> conv1
    rms_norm_channel((__nv_bfloat16*)x.data, (__nv_bfloat16*)rb.norm1_gamma.data,
                     (__nv_bfloat16*)x.data, rb.in_dim, spatial, 1e-12f);
    silu_inplace((__nv_bfloat16*)x.data, (int64_t)rb.in_dim * spatial);

    Tensor after_conv1 = run_causal_conv3d(rb.conv1, x, rb.in_dim, T, H, W);
    x.free_data();

    // Get new dims after conv1 (should be same spatial for stride=1)
    int new_C = rb.out_dim;
    spatial = T * H * W; // same for stride 1, pad 1, k 3

    // norm2 -> silu -> conv2
    rms_norm_channel((__nv_bfloat16*)after_conv1.data, (__nv_bfloat16*)rb.norm2_gamma.data,
                     (__nv_bfloat16*)after_conv1.data, new_C, spatial, 1e-12f);
    silu_inplace((__nv_bfloat16*)after_conv1.data, (int64_t)new_C * spatial);

    Tensor after_conv2 = run_causal_conv3d(rb.conv2, after_conv1, new_C, T, H, W);
    after_conv1.free_data();

    // Residual add
    add_inplace((__nv_bfloat16*)after_conv2.data, (__nv_bfloat16*)h.data,
                (int64_t)new_C * spatial);
    h.free_data();

    return after_conv2;
}

// Run attention block (spatial self-attention)
static Tensor run_attention(const AttentionBlockWeights& ab, Tensor x, int C, int T, int H, int W) {
    int spatial = H * W;
    Tensor identity = x.clone();

    // RMS norm (channel-wise)
    rms_norm_channel((__nv_bfloat16*)x.data, (__nv_bfloat16*)ab.norm_gamma.data,
                     (__nv_bfloat16*)x.data, C, T * H * W, 1e-12f);

    // For single frame (T=1), this is spatial self-attention
    // Permute [C, T, H, W] -> [T, C, H, W] for Conv2d
    // Then to_qkv: Conv2d(C, 3C, 1, 1)
    // For T=1, we can treat as [C, H, W] directly

    // to_qkv: Conv2d 1x1 = just a linear projection
    // x reshaped as [C, H*W] -> matmul with weight [3C, C] -> [3C, H*W]
    Tensor qkv = Tensor::alloc({(int64_t)(3 * C), (int64_t)T, (int64_t)H, (int64_t)W}, DType::BF16);

    // For 1x1 conv, just use linear: treat x as [H*W*T, C] and weight as [3C, C]
    // But our data is [C, T*H*W] (channel-first), so we need to transpose
    // x: [C, spatial] -> transpose to [spatial, C], matmul, transpose back
    int total_spatial = T * H * W;
    Tensor x_t = Tensor::alloc({(int64_t)total_spatial, (int64_t)C}, DType::BF16);
    {
        // Transpose [C, spatial] -> [spatial, C]
        // Actually for conv1x1 we can use cuBLAS directly:
        // out = weight @ x, where weight is [3C, C], x is [C, spatial] -> out is [3C, spatial]
        float alpha = 1.0f, beta = 0.0f;
        CUBLAS_CHECK(cublasGemmEx(
            cublas(), CUBLAS_OP_N, CUBLAS_OP_N,
            total_spatial, 3 * C, C,
            &alpha,
            x.data, CUDA_R_16BF, total_spatial,
            ab.to_qkv_weight.data, CUDA_R_16BF, C,
            &beta,
            qkv.data, CUDA_R_16BF, total_spatial,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
        // Add bias
        if (ab.to_qkv_bias.data) {
            int total = 3 * C * total_spatial;
            int block = 256;
            int grid = (total + block - 1) / block;
            // bias_add for channel-first: qkv[c, s] += bias[c]
            extern __global__ void conv_bias_add_kernel(const __nv_bfloat16* bias, __nv_bfloat16* out, int C, int spatial);
            conv_bias_add_kernel<<<grid, block>>>((__nv_bfloat16*)ab.to_qkv_bias.data,
                                                   (__nv_bfloat16*)qkv.data, 3 * C, total_spatial);
        }
    }
    x_t.free_data();

    // Split QKV: q[C, spatial], k[C, spatial], v[C, spatial]
    // qkv is [3C, spatial]
    auto* qkv_ptr = (__nv_bfloat16*)qkv.data;
    // For attention: reshape to [spatial, C] for q and k, keep v as [C, spatial]

    // q: [C, spatial] -> transpose to [spatial, C]
    // k: [C, spatial] -> transpose to [spatial, C]
    // v: [C, spatial] -> transpose to [spatial, C]
    Tensor q_t2 = Tensor::alloc({(int64_t)total_spatial, (int64_t)C}, DType::BF16);
    Tensor k_t2 = Tensor::alloc({(int64_t)total_spatial, (int64_t)C}, DType::BF16);
    Tensor v_t2 = Tensor::alloc({(int64_t)total_spatial, (int64_t)C}, DType::BF16);

    // For T=1, attention is over spatial dimension with 1 head and dim=C
    // Actually let's use the simple attention function
    // attention expects [BH, S, D] so we have [1, spatial, C]
    // But we need to transpose from [C, spatial] to [spatial, C] first

    // Compute scores = q^T @ k / sqrt(C), softmax, @ v^T
    // q, k: [C, spatial], v: [C, spatial]
    // scores = q^T @ k = [spatial, C] @ [C, spatial] = [spatial, spatial]
    auto* q_ptr = qkv_ptr;
    auto* k_ptr = qkv_ptr + (int64_t)C * total_spatial;
    auto* v_ptr = qkv_ptr + (int64_t)2 * C * total_spatial;

    // scores = (1/sqrt(C)) * q^T @ k
    float* scores;
    CUDA_CHECK(cudaMalloc(&scores, (size_t)total_spatial * total_spatial * sizeof(float)));
    {
        float alpha = 1.0f / sqrtf((float)C);
        float beta = 0.0f;
        // q^T: [spatial, C], k: [C, spatial]
        // scores = q^T @ k = [spatial, spatial]
        CUBLAS_CHECK(cublasGemmEx(
            cublas(), CUBLAS_OP_N, CUBLAS_OP_T,
            total_spatial, total_spatial, C,
            &alpha,
            k_ptr, CUDA_R_16BF, total_spatial,
            q_ptr, CUDA_R_16BF, total_spatial,
            &beta,
            scores, CUDA_R_32F, total_spatial,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
    }

    // Softmax
    softmax(scores, scores, total_spatial, total_spatial);

    // Convert scores to BF16
    __nv_bfloat16* scores_bf16;
    CUDA_CHECK(cudaMalloc(&scores_bf16, (size_t)total_spatial * total_spatial * sizeof(__nv_bfloat16)));
    fp32_to_bf16(scores, scores_bf16, (int64_t)total_spatial * total_spatial);
    CUDA_CHECK(cudaFree(scores));

    // out_cf = v_cf @ weights^T  (row-major: [C, S] @ [S, S] = [C, S])
    // cuBLAS row-major trick: compute weights @ v_cf^T in column-major
    // weights stored row-major = weights^T in column-major, so use OP_T to get weights
    // v_cf stored row-major [C,S] = v_cf^T [S,C] in column-major, use OP_N
    Tensor attn_result = Tensor::alloc({(int64_t)C, (int64_t)T, (int64_t)H, (int64_t)W}, DType::BF16);
    {
        float alpha = 1.0f, beta = 0.0f;
        CUBLAS_CHECK(cublasGemmEx(
            cublas(), CUBLAS_OP_T, CUBLAS_OP_N,
            total_spatial, C, total_spatial,
            &alpha,
            scores_bf16, CUDA_R_16BF, total_spatial,
            v_ptr, CUDA_R_16BF, total_spatial,
            &beta,
            attn_result.data, CUDA_R_16BF, total_spatial,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
    }
    CUDA_CHECK(cudaFree(scores_bf16));
    qkv.free_data();
    q_t2.free_data(); k_t2.free_data(); v_t2.free_data();

    // proj: Conv2d 1x1 (linear projection)
    Tensor proj_result = Tensor::alloc({(int64_t)C, (int64_t)T, (int64_t)H, (int64_t)W}, DType::BF16);
    {
        float alpha = 1.0f, beta = 0.0f;
        CUBLAS_CHECK(cublasGemmEx(
            cublas(), CUBLAS_OP_N, CUBLAS_OP_N,
            total_spatial, C, C,
            &alpha,
            attn_result.data, CUDA_R_16BF, total_spatial,
            ab.proj_weight.data, CUDA_R_16BF, C,
            &beta,
            proj_result.data, CUDA_R_16BF, total_spatial,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
        if (ab.proj_bias.data) {
            int total = C * total_spatial;
            int block = 256;
            int grid = (total + block - 1) / block;
            extern __global__ void conv_bias_add_kernel(const __nv_bfloat16* bias, __nv_bfloat16* out, int C, int spatial);
            conv_bias_add_kernel<<<grid, block>>>((__nv_bfloat16*)ab.proj_bias.data,
                                                   (__nv_bfloat16*)proj_result.data, C, total_spatial);
        }
    }
    attn_result.free_data();

    // Residual add
    add_inplace((__nv_bfloat16*)proj_result.data, (__nv_bfloat16*)identity.data,
                (int64_t)C * total_spatial);
    identity.free_data();
    x.free_data();

    return proj_result;
}

// Run resample (upsample2d or upsample3d)
static Tensor run_resample(const ResampleWeights& rs, Tensor x, int C, int T, int H, int W) {
    // For single image (T=1), upsample3d skips temporal doubling at chunk_idx=0
    // Only spatial upsample + conv2d

    // Spatial 2x upsample
    // x: [C, T, H, W] -> [C, T, 2H, 2W]
    int H2 = H * 2;
    int W2 = W * 2;
    Tensor upsampled = Tensor::alloc({(int64_t)C, (int64_t)T, (int64_t)H2, (int64_t)W2}, DType::BF16);

    // Upsample each [T, H, W] slice per channel (for T=1, it's just [H, W])
    for (int c = 0; c < C; c++) {
        for (int t = 0; t < T; t++) {
            upsample_nearest_2d(
                (__nv_bfloat16*)x.data + (int64_t)(c * T + t) * H * W,
                (__nv_bfloat16*)upsampled.data + (int64_t)(c * T + t) * H2 * W2,
                1, H, W);
        }
    }

    // Conv2d 3x3: need to process each time frame
    // For T=1, treat as [C, H2, W2]
    int C_out = rs.dim / 2; // non-wan2.2: conv outputs dim/2

    // Conv2d weight: [C_out, C, 3, 3]
    Tensor result = Tensor::alloc({(int64_t)C_out, (int64_t)T, (int64_t)H2, (int64_t)W2}, DType::BF16);

    for (int t = 0; t < T; t++) {
        size_t ws = conv2d_workspace_size(1, C, H2, W2, 3, 3, 1, 1, 1, 1);
        Tensor workspace = Tensor::alloc({(int64_t)(ws / 2 + 1)}, DType::BF16);

        conv2d_forward(
            (__nv_bfloat16*)upsampled.data + (int64_t)t * C * H2 * W2,
            (__nv_bfloat16*)rs.conv_weight.data,
            rs.conv_bias.data ? (__nv_bfloat16*)rs.conv_bias.data : nullptr,
            (__nv_bfloat16*)result.data + (int64_t)t * C_out * H2 * W2,
            (__nv_bfloat16*)workspace.data,
            1, C, H2, W2, C_out, 3, 3, 1, 1, 1, 1);
        workspace.free_data();
    }

    upsampled.free_data();
    x.free_data();
    return result;
}

Tensor vae_decode(const VAEDecoderWeights& w, const Tensor& latent, int H_latent, int W_latent) {
    fprintf(stderr, "VAE decode: latent [16, 1, %d, %d]\n", H_latent, W_latent);

    // conv2: CausalConv3d(16, 16, 1,1,1) - post_quant_conv
    int T = 1;
    Tensor x = run_causal_conv3d(w.conv2, latent, 16, T, H_latent, W_latent);

    // conv1: CausalConv3d(16, 384, 3,3,3)
    Tensor x2 = run_causal_conv3d(w.conv1, x, 16, T, H_latent, W_latent);
    x.free_data();
    x = std::move(x2);

    int C = 384;
    int H = H_latent;
    int Wd = W_latent;

    // Middle blocks
    x = run_resblock(w.middle_0, std::move(x), C, T, H, Wd);
    x = run_attention(w.middle_1, std::move(x), C, T, H, Wd);
    x = run_resblock(w.middle_2, std::move(x), C, T, H, Wd);

    // Upsample blocks
    for (size_t i = 0; i < w.upsamples.size(); i++) {
        auto& ub = w.upsamples[i];
        if (ub.type == VAEDecoderWeights::UpsampleBlock::RESBLOCK) {
            x = run_resblock(ub.res, std::move(x), C, T, H, Wd);
            C = ub.res.out_dim;
        } else {
            x = run_resample(ub.resample, std::move(x), C, T, H, Wd);
            C = ub.resample.dim / 2;
            H *= 2;
            Wd *= 2;
        }
    }

    // Head: RMS_norm -> SiLU -> CausalConv3d(96, 3, 3,3,3)
    int spatial = T * H * Wd;
    rms_norm_channel((__nv_bfloat16*)x.data, (__nv_bfloat16*)w.head_norm_gamma.data,
                     (__nv_bfloat16*)x.data, C, spatial, 1e-12f);
    silu_inplace((__nv_bfloat16*)x.data, (int64_t)C * spatial);

    Tensor rgb = run_causal_conv3d(w.head_conv, x, C, T, H, Wd);
    x.free_data();

    fprintf(stderr, "VAE decode done: [3, %d, %d, %d]\n", T, H, Wd);
    return rgb;
}
