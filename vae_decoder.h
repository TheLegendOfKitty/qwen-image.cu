#pragma once
#include "tensor.h"
#include "safetensors.h"
#include "cuda_kernels.cuh"
#include <string>
#include <vector>

// WAN VAE Decoder3d weights
// Architecture: conv2 -> conv1 -> middle -> upsamples -> head
// For non-wan2.2: dim=96, dim_mult=[1,2,4,4], z_dim=16
// dims = [384, 384, 384, 192, 96] (reversed)

struct CausalConv3dWeights {
    Tensor weight; // [out*in, kT, kH, kW] BF16
    Tensor bias;   // [out] BF16
    int in_channels, out_channels;
    int kT, kH, kW;
    int padH, padW;
    int strideT, strideH, strideW;
};

struct ResBlockWeights {
    // norm1 -> silu -> conv1 -> norm2 -> silu -> conv2 + shortcut
    Tensor norm1_gamma;  // RMS_norm gamma
    Tensor norm2_gamma;
    CausalConv3dWeights conv1; // [3,3,3]
    CausalConv3dWeights conv2; // [3,3,3]
    CausalConv3dWeights shortcut; // [1,1,1] if in_dim != out_dim
    bool has_shortcut;
    int in_dim, out_dim;
};

struct AttentionBlockWeights {
    Tensor norm_gamma; // RMS_norm
    Tensor to_qkv_weight; // Conv2d [3*dim, dim, 1, 1]
    Tensor to_qkv_bias;
    Tensor proj_weight;   // Conv2d [dim, dim, 1, 1]
    Tensor proj_bias;
    int dim;
};

struct ResampleWeights {
    Tensor conv_weight;  // Conv2d resample.1
    Tensor conv_bias;
    CausalConv3dWeights time_conv; // for upsample3d
    std::string mode; // "upsample2d" or "upsample3d"
    int dim;
    bool has_time_conv;
};

struct VAEDecoderWeights {
    // Initial convs
    CausalConv3dWeights conv2; // [16, 16, 1, 1, 1]  identity-like
    CausalConv3dWeights conv1; // [16, 384, 3, 3, 3]

    // Middle
    ResBlockWeights middle_0;
    AttentionBlockWeights middle_1;
    ResBlockWeights middle_2;

    // Upsample blocks (flat list matching reference indexing)
    // Non-wan2.2 layout: 15 blocks total
    // i=0: 3× ResBlock(384->384) + Resample("upsample3d", 384)
    // i=1: ResBlock(192->384) + 2× ResBlock(384->384) + Resample("upsample3d", 384)
    // i=2: 3× ResBlock(192->192) + Resample("upsample2d", 192)
    // i=3: 3× ResBlock(96->96)
    struct UpsampleBlock {
        enum Type { RESBLOCK, RESAMPLE } type;
        ResBlockWeights res;
        ResampleWeights resample;
    };
    std::vector<UpsampleBlock> upsamples;

    // Head
    Tensor head_norm_gamma; // RMS_norm
    CausalConv3dWeights head_conv; // [96, 3, 3, 3, 3]

    void load(const SafeTensorsLoader& loader);
    void free_all();
};

// Forward pass: latent [1, 16, 1, H/8, W/8] -> RGB [1, 3, 1, H, W]
// For single image, temporal dim is always 1
Tensor vae_decode(const VAEDecoderWeights& w,
                  const Tensor& latent, // [16, 1, H/8, W/8]
                  int H_latent, int W_latent);
