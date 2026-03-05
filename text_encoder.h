#pragma once
#include "tensor.h"
#include "safetensors.h"
#include "cuda_kernels.cuh"
#include "rope.h"
#include "logging.h"
#include <string>
#include <vector>

struct TextEncoderWeights {
    Tensor embed_tokens; // [152064, 3584]
    Tensor norm_weight;  // [3584] - final RMS norm

    struct Layer {
        Tensor input_layernorm_weight;       // [3584]
        Tensor post_attention_layernorm_weight; // [3584]

        // Attention
        Tensor q_proj_weight; // [3584, 3584]
        Tensor q_proj_bias;   // [3584]
        Tensor k_proj_weight; // [512, 3584]
        Tensor k_proj_bias;   // [512]
        Tensor v_proj_weight; // [512, 3584]
        Tensor v_proj_bias;   // [512]
        Tensor o_proj_weight; // [3584, 3584]

        // MLP (SwiGLU)
        Tensor gate_proj_weight; // [18944, 3584]
        Tensor up_proj_weight;   // [18944, 3584]
        Tensor down_proj_weight; // [3584, 18944]
    };

    std::vector<Layer> layers;

    void load(const SafeTensorsLoader& loader) {
        LOGV("Loading text encoder weights...\n");

        embed_tokens = loader.load_tensor("model.embed_tokens.weight");
        norm_weight = loader.load_tensor("model.norm.weight");

        layers.resize(28);
        for (int i = 0; i < 28; i++) {
            std::string prefix = "model.layers." + std::to_string(i) + ".";
            Layer& l = layers[i];

            l.input_layernorm_weight = loader.load_tensor(prefix + "input_layernorm.weight");
            l.post_attention_layernorm_weight = loader.load_tensor(prefix + "post_attention_layernorm.weight");

            l.q_proj_weight = loader.load_tensor(prefix + "self_attn.q_proj.weight");
            l.q_proj_bias = loader.load_tensor(prefix + "self_attn.q_proj.bias");
            l.k_proj_weight = loader.load_tensor(prefix + "self_attn.k_proj.weight");
            l.k_proj_bias = loader.load_tensor(prefix + "self_attn.k_proj.bias");
            l.v_proj_weight = loader.load_tensor(prefix + "self_attn.v_proj.weight");
            l.v_proj_bias = loader.load_tensor(prefix + "self_attn.v_proj.bias");
            l.o_proj_weight = loader.load_tensor(prefix + "self_attn.o_proj.weight");

            l.gate_proj_weight = loader.load_tensor(prefix + "mlp.gate_proj.weight");
            l.up_proj_weight = loader.load_tensor(prefix + "mlp.up_proj.weight");
            l.down_proj_weight = loader.load_tensor(prefix + "mlp.down_proj.weight");

            if ((i + 1) % 7 == 0)
                LOGV("  Loaded layer %d/28\n", i + 1);
        }
        LOGV("Text encoder weights loaded.\n");
    }

    void free_all() {
        embed_tokens.free_data();
        norm_weight.free_data();
        for (auto& l : layers) {
            l.input_layernorm_weight.free_data();
            l.post_attention_layernorm_weight.free_data();
            l.q_proj_weight.free_data();
            l.q_proj_bias.free_data();
            l.k_proj_weight.free_data();
            l.k_proj_bias.free_data();
            l.v_proj_weight.free_data();
            l.v_proj_bias.free_data();
            l.o_proj_weight.free_data();
            l.gate_proj_weight.free_data();
            l.up_proj_weight.free_data();
            l.down_proj_weight.free_data();
        }
        layers.clear();
    }
};

// Forward pass for the text encoder
// tokens: host vector of token IDs
// Returns: Tensor [1, seq_len, 3584] on GPU
Tensor text_encoder_forward(const TextEncoderWeights& w,
                            const std::vector<int32_t>& token_ids,
                            const char* progress_label = nullptr);
