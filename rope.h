#pragma once
#include <cmath>
#include <vector>
#include <algorithm>
#include <cstdint>

namespace RoPE {

template <class T>
inline std::vector<T> linspace(T start, T end, int num) {
    std::vector<T> result(num);
    if (num == 1) {
        result[0] = start;
        return result;
    }
    T step = (end - start) / (num - 1);
    for (int i = 0; i < num; i++) {
        result[i] = start + i * step;
    }
    return result;
}

// Compute RoPE rotation matrices for given positions and dimension
// Returns [pos_size, half_dim * 4] where each group of 4 is [cos, -sin, sin, cos]
inline std::vector<std::vector<float>> rope(const std::vector<float>& pos, int dim, float theta) {
    int half_dim = dim / 2;

    std::vector<float> scale = linspace(0.0f, (dim * 1.0f - 2.0f) / dim, half_dim);

    std::vector<float> omega(half_dim);
    for (int i = 0; i < half_dim; i++) {
        omega[i] = 1.0f / std::pow(theta, scale[i]);
    }

    size_t pos_size = pos.size();
    std::vector<std::vector<float>> result(pos_size, std::vector<float>(half_dim * 4));
    for (size_t i = 0; i < pos_size; i++) {
        for (int j = 0; j < half_dim; j++) {
            float angle = pos[i] * omega[j];
            result[i][4 * j + 0] = std::cos(angle);
            result[i][4 * j + 1] = -std::sin(angle);
            result[i][4 * j + 2] = std::sin(angle);
            result[i][4 * j + 3] = std::cos(angle);
        }
    }
    return result;
}

// Flatten 2D vector
inline std::vector<float> flatten(const std::vector<std::vector<float>>& vec) {
    std::vector<float> flat;
    for (auto& sub : vec) {
        flat.insert(flat.end(), sub.begin(), sub.end());
    }
    return flat;
}

// Transpose 2D vector
inline std::vector<std::vector<float>> transpose(const std::vector<std::vector<float>>& mat) {
    if (mat.empty()) return {};
    size_t rows = mat.size();
    size_t cols = mat[0].size();
    std::vector<std::vector<float>> t(cols, std::vector<float>(rows));
    for (size_t i = 0; i < rows; i++)
        for (size_t j = 0; j < cols; j++)
            t[j][i] = mat[i][j];
    return t;
}

// Concatenate two ID arrays, interleaving by batch
inline std::vector<std::vector<float>> concat_ids(const std::vector<std::vector<float>>& a,
                                                    const std::vector<std::vector<float>>& b,
                                                    int bs) {
    if (a.empty()) return b;
    if (b.empty()) return a;
    size_t a_len = a.size() / bs;
    size_t b_len = b.size() / bs;
    std::vector<std::vector<float>> ids(a.size() + b.size(), std::vector<float>(3));
    for (int i = 0; i < bs; i++) {
        for (size_t j = 0; j < a_len; j++)
            ids[i * (a_len + b_len) + j] = a[i * a_len + j];
        for (size_t j = 0; j < b_len; j++)
            ids[i * (a_len + b_len) + a_len + j] = b[i * b_len + j];
    }
    return ids;
}

// Embed N-dimensional positions using per-axis RoPE
inline std::vector<float> embed_nd(const std::vector<std::vector<float>>& ids,
                                    int bs, float theta,
                                    const std::vector<int>& axes_dim) {
    auto trans_ids = transpose(ids);
    size_t pos_len = ids.size() / bs;
    size_t num_axes = axes_dim.size();

    int emb_dim = 0;
    for (int d : axes_dim) emb_dim += d / 2;

    std::vector<std::vector<float>> emb(bs * pos_len, std::vector<float>(emb_dim * 2 * 2, 0.0f));
    size_t offset = 0;
    for (size_t i = 0; i < num_axes; i++) {
        auto rope_emb = rope(trans_ids[i], axes_dim[i], theta);
        for (int b = 0; b < bs; b++) {
            for (size_t j = 0; j < pos_len; j++) {
                for (size_t k = 0; k < rope_emb[0].size(); k++) {
                    emb[b * pos_len + j][offset + k] = rope_emb[j][k];
                }
            }
        }
        offset += rope_emb[0].size();
    }

    return flatten(emb);
}

// Generate image patch IDs for Flux-style models
inline std::vector<std::vector<float>> gen_flux_img_ids(int h, int w, int patch_size, int bs,
                                                         int axes_dim_num, int index = 0,
                                                         int h_offset = 0, int w_offset = 0,
                                                         bool scale_rope = false) {
    int h_len = (h + (patch_size / 2)) / patch_size;
    int w_len = (w + (patch_size / 2)) / patch_size;

    std::vector<std::vector<float>> img_ids(h_len * w_len, std::vector<float>(axes_dim_num, 0.0f));

    int h_start = h_offset;
    int w_start = w_offset;

    if (scale_rope) {
        h_start -= h_len / 2;
        w_start -= w_len / 2;
    }

    auto row_ids = linspace<float>((float)h_start, (float)(h_start + h_len - 1), h_len);
    auto col_ids = linspace<float>((float)w_start, (float)(w_start + w_len - 1), w_len);

    for (int i = 0; i < h_len; i++) {
        for (int j = 0; j < w_len; j++) {
            img_ids[i * w_len + j][0] = (float)index;
            img_ids[i * w_len + j][1] = row_ids[i];
            img_ids[i * w_len + j][2] = col_ids[j];
        }
    }

    // Repeat for batch
    std::vector<std::vector<float>> img_ids_rep(bs * img_ids.size(), std::vector<float>(3));
    for (int i = 0; i < bs; i++)
        for (size_t j = 0; j < img_ids.size(); j++)
            img_ids_rep[i * img_ids.size() + j] = img_ids[j];
    return img_ids_rep;
}

// Generate Qwen Image IDs (text + image)
inline std::vector<std::vector<float>> gen_qwen_image_ids(int h, int w, int patch_size, int bs,
                                                            int context_len) {
    int h_len = (h + (patch_size / 2)) / patch_size;
    int w_len = (w + (patch_size / 2)) / patch_size;
    int txt_id_start = std::max(h_len, w_len);

    auto txt_ids = linspace<float>((float)txt_id_start, (float)(context_len + txt_id_start), context_len);

    std::vector<std::vector<float>> txt_ids_rep(bs * context_len, std::vector<float>(3));
    for (int i = 0; i < bs; i++) {
        for (int j = 0; j < context_len; j++) {
            txt_ids_rep[i * context_len + j] = {txt_ids[j], txt_ids[j], txt_ids[j]};
        }
    }

    auto img_ids = gen_flux_img_ids(h, w, patch_size, bs, 3, 0, 0, 0, true);
    return concat_ids(txt_ids_rep, img_ids, bs);
}

// Generate Qwen Image positional embeddings
// Returns flat array with shape [pos_len, axes_dim_sum/2, 2, 2]
inline std::vector<float> gen_qwen_image_pe(int h, int w, int patch_size, int bs,
                                             int context_len, int theta = 10000,
                                             const std::vector<int>& axes_dim = {16, 56, 56}) {
    auto ids = gen_qwen_image_ids(h, w, patch_size, bs, context_len);
    return embed_nd(ids, bs, (float)theta, axes_dim);
}

// ===== Text encoder M-RoPE =====
// For text-only: all 3 axes use same sequential positions
// mrope_section = [16, 24, 24] -> total 64 pairs -> 128 head dim
inline std::vector<float> gen_text_encoder_rope(int seq_len, float theta = 1e6f,
                                                 const std::vector<int>& mrope_section = {16, 24, 24}) {
    // For text-only, positions are sequential [0, 1, ..., seq_len-1]
    std::vector<float> positions(seq_len);
    for (int i = 0; i < seq_len; i++) positions[i] = (float)i;

    // Compute RoPE for each section independently
    // Each section uses 2*section_dim dimensions (section_dim pairs)
    // mrope_section = [16, 24, 24] means:
    //   axis 0: pairs 0..15 (dim 0..31)
    //   axis 1: pairs 16..39 (dim 32..79)
    //   axis 2: pairs 40..63 (dim 80..127)

    int total_pairs = 0;
    for (int s : mrope_section) total_pairs += s;
    // total_pairs = 64, so total dim = 128

    // For each position, compute cos/sin for all pairs
    // Output: [seq_len, 128] of cosines and [seq_len, 128] of sines
    // But we need this in a format usable by the text encoder...

    // The text encoder uses standard RoPE application:
    //   q_rot = q * cos + rotate_half(q) * sin
    // So we return cos and sin arrays: [seq_len, total_pairs]

    int head_dim = total_pairs * 2; // 128
    int pair_offset = 0;
    std::vector<float> cos_vals(seq_len * total_pairs);
    std::vector<float> sin_vals(seq_len * total_pairs);

    int global_pair = 0;
    for (size_t axis = 0; axis < mrope_section.size(); axis++) {
        int section_pairs = mrope_section[axis];

        std::vector<float> freqs(section_pairs);
        for (int i = 0; i < section_pairs; i++) {
            // Use global pair index with total head_dim as denominator
            float scale = (float)(2 * global_pair) / (float)head_dim;
            freqs[i] = 1.0f / std::pow(theta, scale);
            global_pair++;
        }

        for (int s = 0; s < seq_len; s++) {
            for (int p = 0; p < section_pairs; p++) {
                float angle = positions[s] * freqs[p];
                cos_vals[s * total_pairs + pair_offset + p] = std::cos(angle);
                sin_vals[s * total_pairs + pair_offset + p] = std::sin(angle);
            }
        }
        pair_offset += section_pairs;
    }

    // Interleave into [seq_len, total_pairs * 2] as [cos0, cos0, cos1, cos1, ...]
    // Actually for the text encoder, we just need cos and sin separately
    // Return them concatenated: [cos (seq_len * total_pairs), sin (seq_len * total_pairs)]
    std::vector<float> result;
    result.insert(result.end(), cos_vals.begin(), cos_vals.end());
    result.insert(result.end(), sin_vals.begin(), sin_vals.end());
    return result;
}

} // namespace RoPE
