#pragma once
#include <cmath>
#include <vector>
#include <algorithm>
#include <cstdio>

// FlowMatchEulerDiscreteScheduler
// Matches diffusers' FlowMatchEulerDiscreteScheduler for Qwen Image

struct FlowMatchScheduler {
    float shift = 1.0f; // Default shift for time shifting
    int num_train_timesteps = 1000;
    bool use_dynamic_shifting = true;

    // Dynamic shifting parameters (from scheduler_config.json)
    float base_shift = 0.5f;
    float max_shift = 0.9f;
    float shift_terminal = 0.02f;
    int base_image_seq_len = 256;
    int max_image_seq_len = 8192;

    FlowMatchScheduler() = default;

    // Time SNR shift: sigma = alpha * t / (1 + (alpha - 1) * t)
    float time_snr_shift(float alpha, float t) const {
        if (alpha == 1.0f) return t;
        return alpha * t / (1.0f + (alpha - 1.0f) * t);
    }

    // Exponential time shift: exp(mu) / (exp(mu) + (1/t - 1)^sigma)
    // Equivalent to time_snr_shift(exp(mu), t) when sigma=1
    float time_shift_exponential(float mu, float t) const {
        float emu = std::exp(mu);
        return emu / (emu + (1.0f / t - 1.0f));
    }

    // Compute mu for dynamic shift via linear interpolation (matches diffusers pipeline)
    // mu = m * image_seq_len + b, where m = (max_shift - base_shift) / (max_seq - base_seq)
    float calculate_shift_mu(int image_seq_len) const {
        float m = (max_shift - base_shift) / (float)(max_image_seq_len - base_image_seq_len);
        float b = base_shift - m * (float)base_image_seq_len;
        float mu = (float)image_seq_len * m + b;
        return mu;
    }

    // Stretch sigmas so the last one maps to shift_terminal instead of a small value
    void stretch_to_terminal(std::vector<float>& sigmas) const {
        if (shift_terminal <= 0.0f) return;
        int n = (int)sigmas.size();
        if (n < 2) return;
        // one_minus_z = 1 - sigma for each
        // scale_factor = (1 - sigma_last) / (1 - shift_terminal)
        // stretched = 1 - (1 - sigma) / scale_factor
        float scale_factor = (1.0f - sigmas[n - 1]) / (1.0f - shift_terminal);
        for (int i = 0; i < n; i++) {
            sigmas[i] = 1.0f - (1.0f - sigmas[i]) / scale_factor;
        }
    }

    // Get sigma schedule for N steps (matches diffusers Qwen Image pipeline)
    std::vector<float> get_sigmas(int n_steps, int image_seq_len = 0) {
        if (n_steps == 0) return {0.0f};

        // Step 1: Generate initial sigmas as linspace(1.0, 1/N, N)
        // This matches diffusers pipeline: sigmas = np.linspace(1.0, 1/num_inference_steps, num_inference_steps)
        std::vector<float> sigmas(n_steps);
        float start = 1.0f;
        float end = 1.0f / (float)n_steps;
        for (int i = 0; i < n_steps; i++) {
            sigmas[i] = start + (end - start) * (float)i / (float)(n_steps - 1);
        }

        // Step 2: Apply time shifting
        if (use_dynamic_shifting && image_seq_len > 0) {
            // Compute mu via linear interpolation, then apply exponential time shift
            float mu = calculate_shift_mu(image_seq_len);
            fprintf(stderr, "Scheduler: dynamic shift mu=%.4f (exp(mu)=%.4f) for seq_len=%d\n",
                    mu, std::exp(mu), image_seq_len);
            for (int i = 0; i < n_steps; i++) {
                sigmas[i] = time_shift_exponential(mu, sigmas[i]);
            }
        } else {
            // Fixed shift mode
            for (int i = 0; i < n_steps; i++) {
                sigmas[i] = time_snr_shift(shift, sigmas[i]);
            }
        }

        // Step 3: Apply terminal stretching
        if (shift_terminal > 0.0f && use_dynamic_shifting) {
            stretch_to_terminal(sigmas);
        }

        // Append terminal sigma = 0
        sigmas.push_back(0.0f);

        return sigmas;
    }

    // Get timestep from sigma (for model input)
    float sigma_to_timestep(float sigma) const {
        return sigma * (float)num_train_timesteps;
    }
};
