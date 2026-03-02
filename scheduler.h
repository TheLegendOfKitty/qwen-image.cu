#pragma once
#include <cmath>
#include <vector>
#include <algorithm>
#include <cstdio>

// FlowMatchEulerDiscreteScheduler
// Implements the denoising schedule for flow matching models

struct FlowMatchScheduler {
    float shift = 1.0f; // Default shift for time shifting
    int num_train_timesteps = 1000;
    bool use_dynamic_shifting = true;

    // Dynamic shifting parameters (from scheduler config)
    float base_shift = 0.5f;
    float max_shift = 0.9f;
    float shift_terminal = 0.02f;
    int base_image_seq_len = 256;
    int max_image_seq_len = 8192;

    FlowMatchScheduler() = default;

    // Time SNR shift: sigma = shift * t / (1 + (shift - 1) * t)
    float time_snr_shift(float alpha, float t) const {
        if (alpha == 1.0f) return t;
        return alpha * t / (1.0f + (alpha - 1.0f) * t);
    }

    // Convert discrete timestep index to sigma
    float t_to_sigma(float t) const {
        float s = (t + 1.0f) / (float)num_train_timesteps;
        return time_snr_shift(shift, s);
    }

    // Compute dynamic shift based on image sequence length
    float compute_dynamic_shift(int image_seq_len) const {
        // Exponential time shift type (from config)
        // mu = base_shift * exp(log(max_shift / base_shift) * (image_seq_len - base_image_seq_len)
        //                       / (max_image_seq_len - base_image_seq_len))
        float log_ratio = std::log(max_shift / base_shift);
        float frac = (float)(image_seq_len - base_image_seq_len) / (float)(max_image_seq_len - base_image_seq_len);
        float mu = base_shift * std::exp(log_ratio * frac);
        return mu;
    }

    // Get sigma schedule for N steps using discrete scheduler
    std::vector<float> get_sigmas(int n_steps, int image_seq_len = 0) {
        if (use_dynamic_shifting && image_seq_len > 0) {
            shift = compute_dynamic_shift(image_seq_len);
            fprintf(stderr, "Scheduler: dynamic shift = %.4f for seq_len=%d\n", shift, image_seq_len);
        }

        std::vector<float> sigmas;
        sigmas.reserve(n_steps + 1);

        if (n_steps == 0) {
            sigmas.push_back(0.0f);
            return sigmas;
        }

        int t_max = num_train_timesteps - 1;

        if (n_steps == 1) {
            sigmas.push_back(t_to_sigma((float)t_max));
            sigmas.push_back(0.0f);
            return sigmas;
        }

        float step = (float)t_max / (float)(n_steps - 1);
        for (int i = 0; i < n_steps; i++) {
            float t = (float)t_max - step * (float)i;
            sigmas.push_back(t_to_sigma(t));
        }
        sigmas.push_back(0.0f);

        return sigmas;
    }

    // Get timestep from sigma (for model input)
    float sigma_to_timestep(float sigma) const {
        return sigma * (float)num_train_timesteps;
    }
};
