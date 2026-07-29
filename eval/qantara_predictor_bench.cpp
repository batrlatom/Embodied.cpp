// Copyright 2026 SEU-PAISys
// Licensed under the Apache License, Version 2.0.

#include "runtime/model.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace {

template <typename T>
bool read_value(std::ifstream & stream, T * value) {
    return bool(stream.read(reinterpret_cast<char *>(value), sizeof(T)));
}

bool read_floats(std::ifstream & stream, std::vector<float> * values, size_t count) {
    values->resize(count);
    return bool(stream.read(
        reinterpret_cast<char *>(values->data()),
        static_cast<std::streamsize>(count * sizeof(float))));
}

void usage(const char * program) {
    std::fprintf(
        stderr, "usage: %s <qantara.gguf> <parity-input.bin> [iterations]\n",
        program);
}

} // namespace

int main(int argc, char ** argv) {
    if (argc < 3 || argc > 4) {
        usage(argv[0]);
        return 2;
    }
    const int iterations = argc == 4 ? std::max(1, std::atoi(argv[3])) : 1;
    std::ifstream input(argv[2], std::ios::binary);
    uint32_t magic = 0, history = 0, latent_dim = 0, action_block_dim = 0;
    if (!read_value(input, &magic) || !read_value(input, &history) ||
        !read_value(input, &latent_dim) || !read_value(input, &action_block_dim) ||
        magic != 0x51545241u) {
        std::fprintf(stderr, "qantara-bench: invalid parity input\n");
        return 2;
    }
    std::vector<float> latents, actions, video_noise, action_noise;
    if (!read_floats(input, &latents, size_t(history) * latent_dim) ||
        !read_floats(input, &actions, size_t(history - 1) * action_block_dim) ||
        !read_floats(input, &video_noise, action_block_dim) ||
        !read_floats(input, &action_noise, action_block_dim)) {
        std::fprintf(stderr, "qantara-bench: truncated parity input\n");
        return 2;
    }

    vla::Model * model = vla::model_load("", argv[1]);
    if (!model) return 1;
    vla::Inputs values{};
    values.qantara_latents = latents.data();
    values.qantara_history = static_cast<int>(history);
    values.qantara_actions = actions.empty() ? nullptr : actions.data();
    values.qantara_action_blocks = static_cast<int>(history - 1);
    values.qantara_video_noise = video_noise.data();
    values.qantara_action_noise = action_noise.data();

    std::vector<float> result;
    std::vector<float> samples;
    std::vector<float> total_samples;
    samples.reserve(static_cast<size_t>(iterations));
    total_samples.reserve(static_cast<size_t>(iterations));
    for (int i = 0; i < iterations; ++i) {
        result = vla::predict(model, values);
        if (result.empty()) {
            vla::model_free(model);
            return 1;
        }
        samples.push_back(vla::last_stats(model).ms_inference);
        total_samples.push_back(vla::last_stats(model).ms_total);
    }
    std::sort(samples.begin(), samples.end());
    std::sort(total_samples.begin(), total_samples.end());
    std::printf("action");
    for (float value : result) std::printf(" %.9g", value);
    std::printf("\n");
    std::printf(
        "timing_ms inference_median=%.4f inference_min=%.4f "
        "total_median=%.4f total_min=%.4f iterations=%d\n",
        samples[samples.size() / 2], samples.front(),
        total_samples[total_samples.size() / 2], total_samples.front(), iterations);
    vla::model_free(model);
    return 0;
}
