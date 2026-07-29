// Copyright 2026 SEU-PAISys
// Licensed under the Apache License, Version 2.0.

#include "runtime/model.h"

#include <cstdio>
#include <vector>

int main(int argc, char ** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s <qantara.gguf>\n", argv[0]);
        return 2;
    }
    vla::Model * model = vla::model_load("", argv[1]);
    if (!model) return 1;
    const auto & config = vla::model_config(model);
    constexpr int side = 84;
    const int action_block = static_cast<int>(
        config.n_suffix * config.real_action_dim);
    std::vector<float> image(static_cast<size_t>(side * side * 3));
    std::vector<float> video_noise(static_cast<size_t>(action_block));
    std::vector<float> action_noise(static_cast<size_t>(action_block));
    std::vector<float> previous_action;
    std::vector<float> result;
    for (int step = 0; step < 3; ++step) {
        for (size_t i = 0; i < image.size(); ++i) {
            image[i] = static_cast<float>(
                (i * 37u + static_cast<size_t>(step) * 13u) % 256u) / 255.0f;
        }
        for (int i = 0; i < action_block; ++i) {
            video_noise[static_cast<size_t>(i)] =
                (static_cast<float>((i * 17 + step * 7) % 101) - 50.0f) / 29.0f;
            action_noise[static_cast<size_t>(i)] =
                (static_cast<float>((i * 23 + step * 11) % 103) - 51.0f) / 31.0f;
        }
        vla::ImageView view{
            image.data(), side, side, vla::PixelFormat::F32_RGB_01};
        vla::Inputs input{};
        input.images = &view;
        input.n_images = 1;
        input.qantara_session_id = 42;
        input.qantara_reset = step == 0;
        input.qantara_previous_action =
            previous_action.empty() ? nullptr : previous_action.data();
        input.qantara_previous_action_n =
            static_cast<int>(previous_action.size());
        input.qantara_video_noise = video_noise.data();
        input.qantara_action_noise = action_noise.data();
        result = vla::predict(model, input);
        if (result.size() != static_cast<size_t>(action_block)) {
            vla::model_free(model);
            return 1;
        }
        previous_action = result;
    }
    std::printf("action");
    for (float value : result) std::printf(" %.9g", value);
    std::printf("\n");
    const auto & stats = vla::last_stats(model);
    std::printf(
        "timing_ms vision=%.4f inference=%.4f total=%.4f\n",
        stats.ms_vision, stats.ms_inference, stats.ms_total);
    vla::model_free(model);
    return 0;
}
