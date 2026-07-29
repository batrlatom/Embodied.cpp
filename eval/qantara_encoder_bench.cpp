// Copyright 2026 SEU-PAISys
// Licensed under the Apache License, Version 2.0.

#include "runtime/model.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

int main(int argc, char ** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s <qantara.gguf>\n", argv[0]);
        return 2;
    }
    vla::Model * model = vla::model_load("", argv[1]);
    if (!model) return 1;
    constexpr int side = 84;
    std::vector<float> image(static_cast<size_t>(side * side * 3));
    for (size_t i = 0; i < image.size(); ++i) {
        image[i] = static_cast<float>((i * 37u) % 256u) / 255.0f;
    }
    vla::ImageView view{
        image.data(), side, side, vla::PixelFormat::F32_RGB_01};
    vla::Inputs input{};
    input.images = &view;
    input.n_images = 1;
    input.qantara_encode_only = true;
    const std::vector<float> latent = vla::predict(model, input);
    if (latent.empty()) {
        vla::model_free(model);
        return 1;
    }
    std::printf("latent");
    for (float value : latent) std::printf(" %.9g", value);
    std::printf("\n");
    const auto & stats = vla::last_stats(model);
    std::printf(
        "timing_ms vision=%.4f total=%.4f\n",
        stats.ms_vision, stats.ms_total);
    vla::model_free(model);
    return 0;
}
