// Copyright 2026 SEU-PAISys
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

#include "arch.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"
#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <deque>
#include <memory>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

namespace vla {
namespace {

struct GGUFReader {
    gguf_context * gctx = nullptr;
    ggml_context * meta = nullptr;
    FILE * fp = nullptr;
    size_t data_offset = 0;

    bool open(const std::string & path) {
        gguf_init_params params{};
        params.no_alloc = true;
        params.ctx = &meta;
        gctx = gguf_init_from_file(path.c_str(), params);
        fp = std::fopen(path.c_str(), "rb");
        if (!gctx || !fp) return false;
        data_offset = gguf_get_data_offset(gctx);
        return true;
    }
    ~GGUFReader() {
        if (fp) std::fclose(fp);
        if (gctx) gguf_free(gctx);
        if (meta) ggml_free(meta);
    }
    uint32_t u32(const char * key) const {
        return gguf_get_val_u32(gctx, gguf_find_key(gctx, key));
    }
    float f32(const char * key) const {
        return gguf_get_val_f32(gctx, gguf_find_key(gctx, key));
    }
    const ggml_tensor * tensor(const char * name) const {
        return ggml_get_tensor(meta, name);
    }
    bool read(const char * name, void * destination, size_t size) {
        const int64_t index = gguf_find_tensor(gctx, name);
        if (index < 0 || gguf_get_tensor_size(gctx, index) != size) return false;
        const size_t offset = data_offset + gguf_get_tensor_offset(gctx, index);
        return std::fseek(fp, static_cast<long>(offset), SEEK_SET) == 0 &&
               std::fread(destination, 1, size, fp) == size;
    }
};

struct Linear {
    ggml_tensor * weight = nullptr;
    ggml_tensor * bias = nullptr;
};

struct Block {
    ggml_tensor * qkv = nullptr;
    ggml_tensor * q_norm = nullptr;
    ggml_tensor * k_norm = nullptr;
    Linear out_aa, out_az, out_zz, out_za;
    Linear mlp_up, mlp_down;
    Linear condition;
};

struct VisionBlock {
    ggml_tensor * norm1_weight = nullptr;
    ggml_tensor * norm1_bias = nullptr;
    ggml_tensor * qkv_weight = nullptr;
    ggml_tensor * qkv_bias = nullptr;
    Linear out;
    ggml_tensor * norm2_weight = nullptr;
    ggml_tensor * norm2_bias = nullptr;
    Linear mlp_up, mlp_down;
};

struct Session {
    std::deque<std::vector<float>> latents;
    std::deque<std::vector<float>> actions;
};

struct QantaraModel final : ModelArchBase {
    int64_t latent = 0;
    int64_t action_block = 0;
    int64_t max_frames = 0;
    int64_t flow_steps = 0;
    int64_t heads = 0;
    int64_t head_dim = 0;
    int64_t mlp_dim = 0;
    float rms_eps = 1.0e-6f;
    float rope_theta = 10000.0f;
    int64_t image_size = 0;
    int64_t patch_size = 0;
    int64_t vision_heads = 0;
    int64_t vision_depth = 0;

    ggml_backend_t backend = nullptr;
    ggml_context * weights_ctx = nullptr;
    ggml_backend_buffer_t weights_buffer = nullptr;

    ggml_tensor * start = nullptr;
    ggml_tensor * modality = nullptr;
    Linear state_input, action_input, time_in, time_out;
    std::vector<Block> blocks;
    ggml_tensor * final_norm = nullptr;
    Linear state_head_in;
    ggml_tensor * state_bn_scale = nullptr;
    ggml_tensor * state_bn_offset = nullptr;
    Linear state_head_out, action_head;
    ggml_tensor * vision_patch_weight = nullptr;
    ggml_tensor * vision_patch_bias = nullptr;
    ggml_tensor * vision_cls = nullptr;
    ggml_tensor * vision_position = nullptr;
    std::vector<VisionBlock> vision_blocks;
    ggml_tensor * vision_final_norm_weight = nullptr;
    ggml_tensor * vision_final_norm_bias = nullptr;
    Linear encoder_projector_in;
    ggml_tensor * encoder_bn_scale = nullptr;
    ggml_tensor * encoder_bn_offset = nullptr;
    Linear encoder_projector_out;
    std::vector<float> action_mean;
    std::vector<float> action_std;
    std::unordered_map<uint64_t, Session> sessions;
    std::mt19937 rng{0};
    int n_threads = 4;

    QantaraModel() : ModelArchBase(Arch::QANTARA) {}
    ~QantaraModel() override {
        if (weights_buffer) ggml_backend_buffer_free(weights_buffer);
        if (weights_ctx) ggml_free(weights_ctx);
        if (backend) ggml_backend_free(backend);
    }
    std::vector<float> predict(const Inputs & in) override;
    bool encode(const ImageView & image, std::vector<float> * output);
};

ggml_tensor * mm(ggml_context * ctx, ggml_tensor * weight, ggml_tensor * x) {
    ggml_tensor * result = ggml_mul_mat(ctx, weight, x);
    ggml_mul_mat_set_prec(result, GGML_PREC_F32);
    return result;
}

ggml_tensor * linear(ggml_context * ctx, const Linear & layer, ggml_tensor * x) {
    ggml_tensor * result = mm(ctx, layer.weight, x);
    return layer.bias ? ggml_add(ctx, result, layer.bias) : result;
}

ggml_tensor * column(ggml_context * ctx, ggml_tensor * value, int64_t index) {
    return ggml_view_2d(
        ctx, value, value->ne[0], 1, value->nb[1],
        static_cast<size_t>(index) * value->nb[1]);
}

ggml_tensor * concat_columns(
    ggml_context * ctx, const std::vector<ggml_tensor *> & values
) {
    ggml_tensor * result = values.front();
    for (size_t i = 1; i < values.size(); ++i) {
        result = ggml_concat(ctx, result, values[i], 1);
    }
    return result;
}

ggml_tensor * affine_rms(
    ggml_context * ctx, ggml_tensor * x, ggml_tensor * weight, float eps
) {
    return ggml_mul(ctx, ggml_rms_norm(ctx, x, eps), weight);
}

ggml_tensor * attention(
    ggml_context * ctx,
    ggml_tensor * q,
    ggml_tensor * k,
    ggml_tensor * v,
    ggml_tensor * mask,
    int64_t inner,
    int64_t length,
    int64_t head_dim
) {
    ggml_tensor * Q = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));
    ggml_tensor * K = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));
    ggml_tensor * V = ggml_cont(ctx, ggml_permute(ctx, v, 1, 2, 0, 3));
    ggml_tensor * scores = mm(ctx, K, Q);
    ggml_tensor * probabilities = ggml_soft_max_ext(
        ctx, scores, mask, 1.0f / std::sqrt(static_cast<float>(head_dim)), 0.0f);
    ggml_tensor * attended = mm(ctx, V, probabilities);
    return ggml_reshape_2d(
        ctx,
        ggml_cont(ctx, ggml_permute(ctx, attended, 0, 2, 1, 3)),
        inner,
        length);
}

ggml_tensor * build_block(
    ggml_context * ctx,
    const QantaraModel & model,
    const Block & block,
    ggml_tensor * x,
    ggml_tensor * condition,
    ggml_tensor * positions,
    ggml_tensor * state_mask,
    ggml_tensor * action_mask
) {
    const int64_t length = x->ne[1];
    const int64_t hidden = model.cfg.hidden;
    const int64_t inner = model.heads * model.head_dim;

    ggml_tensor * modulation = linear(
        ctx, block.condition, ggml_silu(ctx, condition));
    const size_t row_bytes = static_cast<size_t>(hidden) * sizeof(float);
    auto slice = [&](int index) {
        return ggml_view_2d(
            ctx, modulation, hidden, length, modulation->nb[1],
            static_cast<size_t>(index) * row_bytes);
    };
    ggml_tensor * shift_a = slice(0);
    ggml_tensor * scale_a = slice(1);
    ggml_tensor * gate_a = slice(2);
    ggml_tensor * shift_m = slice(3);
    ggml_tensor * scale_m = slice(4);
    ggml_tensor * gate_m = slice(5);

    ggml_tensor * norm = ggml_rms_norm(ctx, x, model.rms_eps);
    norm = ggml_add(
        ctx, ggml_add(ctx, norm, ggml_mul(ctx, norm, scale_a)), shift_a);
    ggml_tensor * qkv = mm(ctx, block.qkv, norm);
    const size_t qkv_stride = qkv->nb[1];
    const size_t qkv_offset = static_cast<size_t>(inner) * sizeof(float);
    ggml_tensor * q = ggml_cont(ctx, ggml_view_2d(ctx, qkv, inner, length, qkv_stride, 0));
    ggml_tensor * k = ggml_cont(ctx, ggml_view_2d(ctx, qkv, inner, length, qkv_stride, qkv_offset));
    ggml_tensor * v = ggml_cont(ctx, ggml_view_2d(ctx, qkv, inner, length, qkv_stride, 2 * qkv_offset));
    q = ggml_reshape_3d(ctx, q, model.head_dim, model.heads, length);
    k = ggml_reshape_3d(ctx, k, model.head_dim, model.heads, length);
    v = ggml_reshape_3d(ctx, v, model.head_dim, model.heads, length);
    q = affine_rms(ctx, q, block.q_norm, model.rms_eps);
    k = affine_rms(ctx, k, block.k_norm, model.rms_eps);
    q = ggml_rope_ext(
        ctx, q, positions, nullptr, static_cast<int>(model.head_dim),
        GGML_ROPE_TYPE_NEOX, 0, model.rope_theta, 1.0f, 0.0f, 1.0f,
        32.0f, 1.0f);
    k = ggml_rope_ext(
        ctx, k, positions, nullptr, static_cast<int>(model.head_dim),
        GGML_ROPE_TYPE_NEOX, 0, model.rope_theta, 1.0f, 0.0f, 1.0f,
        32.0f, 1.0f);

    ggml_tensor * from_state = attention(
        ctx, q, k, v, state_mask, inner, length, model.head_dim);
    ggml_tensor * from_action = attention(
        ctx, q, k, v, action_mask, inner, length, model.head_dim);
    ggml_tensor * aa = linear(ctx, block.out_aa, from_action);
    ggml_tensor * az = linear(ctx, block.out_az, from_state);
    ggml_tensor * zz = linear(ctx, block.out_zz, from_state);
    ggml_tensor * za = linear(ctx, block.out_za, from_action);
    std::vector<ggml_tensor *> attention_columns;
    attention_columns.reserve(static_cast<size_t>(length));
    for (int64_t token = 0; token < length; ++token) {
        attention_columns.push_back(
            token % 2 == 0
                ? ggml_add(ctx, column(ctx, aa, token), column(ctx, az, token))
                : ggml_add(ctx, column(ctx, zz, token), column(ctx, za, token)));
    }
    ggml_tensor * attention_out = concat_columns(ctx, attention_columns);
    x = ggml_add(ctx, x, ggml_mul(ctx, gate_a, attention_out));

    norm = ggml_rms_norm(ctx, x, model.rms_eps);
    norm = ggml_add(
        ctx, ggml_add(ctx, norm, ggml_mul(ctx, norm, scale_m)), shift_m);
    ggml_tensor * hidden_mlp = ggml_gelu_erf(ctx, linear(ctx, block.mlp_up, norm));
    ggml_tensor * mlp_out = linear(ctx, block.mlp_down, hidden_mlp);
    return ggml_add(ctx, x, ggml_mul(ctx, gate_m, mlp_out));
}

struct QueryResult {
    ggml_tensor * velocity;
    ggml_tensor * state;
};

QueryResult build_query(
    ggml_context * ctx,
    const QantaraModel & model,
    ggml_tensor * z_history,
    ggml_tensor * action_history,
    ggml_tensor * current_action,
    ggml_tensor * current_state,
    ggml_tensor * time_features,
    ggml_tensor * positions,
    ggml_tensor * state_mask,
    ggml_tensor * action_mask
) {
    const int64_t history = z_history->ne[1];
    const int64_t blocks = history + 1;
    const int64_t length = 2 * blocks;

    ggml_tensor * all_states = ggml_concat(ctx, z_history, current_state, 1);
    ggml_tensor * state_tokens = linear(ctx, model.state_input, all_states);
    std::vector<ggml_tensor *> action_columns;
    action_columns.push_back(model.start);
    if (history > 1) {
        ggml_tensor * projected_history = linear(ctx, model.action_input, action_history);
        for (int64_t i = 0; i < history - 1; ++i) {
            action_columns.push_back(column(ctx, projected_history, i));
        }
    }
    action_columns.push_back(linear(ctx, model.action_input, current_action));
    ggml_tensor * action_tokens = concat_columns(ctx, action_columns);

    std::vector<ggml_tensor *> tokens;
    tokens.reserve(static_cast<size_t>(length));
    for (int64_t i = 0; i < blocks; ++i) {
        tokens.push_back(ggml_add(
            ctx, column(ctx, action_tokens, i), column(ctx, model.modality, 0)));
        tokens.push_back(ggml_add(
            ctx, column(ctx, state_tokens, i), column(ctx, model.modality, 1)));
    }
    ggml_tensor * x = concat_columns(ctx, tokens);
    ggml_tensor * condition = linear(
        ctx, model.time_out,
        ggml_silu(ctx, linear(ctx, model.time_in, time_features)));
    for (const Block & block : model.blocks) {
        x = build_block(
            ctx, model, block, x, condition, positions, state_mask, action_mask);
    }
    x = affine_rms(ctx, x, model.final_norm, model.rms_eps);

    ggml_tensor * velocity = linear(
        ctx, model.action_head, column(ctx, x, length - 2));
    ggml_tensor * state_hidden = linear(
        ctx, model.state_head_in, column(ctx, x, length - 1));
    state_hidden = ggml_add(
        ctx,
        ggml_mul(ctx, state_hidden, model.state_bn_scale),
        model.state_bn_offset);
    state_hidden = ggml_gelu_erf(ctx, state_hidden);
    ggml_tensor * delta = linear(ctx, model.state_head_out, state_hidden);
    ggml_tensor * predicted_state = ggml_add(
        ctx, column(ctx, z_history, history - 1), delta);
    return {velocity, predicted_state};
}

std::vector<float> time_features(
    int64_t blocks, float tau_a, float tau_z
) {
    const int64_t length = 2 * blocks;
    std::vector<float> result(static_cast<size_t>(256 * length));
    for (int64_t token = 0; token < length; ++token) {
        float tau = 1.0f;
        if (token == length - 2) tau = tau_a;
        if (token == length - 1) tau = tau_z;
        for (int64_t i = 0; i < 128; ++i) {
            const float frequency = std::exp(
                -std::log(10000.0f) * static_cast<float>(i) / 128.0f);
            const float phase = 1000.0f * tau * frequency;
            result[static_cast<size_t>(token * 256 + i)] = std::cos(phase);
            result[static_cast<size_t>(token * 256 + 128 + i)] = std::sin(phase);
        }
    }
    return result;
}

ggml_tensor * layer_norm(
    ggml_context * ctx,
    ggml_tensor * x,
    ggml_tensor * weight,
    ggml_tensor * bias
) {
    return ggml_add(
        ctx, ggml_mul(ctx, ggml_norm(ctx, x, 1.0e-5f), weight), bias);
}

ggml_tensor * build_vision_block(
    ggml_context * ctx,
    const VisionBlock & block,
    ggml_tensor * x,
    int64_t heads
) {
    const int64_t width = x->ne[0];
    const int64_t sequence = x->ne[1];
    const int64_t head_dim = width / heads;
    ggml_tensor * normalized = layer_norm(
        ctx, x, block.norm1_weight, block.norm1_bias);
    ggml_tensor * qkv = ggml_add(
        ctx, mm(ctx, block.qkv_weight, normalized), block.qkv_bias);
    const size_t stride = qkv->nb[1];
    const size_t offset = static_cast<size_t>(width) * sizeof(float);
    ggml_tensor * q = ggml_cont(
        ctx, ggml_view_2d(ctx, qkv, width, sequence, stride, 0));
    ggml_tensor * k = ggml_cont(
        ctx, ggml_view_2d(ctx, qkv, width, sequence, stride, offset));
    ggml_tensor * v = ggml_cont(
        ctx, ggml_view_2d(ctx, qkv, width, sequence, stride, 2 * offset));
    q = ggml_reshape_3d(ctx, q, head_dim, heads, sequence);
    k = ggml_reshape_3d(ctx, k, head_dim, heads, sequence);
    v = ggml_reshape_3d(ctx, v, head_dim, heads, sequence);
    ggml_tensor * attended = attention(
        ctx, q, k, v, nullptr, width, sequence, head_dim);
    x = ggml_add(ctx, x, linear(ctx, block.out, attended));
    normalized = layer_norm(ctx, x, block.norm2_weight, block.norm2_bias);
    ggml_tensor * hidden = ggml_gelu_erf(
        ctx, linear(ctx, block.mlp_up, normalized));
    return ggml_add(ctx, x, linear(ctx, block.mlp_down, hidden));
}

bool preprocess_qantara_image(
    const ImageView & image, int side, std::vector<float> * output
) {
    if (!image.data || image.w < 1 || image.h < 1 || side < 1) return false;
    constexpr float mean[3] = {0.485f, 0.456f, 0.406f};
    constexpr float stddev[3] = {0.229f, 0.224f, 0.225f};
    output->resize(static_cast<size_t>(3 * side * side));
    auto pixel = [&](int y, int x, int channel) {
        y = std::max(0, std::min(image.h - 1, y));
        x = std::max(0, std::min(image.w - 1, x));
        if (image.format == PixelFormat::U8) {
            const auto * data = static_cast<const uint8_t *>(image.data);
            return data[
                (static_cast<size_t>(y) * image.w + x) * 3 + channel] / 255.0f;
        }
        const auto * data = static_cast<const float *>(image.data);
        return data[(static_cast<size_t>(y) * image.w + x) * 3 + channel];
    };
    for (int y = 0; y < side; ++y) {
        const float source_y =
            (static_cast<float>(y) + 0.5f) * image.h / side - 0.5f;
        const int y_floor = static_cast<int>(std::floor(source_y));
        const int y0 = std::max(0, std::min(image.h - 1, y_floor));
        const int y1 = std::max(0, std::min(image.h - 1, y_floor + 1));
        const float wy = source_y - std::floor(source_y);
        for (int x = 0; x < side; ++x) {
            const float source_x =
                (static_cast<float>(x) + 0.5f) * image.w / side - 0.5f;
            const int x_floor = static_cast<int>(std::floor(source_x));
            const int x0 = std::max(0, std::min(image.w - 1, x_floor));
            const int x1 = std::max(0, std::min(image.w - 1, x_floor + 1));
            const float wx = source_x - std::floor(source_x);
            for (int channel = 0; channel < 3; ++channel) {
                const float top =
                    pixel(y0, x0, channel) * (1.0f - wx) +
                    pixel(y0, x1, channel) * wx;
                const float bottom =
                    pixel(y1, x0, channel) * (1.0f - wx) +
                    pixel(y1, x1, channel) * wx;
                const float value = top * (1.0f - wy) + bottom * wy;
                (*output)[
                    static_cast<size_t>(channel * side * side + y * side + x)] =
                    (value - mean[channel]) / stddev[channel];
            }
        }
    }
    return true;
}

} // namespace

bool QantaraModel::encode(
    const ImageView & image, std::vector<float> * output
) {
    using Clock = std::chrono::high_resolution_clock;
    const auto started = Clock::now();
    const int64_t grid = image_size / patch_size;
    const int64_t patches = grid * grid;
    const int64_t sequence = patches + 1;
    ggml_init_params params{64u * 1024u * 1024u, nullptr, true};
    ggml_context * ctx = ggml_init(params);
    if (!ctx) return false;
    ggml_tensor * pixels = ggml_new_tensor_3d(
        ctx, GGML_TYPE_F32, image_size, image_size, 3);
    ggml_set_input(pixels);
    ggml_tensor * convolution = ggml_conv_2d(
        ctx, vision_patch_weight, pixels,
        static_cast<int>(patch_size), static_cast<int>(patch_size),
        0, 0, 1, 1);
    ggml_tensor * patch_tokens = ggml_transpose(
        ctx, ggml_reshape_2d(ctx, convolution, patches, cfg.hidden));
    patch_tokens = ggml_add(ctx, ggml_cont(ctx, patch_tokens), vision_patch_bias);
    ggml_tensor * x = ggml_concat(ctx, vision_cls, patch_tokens, 1);
    x = ggml_add(ctx, x, vision_position);
    for (const VisionBlock & block : vision_blocks) {
        x = build_vision_block(ctx, block, x, vision_heads);
    }
    x = layer_norm(
        ctx, x, vision_final_norm_weight, vision_final_norm_bias);
    ggml_tensor * cls = column(ctx, x, 0);
    ggml_tensor * projected = linear(ctx, encoder_projector_in, cls);
    projected = ggml_add(
        ctx, ggml_mul(ctx, projected, encoder_bn_scale), encoder_bn_offset);
    projected = ggml_gelu_erf(ctx, projected);
    projected = linear(ctx, encoder_projector_out, projected);
    ggml_set_output(projected);
    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 32768, false);
    ggml_build_forward_expand(graph, projected);
    ggml_gallocr_t allocator = ggml_gallocr_new(
        ggml_backend_get_default_buffer_type(backend));
    if (!allocator || !ggml_gallocr_alloc_graph(allocator, graph)) {
        if (allocator) ggml_gallocr_free(allocator);
        ggml_free(ctx);
        return false;
    }
    std::vector<float> preprocessed;
    if (!preprocess_qantara_image(
            image, static_cast<int>(image_size), &preprocessed)) {
        ggml_gallocr_free(allocator);
        ggml_free(ctx);
        return false;
    }
    ggml_backend_tensor_set(
        pixels, preprocessed.data(), 0, preprocessed.size() * sizeof(float));
    const ggml_status status = ggml_backend_graph_compute(backend, graph);
    output->resize(static_cast<size_t>(latent));
    if (status == GGML_STATUS_SUCCESS) {
        ggml_backend_tensor_get(
            projected, output->data(), 0, output->size() * sizeof(float));
    } else {
        output->clear();
    }
    ggml_gallocr_free(allocator);
    ggml_free(ctx);
    stats.ms_vision += std::chrono::duration<float, std::milli>(
        Clock::now() - started).count();
    return status == GGML_STATUS_SUCCESS;
}

std::vector<float> QantaraModel::predict(const Inputs & in) {
    using Clock = std::chrono::high_resolution_clock;
    const auto started = Clock::now();
    stats = {};
    if (in.qantara_encode_only) {
        if (!in.images || in.n_images != 1) return {};
        std::vector<float> encoded;
        if (!encode(in.images[0], &encoded)) return {};
        stats.ms_total = std::chrono::duration<float, std::milli>(
            Clock::now() - started).count();
        return encoded;
    }
    const bool session_mode = in.qantara_latents == nullptr;
    const float * latent_values = in.qantara_latents;
    const float * action_values = in.qantara_actions;
    int64_t history = in.qantara_history;
    int64_t action_blocks = in.qantara_action_blocks;
    std::vector<float> flattened_latents;
    std::vector<float> flattened_actions;
    if (session_mode) {
        if (!in.images || in.n_images != 1) {
            std::fprintf(stderr, "vla(qantara): session inference needs one image\n");
            return {};
        }
        if (in.qantara_reset) {
            sessions.erase(in.qantara_session_id);
        }
        Session & session = sessions[in.qantara_session_id];
        if (!session.latents.empty()) {
            if (!in.qantara_previous_action ||
                in.qantara_previous_action_n != action_block) {
                std::fprintf(
                    stderr,
                    "vla(qantara): non-initial request needs the executed action block\n");
                return {};
            }
            std::vector<float> normalized(static_cast<size_t>(action_block));
            for (int64_t i = 0; i < action_block; ++i) {
                const int64_t dimension = i % cfg.real_action_dim;
                normalized[static_cast<size_t>(i)] =
                    (in.qantara_previous_action[i] -
                     action_mean[static_cast<size_t>(dimension)]) /
                    action_std[static_cast<size_t>(dimension)];
            }
            session.actions.push_back(std::move(normalized));
        }
        std::vector<float> current_latent;
        if (!encode(in.images[0], &current_latent)) {
            std::fprintf(stderr, "vla(qantara): vision encoding failed\n");
            return {};
        }
        session.latents.push_back(std::move(current_latent));
        while (session.latents.size() >
               static_cast<size_t>(max_frames - 1)) {
            session.latents.pop_front();
            if (!session.actions.empty()) session.actions.pop_front();
        }
        if (session.actions.size() + 1 != session.latents.size()) {
            std::fprintf(stderr, "vla(qantara): session history lost alignment\n");
            return {};
        }
        history = static_cast<int64_t>(session.latents.size());
        action_blocks = static_cast<int64_t>(session.actions.size());
        flattened_latents.reserve(static_cast<size_t>(history * latent));
        for (const auto & value : session.latents) {
            flattened_latents.insert(
                flattened_latents.end(), value.begin(), value.end());
        }
        flattened_actions.reserve(
            static_cast<size_t>(action_blocks * action_block));
        for (const auto & value : session.actions) {
            flattened_actions.insert(
                flattened_actions.end(), value.begin(), value.end());
        }
        latent_values = flattened_latents.data();
        action_values = flattened_actions.empty()
            ? nullptr : flattened_actions.data();
    }
    if (!latent_values || history < 1 || history >= max_frames) {
        std::fprintf(stderr, "vla(qantara): expected 1..%lld history latents\n",
                     static_cast<long long>(max_frames - 1));
        return {};
    }
    if (action_blocks != history - 1 ||
        (history > 1 && !action_values)) {
        std::fprintf(stderr, "vla(qantara): action history is not aligned\n");
        return {};
    }
    const int64_t blocks_count = history + 1;
    const int64_t length = 2 * blocks_count;

    ggml_init_params params{256u * 1024u * 1024u, nullptr, true};
    ggml_context * ctx = ggml_init(params);
    if (!ctx) return {};
    ggml_tensor * z_history = ggml_new_tensor_2d(
        ctx, GGML_TYPE_F32, latent, history);
    ggml_set_input(z_history);
    ggml_tensor * action_history = ggml_new_tensor_2d(
        ctx, GGML_TYPE_F32, action_block, std::max<int64_t>(1, history - 1));
    ggml_set_input(action_history);
    ggml_tensor * video_noise = ggml_new_tensor_2d(
        ctx, GGML_TYPE_F32, action_block, 1);
    ggml_set_input(video_noise);
    ggml_tensor * action_noise = ggml_new_tensor_2d(
        ctx, GGML_TYPE_F32, action_block, 1);
    ggml_set_input(action_noise);
    ggml_tensor * positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, length);
    ggml_set_input(positions);
    ggml_tensor * state_mask = ggml_new_tensor_2d(
        ctx, GGML_TYPE_F16, length, length);
    ggml_set_input(state_mask);
    ggml_tensor * action_mask = ggml_new_tensor_2d(
        ctx, GGML_TYPE_F16, length, length);
    ggml_set_input(action_mask);

    std::vector<ggml_tensor *> times;
    times.reserve(static_cast<size_t>(flow_steps + 1));
    for (int64_t i = 0; i <= flow_steps; ++i) {
        ggml_tensor * value = ggml_new_tensor_2d(
            ctx, GGML_TYPE_F32, 256, length);
        ggml_set_input(value);
        times.push_back(value);
    }

    QueryResult video = build_query(
        ctx, *this, z_history, action_history, video_noise,
        column(ctx, z_history, history - 1), times[0],
        positions, state_mask, action_mask);
    ggml_tensor * action = action_noise;
    for (int64_t step = 0; step < flow_steps; ++step) {
        QueryResult inverse = build_query(
            ctx, *this, z_history, action_history, action, video.state,
            times[static_cast<size_t>(step + 1)],
            positions, state_mask, action_mask);
        action = ggml_add(
            ctx, action,
            ggml_scale(ctx, inverse.velocity, 1.0f / static_cast<float>(flow_steps)));
    }
    ggml_set_output(action);
    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 65536, false);
    ggml_build_forward_expand(graph, action);
    ggml_gallocr_t allocator = ggml_gallocr_new(
        ggml_backend_get_default_buffer_type(backend));
    if (!allocator || !ggml_gallocr_alloc_graph(allocator, graph)) {
        std::fprintf(stderr, "vla(qantara): graph allocation failed\n");
        if (allocator) ggml_gallocr_free(allocator);
        ggml_free(ctx);
        return {};
    }

    ggml_backend_tensor_set(
        z_history, latent_values, 0,
        static_cast<size_t>(latent * history) * sizeof(float));
    if (history > 1) {
        ggml_backend_tensor_set(
            action_history, action_values, 0,
            static_cast<size_t>(action_block * (history - 1)) * sizeof(float));
    }
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::vector<float> generated_video(static_cast<size_t>(action_block));
    std::vector<float> generated_action(static_cast<size_t>(action_block));
    for (float & value : generated_video) value = normal(rng);
    for (float & value : generated_action) value = normal(rng);
    ggml_backend_tensor_set(
        video_noise,
        in.qantara_video_noise ? in.qantara_video_noise : generated_video.data(),
        0, generated_video.size() * sizeof(float));
    ggml_backend_tensor_set(
        action_noise,
        in.qantara_action_noise ? in.qantara_action_noise : generated_action.data(),
        0, generated_action.size() * sizeof(float));

    std::vector<int32_t> position_values(static_cast<size_t>(length));
    for (int64_t i = 0; i < length; ++i) {
        position_values[static_cast<size_t>(i)] = static_cast<int32_t>(i / 2);
    }
    ggml_backend_tensor_set(
        positions, position_values.data(), 0,
        position_values.size() * sizeof(int32_t));
    std::vector<ggml_fp16_t> state_mask_values(
        static_cast<size_t>(length * length));
    std::vector<ggml_fp16_t> action_mask_values(
        static_cast<size_t>(length * length));
    for (int64_t query = 0; query < length; ++query) {
        for (int64_t key = 0; key < length; ++key) {
            const bool causal = key / 2 <= query / 2;
            state_mask_values[static_cast<size_t>(query * length + key)] =
                ggml_fp32_to_fp16(causal && key % 2 == 1 ? 0.0f : -INFINITY);
            action_mask_values[static_cast<size_t>(query * length + key)] =
                ggml_fp32_to_fp16(causal && key % 2 == 0 ? 0.0f : -INFINITY);
        }
    }
    ggml_backend_tensor_set(
        state_mask, state_mask_values.data(), 0,
        state_mask_values.size() * sizeof(ggml_fp16_t));
    ggml_backend_tensor_set(
        action_mask, action_mask_values.data(), 0,
        action_mask_values.size() * sizeof(ggml_fp16_t));
    for (int64_t i = 0; i <= flow_steps; ++i) {
        const float tau_a = i == 0 ? 0.0f : static_cast<float>(i - 1) /
            static_cast<float>(flow_steps);
        const float tau_z = i == 0 ? 0.0f : 1.0f;
        std::vector<float> values = time_features(blocks_count, tau_a, tau_z);
        ggml_backend_tensor_set(
            times[static_cast<size_t>(i)], values.data(), 0,
            values.size() * sizeof(float));
    }

    const auto inference_started = Clock::now();
    const ggml_status status = ggml_backend_graph_compute(backend, graph);
    stats.ms_inference = std::chrono::duration<float, std::milli>(
        Clock::now() - inference_started).count();
    std::vector<float> result(static_cast<size_t>(action_block));
    if (status == GGML_STATUS_SUCCESS) {
        ggml_backend_tensor_get(
            action, result.data(), 0, result.size() * sizeof(float));
        if (session_mode) {
            for (int64_t i = 0; i < action_block; ++i) {
                const int64_t dimension = i % cfg.real_action_dim;
                result[static_cast<size_t>(i)] =
                    result[static_cast<size_t>(i)] *
                    action_std[static_cast<size_t>(dimension)] +
                    action_mean[static_cast<size_t>(dimension)];
                result[static_cast<size_t>(i)] = std::max(
                    -1.0f, std::min(1.0f, result[static_cast<size_t>(i)]));
            }
        }
    } else {
        std::fprintf(stderr, "vla(qantara): graph compute failed (%d)\n",
                     static_cast<int>(status));
        result.clear();
    }
    ggml_gallocr_free(allocator);
    ggml_free(ctx);
    stats.ms_total = std::chrono::duration<float, std::milli>(
        Clock::now() - started).count();
    return result;
}

std::unique_ptr<ModelArchBase> qantara_create(
    const std::string &, const std::string & checkpoint, const std::string &
) {
    GGUFReader reader;
    if (!reader.open(checkpoint)) {
        std::fprintf(stderr, "vla(qantara): failed to open %s\n", checkpoint.c_str());
        return nullptr;
    }
    auto model = std::make_unique<QantaraModel>();
    model->latent = reader.u32("qantara.latent_dim");
    model->cfg.hidden = reader.u32("qantara.hidden_dim");
    model->cfg.n_layers = reader.u32("qantara.depth");
    model->heads = reader.u32("qantara.heads");
    model->head_dim = reader.u32("qantara.head_dim");
    model->mlp_dim = reader.u32("qantara.mlp_dim");
    model->max_frames = reader.u32("qantara.num_frames");
    model->cfg.real_action_dim = reader.u32("qantara.action_dim");
    model->cfg.max_action_dim = model->cfg.real_action_dim;
    model->cfg.n_suffix = reader.u32("qantara.frameskip");
    model->action_block = reader.u32("qantara.action_block_dim");
    model->flow_steps = reader.u32("qantara.flow_steps");
    model->image_size = reader.u32("qantara.image_size");
    model->patch_size = reader.u32("qantara.patch_size");
    model->vision_depth = reader.u32("qantara.vision_depth");
    model->vision_heads = reader.u32("qantara.vision_heads");
    model->cfg.num_steps = static_cast<int>(model->flow_steps);
    model->cfg.head_dim = model->head_dim;
    model->cfg.n_q_heads = model->heads;
    model->cfg.n_kv_heads = model->heads;
    model->cfg.q_full_dim = model->heads * model->head_dim;
    model->cfg.kv_full_dim = model->cfg.q_full_dim;
    model->cfg.expert_h = model->cfg.hidden;
    model->cfg.expert_inter = model->mlp_dim;
    model->cfg.rms_eps = reader.f32("qantara.rms_norm_eps");
    model->rms_eps = model->cfg.rms_eps;
    model->rope_theta = reader.f32("qantara.rope_theta");
    model->cfg.n_img = 1;
    model->cfg.n_lang = 0;
    model->cfg.n_state = 0;
    model->cfg.max_state_dim = 0;
    model->cfg.real_state_dim = 0;

#ifdef GGML_USE_CUDA
    model->backend = ggml_backend_cuda_init(0);
#endif
    if (!model->backend) {
        model->backend = ggml_backend_cpu_init();
        ggml_backend_cpu_set_n_threads(model->backend, model->n_threads);
        std::printf("vla(qantara): backend = CPU\n");
    } else {
        std::printf("vla(qantara): backend = CUDA\n");
    }
    ggml_init_params params{16u * 1024u * 1024u, nullptr, true};
    model->weights_ctx = ggml_init(params);
    if (!model->weights_ctx) return nullptr;
    std::vector<ggml_tensor *> weights;
    auto make = [&](const std::string & name) -> ggml_tensor * {
        const ggml_tensor * source = reader.tensor(name.c_str());
        if (!source) {
            std::fprintf(stderr, "vla(qantara): missing tensor %s\n", name.c_str());
            return nullptr;
        }
        ggml_tensor * tensor = ggml_new_tensor(
            model->weights_ctx, GGML_TYPE_F32, GGML_MAX_DIMS, source->ne);
        ggml_set_name(tensor, name.c_str());
        weights.push_back(tensor);
        return tensor;
    };
    auto make_linear = [&](const std::string & prefix) {
        return Linear{make(prefix + ".weight"), make(prefix + ".bias")};
    };
    model->start = make("qantara.start");
    model->modality = make("qantara.modality");
    model->state_input = make_linear("qantara.state_input");
    model->action_input = make_linear("qantara.action_input");
    model->time_in = make_linear("qantara.time.mlp.0");
    model->time_out = make_linear("qantara.time.mlp.2");
    model->blocks.resize(static_cast<size_t>(model->cfg.n_layers));
    for (int64_t i = 0; i < model->cfg.n_layers; ++i) {
        const std::string prefix = "qantara.blocks." + std::to_string(i);
        Block & block = model->blocks[static_cast<size_t>(i)];
        block.qkv = make(prefix + ".attention.qkv.weight");
        block.q_norm = make(prefix + ".attention.q_norm.weight");
        block.k_norm = make(prefix + ".attention.k_norm.weight");
        block.out_aa = make_linear(prefix + ".attention.out_aa");
        block.out_az = make_linear(prefix + ".attention.out_az");
        block.out_zz = make_linear(prefix + ".attention.out_zz");
        block.out_za = make_linear(prefix + ".attention.out_za");
        block.mlp_up = make_linear(prefix + ".mlp.0");
        block.mlp_down = make_linear(prefix + ".mlp.3");
        block.condition = make_linear(prefix + ".condition.1");
    }
    model->final_norm = make("qantara.norm.weight");
    model->state_head_in = make_linear("qantara.state_head.net.0");
    model->state_bn_scale = make("qantara.state_head.net.1.scale");
    model->state_bn_offset = make("qantara.state_head.net.1.offset");
    model->state_head_out = make_linear("qantara.state_head.net.3");
    model->action_head = make_linear("qantara.action_head");
    model->vision_cls = make("qantara.vision.cls");
    model->vision_position = make("qantara.vision.position");
    model->vision_patch_weight = make("qantara.vision.patch.weight");
    model->vision_patch_bias = make("qantara.vision.patch.bias");
    model->vision_blocks.resize(static_cast<size_t>(model->vision_depth));
    for (int64_t i = 0; i < model->vision_depth; ++i) {
        const std::string prefix =
            "qantara.vision.blk." + std::to_string(i);
        VisionBlock & block = model->vision_blocks[static_cast<size_t>(i)];
        block.norm1_weight = make(prefix + ".norm1.weight");
        block.norm1_bias = make(prefix + ".norm1.bias");
        block.qkv_weight = make(prefix + ".attn_qkv.weight");
        block.qkv_bias = make(prefix + ".attn_qkv.bias");
        block.out = make_linear(prefix + ".attn_out");
        block.norm2_weight = make(prefix + ".norm2.weight");
        block.norm2_bias = make(prefix + ".norm2.bias");
        block.mlp_up = make_linear(prefix + ".ffn_up");
        block.mlp_down = make_linear(prefix + ".ffn_down");
    }
    model->vision_final_norm_weight = make(
        "qantara.vision.norm.weight");
    model->vision_final_norm_bias = make(
        "qantara.vision.norm.bias");
    model->encoder_projector_in = make_linear(
        "qantara.encoder.proj_in");
    model->encoder_bn_scale = make(
        "qantara.encoder.proj_norm.scale");
    model->encoder_bn_offset = make(
        "qantara.encoder.proj_norm.offset");
    model->encoder_projector_out = make_linear(
        "qantara.encoder.proj_out");
    if (std::any_of(weights.begin(), weights.end(), [](ggml_tensor * value) {
            return value == nullptr;
        })) return nullptr;
    model->weights_buffer = ggml_backend_alloc_ctx_tensors(
        model->weights_ctx, model->backend);
    if (!model->weights_buffer) return nullptr;
    for (ggml_tensor * tensor : weights) {
        std::vector<uint8_t> bytes(ggml_nbytes(tensor));
        if (!reader.read(tensor->name, bytes.data(), bytes.size())) {
            std::fprintf(stderr, "vla(qantara): failed to load %s\n", tensor->name);
            return nullptr;
        }
        ggml_backend_tensor_set(tensor, bytes.data(), 0, bytes.size());
    }
    model->action_mean.resize(
        static_cast<size_t>(model->cfg.real_action_dim));
    model->action_std.resize(
        static_cast<size_t>(model->cfg.real_action_dim));
    if (!reader.read(
            "qantara.action_mean", model->action_mean.data(),
            model->action_mean.size() * sizeof(float)) ||
        !reader.read(
            "qantara.action_std", model->action_std.data(),
            model->action_std.size() * sizeof(float))) {
        std::fprintf(stderr, "vla(qantara): failed to load action statistics\n");
        return nullptr;
    }
    std::printf(
        "vla(qantara): loaded predictor (%.1f MiB, frames=%lld, flow_steps=%lld)\n",
        ggml_backend_buffer_get_size(model->weights_buffer) / (1024.0 * 1024.0),
        static_cast<long long>(model->max_frames),
        static_cast<long long>(model->flow_steps));
    return model;
}

} // namespace vla
