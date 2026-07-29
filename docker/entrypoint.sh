#!/usr/bin/env bash
set -euo pipefail

ROOT=/opt/embodied
MODEL_CACHE_DIR="${MODEL_CACHE_DIR:-/models}"
MODEL="${MODEL:-pi05}"
PORT="${PORT:-5555}"
HY_VLA_FILE="${HY_VLA_FILE:-Hy-Embodied-0.5-VLA-RoboTwin/Hy-Embodied-0.5-VLA-RoboTwin_q4_K.gguf}"
ASSETS="${ROOT}/docker/model_assets.sh"
PYTHON="${ROOT}/eval/sim/libero/libero_uv/bin/python"
for candidate in \
    "${ROOT}/eval/sim/libero/libero_uv/.venv/bin/python" \
    "${ROOT}/eval/sim/libero/libero_uv/.venv/bin/python3" \
    "${ROOT}/eval/sim/libero/libero_uv/.venv/bin/python3.10" \
    "${ROOT}/eval/sim/libero/libero_uv/bin/python"; do
    if [[ -x "${candidate}" ]]; then
        PYTHON="${candidate}"
        break
    fi
done

normalize_model() {
    case "$1" in
        pi05|pi0.5) echo pi05 ;;
        hy_vla|hy-vla|hy) echo hy_vla ;;
        groot_n1|groot-n1|groot) echo groot_n1 ;;
        lingbot_va|lingbot-va|lingbot) echo lingbot_va ;;
        *)
            echo "unsupported MODEL=$1; choose pi05, hy_vla, groot_n1, or lingbot_va" >&2
            exit 2
            ;;
    esac
}

MODEL="$(normalize_model "${MODEL}")"

prepare_assets() {
    MODEL_CACHE_DIR="${MODEL_CACHE_DIR}" MODEL="${MODEL}" \
        "${ASSETS}" "${MODEL}"
}

server_env=(
    env
    LD_LIBRARY_PATH="${ROOT}/build:${ROOT}/build/bin:${LD_LIBRARY_PATH:-}"
)

case "${1:-server}" in
    prepare)
        prepare_assets
        ;;
    server)
        shift || true
        prepare_assets
        case "${MODEL}" in
            pi05)
                exec "${server_env[@]}" "${ROOT}/build/vla-pi05-server" \
                    --bind "tcp://*:${PORT}" \
                    "${MODEL_CACHE_DIR}/pi05/pi05-mmproj.gguf" \
                    "${MODEL_CACHE_DIR}/pi05/pi05.gguf" "$@"
                ;;
            hy_vla)
                exec "${server_env[@]}" "${ROOT}/build/vla-hy-vla-server" \
                    --bind "tcp://*:${PORT}" \
                    "${MODEL_CACHE_DIR}/hy_vla/${HY_VLA_FILE##*/}" "$@"
                ;;
            groot_n1)
                exec "${server_env[@]}" \
                    VLA_GROOT_WEIGHT_DTYPE="${VLA_GROOT_WEIGHT_DTYPE:-bf16}" \
                    GGML_CUDA_FORCE_CUBLAS_COMPUTE_32F="${GGML_CUDA_FORCE_CUBLAS_COMPUTE_32F:-1}" \
                    "${ROOT}/build/vla-groot-n1-server" \
                    --bind "tcp://*:${PORT}" \
                    --backbone "${MODEL_CACHE_DIR}/groot_n1/qwen3vl-backbone-bf16.gguf" \
                    "${MODEL_CACHE_DIR}/groot_n1/qwen3vl-mmproj-bf16.gguf" \
                    "${MODEL_CACHE_DIR}/groot_n1/groot-n1.7-libero-object-action-head-bf16.gguf" "$@"
                ;;
            lingbot_va)
                exec "${server_env[@]}" \
                    VLA_LINGBOT_RUNTIME_CUDA="${VLA_LINGBOT_RUNTIME_CUDA:-1}" \
                    VLA_LINGBOT_REQUIRE_CUDA="${VLA_LINGBOT_REQUIRE_CUDA:-1}" \
                    VLA_LINGBOT_RESIDENT_BLOCK_DTYPE="${VLA_LINGBOT_RESIDENT_BLOCK_DTYPE:-q4_K}" \
                    VLA_LINGBOT_PREDICT_TEXT_ENCODER="${VLA_LINGBOT_PREDICT_TEXT_ENCODER:-1}" \
                    VLA_LINGBOT_TEXT_GGUF="${VLA_LINGBOT_TEXT_GGUF:-${MODEL_CACHE_DIR}/lingbot_va/lingbot_va_text_encoder_bf16.gguf}" \
                    VLA_LINGBOT_VAE_GGUF="${VLA_LINGBOT_VAE_GGUF:-${MODEL_CACHE_DIR}/lingbot_va/lingbot_va_vae_encoder_f32.gguf}" \
                    "${ROOT}/build/wam-lingbot-server" \
                    --bind "tcp://*:${PORT}" \
                    "${MODEL_CACHE_DIR}/lingbot_va/lingbot_va_transformer_wan_q4_K.gguf" "$@"
                ;;
        esac
        ;;
    libero)
        shift || true
        prepare_assets
        case "${MODEL}" in
            lingbot_va)
                exec "${PYTHON}" "${ROOT}/eval/client/run_sim_client_direct.py" \
                    --arch lingbot_va \
                    --tokenizer "${MODEL_CACHE_DIR}/lingbot_va/tokenizer" \
                    --vla-addr "tcp://127.0.0.1:${PORT}" "$@"
                ;;
            pi05)
                exec "${PYTHON}" "${ROOT}/eval/client/run_sim_client_direct.py" \
                    --arch pi05 \
                    --tokenizer "${MODEL_CACHE_DIR}/pi05/paligemma_tokenizer.model" \
                    --vla-addr "tcp://127.0.0.1:${PORT}" "$@"
                ;;
            *)
                echo "LIBERO client wiring is currently provided for MODEL=pi05 or lingbot_va" >&2
                exit 2
                ;;
        esac
        ;;
    shell|bash)
        shift || true
        exec /bin/bash "$@"
        ;;
    *)
        exec "$@"
        ;;
esac
