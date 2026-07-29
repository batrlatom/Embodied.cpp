#!/usr/bin/env bash
set -euo pipefail

ROOT=/opt/embodied
SERVER="${ROOT}/build/vla-pi05-server"
MM_PROJ="${ROOT}/checkpoints/pi05/pi05-mmproj.gguf"
POLICY="${ROOT}/checkpoints/pi05/pi05.gguf"
PYTHON="${ROOT}/eval/sim/libero/libero_uv/.venv/bin/python"

case "${1:-server}" in
    server)
        shift || true
        exec env LD_LIBRARY_PATH="${ROOT}/build:${ROOT}/build/bin:${LD_LIBRARY_PATH:-}" \
            "${SERVER}" "${MM_PROJ}" "${POLICY}" "$@"
        ;;
    libero)
        shift || true
        exec "${PYTHON}" "${ROOT}/eval/client/run_sim_client_direct.py" \
            --arch pi05 \
            --tokenizer lerobot/pi05_libero \
            --vla-addr tcp://127.0.0.1:5555 \
            "$@"
        ;;
    shell|bash)
        shift || true
        exec /bin/bash "$@"
        ;;
    *)
        exec "$@"
        ;;
esac
