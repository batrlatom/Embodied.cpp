#!/usr/bin/env bash
set -euo pipefail

MODEL_CACHE_DIR="${MODEL_CACHE_DIR:-/models}"
HF_CACHE_DIR="${HF_HOME:-${MODEL_CACHE_DIR}/.cache/huggingface}"
PI05_REPO="${PI05_HF_REPO:-SEU-PAISys/Embodied.cpp}"
PI05_REVISION="${PI05_HF_REVISION:-main}"
EMBODIED_REPO="${EMBODIED_HF_REPO:-SEU-PAISys/Embodied.cpp}"
EMBODIED_REVISION="${EMBODIED_HF_REVISION:-main}"
LINGBOT_REPO="${LINGBOT_HF_REPO:-robbyant/lingbot-va-posttrain-libero-long}"
MODEL="${1:-${MODEL:-pi05}}"

case "${MODEL}" in
    pi05|pi0.5) MODEL=pi05 ;;
    hy_vla|hy-vla|hy) MODEL=hy_vla ;;
    groot_n1|groot-n1|groot) MODEL=groot_n1 ;;
    lingbot_va|lingbot-va|lingbot) MODEL=lingbot_va ;;
    *)
        echo "unsupported MODEL=${MODEL}; choose pi05, hy_vla, groot_n1, or lingbot_va" >&2
        exit 2
        ;;
esac

mkdir -p "${MODEL_CACHE_DIR}" "${HF_CACHE_DIR}"
exec 9>"${MODEL_CACHE_DIR}/.download.lock"
flock 9

download_hf() {
    local repo="$1"
    local revision="$2"
    local filename="$3"
    local destination="$4"

    if [[ -s "${destination}" ]]; then
        echo "[assets] cached ${destination}"
        return
    fi

    mkdir -p "$(dirname "${destination}")"
    echo "[assets] downloading ${repo}:${filename}"
HF_HOME="${HF_CACHE_DIR}" python - "${repo}" "${revision}" "${filename}" "${destination}" <<'PY'
import os
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

repo, revision, filename, destination = sys.argv[1:]
path = Path(hf_hub_download(
    repo_id=repo,
    filename=filename,
    revision=revision,
    cache_dir=os.environ["HF_HOME"],
))
target = Path(destination)
target.parent.mkdir(parents=True, exist_ok=True)
tmp = target.with_name(target.name + ".tmp")
source = path.resolve()
try:
    os.link(source, tmp)
except OSError:
    # This fallback is useful when the HF cache and model directory are on
    # different filesystems. The normal /models layout uses a hard link and
    # therefore does not duplicate multi-GiB model files.
    import shutil
    shutil.copyfile(source, tmp)
tmp.replace(target)
PY
}

download_url() {
    local url="$1"
    local destination="$2"
    if [[ -s "${destination}" ]]; then
        echo "[assets] cached ${destination}"
        return
    fi
    mkdir -p "$(dirname "${destination}")"
    echo "[assets] downloading ${url}"
    wget -q --show-progress -O "${destination}.tmp" "${url}"
    mv "${destination}.tmp" "${destination}"
}

prepare_pi05() {
    local dir="${MODEL_CACHE_DIR}/pi05"
    download_hf "${PI05_REPO}" "${PI05_REVISION}" \
        "pi05_libero_finetuned_v044/pi05-mmproj.gguf" \
        "${dir}/pi05-mmproj.gguf"
    download_hf "${PI05_REPO}" "${PI05_REVISION}" \
        "pi05_libero_finetuned_v044/pi05.gguf" \
        "${dir}/pi05.gguf"
    download_url \
        "https://storage.googleapis.com/big_vision/paligemma_tokenizer.model" \
        "${dir}/paligemma_tokenizer.model"
}

prepare_hy_vla() {
    local dir="${MODEL_CACHE_DIR}/hy_vla"
    local file="${HY_VLA_FILE:-Hy-Embodied-0.5-VLA-RoboTwin/Hy-Embodied-0.5-VLA-RoboTwin_q4_K.gguf}"
    download_hf "${EMBODIED_REPO}" "${EMBODIED_REVISION}" "${file}" \
        "${dir}/$(basename "${file}")"
}

prepare_groot_n1() {
    local dir="${MODEL_CACHE_DIR}/groot_n1"
    download_hf "${EMBODIED_REPO}" "${EMBODIED_REVISION}" \
        "groot-n1/qwen3vl-backbone-bf16.gguf" \
        "${dir}/qwen3vl-backbone-bf16.gguf"
    download_hf "${EMBODIED_REPO}" "${EMBODIED_REVISION}" \
        "groot-n1/qwen3vl-mmproj-bf16.gguf" \
        "${dir}/qwen3vl-mmproj-bf16.gguf"
    download_hf "${EMBODIED_REPO}" "${EMBODIED_REVISION}" \
        "groot-n1/groot-n1.7-libero-object-action-head-bf16.gguf" \
        "${dir}/groot-n1.7-libero-object-action-head-bf16.gguf"
}

prepare_lingbot_va() {
    local dir="${MODEL_CACHE_DIR}/lingbot_va"
    download_hf "${EMBODIED_REPO}" "${EMBODIED_REVISION}" \
        "lingbot-va-LIBERO/lingbot_va_transformer_wan_q4_K.gguf" \
        "${dir}/lingbot_va_transformer_wan_q4_K.gguf"
    download_hf "${EMBODIED_REPO}" "${EMBODIED_REVISION}" \
        "lingbot-va-LIBERO/lingbot_va_text_encoder_bf16.gguf" \
        "${dir}/lingbot_va_text_encoder_bf16.gguf"
    download_hf "${EMBODIED_REPO}" "${EMBODIED_REVISION}" \
        "lingbot-va-LIBERO/lingbot_va_vae_encoder_f32.gguf" \
        "${dir}/lingbot_va_vae_encoder_f32.gguf"

    local tokenizer_dir="${dir}/tokenizer"
    if [[ ! -f "${tokenizer_dir}/tokenizer.json" ]]; then
        echo "[assets] downloading ${LINGBOT_REPO}:tokenizer"
        HF_HOME="${HF_CACHE_DIR}" python - "${LINGBOT_REPO}" "${dir}" <<'PY'
import os
import sys
from huggingface_hub import snapshot_download

repo, local_dir = sys.argv[1:]
snapshot_download(
    repo_id=repo,
    local_dir=local_dir,
    cache_dir=os.environ["HF_HOME"],
    allow_patterns=["tokenizer/*"],
)
PY
    else
        echo "[assets] cached ${tokenizer_dir}"
    fi
}

case "${MODEL}" in
    pi05)       prepare_pi05 ;;
    hy_vla)     prepare_hy_vla ;;
    groot_n1)   prepare_groot_n1 ;;
    lingbot_va) prepare_lingbot_va ;;
esac

echo "[assets] ready: ${MODEL} in ${MODEL_CACHE_DIR}/${MODEL}"
