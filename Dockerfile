# syntax=docker/dockerfile:1.7

FROM nvcr.io/nvidia/pytorch:25.12-py3 AS base

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
    MUJOCO_GL=egl \
    LIBERO_CONFIG_PATH=/root/.libero \
    PYTHONUNBUFFERED=1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics

WORKDIR /opt/embodied

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        cppzmq-dev \
        ffmpeg \
        git \
        iproute2 \
        libegl1-mesa-dev \
        libgl1-mesa-dev \
        libglu1-mesa-dev \
        libprotobuf-dev \
        libzmq3-dev \
        ninja-build \
        pkg-config \
        protobuf-compiler \
        wget \
    && rm -rf /var/lib/apt/lists/* \
    && python -m pip install --no-cache-dir uv huggingface_hub

# This stage is independent of the CUDA architecture. It is reused when only
# CUDA_ARCHITECTURES changes, so llama.cpp is not cloned or patched again.
FROM base AS source
ARG LLAMA_REF=b9016
COPY . .
RUN LLAMA_REF="${LLAMA_REF}" LLAMA_PATCH_PROFILE=pi05 \
        bash patches/init_third_party.sh

# Model and tokenizer assets are independent of the C++ build. The BuildKit
# cache keeps Hugging Face downloads across architecture rebuilds.
FROM base AS assets
ARG PI05_HF_REPO=SEU-PAISys/Embodied.cpp
ARG PI05_HF_REVISION=main

RUN --mount=type=cache,id=embodied-hf,target=/root/.cache/huggingface \
    mkdir -p /root/.cache/openpi /opt/embodied/checkpoints/pi05 \
    && wget -q --show-progress \
        -O /root/.cache/openpi/paligemma_tokenizer.model \
        https://storage.googleapis.com/big_vision/paligemma_tokenizer.model \
    && test "$(stat -c '%s' /root/.cache/openpi/paligemma_tokenizer.model)" = "4264023" \
    && PI05_HF_REPO="${PI05_HF_REPO}" PI05_HF_REVISION="${PI05_HF_REVISION}" \
        python - <<'PY'
import os
from huggingface_hub import hf_hub_download

repo_id = os.environ["PI05_HF_REPO"]
revision = os.environ["PI05_HF_REVISION"]
local_dir = "/opt/embodied/checkpoints/pi05"

for filename in (
    "pi05_libero_finetuned_v044/pi05-mmproj.gguf",
    "pi05_libero_finetuned_v044/pi05.gguf",
):
    hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        revision=revision,
        local_dir=local_dir,
    )
PY
RUN mv /opt/embodied/checkpoints/pi05/pi05_libero_finetuned_v044/pi05-mmproj.gguf \
        /opt/embodied/checkpoints/pi05/pi05-mmproj.gguf \
    && mv /opt/embodied/checkpoints/pi05/pi05_libero_finetuned_v044/pi05.gguf \
        /opt/embodied/checkpoints/pi05/pi05.gguf \
    && rm -rf /opt/embodied/checkpoints/pi05/pi05_libero_finetuned_v044

# The simulator environment is also independent of the CUDA architecture.
# Cache uv's package downloads so a later rebuild does not redownload wheels.
FROM base AS libero
COPY eval/sim/libero/setup_libero.sh eval/sim/libero/setup_libero.sh
RUN --mount=type=cache,id=embodied-uv,target=/root/.cache/uv \
    git -c http.version=HTTP/1.1 clone --depth 1 \
        https://github.com/Lifelong-Robot-Learning/LIBERO.git \
        eval/sim/libero/LIBERO \
    && bash eval/sim/libero/setup_libero.sh

# Only this stage depends on the selected GPU architecture. For the RTX 3090
# and RTX 5080 together, pass --build-arg 'CUDA_ARCHITECTURES=86;120'.
FROM base AS build
ARG CUDA_ARCHITECTURES=86;120
COPY . .
COPY --from=source /opt/embodied/third_party/llama.cpp \
    /opt/embodied/third_party/llama.cpp
RUN cmake -S . -B build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES}" \
        -DGGML_CUDA=ON \
        -DGGML_CUDA_FAST_MATH=OFF \
        -DGGML_CUDA_FORCE_CUBLAS=ON \
        -DMODEL_BUILD_VLA_PI05=ON \
    && cmake --build build --target vla-pi05-server --parallel

FROM base
COPY . .
COPY --from=build /opt/embodied/build /opt/embodied/build
COPY --from=libero /opt/embodied/eval/sim/libero/LIBERO \
    /opt/embodied/eval/sim/libero/LIBERO
COPY --from=libero /opt/embodied/eval/sim/libero/libero_uv \
    /opt/embodied/eval/sim/libero/libero_uv
COPY --from=assets /root/.cache/openpi \
    /root/.cache/openpi
COPY --from=assets /opt/embodied/checkpoints/pi05 \
    /opt/embodied/checkpoints/pi05

COPY docker/entrypoint.sh /usr/local/bin/embodied-entrypoint
RUN chmod 0755 /usr/local/bin/embodied-entrypoint

EXPOSE 5555
ENTRYPOINT ["/usr/local/bin/embodied-entrypoint"]
CMD ["server"]
