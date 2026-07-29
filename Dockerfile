# syntax=docker/dockerfile:1

FROM nvcr.io/nvidia/pytorch:25.12-py3

ARG CUDA_ARCHITECTURES=86
ARG LLAMA_REF=b9016
ARG PI05_HF_REPO=SEU-PAISys/Embodied.cpp
ARG PI05_HF_REVISION=main

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

# Keep heavyweight local artifacts out of the build context. The source tree,
# patch files, and the LIBERO launcher are still copied into the image.
COPY . .

# Prepare the exact llama.cpp revision and the pi0.5 patch profile used by the
# CUDA server tested in this repository.
RUN LLAMA_REF="${LLAMA_REF}" LLAMA_PATCH_PROFILE=pi05 \
        bash patches/init_third_party.sh

RUN cmake -S . -B build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES}" \
        -DGGML_CUDA=ON \
        -DGGML_CUDA_FAST_MATH=OFF \
        -DGGML_CUDA_FORCE_CUBLAS=ON \
        -DMODEL_BUILD_VLA_PI05=ON \
    && cmake --build build --target vla-pi05-server --parallel

# Install the simulator and its pinned Python runtime. Clone explicitly with
# HTTP/1.1 because GitHub's HTTP/2 transfer can be unreliable for this repo.
RUN git -c http.version=HTTP/1.1 clone --depth 1 \
        https://github.com/Lifelong-Robot-Learning/LIBERO.git \
        eval/sim/libero/LIBERO \
    && bash eval/sim/libero/setup_libero.sh

# The C++ client uses the PaliGemma SentencePiece model. Cache it at the
# fallback path understood by eval/client/vla_cpp_client.py.
RUN mkdir -p /root/.cache/openpi \
    && python - <<'PY'
from huggingface_hub import hf_hub_download

hf_hub_download(
    repo_id="google/paligemma-3b-pt-224",
    filename="tokenizer.model",
    revision="main",
    local_dir="/root/.cache/openpi",
)
PY
RUN mv /root/.cache/openpi/tokenizer.model \
        /root/.cache/openpi/paligemma_tokenizer.model

# Download the model artifacts used by the documented LIBERO pi0.5 command.
RUN mkdir -p /opt/embodied/checkpoints/pi05 \
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

COPY docker/entrypoint.sh /usr/local/bin/embodied-entrypoint
RUN chmod 0755 /usr/local/bin/embodied-entrypoint

EXPOSE 5555
ENTRYPOINT ["/usr/local/bin/embodied-entrypoint"]
CMD ["server"]
