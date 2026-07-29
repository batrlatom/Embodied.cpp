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
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
    MODEL_CACHE_DIR=/models \
    HF_HOME=/models/.cache/huggingface

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

# All model implementations use the combined llama.cpp patch profile. This
# stage is independent of the CUDA architecture and is reused across builds.
FROM base AS source
ARG LLAMA_REF=b9016
COPY . .
RUN LLAMA_REF="${LLAMA_REF}" LLAMA_PATCH_PROFILE=all \
        bash patches/init_third_party.sh

# The simulator environment is independent of the CUDA architecture.
FROM base AS libero
COPY eval/sim/libero/setup_libero.sh eval/sim/libero/setup_libero.sh
RUN --mount=type=cache,id=embodied-uv,target=/root/.cache/uv \
    git -c http.version=HTTP/1.1 clone --depth 1 \
        https://github.com/Lifelong-Robot-Learning/LIBERO.git \
        eval/sim/libero/LIBERO \
    && bash eval/sim/libero/setup_libero.sh

# Build every currently supported server once. Only this stage depends on the
# selected CUDA architectures. We intentionally do not download model weights
# here; they are fetched on first run into the persistent /models directory.
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
        -DMODEL_BUILD_VLA_HY_VLA=ON \
        -DMODEL_BUILD_VLA_GROOT_N1=ON \
        -DMODEL_BUILD_WAM_LINGBOT_VA=ON \
    && cmake --build build --target \
        vla-pi05-server \
        vla-hy-vla-server \
        vla-groot-n1-server \
        wam-lingbot-server \
        --parallel

# Keep the LIBERO environment as the final stage's base. uv may create the
# Python 3.10 interpreter in its managed runtime directory; copying only the
# venv would leave its python symlink pointing at a missing interpreter.
FROM libero
COPY . .
COPY --from=build /opt/embodied/build /opt/embodied/build

COPY docker/model_assets.sh /usr/local/bin/embodied-model-assets
COPY docker/entrypoint.sh /usr/local/bin/embodied-entrypoint
RUN chmod 0755 /usr/local/bin/embodied-model-assets \
    /usr/local/bin/embodied-entrypoint \
    && mkdir -p /models

VOLUME ["/models"]
EXPOSE 5555
ENTRYPOINT ["/usr/local/bin/embodied-entrypoint"]
CMD ["server"]
