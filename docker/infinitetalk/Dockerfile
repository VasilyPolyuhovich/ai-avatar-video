# ComfyUI + InfiniteTalk (Wan2.1-14B I2V backbone).
# Weights are NOT baked in -- mount the RunPod network volume at /workspace and
# run scripts/download_weights_infinitetalk.sh once (see docs/infra-notes.md).
#
# Pinned (2026-07-21):
#   ComfyUI                    v0.28.0
#   InfiniteTalk (comfyui br.) 7089f0a7c84d2cb20709599fb08444ab3bc837ce (2025-08-19)
#
# NOTE: the InfiniteTalk "comfyui" branch IS a fork of Kijai's
# ComfyUI-WanVideoWrapper with InfiniteTalk support merged in (its own
# README: "This project is based on ComfyUI-WanVideoWrapper") -- it is NOT
# a small companion pack meant to sit alongside a separate WanVideoWrapper
# install. An earlier version of this Dockerfile cloned both, which would
# have registered the same ComfyUI node classes twice. Install only this one.
FROM nvidia/cuda:12.9.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3-pip git ffmpeg libgl1 libglib2.0-0 \
        openssh-server curl ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && mkdir -p /run/sshd

WORKDIR /opt
RUN git clone --depth 1 --branch v0.28.0 https://github.com/comfyanonymous/ComfyUI.git /opt/ComfyUI

WORKDIR /opt/ComfyUI
RUN python3 -m pip install --upgrade pip \
    && pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu129 \
    && pip install -r requirements.txt

RUN mkdir -p custom_nodes \
    && git clone --branch comfyui https://github.com/MeiGen-AI/InfiniteTalk.git custom_nodes/InfiniteTalk \
    && cd custom_nodes/InfiniteTalk \
    && git checkout 7089f0a7c84d2cb20709599fb08444ab3bc837ce \
    && pip install -r requirements.txt

# Bundle the maintainers' own example workflow (single-speaker InfiniteTalk)
# so the pod has a known-working node graph to load in ComfyUI immediately,
# instead of hand-wiring nodes blind on rented GPU time.
RUN mkdir -p /opt/workflows \
    && curl -fsSL -o /opt/workflows/infinitetalk_single_example.json \
       https://raw.githubusercontent.com/MeiGen-AI/InfiniteTalk/comfyui/example_workflows/wanvideo_infinitetalk_single_example.json

# Triton has prebuilt manylinux wheels. SageAttention is intentionally NOT
# compiled here -- it needs the target GPU's real arch (TORCH_CUDA_ARCH_LIST);
# baking the wrong one would silently underperform or fail to load. Build it
# once on the actual pod instead (see docs/infra-notes.md).
RUN pip install triton

RUN mkdir -p output input

COPY entrypoint.sh supervisor.sh /opt/
RUN chmod +x /opt/entrypoint.sh /opt/supervisor.sh

EXPOSE 8188 22
ENTRYPOINT ["/opt/entrypoint.sh"]
