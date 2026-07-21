# ComfyUI + WanVideoWrapper + InfiniteTalk (Wan2.1-14B I2V backbone).
# Weights are NOT baked in -- mount the RunPod network volume at /workspace and
# run scripts/download_weights_infinitetalk.sh once (see docs/infra-notes.md).
#
# Pinned (2026-07-21):
#   ComfyUI                    v0.28.0
#   ComfyUI-WanVideoWrapper    088128b224242e110d3906c6750e9a3a348a659b (2026-05-24)
#   InfiniteTalk (comfyui br.) 7089f0a7c84d2cb20709599fb08444ab3bc837ce (2025-08-19)
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
    && git clone https://github.com/kijai/ComfyUI-WanVideoWrapper.git custom_nodes/ComfyUI-WanVideoWrapper \
    && cd custom_nodes/ComfyUI-WanVideoWrapper \
    && git checkout 088128b224242e110d3906c6750e9a3a348a659b \
    && pip install -r requirements.txt

RUN git clone --branch comfyui https://github.com/MeiGen-AI/InfiniteTalk.git custom_nodes/InfiniteTalk \
    && cd custom_nodes/InfiniteTalk \
    && git checkout 7089f0a7c84d2cb20709599fb08444ab3bc837ce \
    && pip install -r requirements.txt

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
