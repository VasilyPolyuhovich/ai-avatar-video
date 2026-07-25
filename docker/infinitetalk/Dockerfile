# ComfyUI + InfiniteTalk (Wan2.1-14B I2V backbone).
# Weights are NOT baked in -- mount the RunPod network volume at /workspace and
# run scripts/download_weights_infinitetalk.sh once (see docs/infra-notes.md).
#
# Pinned (2026-07-21, torch/kornia pins added 2026-07-24 after real pod testing):
#   ComfyUI                    v0.28.0
#   InfiniteTalk (comfyui br.) 7089f0a7c84d2cb20709599fb08444ab3bc837ce (2025-08-19)
#   torch/vision/audio          2.8.0 / 0.23.0 / 2.8.0 (+cu129)
#   kornia                      0.7.1 (ComfyUI's own dep, see NOTE below)
#
# NOTE: the InfiniteTalk "comfyui" branch IS a fork of Kijai's
# ComfyUI-WanVideoWrapper with InfiniteTalk support merged in (its own
# README: "This project is based on ComfyUI-WanVideoWrapper") -- it is NOT
# a small companion pack meant to sit alongside a separate WanVideoWrapper
# install. An earlier version of this Dockerfile cloned both, which would
# have registered the same ComfyUI node classes twice. Install only this one.
#
# NOTE on python3.11 + the sitecustomize shim: Ubuntu 22.04's apt archive has
# never shipped a real python3.11 -- its "universe" package is permanently
# frozen at `3.11.0~rc1-1~22.04` (an August-2022 pre-release), missing
# `sys.get_int_max_str_digits`/`set_int_max_str_digits` (added later in the
# 3.11.0 final release; torch's `_dynamo` polyfills import unconditionally
# needs them). Switching to python3.10 (jammy's real, fully-released default)
# was tried and REJECTED: ComfyUI's own `kornia>=0.7.1` requirement resolves
# to 0.8.x, which requires Python >=3.11 -- 3.10 can't install it at all.
# So: stay on the rc1 python3.11, and paper over the one missing stdlib
# feature with a sitecustomize.py shim (Python auto-imports this at every
# interpreter start, from any of the paths in `python3 -c "import site;
# print(site.getsitepackages())"` -- there's already a Debian-shipped one at
# /usr/lib/python3.11/sitecustomize.py that takes precedence over
# dist-packages, so APPEND to that file, don't create a new one elsewhere or
# it'll be silently shadowed). The shim functions must match torch's expected
# signature exactly (param name `maxdigits: int`, return `-> None`) --
# torch's `@substitute_in_graph` decorator inspects it and raises TypeError
# on a bare `lambda n: None`.
#
# NOTE on the torch pin: `pip install torch torchvision torchaudio
# --index-url .../cu129` with no version grabs whatever's newest at build
# time -- it resolved to 2.13.0 on 2026-07-24 and segfaulted (SIGSEGV, not a
# catchable exception) inside kornia's `@torch.jit.script` decorator at
# import time. Downgrading torch alone (tried 2.8.0, the *oldest* cu129
# build) did NOT fix the segfault -- confirmed on the actual pod. The real
# fix is the kornia pin below; the torch pin is kept anyway as separate good
# practice (avoid silent drift onto an untested "latest").
#
# NOTE on the kornia pin: ComfyUI's requirements.txt only says `kornia>=0.7.1`,
# so pip resolved the newest match, 0.8.3, which segfaults inside
# `kornia/geometry/epipolar/essential.py`'s `@torch.jit.script`-decorated
# function at import time -- used by comfy_extras/nodes_post_processing.py
# for an RGB<->LAB color node, nothing InfiniteTalk/WanVideo itself needs.
# kornia==0.7.1 (the oldest version satisfying ComfyUI's own constraint) was
# verified NOT to crash on the actual pod/GPU. Installed as a separate `pip
# install kornia==0.7.1` AFTER requirements.txt so it overrides whatever
# version pip resolved there.
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
    && mkdir -p /run/sshd \
    && printf '%s\n' \
        'import sys as _sys' \
        'if not hasattr(_sys, "get_int_max_str_digits"):' \
        '    def _get_int_max_str_digits() -> int:' \
        '        return 4300' \
        '    _sys.get_int_max_str_digits = _get_int_max_str_digits' \
        'if not hasattr(_sys, "set_int_max_str_digits"):' \
        '    def _set_int_max_str_digits(maxdigits: int) -> None:' \
        '        pass' \
        '    _sys.set_int_max_str_digits = _set_int_max_str_digits' \
        >> /usr/lib/python3.11/sitecustomize.py

WORKDIR /opt
RUN git clone --depth 1 --branch v0.28.0 https://github.com/comfyanonymous/ComfyUI.git /opt/ComfyUI

WORKDIR /opt/ComfyUI
RUN python3 -m pip install --upgrade pip \
    && pip install torch==2.8.0+cu129 torchvision==0.23.0+cu129 torchaudio==2.8.0+cu129 \
       --index-url https://download.pytorch.org/whl/cu129 \
    && pip install -r requirements.txt \
    && pip install kornia==0.7.1 \
    && python3 -c "import kornia.geometry.epipolar.essential" \
    && python3 -c "import torch._dynamo"

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
