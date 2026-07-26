# ComfyUI + LongCat-Video-Avatar-1.5 via kijai/ComfyUI-WanVideoWrapper.
# Weights are NOT baked in -- mount the RunPod network volume at /workspace and
# run scripts/download_weights_longcat_comfyui.sh once (see docs/infra-notes.md).
#
# Why this image exists (2026-07-26): the plain-CLI LongCat integration
# (docker/longcat-avatar) only exposes a binary --use_distill switch --
# on: fast, but text/audio guidance scale is forced to 1.0 (prompt has no
# effect, articulation looks exaggerated); off: the only alternative is a
# hardcoded jump to 50 full-CFG steps, ~17x the cost, and a real live test
# came back "повне лайно" (garbage). WanVideoWrapper's own LongCat Avatar
# support exposes `audio_scale`/`audio_cfg_scale` as continuous, independent
# values with no hardcoded distill-forces-scale=1.0 override -- the actual
# middle ground the plain CLI cannot offer at any price. See
# docs/decisions.md for the full writeup.
#
# Pinned (2026-07-26):
#   ComfyUI                       v0.28.0 (same as docker/infinitetalk)
#   ComfyUI-WanVideoWrapper       088128b224242e110d3906c6750e9a3a348a659b (main HEAD as of this pin)
#   ComfyUI-KJNodes               e27a505b3ba6ce42687fe00500deda103d9d6071 (same pin as docker/infinitetalk)
#   ComfyUI-VideoHelperSuite      4ee72c065db22c9d96c2427954dc69e7b908444b (same pin as docker/infinitetalk)
#   ComfyUI-MelBandRoFormer       92c86854e6654f4aacc97484471af95c98ea16d4 (main HEAD as of this pin)
#   torch/vision/audio            2.8.0 / 0.23.0 / 2.8.0 (+cu129) -- same as docker/infinitetalk
#   kornia                        0.7.1 -- same segfault-avoidance pin as docker/infinitetalk
#
# NOTE on reusing docker/infinitetalk's torch/kornia pins: WanVideoWrapper's
# own requirements.txt (ftfy, accelerate, einops, diffusers, peft,
# sentencepiece, protobuf, pyloudnorm, gguf, opencv-python, scipy) declares
# no torch/cuda version itself -- it expects ComfyUI's own torch install to
# already be present, same assumption docker/infinitetalk's InfiniteTalk
# fork makes. Since this image is ALSO ComfyUI v0.28.0, the exact same
# kornia 0.8.3-vs-0.7.1 segfault risk applies (ComfyUI's own
# `kornia>=0.7.1` requirement, unpinned, resolves to the newer, crashing
# version) -- keep the identical pin, don't re-derive it.
#
# NOTE on python3.11 + the sitecustomize shim: identical root cause and fix
# as docker/infinitetalk -- see that Dockerfile's own note for the full
# explanation (Ubuntu 22.04's apt python3.11 is frozen at an August-2022
# rc1 missing sys.get_int_max_str_digits, needed unconditionally by
# torch._dynamo; python3.10 is not viable since ComfyUI's kornia>=0.7.1
# needs Python >=3.11).
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

# ComfyUI-WanVideoWrapper: LongCat-Video-Avatar-1.5 support lives here
# (LongCat/nodes.py -- WanVideoLongCatAvatarExtendEmbeds,
# LongCatAvatarWhisperEmbeds -- plus the shared WanVideo* sampler/loader/
# scheduler nodes and HuMo/nodes.py's WhisperModelLoader, which
# LongCatAvatarWhisperEmbeds consumes). This is a big, actively-developed
# repo (6631 stars, pushed to main the same day this was pinned) -- pin
# hard to a commit, don't track a branch, since the workflow this project
# builds against (scripts/comfyui_workflow_to_api.py's overrides) depends
# on exact node schemas that could shift upstream at any time.
RUN cd custom_nodes \
    && git clone https://github.com/kijai/ComfyUI-WanVideoWrapper.git \
    && cd ComfyUI-WanVideoWrapper && git checkout 088128b224242e110d3906c6750e9a3a348a659b \
    && pip install -r requirements.txt

# Same pins as docker/infinitetalk -- ImageResizeKJv2/SetNode/GetNode/
# INTConstant/FloatConstant (KJNodes) and VHS_VideoCombine
# (VideoHelperSuite) are both used by the LongCat Avatar example workflow
# too, confirmed by inspecting it directly before writing this Dockerfile.
RUN cd custom_nodes \
    && git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git \
    && cd ComfyUI-VideoHelperSuite && git checkout 4ee72c065db22c9d96c2427954dc69e7b908444b \
    && pip install -r requirements.txt
RUN cd custom_nodes \
    && git clone https://github.com/kijai/ComfyUI-KJNodes.git \
    && cd ComfyUI-KJNodes && git checkout e27a505b3ba6ce42687fe00500deda103d9d6071 \
    && pip install -r requirements.txt

# Vocal separation (isolates vocals from the driving audio before Whisper
# encoding, same purpose as the plain-CLI pipeline's Kim_Vocal_2 ONNX step,
# different model/implementation). Used by the example workflow's
# MelBandRoFormerSampler/MelBandRoFormerModelLoader nodes.
RUN cd custom_nodes \
    && git clone https://github.com/kijai/ComfyUI-MelBandRoFormer.git \
    && cd ComfyUI-MelBandRoFormer && git checkout 92c86854e6654f4aacc97484471af95c98ea16d4 \
    && pip install -r requirements.txt

# Triton has prebuilt manylinux wheels. SageAttention is intentionally NOT
# compiled here -- it needs the target GPU's real arch (TORCH_CUDA_ARCH_LIST);
# baking the wrong one would silently underperform or fail to load. Build it
# once on the actual pod instead (see docs/infra-notes.md). Also: a note
# baked into the LongCat Avatar example workflow itself warns "sageattention
# 1.0.6 does NOT work" with these models -- verify the actual installed
# version's compatibility live before relying on it; 'sageattn' is only one
# of several attention-backend choices WanVideoModelLoader exposes, not a
# hard requirement.
RUN pip install triton

RUN mkdir -p output input

COPY entrypoint.sh supervisor.sh /opt/
RUN chmod +x /opt/entrypoint.sh /opt/supervisor.sh

EXPOSE 8188 22
ENTRYPOINT ["/opt/entrypoint.sh"]
