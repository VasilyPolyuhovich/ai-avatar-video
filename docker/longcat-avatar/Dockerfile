# LongCat-Video-Avatar-1.5 standalone inference -- A/B counterpart to
# docker/infinitetalk (see docs/ab-test-plan.md). No ComfyUI; it's a batch-job
# container -- connect over SSH and torchrun run_demo_avatar_*.py manually.
#
# Pinned: LongCat-Video @ 6b3f4b8582a8bc3f20f795735f5383716c4ba794 (2026-05-27)
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3.10-venv python3-pip git ffmpeg libgl1 libglib2.0-0 \
        libsndfile1 openssh-server curl ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && mkdir -p /run/sshd

WORKDIR /opt
RUN git clone https://github.com/meituan-longcat/LongCat-Video.git /opt/LongCat-Video \
    && cd /opt/LongCat-Video \
    && git checkout 6b3f4b8582a8bc3f20f795735f5383716c4ba794

WORKDIR /opt/LongCat-Video
RUN python3 -m pip install --upgrade pip \
    && pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
       --index-url https://download.pytorch.org/whl/cu124

# flash-attn from source, against the already-installed torch. Two prebuilt
# GitHub-release wheels (cxx11abiTRUE and FALSE, matching the "cu12torch2.6"
# tag) were tried first and BOTH failed identically at import time:
#   undefined symbol: _ZN3c105ErrorC2ENS_14SourceLocationE...
# (a c10::Error/std::string ABI mismatch) -- so this isn't the cxx11-ABI
# flag the wheel names advertise, it's those release wheels not actually
# matching torch==2.6.0+cu124 as resolved today. Building from source
# against the real installed torch sidesteps the whole guessing game.
# `--no-build-isolation` is required: flash-attn's setup.py imports torch
# directly, and pip's default isolated build venv doesn't have it (that's
# what broke the plain `pip install -r requirements.txt` attempt before
# this one). The trade-off: disabling build isolation also disables pip's
# automatic install of the package's declared build-time deps (normally
# pulled into that isolated venv from pyproject.toml's build-system.requires)
# -- so packaging/setuptools/wheel/psutil (all imported by flash-attn's
# setup.py) must already be in the main env, or metadata generation fails
# with a plain ModuleNotFoundError before compilation even starts (as it did
# here for `packaging`). ninja + capped MAX_JOBS keep the actual compile from
# OOMing the runner; TORCH_CUDA_ARCH_LIST is pinned to A100/H100 (this
# project's only target GPUs) instead of building every architecture.
# Expect ~20-40min for the compile itself.
RUN pip install ninja packaging setuptools wheel psutil
ENV MAX_JOBS=2 \
    TORCH_CUDA_ARCH_LIST="8.0;9.0"
# requirements_avatar.txt has an upstream bug: `libsndfile1==0.0.1` is a pip
# line for what is actually the system C library (already installed via apt
# above, matching soundfile/librosa's actual runtime need) -- no such PyPI
# package exists, so `pip install -r requirements_avatar.txt` fails outright.
# Strip that one line rather than patch the vendored file.
RUN pip install --no-build-isolation -r requirements.txt \
    && grep -v '^libsndfile1==' requirements_avatar.txt > requirements_avatar.filtered.txt \
    && pip install -r requirements_avatar.filtered.txt \
    && python3 -c "import flash_attn; print('flash_attn', flash_attn.__version__, 'import ok')"

RUN mkdir -p weights outputs

COPY entrypoint.sh /opt/
RUN chmod +x /opt/entrypoint.sh

EXPOSE 22
ENTRYPOINT ["/opt/entrypoint.sh"]
