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

# Prebuilt flash-attn wheel, installed BEFORE `pip install -r requirements.txt`
# (which also pins flash-attn==2.7.4.post1): pip sees the exact version
# already satisfied and skips it, instead of trying to build it from source --
# which fails here anyway, since flash-attn's setup.py needs torch importable
# in its OWN build-isolation venv, not just the outer one (a 30-60min compile
# with matching nvcc arch flags is the fallback if this build ever breaks).
# Try the newer C++11-ABI build first (matches recent official PyTorch pip
# wheels), fall back to the pre-C++11-ABI one. Verified by import at build
# time; there's no GPU in CI so actual forward-pass execution stays
# unverified until run on a real pod.
RUN ( pip install "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiTRUE-cp310-cp310-linux_x86_64.whl" \
      && python3 -c "import flash_attn; print('flash_attn', flash_attn.__version__, 'cxx11abiTRUE ok')" ) \
    || ( pip install "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl" \
      && python3 -c "import flash_attn; print('flash_attn', flash_attn.__version__, 'cxx11abiFALSE ok')" )

RUN pip install -r requirements.txt \
    && pip install -r requirements_avatar.txt

RUN mkdir -p weights outputs

COPY entrypoint.sh /opt/
RUN chmod +x /opt/entrypoint.sh

EXPOSE 22
ENTRYPOINT ["/opt/entrypoint.sh"]
