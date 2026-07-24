#!/usr/bin/env bash
# Run ON the pod, after the network volume is mounted at /workspace. Filenames
# and source repos cross-checked 2026-07-24 against the maintainers' own
# example workflow (MeiGen-AI/InfiniteTalk, comfyui branch,
# example_workflows/wanvideo_infinitetalk_single_example.json) and the HF
# repo listings.
#
# Needs the huggingface_hub CLI. `huggingface-cli` was renamed to `hf` in a
# recent huggingface_hub release and the old name stopped working entirely
# (not just deprecated -- it errors out) on the image as actually deployed
# 2026-07-24, despite pod_up.py/the Dockerfile never pinning a version. Auto-
# detect whichever is present rather than hardcoding one.
set -euo pipefail
MODELS=${1:-/workspace/models}
mkdir -p "$MODELS/diffusion_models" "$MODELS/text_encoders" \
         "$MODELS/clip_vision" "$MODELS/vae" "$MODELS/audio_encoders"

if command -v hf >/dev/null 2>&1; then
  HF_CLI=hf
elif command -v huggingface-cli >/dev/null 2>&1; then
  HF_CLI=huggingface-cli
else
  echo "neither 'hf' nor 'huggingface-cli' found -- pip install -U 'huggingface_hub[cli]'" >&2
  exit 1
fi

# `<cli> download <repo> <file> --local-dir <dir>` preserves the file's path
# *within the repo* under <dir> (e.g. a repo file at
# split_files/diffusion_models/x.safetensors lands at
# <dir>/split_files/diffusion_models/x.safetensors) -- not flat, which is
# what ComfyUI's model folders need. Download to a scratch dir and move just
# the file out, so the result always lands as <destdir>/<basename>.
fetch_flat() {
  local repo="$1" file="$2" destdir="$3"
  local tmp; tmp=$(mktemp -d)
  "$HF_CLI" download "$repo" "$file" --local-dir "$tmp"
  mv "$tmp/$file" "$destdir/$(basename "$file")"
  rm -rf "$tmp"
}

echo "== Wan2.1 I2V 14B 720p backbone (fp16, ~30.5GB) =="
fetch_flat Comfy-Org/Wan_2.1_ComfyUI_repackaged \
  split_files/diffusion_models/wan2.1_i2v_720p_14B_fp16.safetensors \
  "$MODELS/diffusion_models"

echo "== UMT5-XXL text encoder (bf16, NOT scaled, ~11.4GB per brief) =="
fetch_flat Kijai/WanVideo_comfy umt5-xxl-enc-bf16.safetensors "$MODELS/text_encoders"

echo "== CLIP vision (clip_vision_h, fp16, ~1.26GB -- the exact filename the"
echo "   example workflow's CLIPVisionLoader expects) =="
fetch_flat Comfy-Org/Wan_2.1_ComfyUI_repackaged \
  split_files/clip_vision/clip_vision_h.safetensors "$MODELS/clip_vision"

echo "== Wan2.1 VAE (bf16, ~254MB -- the example workflow's default is the"
echo "   fp32 variant; bf16 works too and is 1/2 the download) =="
fetch_flat Kijai/WanVideo_comfy Wan2_1_VAE_bf16.safetensors "$MODELS/vae"

echo "== InfiniteTalk audio-conditioning weights, single-speaker, fp16"
echo "   (~4.8GB -- matches the brief's Wan2_1-InfiniteTalk-Single_fp16 spec;"
echo "   note the upstream filename has a typo, 'InfiniTetalk') =="
fetch_flat Kijai/WanVideo_comfy \
  InfiniteTalk/Wan2_1-InfiniTetalk-Single_fp16.safetensors "$MODELS/diffusion_models"

echo "== Chinese-wav2vec2-base audio encoder (feature extractor, not ASR --"
echo "   works cross-lingual for InfiniteTalk's audio conditioning). The"
echo "   workflow's DownloadAndLoadWav2VecModel node can also fetch this"
echo "   itself on first run; pre-downloading here just avoids that wait. =="
"$HF_CLI" download TencentGameMate/chinese-wav2vec2-base \
  --local-dir "$MODELS/audio_encoders/chinese-wav2vec2-base"

echo "Done. Total size:"
du -sh "$MODELS"

cat <<'NOTE'

NOTE: the bundled example workflow (/opt/workflows/infinitetalk_single_example.json
in the image) defaults to a lighter 480p fp8 model + a step-distill LoRA, for a
fast demo. This project's brief wants max quality (720p fp16, no distill LoRA),
so after loading the workflow in ComfyUI, repoint two dropdowns:
  - WanVideoModelLoader -> wan2.1_i2v_720p_14B_fp16.safetensors
  - WanVideoLoraSelect   -> None / bypass the node (optional: keep it if you
                            want faster sampling and are OK trading a little
                            quality for it -- worth an A/B of its own later)
NOTE
