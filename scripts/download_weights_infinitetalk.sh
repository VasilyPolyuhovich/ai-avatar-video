#!/usr/bin/env bash
# Run ON the pod, after the network volume is mounted at /workspace. Not run
# yet -- no pod exists at the time this was written (prep phase). Written
# from verified HF repo listings (checked 2026-07-21); not execution-tested.
#
# Needs `huggingface-cli` (pip install -U "huggingface_hub[cli]").
set -euo pipefail
MODELS=${1:-/workspace/models}
mkdir -p "$MODELS/diffusion_models" "$MODELS/text_encoders" \
         "$MODELS/clip_vision" "$MODELS/vae" "$MODELS/audio_encoders"

if ! command -v huggingface-cli >/dev/null 2>&1; then
  echo "huggingface-cli not found -- pip install -U 'huggingface_hub[cli]'" >&2
  exit 1
fi

echo "== Wan2.1 I2V 14B 720p backbone (fp16, ~30.5GB) =="
huggingface-cli download Comfy-Org/Wan_2.1_ComfyUI_repackaged \
  split_files/diffusion_models/wan2.1_i2v_720p_14B_fp16.safetensors \
  --local-dir "$MODELS/diffusion_models"

echo "== UMT5-XXL text encoder (bf16, NOT scaled, ~11.4GB per brief) =="
huggingface-cli download Kijai/WanVideo_comfy umt5-xxl-enc-bf16.safetensors \
  --local-dir "$MODELS/text_encoders"

echo "== CLIP vision (open-clip xlm-roberta ViT-H, fp16, ~1.26GB) =="
huggingface-cli download Kijai/WanVideo_comfy \
  open-clip-xlm-roberta-large-vit-huge-14_visual_fp16.safetensors \
  --local-dir "$MODELS/clip_vision"

echo "== Wan2.1 VAE (bf16, ~254MB) =="
huggingface-cli download Kijai/WanVideo_comfy Wan2_1_VAE_bf16.safetensors \
  --local-dir "$MODELS/vae"

echo "== InfiniteTalk audio-conditioning weights, single-speaker (~2.7GB) =="
huggingface-cli download MeiGen-AI/InfiniteTalk comfyui/infinitetalk_single.safetensors \
  --local-dir "$MODELS/diffusion_models"

echo "== Chinese-wav2vec2-base audio encoder (feature extractor, not ASR --"
echo "   works cross-lingual for InfiniteTalk's audio conditioning) =="
huggingface-cli download TencentGameMate/chinese-wav2vec2-base \
  --local-dir "$MODELS/audio_encoders/chinese-wav2vec2-base"

echo "Done. Total size:"
du -sh "$MODELS"
