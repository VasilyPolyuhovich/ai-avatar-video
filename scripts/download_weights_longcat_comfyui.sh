#!/usr/bin/env bash
# Run ON the pod, after the network volume is mounted at /workspace. Filenames
# and source repos cross-checked 2026-07-26 against Kijai/ComfyUI-WanVideoWrapper's
# LongCat Avatar example workflow and the actual HF repo listings (not guessed --
# the example workflow references LongCat_distill_lora_rank128_bf16.safetensors,
# which does NOT exist in the current Kijai/LongCat-Video_comfy repo listing;
# the real file is LongCat_distill_lora_alpha64_bf16.safetensors).
#
# NOTE: Wan2.1 VAE and the UMT5 text encoder are the SAME files already
# downloaded for docker/infinitetalk (same repo, same filenames, same target
# dirs on this shared network volume) -- fetch_flat's skip-if-exists check
# means this is a no-op re-download if InfiniteTalk's weights are still on
# the volume, not a wasted duplicate fetch.
set -euo pipefail
MODELS=${1:-/workspace/models}
mkdir -p "$MODELS/diffusion_models" "$MODELS/text_encoders" \
         "$MODELS/vae" "$MODELS/audio_encoders" "$MODELS/loras"

if command -v hf >/dev/null 2>&1; then
  HF_CLI=hf
elif command -v huggingface-cli >/dev/null 2>&1; then
  HF_CLI=huggingface-cli
else
  echo "neither 'hf' nor 'huggingface-cli' found -- pip install -U 'huggingface_hub[cli]'" >&2
  exit 1
fi

# Same fetch_flat as scripts/download_weights_infinitetalk.sh: flattens a
# repo's internal path into destdir, downloads into a scratch dir *inside*
# destdir (same filesystem, avoids the cross-fs mv hang documented there),
# skips entirely if the destination file already exists.
fetch_flat() {
  local repo="$1" file="$2" destdir="$3"
  local dest="$destdir/$(basename "$file")"
  if [ -f "$dest" ]; then
    echo "  already have $dest, skipping"
    return
  fi
  local tmp; tmp=$(mktemp -d "$destdir/.dl-XXXXXX")
  "$HF_CLI" download "$repo" "$file" --local-dir "$tmp" --max-workers 4
  mv "$tmp/$file" "$dest"
  rm -rf "$tmp"
}

echo "== LongCat-Video-Avatar-1.5 diffusion model (bf16, ComfyUI-native conversion) =="
fetch_flat Kijai/LongCat-Video_comfy \
  Avatar/LongCat-Avatar_comfy_bf16.safetensors "$MODELS/diffusion_models"

echo "== LongCat distill LoRA (bf16, alpha64 -- NOTE: the example workflow's own"
echo "   JSON references a 'rank128' filename that does not exist in the repo;"
echo "   this is the real one, verified against the actual HF file listing) =="
fetch_flat Kijai/LongCat-Video_comfy \
  LongCat_distill_lora_alpha64_bf16.safetensors "$MODELS/loras"

echo "== Wan2.1 VAE (bf16) -- same file docker/infinitetalk already uses;"
echo "   skipped automatically if already present on this volume =="
fetch_flat Kijai/WanVideo_comfy Wan2_1_VAE_bf16.safetensors "$MODELS/vae"

echo "== UMT5-XXL text encoder (bf16) -- same file docker/infinitetalk already"
echo "   uses; skipped automatically if already present on this volume =="
fetch_flat Kijai/WanVideo_comfy umt5-xxl-enc-bf16.safetensors "$MODELS/text_encoders"

echo "== Whisper-large-v3 encoder-only (fp16, ~1.7GB) -- pre-converted to the"
echo "   single-file 'model.*'-prefixed format WhisperModelLoader expects, so"
echo "   no local conversion script is needed =="
fetch_flat Kijai/WanVideo_comfy \
  HuMo/whisper_large_v3_encoder_fp16.safetensors "$MODELS/audio_encoders"

echo "== Mel-Band RoFormer vocal separator (fp32, ~913MB) -- goes in"
echo "   diffusion_models per the model's own ComfyUI node convention, not a"
echo "   dedicated folder =="
fetch_flat Kijai/MelBandRoFormer_comfy \
  MelBandRoformer_fp32.safetensors "$MODELS/diffusion_models"

echo "Done. Total size:"
du -sh "$MODELS"
