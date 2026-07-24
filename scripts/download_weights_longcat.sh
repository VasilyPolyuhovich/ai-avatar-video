#!/usr/bin/env bash
# Run ON the pod, after the network volume is mounted at /workspace. Written
# from the verified HF repo id.
#
# Needs the huggingface_hub CLI. `huggingface-cli` was renamed to `hf` in a
# recent huggingface_hub release and the old name stopped working entirely
# on the image as actually deployed 2026-07-24 -- auto-detect whichever is
# present rather than hardcoding one (see download_weights_infinitetalk.sh).
set -euo pipefail
WEIGHTS=${1:-/workspace/longcat-weights}
mkdir -p "$WEIGHTS"

if command -v hf >/dev/null 2>&1; then
  HF_CLI=hf
elif command -v huggingface-cli >/dev/null 2>&1; then
  HF_CLI=huggingface-cli
else
  echo "neither 'hf' nor 'huggingface-cli' found -- pip install -U 'huggingface_hub[cli]'" >&2
  exit 1
fi

echo "== LongCat-Video base checkpoint =="
"$HF_CLI" download meituan-longcat/LongCat-Video \
  --local-dir "$WEIGHTS/LongCat-Video"

echo "== LongCat-Video-Avatar-1.5 (avatar head, MIT license) =="
"$HF_CLI" download meituan-longcat/LongCat-Video-Avatar-1.5 \
  --local-dir "$WEIGHTS/LongCat-Video-Avatar-1.5"

echo "Done. Total size:"
du -sh "$WEIGHTS"
