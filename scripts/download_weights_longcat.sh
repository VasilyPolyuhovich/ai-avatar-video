#!/usr/bin/env bash
# Run ON the pod, after the network volume is mounted at /workspace. Not run
# yet (prep phase, no pod exists). Written from the verified HF repo id;
# not execution-tested.
set -euo pipefail
WEIGHTS=${1:-/workspace/longcat-weights}
mkdir -p "$WEIGHTS"

if ! command -v huggingface-cli >/dev/null 2>&1; then
  echo "huggingface-cli not found -- pip install -U 'huggingface_hub[cli]'" >&2
  exit 1
fi

echo "== LongCat-Video base checkpoint =="
huggingface-cli download meituan-longcat/LongCat-Video \
  --local-dir "$WEIGHTS/LongCat-Video"

echo "== LongCat-Video-Avatar-1.5 (avatar head, MIT license) =="
huggingface-cli download meituan-longcat/LongCat-Video-Avatar-1.5 \
  --local-dir "$WEIGHTS/LongCat-Video-Avatar-1.5"

echo "Done. Total size:"
du -sh "$WEIGHTS"
