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

# --max-workers 4 (default 8): this session hit repeated silent corruption
# during these downloads -- files ending up 2-3MB LARGER than their real HF
# size, in both the xet fast-transfer backend and the classic HTTP
# downloader (HF_HUB_DISABLE_XET=1), plus assorted hangs/IO errors elsewhere
# on this network volume under concurrent write load (see docs/decisions.md,
# 2026-07-25 root-cause writeup). Never fully proven, but the corruption
# pattern (always under multi-file/multi-threaded downloads, never a single
# small file) points at the MooseFS-backed volume's write path, not at any
# one tool. Halving the worker count is a cheap mitigation; if corruption
# recurs even at 4, drop to 1 and treat it as confirmation.
echo "== LongCat-Video base checkpoint (~83GB) =="
"$HF_CLI" download meituan-longcat/LongCat-Video \
  --max-workers 4 \
  --local-dir "$WEIGHTS/LongCat-Video"

# The full LongCat-Video-Avatar-1.5 repo is ~74.9GB, but ~31GB of that is
# dead weight for a single-A100-80GB, full-precision run: base_model_int8/
# (~16GB, INT8-quantized duplicate -- only relevant with --use_int8, which
# we don't need with 80GB VRAM available) and 5 of 6 duplicate Whisper-
# large-v3 formats (flax_model.msgpack, pytorch_model.bin, the fp32-split
# variants -- ~15GB) alongside the one we actually load,
# whisper-large-v3/model.safetensors. Checked actual per-file sizes via the
# HF API before writing this (2026-07-25) rather than blind-downloading the
# whole repo, after downloading the base checkpoint that way turned out
# ~2x the originally-assumed size.
echo "== LongCat-Video-Avatar-1.5 (avatar head, MIT license, ~44GB after excludes) =="
"$HF_CLI" download meituan-longcat/LongCat-Video-Avatar-1.5 \
  --max-workers 4 \
  --exclude "base_model_int8/*" \
            "whisper-large-v3/flax_model.msgpack" \
            "whisper-large-v3/pytorch_model.bin" \
            "whisper-large-v3/pytorch_model.fp32-*.bin" \
            "whisper-large-v3/model.fp32-*.safetensors" \
  --local-dir "$WEIGHTS/LongCat-Video-Avatar-1.5"

echo "Done. Total size:"
du -sh "$WEIGHTS"
