#!/usr/bin/env bash
# Run ON the pod, from /opt/LongCat-Video (the image's baked-in repo checkout).
# Wraps run_demo_avatar_single_audio_to_video.py with the flags this project
# verified actually work -- see docs/decisions.md#2-generation-stack for the
# 2026-07-25 finding that motivated this script: the demo's own example
# invocation uses --stage_1=at2v, which is audio+text-to-video with NO image
# conditioning at all (cond_image is never read on that code path) -- it
# produces a plausible-looking talking head of a completely different,
# unrelated face. --stage_1=ai2v (the script's own argparse default) is the
# only mode that actually loads and conditions on the reference photo.
#
# Also auto-computes --num_segments from the audio's real duration: a single
# segment is only num_frames/save_fps = 93/25 = 3.72s of video regardless of
# audio length (avatar-v1.5 constants, hardcoded in the upstream script) --
# leaving num_segments at its default of 1 truncates any longer phrase
# mid-sentence (confirmed live). Each additional segment adds
# (num_frames-num_cond_frames)/save_fps = 3.2s via the long-video
# continuation path.
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage: run_longcat_avatar.sh --image PATH --audio PATH [options]

Required:
  --image PATH            Reference photo (identity source).
  --audio PATH            Driving audio (any format ffmpeg reads).

Options:
  --prompt TEXT            Gesture/scene prompt (default: neutral talking-head).
  --checkpoint-dir PATH     Default: /workspace/longcat-weights/LongCat-Video-Avatar-1.5
  --resolution 480p|720p    Default: 480p
  --num-segments N          Default: auto-computed from audio duration.
  --output-dir PATH         Default: ./outputs_avatar_single
  --no-distill              Disable the 8-step distilled LoRA -- runs the
                             full 50-step sampler instead, which is the ONLY
                             way text_guidance_scale/audio_guidance_scale
                             (upstream defaults 4.0/4.0) actually matter.
                             Roughly 17x longer render per segment, not just
                             the 6x the step-count ratio alone would suggest
                             -- see the note below.
EOF
  exit 1
}

# NOTE on --no-distill: the upstream script (run_demo_avatar_single_audio_to_video.py)
# unconditionally overrides BOTH text_guidance_scale and audio_guidance_scale
# to 1.0 whenever --use_distill is passed, regardless of any other flag --
# there is no supported combination of "fast 8-step distilled sampling" +
# "guidance scale actually has an effect". Confirmed 2026-07-25 after a user
# report that prompts seemed to do nothing and articulation looked
# exaggerated in every render -- that's expected, not a bug, given
# text_guidance_scale=1.0/audio_guidance_scale=1.0 in the default (distilled)
# mode this script has always used. --no-distill is the only way to get that
# control back -- but it costs roughly 17x the render time per segment, NOT
# ~6x as the 50-vs-8 step-count ratio alone would suggest: a non-distilled
# step also takes ~2.7x longer on its own (full classifier-free guidance
# runs two forward passes per step -- conditional + unconditional -- vs the
# distilled DMD LoRA's one), confirmed live (~38.5s/step vs ~14.5s/step).
# scripts/generate_avatar_video.py's caller-side --timeout auto-scales to
# 10800s for --no-distill for exactly this reason -- an earlier version
# left the short distilled-mode default in place and it killed a real
# --no-distill render at 82% through segment 1 of 3.
IMAGE=""
AUDIO=""
PROMPT="A person speaks calmly to the camera in a steady, natural tone, with a neutral, composed expression."
CHECKPOINT_DIR="/workspace/longcat-weights/LongCat-Video-Avatar-1.5"
RESOLUTION="480p"
NUM_SEGMENTS=""
OUTPUT_DIR="./outputs_avatar_single"
DISTILL_FLAG="--use_distill"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --image) IMAGE="$2"; shift 2 ;;
    --audio) AUDIO="$2"; shift 2 ;;
    --prompt) PROMPT="$2"; shift 2 ;;
    --checkpoint-dir) CHECKPOINT_DIR="$2"; shift 2 ;;
    --resolution) RESOLUTION="$2"; shift 2 ;;
    --num-segments) NUM_SEGMENTS="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --no-distill) DISTILL_FLAG=""; shift ;;
    -h|--help) usage ;;
    *) echo "Unknown arg: $1" >&2; usage ;;
  esac
done

if [[ -z "$DISTILL_FLAG" ]]; then
  echo "--no-distill: running the full 50-step sampler (text/audio guidance" >&2
  echo "scale 4.0/4.0 apply) -- expect roughly 17x the usual per-segment render time." >&2
fi

[[ -n "$IMAGE" && -n "$AUDIO" ]] || usage
[[ -f "$IMAGE" ]] || { echo "No such image: $IMAGE" >&2; exit 1; }
[[ -f "$AUDIO" ]] || { echo "No such audio: $AUDIO" >&2; exit 1; }

if [[ -z "$NUM_SEGMENTS" ]]; then
  AUDIO_DURATION=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$AUDIO")
  # num_frames=93, save_fps=25, num_cond_frames=13 -- avatar-v1.5 constants
  # hardcoded in run_demo_avatar_single_audio_to_video.py, not exposed as flags.
  NUM_SEGMENTS=$(python3 -c "
import math
d = $AUDIO_DURATION
first = 93/25
per_extra = (93-13)/25
segs = 1 if d <= first else 1 + math.ceil((d - first) / per_extra)
print(segs)
")
  echo "Audio duration ${AUDIO_DURATION}s -> --num_segments=$NUM_SEGMENTS" >&2
fi

INPUT_JSON=$(mktemp /tmp/longcat_input_XXXXXX.json)
python3 -c "
import json
json.dump({
    'prompt': '''$PROMPT''',
    'cond_image': '''$IMAGE''',
    'cond_audio': {'person1': '''$AUDIO'''},
}, open('$INPUT_JSON', 'w'))
"

if [[ "$NUM_SEGMENTS" -eq 1 ]]; then
  echo "Final output will be: $OUTPUT_DIR/ai2v_demo_1.mp4" >&2
else
  echo "Final output will be: $OUTPUT_DIR/video_continue_${NUM_SEGMENTS}.mp4 (the upstream script names each intermediate segment video_continue_N.mp4 -- only the highest N is the complete video)" >&2
fi

cd /opt/LongCat-Video
exec torchrun --nproc_per_node=1 run_demo_avatar_single_audio_to_video.py \
  --checkpoint_dir="$CHECKPOINT_DIR" \
  --stage_1=ai2v \
  --num_segments="$NUM_SEGMENTS" \
  --input_json="$INPUT_JSON" \
  $DISTILL_FLAG \
  --model_type avatar-v1.5 \
  --resolution "$RESOLUTION" \
  --output_dir "$OUTPUT_DIR"
