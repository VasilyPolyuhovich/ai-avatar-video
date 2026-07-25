# A/B test plan: InfiniteTalk vs LongCat-Video-Avatar-1.5

Goal: pick one stack as the default pipeline, without over-committing before
seeing either one run on the actual target photo.

## Inputs (held identical across both runs)
- One photo (the brief's "one clear, well-lit portrait" requirement).
- One audio clip: the ElevenLabs Ukrainian voice-clone test phrase, ~10s.
- Same gesture prompt where the stack supports text-conditioned motion, e.g.
  *"A person speaks warmly, occasionally gesturing with one hand while
  explaining."*

## Procedure
1. **480p first-look**, both stacks, 2-3 seeds each where the stack supports
   seeding. Cheapest signal on lip sync + identity before spending on 720p.
2. Score each result (see rubric below).
3. Render the better-scoring stack's best seed at **720p** (InfiniteTalk) or
   `--resolution 720P` (LongCat) for the final candidate.
4. Frame-interpolate (RIFE, doubled FPS) to remove flicker, export square
   MP4/H.264 ≤60s per the brief's output spec.

### How each stack is actually invoked
- **InfiniteTalk**: the plan was to open ComfyUI's UI (SSH tunnel to :8188)
  and drive it via Claude for Chrome browser automation, but the extension
  wasn't reachable from the pod session (2026-07-25) -- worked around it
  entirely via ComfyUI's own HTTP API instead, no browser needed:
  `scripts/comfyui_workflow_to_api.py` converts the bundled
  `/opt/workflows/infinitetalk_single_example.json` (its raw UI/graph
  format) into ComfyUI's API prompt format using the pod's live
  `/object_info` schemas, then `POST /prompt` triggers the render directly.
  Test image/audio (`face_8.jpg`/`celyj.mp3`) already sit in
  `/workspace/input`, auto-discovered by ComfyUI's upload-widget combo
  lists. **Don't reuse the demo's sampler settings as-is**: it defaults to
  a lighter 480p fp8 model + `steps=4, cfg=1.0, scheduler=flowmatch_distill`
  tuned specifically for a step-distillation LoRA -- if you skip that LoRA
  (as the script's overrides do, for a full-fp16 quality run) but keep
  those sampler settings, the diffusion process is starved of steps and
  the output is blurry/unstable (confirmed live: first real render looked
  terrible for exactly this reason). The script's
  `apply_infinitetalk_single_overrides()` already reverts to
  `WanVideoSampler`'s own non-distilled defaults (steps=30, cfg=6.0,
  scheduler=unipc) and points the model loaders at the 720p fp16 weights.
- **LongCat**: SSH in, write an `input_json` (schema confirmed against
  `meituan-longcat/LongCat-Video`'s `assets/avatar/single_example_1.json`):
  ```json
  {"prompt": "<gesture prompt>", "cond_image": "/path/to/photo.png",
   "cond_audio": {"person1": "/path/to/audio.mp3"}}
  ```
  then `torchrun run_demo_avatar_single_audio_to_video.py --checkpoint_dir=/workspace/longcat-weights/LongCat-Video-Avatar-1.5 --stage_1=ai2v --num_segments=4 --input_json=<path> --use_distill --model_type avatar-v1.5 --resolution 480p` (drop `--use_int8` unless VRAM is tight on the chosen GPU).
  **`--stage_1` must be `ai2v` (the script's own default), not `at2v`** — `at2v`
  is audio+text-to-video with **no image conditioning at all** (`cond_image`
  is never read in that code path); only `ai2v` loads `input_data['cond_image']`
  and actually passes it into the pipeline. Running `at2v` produces a
  plausible-looking talking head that ignores the reference photo entirely —
  confirmed live 2026-07-25: the first real render used `at2v` by mistake and
  came back as a completely different, unrelated face. Also note `--resolution`
  takes **lowercase** `480p`/`720p`, and a single segment is only
  `num_frames/save_fps` = 93/25 ≈ **3.7s** of video regardless of audio
  length — `--num_segments` must be raised (each extra segment adds
  `(num_frames-num_cond_frames)/save_fps` ≈ 3.2s) to cover the full clip, or
  the audio gets cut off mid-phrase (also confirmed live: same first-render
  mistake).

## Rubric (1-5 each, notes required for any score ≤3)
| Criterion | What to look for |
|---|---|
| Lip-sync accuracy | Mouth shapes track phonemes, not just "moves when audio plays" |
| Identity stability | Face doesn't drift/warp over the clip length |
| Gesture naturalism | Hands/fingers — the brief's known weak point for this model class |
| Artifacts | Flicker, ghosting, background warping |
| Render time | Wall-clock on the same GPU, 480p and 720p separately |
| Peak VRAM | From `nvidia-smi` during the render, both stacks |

## Decision rule
Winner by lip-sync + identity stability (the two hardest failure modes to
paper over) unless render time/VRAM makes the other stack impractical on a
single A100 80GB. Keep both Docker images regardless — the cost of keeping
them is just registry storage, and a second opinion is cheap once built.

## Result (2026-07-25): LongCat-Video-Avatar-1.5 wins

Both stacks rendered on the same test inputs (`face_8.jpg` synthetic face,
`celyj.mp3` 12s placeholder audio, same A100 80GB pod class):

| | InfiniteTalk (2nd render, post sampler-fix) | LongCat (post `ai2v`/`--num_segments` fix) |
|---|---|---|
| Lip-sync accuracy | Rated "значно краще" than v1, but still didn't match phonemes well | Rated "практично ідеальний" — clearly better |
| Identity stability | Earring/identity jitter across the 5 stitched windows | Stable, correct identity once `ai2v` was used |
| Render time (480p) | ~43 min (5 windows × ~8.6 min) | ~10 min (4 segments, incl. one-time compile warmup) |
| Invocation | ComfyUI HTTP API (`scripts/comfyui_workflow_to_api.py`) | Plain CLI (`scripts/run_longcat_avatar.sh`) |

**Decision: LongCat-Video-Avatar-1.5, `--stage_1=ai2v`, is the pipeline going
forward.** See `decisions.md#2-generation-stack` for the exact locked
invocation and the two bugs that had to be fixed first (`at2v` silently
drops the reference photo; default `--num_segments=1` truncates any audio
longer than ~3.7s). Both Docker images are kept, but LongCat is what any
future UI/automation should be built against.
