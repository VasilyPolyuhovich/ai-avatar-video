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

## Not yet run
This plan is written; no render has happened yet (no pod has been deployed —
see `infra-notes.md` for what "deploying" actually takes once we're ready).
