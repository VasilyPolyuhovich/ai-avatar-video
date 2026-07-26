# Writing a good prompt for LongCat-Video-Avatar-1.5

This is specific to this project's proven-good configuration: the
distilled 8-step sampler with default guidance (`--use_distill`, no
`--no-distill`, no custom `--text-guidance-scale`/`--audio-guidance-scale`).
See `docs/decisions.md#2-generation-stack` for why that combination is the
only one confirmed to produce clean output — everything below assumes it.

## What the prompt actually does

- The **positive prompt** fully drives the model's single forward pass at
  guidance=1.0 — it is not weakened or skipped. Confirmed by reading the
  actual sampling loop (`pipeline_longcat_video_avatar.py`): the
  conditional prediction (which uses your prompt) is used directly, with
  no extra unconditional pass to blend against.
- The **negative prompt is completely inert** at guidance=1.0 — the
  unconditional branch that would consume it is never computed. Don't
  bother phrasing anything as "avoid X" expecting a negative-prompt
  mechanism; there isn't one active here. Put restraint language in the
  positive prompt instead ("keeps a relaxed, mostly still face" rather
  than "no exaggerated movement").
- The **driving audio** controls timing (when the mouth moves) and some of
  the motion's intensity. The prompt controls demeanor, tone, and how
  restrained or expressive the delivery looks — the two combine, they
  don't override each other.

## Template

```
[Who / brief subject description], speaks [manner/tone] to the camera.
[Explicit restraint or energy instruction for the mouth/face].
[Anything about pose/camera stability, if relevant].
```

## Worked example (tested, confirmed to measurably reduce over-articulation)

> "A tired woman speaks softly and slowly with minimal, subtle lip
> movement. Her face stays mostly still and relaxed between words, small
> restrained mouth motion, no exaggerated mouth opening, calm low-energy
> delivery, natural micro-expressions only."

Compared side-by-side against the generic default prompt on the same
image+audio, this reduced the "whole face performing every phoneme" look
noticeably. It did **not** eliminate a separate, unrelated artifact (see
below) — the prompt only affects things the model actually attends to
during generation; it cannot fix a structural sampling artifact.

## Known limitation the prompt cannot fix

The distilled sampler tends toward a slight "closing" smile at the very
end of the final generated segment, regardless of prompt content — a
recurring model behavior, not something under prompt control (see
`docs/decisions.md`'s "Closing-smile artifact" entry). If your driving
audio ends with a decent pause before you actually need the last frame,
`run_longcat_avatar.sh --end-trim-s` can crop it off — but only use it
after watching the untrimmed result, since the artifact's onset can
overlap with real trailing speech and there is no universally safe value.

## Iteration tips

- Change one thing at a time (prompt text, or driving audio, or reference
  photo) so you can tell what actually moved the needle.
- Keep prompts to roughly one paragraph, similar length to the example
  above. Much longer prompts haven't been validated.
- Favor positive, concrete phrasing over negation ("stays still" beats
  "doesn't move") — the model has no negative-prompt mechanism active
  here, so double down on saying what you DO want.
- Don't reach for `--no-distill` or custom guidance-scale flags to get
  "more control" — both are confirmed to distort facial geometry on this
  checkpoint. The prompt is the correct, safe lever for demeanor and
  expression; guidance scale is not.
