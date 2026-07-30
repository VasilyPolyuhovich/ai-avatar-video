# AI Avatar Video

Self-hosted talking-head video generation (photo + audio → lip-synced video,
sized for a Telegram video note), on RunPod GPUs.

**Status: A/B decided, generation pipeline automated and colleague-ready.**
The A/B test is **decided**: **LongCat-Video-Avatar-1.5** beat InfiniteTalk
(better lip-sync, correct identity preservation, faster renders) — see
[`docs/decisions.md`](docs/decisions.md#2-generation-stack--ab-decided-2026-07-25-longcat-video-avatar-15-wins).
The full deploy → render → retrieve → terminate pipeline is proven working
end-to-end and is a one-command tool, with a CLI, a reconnect-safe local web
UI, and a headless CLI mode for the UI script too.

## Getting started

New here? Do these in order — full detail for all three lives in
**[`docs/generate-video.md`](docs/generate-video.md), start there**:

1. **Get repo + credential access.** Ask the repo owner
   ([VasilyPolyuhovich](https://github.com/VasilyPolyuhovich)) for a
   collaborator invite and a copy of the shared `~/.runpod-key-video`
   account key (sent out-of-band, never over plain chat/email).
2. **Install prerequisites and generate your own SSH keypair.** Python
   3.10+, `git`, an SSH client, then `python3 -m venv .venv && pip install
   -r requirements.txt`. The keypair is per-person and local-only — never
   shared, never committed.
3. **Generate a video** — any of the three ways below.

Already set up? Jump straight in:

```bash
# CLI: deploys a pod, renders, downloads the result, terminates the pod
python3 scripts/generate_avatar_video.py --image photo.jpg --audio voice.mp3

# Local web UI: reload/reopen the page any time, it reconnects automatically
python3 scripts/app_gradio.py

# Or app_gradio.py's own headless CLI mode -- no browser, same warm-pod code path
python3 scripts/app_gradio.py --image photo.jpg --audio voice.mp3
```

Requires `~/.runpod-key-video` (account key) and
`~/.runpod/ssh/runpodctl-video-ssh-key[.pub]` locally — see
[`docs/generate-video.md`](docs/generate-video.md) for how to get both.

## What exists

- A **250GB network volume** on RunPod (`fl7pl7z0sz`, US-MD-1), holding both
  stacks' model weights (verified intact). Two RunPod accounts exist for
  this project; see
  [`docs/infra-notes.md`](docs/infra-notes.md#two-runpod-accounts) for which
  one is active by default and how to target the other.
- Two **Docker images**, built and verified working by this repo's CI on
  every push to `docker/**`, and both real-pod-tested with actual renders:
  - `ghcr.io/vasilypolyuhovich/ai-avatar-longcat:latest` — **the winning
    stack**, LongCat-Video-Avatar-1.5, standalone (no ComfyUI).
  - `ghcr.io/vasilypolyuhovich/ai-avatar-infinitetalk:latest` — ComfyUI +
    `ComfyUI-WanVideoWrapper` + InfiniteTalk (Wan2.1-14B I2V backbone). Kept
    for reference; not the default going forward.
  - Both are commit-pinned (exact SHAs in each Dockerfile's header) so a pull
    later reproduces exactly what was tested, not whatever the upstream repos
    happen to contain by then.
- `scripts/generate_avatar_video.py` — deploys a fresh on-demand pod, renders
  a photo+audio+prompt into a video, retrieves it, and always terminates the
  pod (even on error/Ctrl-C). `scripts/app_gradio.py` is a minimal local UI
  on top of it. See `docs/generate-video.md` for full usage, including how
  colleagues get set up with a shared account credential.
- `scripts/pod_up.py` — the lower-level building block (GPU ranking, deploy,
  verified boot, auto-retry) both the orchestrator and manual ops use
  directly.

## Layout

```
docker/longcat-avatar/          Dockerfile + entrypoint for the LongCat standalone stack (winner)
docker/infinitetalk/            Dockerfile + entrypoint + supervisor for the ComfyUI stack (reference)
scripts/generate_avatar_video.py  One-command generation: deploy -> render -> retrieve -> terminate
scripts/app_gradio.py           Minimal local web UI on top of generate_avatar_video.py
scripts/run_longcat_avatar.sh   The verified LongCat invocation (runs on the pod)
scripts/pod_up.py               Deploy a pod: cheapest in-stock GPU, verified boot, auto-retry
scripts/download_weights_*.sh   Pull each stack's weights onto the volume (run on the pod)
scripts/autostop.sh             Daily self-stop (run on the pod; needs the account key)
scripts/check_balance.sh        Local balance/runway check
requirements.txt                Local tooling deps (gradio, for app_gradio.py only)
docs/                           Decisions, A/B test plan, infra notes, generate-video how-to
.github/workflows/               CI: build + push both images to GHCR
```

## Documentation map

| File | Read this for |
|---|---|
| [`docs/generate-video.md`](docs/generate-video.md) | **Start here.** One-time setup + day-to-day usage: CLI, web UI, headless UI-script CLI mode, cost, troubleshooting. |
| [`docs/decisions.md`](docs/decisions.md) | Locked decisions and *why* — generation stack, network volume, UI design — a dated changelog. |
| [`docs/ab-test-plan.md`](docs/ab-test-plan.md) | The A/B methodology and results that picked LongCat over InfiniteTalk. |
| [`docs/infra-notes.md`](docs/infra-notes.md) | Deeper RunPod reference: SSH recipes, GPU/CUDA choices, persistence, cost control, registry auth. |
| [`docs/prompt-guide.md`](docs/prompt-guide.md) | How to write a prompt that actually changes tone/expression on this checkpoint. |
| [`ai-video-message-brief.md`](ai-video-message-brief.md) | Original pre-implementation project brief (goal/spec). Historical — some decisions in it are since superseded, see `decisions.md`. |
