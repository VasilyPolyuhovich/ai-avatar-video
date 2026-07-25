# AI Avatar Video

Self-hosted talking-head video generation (photo + audio → lip-synced video,
sized for a Telegram video note), on RunPod GPUs. Full goal/spec/constraints:
[`ai-video-message-brief.md`](ai-video-message-brief.md). Locked decisions and
their reasoning: [`docs/decisions.md`](docs/decisions.md).

## Status: A/B decided, generation pipeline automated

The A/B test is **decided**: **LongCat-Video-Avatar-1.5** beat InfiniteTalk
(better lip-sync, correct identity preservation, faster renders) — see
[`docs/decisions.md`](docs/decisions.md#2-generation-stack--ab-decided-2026-07-25-longcat-video-avatar-15-wins).
The full deploy → render → retrieve → terminate pipeline is proven working
end-to-end and is now a one-command tool.

**Want to generate a video?** Start at
[`docs/generate-video.md`](docs/generate-video.md) — it covers setup and
day-to-day usage for both the CLI and a local point-and-click UI.

What exists:

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

See [`docs/ab-test-plan.md`](docs/ab-test-plan.md) for the full comparison
that decided the winner, and [`docs/infra-notes.md`](docs/infra-notes.md)
for the persistence/cost/SSH details specific to this setup.

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

## Quick start

```bash
# Generate a video (see docs/generate-video.md for full setup first)
python3 scripts/generate_avatar_video.py --image photo.jpg --audio voice.mp3

# Or the local web UI
python3 scripts/app_gradio.py
```

Requires `~/.runpod-key-video` (account key) and
`~/.runpod/ssh/runpodctl-video-ssh-key[.pub]` locally — see
[`docs/generate-video.md`](docs/generate-video.md) for the full one-time
setup (including for colleagues) and
[`docs/infra-notes.md`](docs/infra-notes.md#two-runpod-accounts) for the
account details.
