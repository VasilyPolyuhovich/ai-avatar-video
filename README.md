# AI Avatar Video

Self-hosted talking-head video generation (photo + audio → lip-synced video,
sized for a Telegram video note), on RunPod GPUs. Full goal/spec/constraints:
[`ai-video-message-brief.md`](ai-video-message-brief.md). Locked decisions and
their reasoning: [`docs/decisions.md`](docs/decisions.md).

## Status: prepared, not deployed

Everything in this repo was built **without renting a GPU pod**. What exists:

- A **150GB network volume** on RunPod (`njvmaowlqv`, EUR-IS-1) — created, empty.
  Two RunPod accounts exist for this project; see
  [`docs/infra-notes.md`](docs/infra-notes.md#two-runpod-accounts) for which
  one is active by default and how to target the other.
- Two **Docker images**, built and verified working by this repo's CI on
  every push to `docker/**`:
  - `ghcr.io/vasilypolyuhovich/ai-avatar-infinitetalk:latest` — ComfyUI +
    `ComfyUI-WanVideoWrapper` + InfiniteTalk (Wan2.1-14B I2V backbone).
  - `ghcr.io/vasilypolyuhovich/ai-avatar-longcat:latest` — LongCat-Video-Avatar-1.5,
    standalone (no ComfyUI). Its `flash-attn` build needed 6 CI iterations to
    get right (see the Dockerfile's comments) — the image now builds cleanly
    and `import flash_attn` is verified to succeed.
  - Both are commit-pinned (exact SHAs in each Dockerfile's header) so a pull
    later reproduces exactly what was tested, not whatever the upstream repos
    happen to contain by then.
- No model weights are downloaded or baked in anywhere — they're multi-GB and
  belong on the volume, not the image or this git history. Download scripts
  are written (`scripts/download_weights_*.sh`) but not execution-tested.
- No pod has been created. `scripts/pod_up.py` deploys one on request.

See [`docs/ab-test-plan.md`](docs/ab-test-plan.md) for why there are two
images and how the winner gets picked, and
[`docs/infra-notes.md`](docs/infra-notes.md) for the persistence/cost/SSH
details specific to this setup.

## Layout

```
docker/infinitetalk/     Dockerfile + entrypoint + supervisor for the ComfyUI stack
docker/longcat-avatar/   Dockerfile + entrypoint for the LongCat standalone stack
scripts/pod_up.py        Deploy a pod: cheapest in-stock GPU, verified boot, auto-retry
scripts/download_weights_*.sh   Pull each stack's weights onto the volume (run on the pod)
scripts/autostop.sh      Daily self-stop (run on the pod; needs the account key)
scripts/check_balance.sh Local balance/runway check
docs/                    Decisions, A/B test plan, infra notes
.github/workflows/       CI: build + push both images to GHCR
```

## When it's time to actually deploy

```bash
# 1. Rank live GPU stock/price without spending anything
python3 scripts/pod_up.py --dry-run

# 2. Deploy the InfiniteTalk image (default), verify it boots, print the URL
python3 scripts/pod_up.py --wait

# 3. To try the LongCat image instead
IMAGE=ghcr.io/vasilypolyuhovich/ai-avatar-longcat:latest \
  PORTS=22/tcp python3 scripts/pod_up.py

# 4. Once SSH'd in (suffix from the RunPod Connect panel -- see docs/infra-notes.md):
bash scripts/download_weights_infinitetalk.sh   # or download_weights_longcat.sh

# 5. When done for the session -- just Stop/Terminate in the console,
#    or use scripts/check_balance.sh's account key to call podStop via the API
```

Requires `~/.runpod-key-video` (account key) and
`~/.runpod/ssh/runpodctl-video-ssh-key[.pub]` locally — see
[`docs/infra-notes.md`](docs/infra-notes.md#two-runpod-accounts).
