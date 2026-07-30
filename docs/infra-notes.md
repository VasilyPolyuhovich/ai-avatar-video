# Infra notes

Deeper reference material on this project's RunPod setup — SSH recipes,
GPU/CUDA choices, persistence layout, cost control, registry auth. If
you're setting up a new machine or just want to generate a video, start at
[`generate-video.md`](generate-video.md) instead; come back here when you
need the "why" behind something or want to do a manual/SSH-level operation.

## Two RunPod accounts

This project now has two RunPod accounts available, kept side by side (no
"switching" a single credential file back and forth):

| | Account key file | SSH key | Network volume |
|---|---|---|---|
| **Active (default)** | `~/.runpod-key-video` | `~/.runpod/ssh/runpodctl-video-ssh-key[.pub]` | `fl7pl7z0sz`, 250GB (resized from 150GB 2026-07-25), US-MD-1 |
| Original | `~/.runpod-key` | `~/.runpod/ssh/runpodctl-ssh-key[.pub]` | none (`plk85ofiny` 404'd, deleted at some point — cause unknown, nothing was stored on it) |

`scripts/pod_up.py` and `scripts/check_balance.sh` default to the active
account. To target the original account instead, override **both**
`ACCOUNT_KEY_FILE` and `SSH_PUBKEY_FILE` together (they must point at the
same account's key pair, or the deployed pod will authorize a key that
doesn't match whichever account you actually billed):
```bash
ACCOUNT_KEY_FILE=~/.runpod-key SSH_PUBKEY_FILE=~/.runpod/ssh/runpodctl-ssh-key.pub \
  python3 scripts/pod_up.py --dry-run
```
Both SSH keys are RSA (RunPod's proxy only accepts legacy ssh-rsa). Neither
needs registering in the RunPod console's "SSH Public Keys" for this
project's own images — `pod_up.py` passes the pubkey directly as the pod's
`PUBLIC_KEY` env, and each image's `entrypoint.sh` authorizes it itself.
Console registration only matters if you also want RunPod's own
templates/web-terminal to accept the same key.

## Persistence — why the Docker-image approach was chosen here

| Layer | Survives Stop→Start | Survives Delete |
|---|---|---|
| container root `/` | ❌ | ❌ |
| network volume `/workspace` (`fl7pl7z0sz`) | ✅ | ✅ |

Both `docker/infinitetalk` and `docker/longcat-avatar` bake the **app** (ComfyUI
+ nodes, or the LongCat repo + deps) into the image, and mount only
**weights + outputs** from the network volume at `/workspace`. That means
recovery after a Stop→Start is just **the container restarting from the
image** — no manual reinstall/venv-relaunch step, unlike a bare-venv-on-volume
setup. The trade-off: any code/dependency change needs a new image build
(CI handles this — see `.github/workflows/docker-build.yml`), not a `git pull`
on the pod.

## GPU / CUDA
- Target: **A100 80GB or H100, Secure Cloud**. `pod_up.py` defaults
  `MIN_VRAM=80`.
- `--min-cuda-version 12.9` equivalent is baked into the InfiniteTalk image's
  base (`nvidia/cuda:12.9.1-devel-ubuntu22.04`); LongCat's image pins
  `12.4.1-devel-ubuntu22.04` to match its `torch==2.6.0+cu124` requirement.
  Landing on an old-driver host would break SageAttention/Triton the same way
  `libcudart.so.13` breaks vLLM — new-driver A100/H100 avoids the whole class
  of problem.
- **SageAttention is intentionally not built into the image.** It compiles
  against a specific GPU architecture; baking it for the wrong arch either
  silently underperforms or fails to load. Build it once, on the actual pod,
  after landing on a host (`pip install sageattention` or build from source
  with `TORCH_CUDA_ARCH_LIST` set to that host's actual arch).

## Ports
- InfiniteTalk image: ComfyUI on **8188** (prefer an SSH tunnel —
  `ssh -N -L 8188:localhost:8188 ...` — over the public proxy, which
  terminates TLS at RunPod's edge). SSH on 22.
- LongCat image: no server, it's a batch-job container (`torchrun
  run_demo_avatar_*.py` via SSH/`docker exec`). SSH on 22 only.
- Both images run `sshd` and accept `$PUBLIC_KEY` (the pod's env var) into
  `/root/.ssh/authorized_keys` — `pod_up.py` populates it from
  `~/.runpod/ssh/runpodctl-video-ssh-key.pub` (the active account's key; see
  "Two RunPod accounts" above).

## SSH recipe (once a pod exists)
```bash
ssh -tt -i ~/.runpod/ssh/runpodctl-video-ssh-key \
  -o PubkeyAcceptedAlgorithms=+ssh-rsa \
  -o StrictHostKeyChecking=no \
  <POD_ID>-<SUFFIX>@ssh.runpod.io
```
`<SUFFIX>` changes on recreate — copy it from the RunPod console's Connect
panel each time, don't assume it's stable.

## Cost control
- No idle auto-stop exists on RunPod — the GPU stays reserved even at 0%
  util. **Terminate when a render session is done**; don't leave the pod
  Stopped-but-billing-storage longer than needed for the next session, or
  Running at all when idle.
- `scripts/autostop.sh` (daily self-stop) needs the **account key**, not the
  pod's injected `$RUNPOD_API_KEY` (restricted, can't stop pods). The human
  writes that key file into the pod directly (never over `ssh -tt` — the PTY
  echoes it into the transcript); automation only reads/verifies it.
- `scripts/check_balance.sh` runs locally against the account key to check
  runway before/after a session.
- ⚠️ Stopping (not terminating) a Secure pod can strand on "not enough free
  GPUs on the host" if that host fills up. For a scarce GPU, prefer
  **terminate + `pod_up.py` fresh deploy** (network volume reattaches to any
  host) over Stop→Resume.

## Registry — the images are PRIVATE, not public

`ghcr.io/vasilypolyuhovich/ai-avatar-infinitetalk` and `-longcat` default to
**private** visibility on GHCR — `docker/build-push-action`'s `GITHUB_TOKEN`
can publish a package but cannot flip its visibility (GitHub only exposes
that toggle in the package's own web settings, not via the REST API; a
`PATCH /user/packages/container/{name}` attempt 404s). This caused a full
afternoon of chasing phantom "broken hosts" (2026-07-24, see
`decisions.md#3-network-volume`) before the real error surfaced: RunPod's
pod-init failure emails say `IMAGE_AUTH_ERROR: unauthorized`, but the
GraphQL API gives no hint of *why* a pod never started — uptime just sits
at 0, identical to an actually-broken host. **If a deploy ever silently
fails to start again, check the RunPod console/email for the real reason
before assuming host flakiness.**

Fixed via RunPod's `saveRegistryAuth` GraphQL mutation (lowercase owner name
is mandatory, or the mutation silently leaves a broken record) — credential id
`cmrzcm01x0079uy8y4v8536wo`, already the default `REGISTRY_AUTH_ID` in
`pod_up.py`. No action needed for normal use. To make the packages public
instead (removing the need for this credential entirely), it has to be done
by hand: GitHub → the repo → Packages → each package → Package settings →
Change visibility.
