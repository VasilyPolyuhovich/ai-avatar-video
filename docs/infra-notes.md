# Infra notes (project-specific distillation of the `runpod-pod-ops` skill)

Full detail lives in the skill (`~/.claude/skills/runpod-pod-ops/`); this file
only records the choices specific to this project.

## Persistence — why the Docker-image approach was chosen here

| Layer | Survives Stop→Start | Survives Delete |
|---|---|---|
| container root `/` | ❌ | ❌ |
| network volume `/workspace` (`plk85ofiny`) | ✅ | ✅ |

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
  `~/.runpod/ssh/runpodctl-ssh-key.pub`.

## SSH recipe (once a pod exists)
```bash
ssh -tt -i ~/.runpod/ssh/runpodctl-ssh-key \
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

## Registry
Images are public on GHCR (`ghcr.io/vasilypolyuhovich/ai-avatar-*`) once CI
builds them — no registry auth needed to pull. If that changes, see the
skill's `ssh-and-api.md` for `saveRegistryAuth` (lowercase owner name is
mandatory, or the mutation silently leaves a broken record).
