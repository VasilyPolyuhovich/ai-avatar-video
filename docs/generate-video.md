# Generating a video

This is the how-to for turning a photo + audio clip + prompt into a
lip-synced avatar video, using the LongCat-Video-Avatar-1.5 pipeline that
won the A/B test (see [`decisions.md`](decisions.md)). It covers both the
one-time setup a new person needs and the day-to-day usage.

**Scope note**: this is a local-tool phase for a small, trusted group
sharing one RunPod account's credentials. There is no hosted service and no
per-user login. If that stops being appropriate (more people, less trust,
need for individual usage tracking), the next phase is a proper
Telegram-bot or web-app front end with its own auth — not built yet, see
`decisions.md`'s "Still open" section.

## A. One-time setup

Do this once per person. None of it requires the repo owner to be present
afterward — it's a from-scratch setup for anyone with a clean laptop.

### 1. Get the code

This is a **private** GitHub repository. Ask the repo owner
([VasilyPolyuhovich](https://github.com/VasilyPolyuhovich)) to add you as a
collaborator, accept the invite, then:

```bash
git clone https://github.com/VasilyPolyuhovich/ai-avatar-video.git
cd ai-avatar-video
```

### 2. Install prerequisites

- **Python 3.10+** (`python3 --version`)
- **git** (you just used it)
- An **`ssh`/`scp` client** — already present on macOS and Linux. On
  Windows, use WSL or Git Bash's bundled OpenSSH; a native `cmd.exe`/
  PowerShell without OpenSSH installed won't work.

### 3. Get the shared RunPod account key

All generation on this project bills to one existing RunPod account. The
repo owner will send you a copy of the `~/.runpod-key-video` file **out of
band** — a password manager, an encrypted note, anything other than plain
Slack/email text, and never committed to git.

**Treat this file like a password.** Anyone with it can deploy pods and
spend money on the shared account, and can see/stop everyone else's pods
too. Save it to exactly this path on your machine:

```
~/.runpod-key-video
```

### 4. Generate your own SSH keypair

This part is **not shared** — each person generates their own local
keypair. It never leaves your machine (only the public half gets sent to a
pod, automatically, at deploy time):

```bash
mkdir -p ~/.runpod/ssh
ssh-keygen -t rsa -b 4096 -f ~/.runpod/ssh/runpodctl-video-ssh-key -N ""
```

(RSA specifically — this project's tooling and RunPod's own proxy assume
it; a newer key type like ed25519 won't work here.)

### 5. Install Python dependencies (for the Gradio UI)

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

(The CLI tool itself, `scripts/generate_avatar_video.py`, has no extra
dependencies beyond Python's standard library — this step is only needed if
you want the point-and-click UI.)

That's the whole one-time setup. Steps 3-5 are also everything a new
colleague needs — nothing about your local repo clone, keys, or venv is
shared with anyone else's.

## B. Generating a video

### Option 1: command line

```bash
python3 scripts/generate_avatar_video.py \
  --image path/to/photo.jpg \
  --audio path/to/voice.mp3 \
  --prompt "A person speaks calmly to the camera in a steady, natural tone." \
  --resolution 480p
```

This deploys a fresh GPU pod, uploads your files, runs the render, downloads
the result to `outputs/<job-id>.mp4`, and terminates the pod — all
automatically. Progress streams live to your terminal; a real render takes
roughly **10-16 minutes** total (a few minutes for the pod to boot and
become reachable, then ~10 minutes of actual rendering). Silence for a
minute or two during pod boot is normal; total silence for much longer than
that is not — see Troubleshooting below.

Useful flags:
- `--resolution 720p` — higher quality, slower.
- `--output path/to/result.mp4` — choose where the file lands.
- `--dry-run` — **free**, no pod deployed. Prints the GPU candidates and the
  exact command that would run, so you can sanity-check your inputs before
  spending anything.
- `--json` — print the result as one JSON line instead of a summary
  sentence (useful for scripting).

### Option 2: local web UI (Gradio)

```bash
source .venv/bin/activate    # if not already active
python3 scripts/app_gradio.py
```

Open `http://127.0.0.1:7860` in your browser. Upload a photo and an audio
clip, optionally type a prompt, choose a resolution, click **Generate**.
The status box streams the same live progress as the CLI; the finished
video appears in the player when done. This only runs on your own machine —
nobody else can reach it.

### Cost

Everyone's usage draws from the **same shared account balance**. Rough math:
an A100 80GB runs about **$1.40-1.50/hr**, and a typical video takes
**~10-16 minutes end-to-end** (pod boot + render), so expect roughly
**$0.25-0.40 per video**. 720p and longer audio clips cost more (more
segments, more render time).

Check remaining balance any time with:

```bash
scripts/check_balance.sh
```

If you're about to generate a large batch, check balance first.

### Troubleshooting

- **The pod always terminates itself**, even if the run fails or you hit
  Ctrl-C — you shouldn't need to clean anything up manually. If a run
  prints `CRITICAL: pod ... may STILL BE RUNNING AND BILLING`, that's the
  one case where it didn't — check the
  [RunPod console](https://www.runpod.io/console/pods) or
  `scripts/check_balance.sh` and terminate it by hand.
- **`--dry-run` fails immediately** with a GPU-availability error — no
  matching A100/H100 is in stock right now; try again later, or ask the
  repo owner about raising `--max-price`.
- **A run fails partway through** — the error message prints to your
  terminal (and the Gradio status box). The underlying pipeline's known
  gotchas (`--stage_1=ai2v` vs `at2v`, `--num_segments` sizing) are already
  handled by `scripts/run_longcat_avatar.sh` and won't recur here, but if
  you ever end up running pieces of this by hand over SSH, read
  [`decisions.md`](decisions.md#2-generation-stack--ab-decided-2026-07-25-longcat-video-avatar-15-wins)
  first.
- **General infra questions** (SSH details, GPU/region behavior, registry
  auth) — see [`infra-notes.md`](infra-notes.md).
