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
- `--no-distill` — disable the 8-step distilled sampler and run the full
  50-step one instead. **This is the only way `--prompt` (and guidance
  scale generally) actually affects the output** — the distilled mode this
  tool uses by default forces both text and audio guidance to their
  minimum regardless of what you ask for, which is also why articulation
  can look exaggerated with the default settings. **Costs roughly 17x the
  usual render time/cost per segment** (not just the ~6x the step-count
  ratio alone would suggest — a non-distilled step is also individually
  ~2.7x slower, confirmed live: ~38.5s/step vs ~14.5s/step, since full
  guidance runs two forward passes per step instead of one). `--timeout`
  auto-scales to 3 hours for `--no-distill` runs for exactly this reason —
  don't override it downward unless you know your audio is short.
- `--audio-gain-db -6` (or similar) — apply a volume reduction to a local
  copy of your audio before upload. Quieter, flatter delivery tends to
  produce less exaggerated mouth articulation, since the model's motion
  signal correlates with the driving audio's energy. Free to try (no extra
  GPU time), worth experimenting with before reaching for `--no-distill`.

### Option 2: local web UI (Gradio)

```bash
source .venv/bin/activate    # if not already active
python3 scripts/app_gradio.py
```

Open `http://127.0.0.1:7860` in your browser. Upload a photo and an audio
clip, optionally type a prompt, choose a resolution, click **Generate**.
An **Advanced** section has the same `--no-distill`/audio-gain controls as
the CLI. This only runs on your own machine — nobody else can reach it.

**Unlike the CLI, the UI keeps one pod alive across a session** instead of
redeploying for every click — the first Generate pays the usual ~10-16 min
(pod boot + the one-time ~5 min `torch.compile` warmup), later clicks in
the same session skip both and just render (typically a few minutes). The
pod is destroyed automatically on any of:
- Clicking **Stop pod**.
- Any render error (a crash can leave the GPU in a bad state, so the next
  attempt gets a clean pod rather than reusing a possibly-broken one).
- Sitting idle for **15 minutes** by default (override with
  `IDLE_TIMEOUT_S=1800 python3 scripts/app_gradio.py` for 30 min, etc.) —
  the safety net for leaving the browser tab open without clicking Stop;
  closing the *tab* does **not** stop the pod by itself, since the local
  `app_gradio.py` process keeps running in the background regardless.
- Ctrl-C, or closing the terminal running `app_gradio.py` (best effort).

**Not covered**: force-killing the process (`kill -9`), or the machine
crashing/sleeping for a long time — none of the above triggers can fire in
those cases. If you're ever unsure, check
`scripts/check_balance.sh` or the [RunPod console](https://www.runpod.io/console/pods)
directly.

The status line at the top of the page shows the current session pod (if
any) and running cost; it refreshes automatically every ~10s, including
after an idle-timeout auto-termination.

### Cost

Everyone's usage draws from the **same shared account balance**. Rough math:
an A100 80GB runs about **$1.40-1.50/hr**. For the **CLI** (one pod per
call), a typical video costs roughly **$0.25-0.40** (~10-16 min end-to-end).
For the **Gradio UI**, per-click cost is *not* the same as session cost once
a pod is reused — a click after the first can be much cheaper (no redeploy/
recompile), but the pod keeps billing during any idle gaps between clicks
too. Trust the UI's own "session so far: ~$X" figure over a per-video
estimate. 720p and longer audio clips cost more either way (more segments,
more render time); `--no-distill` costs roughly **17x** a normal render per
segment, not just the ~6x its step count alone would suggest — budget
accordingly (e.g. a 3-segment `--no-distill` clip took over 26 minutes of
GPU time just to reach 82% of segment 1 in testing).

Check remaining balance any time with:

```bash
scripts/check_balance.sh
```

If you're about to generate a large batch, check balance first.

### Troubleshooting

- **The CLI's pod always terminates itself**, even if the run fails or you
  hit Ctrl-C — you shouldn't need to clean anything up manually there. The
  **Gradio UI**'s session pod terminates on the four triggers listed above
  (not literally every click) — see that section if a pod seems to be
  lingering. Either way, if a run prints
  `CRITICAL: pod ... may STILL BE RUNNING AND BILLING`, that's the one case
  automatic cleanup didn't work — check the
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
- **Articulation/mimicry looks exaggerated, or the prompt seems to do
  nothing** — expected in the default distilled mode (see `--no-distill`
  above); try `--audio-gain-db` first (free), then `--no-distill` (~17x
  render cost per segment — budget time/money accordingly) if that's not
  enough.
- **A `--no-distill` run got killed with "remote command exceeded ...s --
  killed"** — the render was very likely progressing fine, just slower than
  the timeout budget. `--timeout` auto-scales to 10800s (3 hours) for
  `--no-distill` automatically; for unusually long audio (many segments),
  pass an even larger `--timeout` explicitly. The pod still terminates
  correctly when this happens (cost-safety held, GPU time already spent on
  that attempt is real but not compounding).
- **General infra questions** (SSH details, GPU/region behavior, registry
  auth) — see [`infra-notes.md`](infra-notes.md).
