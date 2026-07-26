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
- `--prompt` — the real, working lever for tone and expression. It fully
  drives the model at this tool's default settings (not weakened or
  ignored) — see `docs/prompt-guide.md` for what works and what doesn't
  (short version: one consistent idea per prompt, avoid contradictory
  traits like "ironic but relaxed" in the same sentence — that confuses
  the model into erratic mouth movement, confirmed live).
- `--end-trim-s N` — crops N seconds off the very end of the final output.
  Opt-in mitigation for a confirmed model quirk (a slight "closing smile"
  at the end of the last segment) — **do not set a default value blindly**,
  preview the untrimmed render first. The artifact's onset can overlap
  with genuine trailing speech, so a value safe for one clip can cut real
  words on another (confirmed live 2026-07-26).
- `--no-distill` and `--audio-gain-db` still exist for backward
  compatibility but are **not recommended**: `--no-distill` (full 50-step
  sampler, ~17x slower per segment) is confirmed to distort facial
  geometry on this checkpoint regardless of step count, and
  `--audio-gain-db` is a confirmed no-op — LongCat normalizes the driving
  audio's loudness unconditionally before encoding, so the value never
  reaches the model. See `docs/decisions.md#2-generation-stack` for both
  findings.

### Option 2: local web UI (Gradio)

```bash
source .venv/bin/activate    # if not already active
python3 scripts/app_gradio.py
```

Open `http://127.0.0.1:7860` in your browser. Upload a photo and an audio
clip, type a prompt (see the built-in examples or `docs/prompt-guide.md`),
choose a resolution, click **Generate**. An **Advanced** section has the
same end-trim/`--no-distill` controls as the CLI, clearly marked with
which ones are actually recommended. This only runs on your own machine —
nobody else can reach it.

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
- **Articulation/mimicry looks exaggerated** — the prompt IS working and
  does have real effect; rewrite it to explicitly ask for restrained
  articulation (see `docs/prompt-guide.md`'s worked example) rather than
  reaching for `--no-distill` or `--audio-gain-db` — both are confirmed to
  either not help or actively make things worse (see `decisions.md`).
  Keep the prompt to one consistent idea; mixing contradictory traits in
  one sentence tends to produce erratic, unstable mouth movement instead
  of a compromise between them.
- **A `--no-distill` run failed with "remote command exceeded ...s"** — the
  render was very likely progressing fine, just slower than the timeout
  budget. `--timeout` auto-scales to 10800s (3 hours) for `--no-distill`
  automatically; for unusually long audio (many segments), pass an even
  larger `--timeout` explicitly. The pod still terminates correctly when
  this happens (cost-safety held, GPU time already spent on that attempt is
  real but not compounding).
- **Your local network hiccups mid-render (WiFi drop, laptop sleep, VPN
  reconnect)** — this used to kill the whole render (and terminate the
  pod), even though the actual generation on the pod might have kept going
  fine. Fixed: the render now runs detached on the pod and is monitored via
  reconnecting polls rather than one fragile SSH stream held open for the
  whole render — a network blip just costs a few missed status updates,
  tolerated up to **30 minutes** of lost contact before giving up. If you
  see `lost contact with pod for over 1800s`, that's this backstop finally
  giving up after real, prolonged connectivity loss, not a first-hiccup
  overreaction.
- **General infra questions** (SSH details, GPU/region behavior, registry
  auth) — see [`infra-notes.md`](infra-notes.md).
