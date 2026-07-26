# Locked decisions

Dated 2026-07-21 (prep phase — no pod deployed yet).

## 1. Audio
**TTS voice clone via ElevenLabs**, Ukrainian. Not own recorded voice, not an
open-source TTS clone (Fish Speech / CosyVoice2 / IndexTTS-2 don't have
confirmed Ukrainian support as of this check).

## 2. Generation stack — A/B decided 2026-07-25: **LongCat-Video-Avatar-1.5 wins**

- **Winner: LongCat-Video-Avatar-1.5** (Meituan/MeiGen-AI, released
  2026-05-21, MIT license). Whisper-Large-v3 audio conditioning (genuinely
  multilingual, unlike InfiniteTalk's Chinese-specific wav2vec2), 8-step
  distilled inference (`--use_distill`). Standalone Python script, no
  ComfyUI. User verdict on the corrected render (`--stage_1=ai2v`, see
  below): "практично ідеальний" (practically perfect) — clearly better
  lip-sync/mimicry than InfiniteTalk's best render, and correct identity
  preservation once the invocation bug was fixed.
  - **Exact working invocation** (also encoded in
    `scripts/run_longcat_avatar.sh`, which additionally auto-computes
    `--num_segments` from the input audio's real duration):
    ```bash
    torchrun --nproc_per_node=1 run_demo_avatar_single_audio_to_video.py \
      --checkpoint_dir=/workspace/longcat-weights/LongCat-Video-Avatar-1.5 \
      --stage_1=ai2v \
      --num_segments=<computed, see below> \
      --input_json=<path to {"prompt","cond_image","cond_audio":{"person1":...}}> \
      --use_distill --model_type avatar-v1.5 --resolution 480p
    ```
  - **`--stage_1` must be `ai2v`, never `at2v`.** `at2v` is audio+text-to-video
    with **no image conditioning at all** — `cond_image` is never read on
    that code path. The first real attempt used `at2v` by mistake (copied
    from a doc example without checking which stage actually consumes the
    photo) and produced a fluent, well-lip-synced video of a **completely
    different, unrelated person** — confirmed live 2026-07-25, user caught
    it immediately from the output ("це зовсім інша людина"). `ai2v` is the
    upstream script's own argparse default.
  - **`--num_segments` must cover the audio length**, not be left at its
    default of 1. avatar-v1.5's hardcoded constants are `num_frames=93`,
    `save_fps=25`, `num_cond_frames=13` — one segment is only
    `93/25 = 3.72s` of video regardless of audio length; each additional
    segment (via the long-video continuation / KV-cache path) adds
    `(93-13)/25 = 3.2s`. Formula: `segments = 1 + ceil((audio_seconds - 3.72)
    / 3.2)`. Our 12.05s test clip needed 4 segments (~13.3s of video,
    correctly covers it). Leaving this at 1 truncates the phrase mid-sentence
    — also confirmed live on the same bad first render.
  - Render time: ~10 min wall-clock for a 4-segment 480p clip on one A100
    80GB (checkpoint load + vocal separation + torch.compile warmup ~5min
    one-time cost, ~2min/segment after that).
- **Runner-up: Wan2.1-I2V-14B (720p) + InfiniteTalk**, ComfyUI, via
  `ComfyUI-WanVideoWrapper` (Kijai). Verified 2026-07: InfiniteTalk has **no**
  Wan2.2 backbone — the repo is frozen on `Wan2.1-I2V-14B-480P` as its base
  weights (the ComfyUI branch additionally supports the 720p checkpoint).
  Second render (after fixing a sampler-settings mismatch) was rated
  "значно краще" (much better) but still had lip-sync/language-mismatch and
  per-window motion-instability complaints that LongCat doesn't share.
- Both images are kept (registry storage is cheap, and a second opinion
  stays available) — `docker/infinitetalk`, `docker/longcat-avatar`. See
  `ab-test-plan.md` for the full comparison protocol/rubric this decision
  was based on.

## 3. Network volume
Created 2026-07-24: id **`fl7pl7z0sz`**, name `ai-avatar-video`, DC
**US-MD-1** (Secure), on the account this project now uses by default
(`~/.runpod-key-video` — see `infra-notes.md`). A network volume is
DC-locked — any pod deploy that attaches it must pin `dataCenterId:
US-MD-1`.

**Resized 150GB → 250GB on 2026-07-25** (`PATCH /v1/networkvolumes/{id}`
`{"size": 250}`, in place, no data loss, confirmed) after LongCat's actual
weight footprint (both repos combined, post-`--exclude`) turned out bigger
than the original ~30-40GB estimate — the volume hit 95% full (50GB
InfiniteTalk + 91GB LongCat-partial vs 150GB cap) mid-download and started
throwing `OSError: [Errno 122] Disk quota exceeded`. `df -h /workspace`
is **not** useful for checking free space on this volume — it reports the
MooseFS cluster's total capacity (hundreds of TB), not the per-volume
quota; only an actual quota-exceeded error or the RunPod console's own
volume-size field tell the truth.

### Weight-download corruption — root cause (2026-07-25)
Multiple LongCat weight downloads on this volume silently corrupted files
(final size 2-3MB **larger** than the real HuggingFace size — not a
truncation, an overwrite/append artifact), across both the `xet` fast-
transfer backend and the classic HTTP downloader (`HF_HUB_DISABLE_XET=1`),
plus separate unrelated hangs (a cross-filesystem `mv`, a `.gitignore.lock`
wait) earlier in the same session. The common factor across every incident
is **concurrent multi-threaded writes to this specific MooseFS-backed
network volume** — `hf download`'s default 8 parallel workers, each writing
a different large shard at once. One incident (a `hf_xet` "background
writer channel closed" crash) happened right at the 142GB/150GB quota
boundary and was very likely just a badly-surfaced disk-quota error, not
evidence of xet itself being unreliable — but a *later* crash, same
symptom (bytes added, not lost), happened well after the resize with
plenty of headroom, using the plain HTTP downloader, which rules out quota
as the sole explanation. Working theory: the volume's write-durability
under concurrent load is the weak point, independent of which HF download
backend is in use. Mitigation applied: `scripts/download_weights_longcat.sh`
now passes `--max-workers 4` (default is 8) to both downloads, and every
download should be followed by a full per-file size check against the HF
API before trusting the weights (ad hoc Python one-liner used this
session; not yet turned into a standing script — see "Still open" below).

(History, all same day, 2026-07-24 — the volume was empty at every move so
none of this cost anything but time:
1. First created in **EU-RO-1**: zero A100/H100 Secure stock at deploy
   time (`lowestPrice(...).stockStatus` was `None` for every variant,
   confirmed per-DC, not just the global figure a plain `gpuTypes` query
   shows).
2. Moved to **EUR-IS-1** (had A100-SXM4-80GB at Low stock). ~9 deploy
   attempts there and in US-MD-1 (see step 3) all failed with uptime
   stuck at 0, which `pod_up.py` reports as "broken host" — every single
   one blamed on a *different* host/machineId, which looked exactly like
   the DC having systemically bad hardware. **It wasn't.** RunPod's
   pod-init emails (which the API doesn't surface, only the console/email
   does) all said the same thing: `IMAGE_AUTH_ERROR: unauthorized to use
   image ghcr.io/vasilypolyuhovich/ai-avatar-infinitetalk:latest`. The
   GHCR packages defaulted to **private** visibility — `docker/build-push-
   action`'s `GITHUB_TOKEN` can publish a package but can't change its
   visibility (that's web-UI-only), so every pull attempt failed
   identically regardless of host or region. Fixed by registering the
   image registry credentials with RunPod (`saveRegistryAuth`) instead of
   chasing a host-reliability ghost.
3. Moved again to **US-MD-1** (A100-SXM4-80GB, Low stock) before the real
   cause was found, on the assumption the whole EUR-IS-1 region was
   unreliable. In hindsight that move was unnecessary — the auth fix alone
   would have worked in EUR-IS-1 too — but it's where the volume actually
   ended up, and moving it again would cost real time for no benefit.)

(An earlier volume, `plk85ofiny`, was created 2026-07-21 on a different
account and no longer exists — 404 from the API when checked 2026-07-24,
cause unknown. Nothing was ever downloaded onto it.)

## 4. Output format (unchanged from brief)
Square MP4/H.264, ≤60s, diameter 384-640px; thumbnail JPEG ≤200KB/≤320px.
Telegram rounds client-side — no manual circular masking needed.

## 5. Generation UI (decided 2026-07-25)

Now that the pipeline shape is locked (`scripts/run_longcat_avatar.sh`),
built a one-command generator on top of it: `scripts/generate_avatar_video.py`
(CLI, importable) + `scripts/app_gradio.py` (minimal local UI). Full usage
in [`generate-video.md`](generate-video.md). Two scoping decisions made via
AskUserQuestion, both explicitly trading isolation/scale for simplicity
given the current small, trusted user base:

- **GPU lifecycle: on-demand per request for the CLI; session-scoped reuse
  for the Gradio UI (amended 2026-07-25).** The CLI (`generate_avatar_video()`)
  keeps the original rule unchanged: deploy a fresh pod, render, retrieve,
  terminate, every single call — nothing bills while idle, and it must
  **always** terminate the pod, including on crash or Ctrl-C (see
  `_safe_terminate`). The Gradio UI (`scripts/app_gradio.py`) **partially
  reverses this** via a new `PodSession` class: it keeps one pod alive
  across several "Generate" clicks, because a real session that day showed
  the cost of the original rule in practice — every fresh pod repays a ~5
  min one-time `torch.compile` kernel-compilation cost (cached only on that
  pod's local disk), which is wasteful when generating several videos in a
  row. `PodSession` still guarantees termination on: an explicit Stop
  button, any render error (a crash can leave the GPU in a bad state —
  this session hit a real `CUDA error: device(s) is/are busy or
  unavailable` mid-render), an idle timeout (default 15 min,
  `IDLE_TIMEOUT_S` override), and the local process exiting (`atexit` for
  Ctrl-C/normal exit, explicit `SIGTERM`/`SIGHUP` handlers for the
  terminal closing — plain `atexit` alone does *not* cover signal-based
  termination). **Explicitly accepted, not solved**: `SIGKILL`, or the
  machine crashing/sleeping for a long time, bypass all four triggers —
  `scripts/check_balance.sh`/the RunPod console remain the backstop. See
  `scripts/generate_avatar_video.py`'s `PodSession` docstring for the full
  locking-safety contract (an earlier draft had two real self-deadlock
  bugs, caught in review before implementation).
- **Colleague access: a shared RunPod account key**, not individual
  accounts or a RunPod Team. Simpler onboarding (no per-person account
  setup, no billing split) at the explicit cost of usage isolation —
  anyone with the key can see and stop anyone else's pods, and everyone
  bills to the same balance. Each person still generates their **own** SSH
  keypair locally (no sharing needed there — deploy-time key authorization
  already works per-deploy with zero code changes). If this stops being an
  acceptable tradeoff (more people, less trust), revisit as a RunPod Team
  or individual accounts before scaling further.
- Both remain a **local CLI/UI tool per person**, not a hosted service —
  the Telegram-bot/web-app phase (with real per-user auth, no shared-secret
  distribution) is still deliberately future work, not started.

### Prompt/guidance-scale limitation in distilled mode, and mitigations (2026-07-25)

User-reported: generated videos have exaggerated articulation/mimicry, and
the `--prompt` field seems to have no effect. Confirmed as expected
behavior, not a bug: `run_demo_avatar_single_audio_to_video.py` (upstream)
unconditionally overrides **both** `text_guidance_scale` and
`audio_guidance_scale` to `1.0` whenever `--use_distill` is passed
(regardless of any other flag) — and `scripts/run_longcat_avatar.sh` always
passed `--use_distill` (the whole point of the A/B-winning config was the
8-step distilled sampler's speed/cost). There is **no supported combination**
of "fast distilled sampling" + "guidance scale actually matters" in the
upstream script as written. Two mitigations added, both opt-in:
- `--no-distill` (`run_longcat_avatar.sh`, threaded through
  `generate_avatar_video.py`/`PodSession`/the Gradio UI's Advanced section):
  disables the distilled LoRA entirely, restoring the full 50-step sampler
  and the upstream defaults (`text_guidance_scale`/`audio_guidance_scale`
  = 4.0) — the only real lever for prompt/guidance control. **Costs roughly
  17x the usual render time/cost per segment**, confirmed live 2026-07-25 —
  not just the ~6x the 50-vs-8 step-count ratio alone would suggest, since
  a non-distilled step is also individually ~2.7x slower (full
  classifier-free guidance runs two forward passes per step vs the
  distilled DMD LoRA's one; observed ~38.5s/step vs ~14.5s/step). Not the
  default. `generate_avatar_video.py`'s `--timeout` auto-scales from 1800s
  to 10800s when `--no-distill` is set for exactly this reason — the first
  real test run hit the *old*, too-short default and got killed by the
  local hard-timeout safeguard at 82% through segment 1 of 3, wasting the
  GPU time already spent (the render itself was working fine; the timeout,
  sized for distilled-mode expectations, was the bug).
- `--audio-gain-db`: applies an `ffmpeg volume` filter to a **local** copy
  of the driving audio before upload (original file untouched). Cheap,
  free-of-GPU-cost lever based on the empirical observation that the
  model's motion signal correlates with the driving audio's energy —
  quieter/flatter audio tends to reduce exaggerated articulation without
  needing `--no-distill` at all. Worth trying first.

### Resilient remote render: survive local network drops (2026-07-26)

A real `--no-distill` render (up to ~2 hours) got killed twice by a
transient LOCAL network problem (client WiFi/laptop sleep/VPN reconnect —
nothing to do with the pod), burning real GPU spend both times. Root cause:
the render ran via one long-lived foreground SSH exec held open for the
entire render; when that single connection dropped, sshd almost certainly
delivered SIGHUP to the remote process tree, killing the actual `torchrun`
render on the pod, not just the local monitoring view of it — and the
resulting exception then triggered pod termination via `PodSession`'s
error-handling contract, destroying a render that may have been
progressing completely normally.

**Fix**: `render_on_pod` now launches the remote command **detached**
(`setsid nohup ... > log 2>&1 < /dev/null & disown` — survives the SSH
session dying) and monitors it via short-lived reconnecting polls
(`run_remote_detached`, byte-offset `tail -c +N` reads) instead of one
fragile stream. A local network blip now costs a few missed poll cycles,
tolerated up to 30 minutes of lost contact, not a dead render. Reviewed via
a Plan-agent pass before implementation (same practice as the `PodSession`
work) — caught two real bugs in the first draft: a marker-based
full-log-refetch design that silently corrupted the incremental display
stream by one character per poll (fixed with authoritative byte-offset
tracking, verified against a local simulation including a poll landing
mid-multi-byte-UTF-8-character), and a redesign that would have broken
`output_filename` extraction for every single render (the new function
returns accumulated text once rather than streaming lines, so
`render_on_pod` now scans the *returned* text, not a per-line generator).

Unchanged, explicitly not solved by this fix: total local-process death
(`kill -9`, dead battery, OS crash) still leaves the pod running with
nothing to terminate it — `check_balance.sh`/RunPod console remain the
backstop, same as before.

## Still open
- Whether SageAttention gets built on top of the LongCat image (deferred to
  the actual pod — see `infra-notes.md`).
- Turn the ad hoc weight-integrity check (size-compare every local file
  against the HF API) into a real script, and consider running it as a
  standard last step of both download scripts rather than only on-demand
  after a failure.
- Get a real ElevenLabs Ukrainian voice-clone sample to replace the
  placeholder Russian `celyj.mp3` test clip.
- Pod for the A/B was terminated 2026-07-25 after the winning render
  (network volume `fl7pl7z0sz` persists with all weights intact) —
  `scripts/generate_avatar_video.py` deploys fresh pods on request now, no
  manual `pod_up.py` step needed for normal generation.
- A real paid end-to-end run of `generate_avatar_video.py` happened
  2026-07-25 (via the CLI) — hit a real `CUDA error: device(s) is/are busy
  or unavailable` mid-render on one attempt, and confirmed the
  `finally`-terminate contract held (pod correctly torn down, no orphaned
  billing) even on that real failure. A deliberate Ctrl-C negative test,
  and a real end-to-end exercise of the Gradio UI's `PodSession` (session
  reuse across clicks, Stop button, idle-timeout firing, Ctrl-C/SIGTERM
  handling), are still outstanding — only free checks (`--dry-run`, import
  smoke tests, manual lock-invariant review) have been run against the
  `PodSession` code so far.
- Public/hosted colleague interface (Telegram bot or web app, with real
  per-user auth) — future phase, not started.
