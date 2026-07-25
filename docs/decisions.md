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

## Still open
- Whether SageAttention gets built on top of the LongCat image (deferred to
  the actual pod — see `infra-notes.md`).
- Turn the ad hoc weight-integrity check (size-compare every local file
  against the HF API) into a real script, and consider running it as a
  standard last step of both download scripts rather than only on-demand
  after a failure.
- Build a simple end-user UI in front of LongCat (upload photo + audio +
  prompt, get back the Telegram video note) — the pipeline shape is now
  locked in (`scripts/run_longcat_avatar.sh`), so this can start.
- Get a real ElevenLabs Ukrainian voice-clone sample to replace the
  placeholder Russian `celyj.mp3` test clip.
- Pod for the A/B was terminated 2026-07-25 after the winning render
  (network volume `fl7pl7z0sz` persists with all weights intact) — next
  session needs a fresh `pod_up.py` deploy against the `longcat-avatar`
  image before any further rendering.
