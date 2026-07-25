# Locked decisions

Dated 2026-07-21 (prep phase — no pod deployed yet).

## 1. Audio
**TTS voice clone via ElevenLabs**, Ukrainian. Not own recorded voice, not an
open-source TTS clone (Fish Speech / CosyVoice2 / IndexTTS-2 don't have
confirmed Ukrainian support as of this check).

## 2. Generation stack — A/B on the same volume
- **Baseline: Wan2.1-I2V-14B (720p) + InfiniteTalk**, ComfyUI, via
  `ComfyUI-WanVideoWrapper` (Kijai). Verified 2026-07: InfiniteTalk has **no**
  Wan2.2 backbone — the repo is frozen on `Wan2.1-I2V-14B-480P` as its base
  weights (the ComfyUI branch additionally supports the 720p checkpoint).
- **Challenger: LongCat-Video-Avatar-1.5** (Meituan/MeiGen-AI, released
  2026-05-21, MIT license). Swaps Wav2Vec2 for Whisper-Large-v3 audio
  conditioning, 8-step distilled inference (`--use_distill`), optional INT8
  (`--use_int8`) to fit VRAM. Standalone Python scripts, no ComfyUI.
- Both get a Docker image (`docker/infinitetalk`, `docker/longcat-avatar`) so
  either can be pulled and run without a rebuild. See `ab-test-plan.md` for
  the comparison protocol and decision rule.

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
- Which stack wins the A/B (InfiniteTalk has rendered twice, second pass
  much better after a sampler-settings fix; LongCat render is in progress
  2026-07-25 on verified-intact weights, not yet compared).
- Whether SageAttention gets built on top of either image (deferred to the
  actual pod — see `infra-notes.md`).
- Turn the ad hoc weight-integrity check (size-compare every local file
  against the HF API) into a real script, and consider running it as a
  standard last step of both download scripts rather than only on-demand
  after a failure.
- Once a stack wins the A/B: build a simple end-user UI (upload photo +
  audio, get back the Telegram video note) in front of whichever pipeline
  is kept — not started yet, deliberately deferred until the A/B result is
  in so it isn't built against the losing stack's invocation shape.
