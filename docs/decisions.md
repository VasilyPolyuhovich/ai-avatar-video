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
Created 2026-07-24: id **`wrqr1689to`**, name `ai-avatar-video`, **150GB**,
DC **EU-RO-1** (Secure), on the account this project now uses by default
(`~/.runpod-key-video` — see `infra-notes.md`). Sized for both stacks'
weights (~50-70GB InfiniteTalk + ~30-40GB LongCat) plus outputs/cache
headroom. A network volume is DC-locked — any pod deploy that attaches it
must pin `dataCenterId: EU-RO-1`.

(An earlier volume, `plk85ofiny`, was created 2026-07-21 on a different
account and no longer exists — 404 from the API when checked 2026-07-24,
cause unknown. Nothing was ever downloaded onto it.)

## 4. Output format (unchanged from brief)
Square MP4/H.264, ≤60s, diameter 384-640px; thumbnail JPEG ≤200KB/≤320px.
Telegram rounds client-side — no manual circular masking needed.

## Still open
- Which stack wins the A/B (decided after the first real render, not yet run).
- Whether SageAttention gets built on top of either image (deferred to the
  actual pod — see `infra-notes.md`).
