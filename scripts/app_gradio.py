#!/usr/bin/env python3
"""Local UI for scripts/generate_avatar_video.py: upload a photo and audio
clip, type a prompt, click Generate, watch progress, get a video back.
Also runnable headless via CLI flags (--image/--audio/...) as an
alternative to opening the browser -- see the bottom of this file.

Runs on http://127.0.0.1:7860 -- localhost only by default, matches
this project's current phase (see docs/generate-video.md): each user (you or
a colleague) runs this locally on their own machine with their own copy of
the shared RunPod account key. There is no auth here and none is needed for
that reason; a public/hosted version is a deliberately separate later phase.
Optional remote access over an existing Tailscale tailnet: see BIND_ALL
below and docs/generate-video.md.

Unlike the CLI (one pod per invocation, always terminated), the web-UI mode
keeps ONE pod alive across a session via PodSession (scripts/generate_avatar_video.py)
so repeat "Generate" clicks skip the ~5 min torch.compile warmup cost of a
fresh pod. The pod is destroyed on: the Stop pod button, any render error,
an idle timeout (default 15 min, override with IDLE_TIMEOUT_S), or this
process exiting (Ctrl-C, SIGTERM, or the terminal closing -- best effort;
see docs/decisions.md#5-generation-ui for the accepted residual risk of a
SIGKILL/crash/sleep not being covered).

Render progress/results live in a server-side RenderState object (not tied
to any one browser request), polled by a Timer -- so a page refresh, a
closed-then-reopened tab, or a dropped connection (confirmed live: a
Tailscale-connected phone losing its stream mid-render) reconnects to
whatever's actually happening instead of showing a dead/empty page. This is
in-memory only, scoped to this one running process -- it does NOT survive
an app_gradio.py restart; see docs/decisions.md.

    pip install -r requirements.txt
    python3 scripts/app_gradio.py                       # web UI
    python3 scripts/app_gradio.py --image X --audio Y    # headless CLI
"""
import argparse
import atexit
import signal
import sys
import threading
import time
import uuid
from pathlib import Path

import gradio as gr

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import pod_up  # noqa: E402
from generate_avatar_video import PodSession  # noqa: E402

MAX_LOG_LINES = 400
IDLE_TIMEOUT_S = int(pod_up.env("IDLE_TIMEOUT_S", "900") or "900")

EXAMPLE_PROMPTS = [
    ["A person speaks calmly to the camera in a steady, natural tone, with a neutral, composed expression."],
    ["A tired woman speaks softly and slowly with minimal, subtle lip movement. Her face stays mostly "
     "still and relaxed between words, small restrained mouth motion, no exaggerated mouth opening, "
     "calm low-energy delivery, natural micro-expressions only."],
    ["A person speaks with warm, friendly energy, animated but controlled expression, natural hand-free "
     "conversational delivery, looking directly at the camera."],
]

session = PodSession(idle_timeout_s=IDLE_TIMEOUT_S)


class RenderState:
    """Shared, in-process render-progress state -- deliberately NOT gr.State
    (that's per-browser-session in this Gradio version, the wrong shape for
    state that must survive a reload and be visible to every connection).
    Same "shared object + Timer polling" pattern as PodSession/status_md
    below, applied to render progress and last-used inputs too. Guarded by
    its own lock, kept separate from PodSession._lock (never nested --
    status_cb callbacks fire from inside render_on_pod/deploy_pod/
    wait_for_ssh, and from PodSession.terminate()/_watchdog_loop, all of
    which run outside PodSession._lock by design; see PodSession's own
    _terminate_locked docstring in generate_avatar_video.py)."""

    def __init__(self):
        self._lock = threading.Lock()
        self.status = "idle"  # idle | running | done | error
        self.log_lines = []
        self.result_path = None
        self.error_text = None
        self.job_token = None
        self.last_image = None
        self.last_audio = None
        self.last_prompt = EXAMPLE_PROMPTS[0][0]
        self.last_resolution = "480p"
        self.last_no_distill = False
        self.last_end_trim_s = 0

    def try_start(self, image, audio, prompt, resolution, no_distill, end_trim_s):
        """Atomic gate: returns a job_token to proceed, or None if a render
        is already running. Callers must NOT spawn a worker thread on None
        -- prevents a double-click/two-tabs race where a second worker's
        PodSession._busy RuntimeError would otherwise clobber the first,
        still-running job's displayed state."""
        with self._lock:
            if self.status == "running":
                return None
            self.status = "running"
            # Seeded with a placeholder line (not []) so the click's own
            # immediate response (handle_submit -> refresh_from_state, called
            # synchronously right after this) has something to show instead
            # of a blank status panel -- the background thread's first real
            # log line may take a moment to land, especially on an
            # already-warm pod with no deploy/SSH wait ahead of it.
            self.log_lines = ["Starting ..."]
            self.result_path = None
            self.error_text = None
            self.job_token = uuid.uuid4().hex
            self.last_image, self.last_audio, self.last_prompt = image, audio, prompt
            self.last_resolution = resolution
            self.last_no_distill, self.last_end_trim_s = no_distill, end_trim_s
            return self.job_token

    def append_log(self, token, line):
        with self._lock:
            if token == self.job_token:
                self.log_lines.append(line)

    def finish(self, token, result_path, summary_line):
        with self._lock:
            if token == self.job_token:
                self.status = "done"
                self.result_path = result_path
                self.log_lines.append(summary_line)

    def fail(self, token, error_text):
        with self._lock:
            if token == self.job_token:
                self.status = "error"
                self.error_text = error_text

    def snapshot(self):
        with self._lock:
            return dict(
                status=self.status, job_token=self.job_token,
                log_text="\n".join(self.log_lines[-MAX_LOG_LINES:]),
                result_path=self.result_path, error_text=self.error_text,
                last_image=self.last_image, last_audio=self.last_audio,
                last_prompt=self.last_prompt, last_resolution=self.last_resolution,
                last_no_distill=self.last_no_distill, last_end_trim_s=self.last_end_trim_s)


render_state = RenderState()


# Foreign-pod detection itself lives in pod_up.py (pod_up.find_other_pod,
# pod_up.make_foreign_pod_check, pod_up.ForeignPodDetected) -- shared with
# generate_avatar_video.py's own standalone CLI, which needs the exact same
# check (see its _cli()). Only the UI-facing bits (banner display, caching
# for the polling Timers below) stay here.

_foreign_pod_cache = {"checked_at": 0.0, "result": None}
_FOREIGN_POD_CACHE_TTL_S = 8  # under the 10s banner Timer's interval, so a
# poll is never more than one interval stale. Without this, every open
# browser tab/device independently re-runs the same account-wide RunPod API
# call every 10s -- N tabs means N identical calls; this collapses them to
# ~1 real call per TTL window regardless of tab count. Deliberately NOT used
# by pod_up.make_foreign_pod_check() itself (the pre_deploy_check backstop
# right before an actual deploy) -- that one needs the freshest possible
# read for its race-resistance purpose, not a cached one.


def _find_foreign_pod():
    """The first OTHER project pod on the shared account (not this
    session's own pod_id), or None -- for UI display only (the banner and
    handle_submit's own fast inline check). Cached briefly (see above) and
    fails OPEN (returns None) on a transient RunPod API error -- this is a
    safety-net convenience check, not the only protection (PodSession._busy
    still guards this process's own concurrent renders); a flaky API call
    shouldn't make the whole app unusable."""
    now = time.monotonic()
    if now - _foreign_pod_cache["checked_at"] < _FOREIGN_POD_CACHE_TTL_S:
        return _foreign_pod_cache["result"]
    try:
        result = pod_up.find_other_pod(session.account_key, exclude_pod_id=session.pod_id)
    except Exception as e:
        print(f"[app_gradio] foreign-pod check failed ({e}) -- proceeding without it")
        result = None
    _foreign_pod_cache["checked_at"], _foreign_pod_cache["result"] = now, result
    return result


def make_foreign_pod_check(override):
    """Built fresh per Generate click with that click's override state --
    passed through as PodSession.render()'s pre_deploy_check, called right
    before the actual deploy call (the real, race-resistant gate; the
    check in handle_submit below is the fast, page-level UX version). Does
    its own fresh, uncached lookup -- see pod_up.make_foreign_pod_check."""
    return pod_up.make_foreign_pod_check(session.account_key, exclude_pod_id=session.pod_id, override=override)


def _run_render(token, image, audio, prompt, resolution, no_distill, end_trim_s, override):
    t0 = time.monotonic()
    try:
        local_path, _job_id = session.render(
            image, audio, prompt=prompt or None, resolution=resolution,
            no_distill=no_distill, end_trim_s=end_trim_s or None,
            pre_deploy_check=make_foreign_pod_check(override),
            status_cb=lambda line: render_state.append_log(token, line))
        elapsed_s = time.monotonic() - t0
        # snapshot_cost_info() reads gpu_id/gpu_price/cost atomically under
        # PodSession._lock -- a plain session.gpu_id/session.gpu_price read
        # here could race a concurrent Stop-pod click (session.terminate())
        # clearing both to None between the two attribute reads.
        gpu_id, gpu_price, cost_so_far = session.snapshot_cost_info()
        summary = (f"\n\nThis render: {elapsed_s / 60:.1f} min. "
                   f"Session so far: ~${cost_so_far:.2f} ({gpu_id} @ ${gpu_price}/hr).")
        render_state.finish(token, local_path, summary)
    except Exception as e:
        render_state.fail(token, str(e))


def refresh_from_state(seen_token):
    """Reads render_state -- used both as the Generate click's immediate
    response and as the fast Timer's recurring poll, so ANY connected page
    (including one that just reconnected) converges on the same view.
    seen_token is a small per-browser gr.State display cursor (the only
    per-session state in this app) -- without it, re-sending an unchanged
    gr.Video value on every tick risks resetting playback in the browser
    that's already watching it."""
    s = render_state.snapshot()
    video_update = gr.update()
    new_seen = seen_token
    if seen_token != s["job_token"]:
        if s["status"] == "done" and s["result_path"]:
            video_update = s["result_path"]
            new_seen = s["job_token"]
        else:
            # A newer job has started since this browser last caught up
            # (job_token changed) and it isn't done yet -- clear any stale
            # video from a PREVIOUS job rather than leaving it visibly
            # displayed through this job's whole run (and through a
            # possible failure, which would otherwise look like the old
            # video IS the new job's result). new_seen deliberately stays
            # unchanged here (not advanced to s["job_token"]) so the "done"
            # branch above still fires and shows the real result once this
            # job actually finishes.
            video_update = None
    status_text = s["log_text"]
    if s["status"] == "error":
        status_text = (status_text + f"\n\nFAILED: {s['error_text']}").strip()
    btn_update = gr.update(interactive=(s["status"] != "running"))
    return video_update, status_text, btn_update, new_seen


def restore_inputs():
    s = render_state.snapshot()
    return (s["last_image"], s["last_audio"], s["last_prompt"],
            s["last_resolution"], s["last_no_distill"], s["last_end_trim_s"])


def refresh_foreign_pod_banner():
    foreign = _find_foreign_pod()
    if foreign is None:
        return gr.update(visible=False), gr.update(visible=False, value=False)
    status_note = f", статус: `{foreign['desiredStatus']}`" if foreign.get("desiredStatus") else ""
    msg = (f"⚠️ **Виявлено активний под на спільному акаунті:** `{foreign['id']}` "
           f"(`{foreign['name']}`{status_note}). Схоже, десь уже триває рендер (можливо, колега, "
           f"або ваша ж інша вкладка/процес) -- або це просто зупинений, але не термінований "
           f"под з минулого разу. **Generate заблоковано**, поки ви явно не підтвердите "
           f"продовження нижче.")
    return gr.update(value=msg, visible=True), gr.update(visible=True)


def handle_submit(image, audio, prompt, resolution, no_distill, end_trim_s, override, seen_token):
    # override_in is always reset to False in the return value below --
    # it's a per-click, single-use confirmation (approved design: "not a
    # sticky checkbox someone forgets is on"). Without this, ticking it
    # once would silently keep overriding the foreign-pod block on every
    # later click too, since refresh_foreign_pod_banner()'s "still found"
    # branch deliberately leaves the checkbox's live value alone (so a
    # click racing a poll doesn't get its own tick wiped out first).
    if not image or not audio:
        return None, "Please provide both a photo and an audio clip.", gr.update(), seen_token, gr.update(value=False)

    if not override:
        foreign = _find_foreign_pod()
        if foreign is not None:
            status_note = f", статус: `{foreign['desiredStatus']}`" if foreign.get("desiredStatus") else ""
            msg = (f"⚠️ Знайдено активний под `{foreign['id']}` (`{foreign['name']}`{status_note}) на "
                   f"спільному акаунті. Позначте прапорець підтвердження вище і натисніть "
                   f"Generate ще раз, якщо все одно хочете продовжити.")
            # visible=True here too, not just value=False -- this inline
            # check can be the FIRST thing to detect a foreign pod (between
            # two 10s refresh_foreign_pod_banner ticks), in which case the
            # checkbox this message tells the user to tick would otherwise
            # still be visible=False from the last banner refresh.
            return None, msg, gr.update(), seen_token, gr.update(value=False, visible=True)

    token = render_state.try_start(image, audio, prompt, resolution, no_distill, end_trim_s)
    if token is None:
        video_update, status_text, btn_update, new_seen = refresh_from_state(seen_token)
        return video_update, status_text, btn_update, new_seen, gr.update(value=False)

    threading.Thread(
        target=_run_render,
        args=(token, image, audio, prompt, resolution, no_distill, end_trim_s, override),
        daemon=True).start()
    video_update, status_text, btn_update, new_seen = refresh_from_state(seen_token)
    return video_update, status_text, btn_update, new_seen, gr.update(value=False)


def handle_stop():
    if session.pod_id is None:
        return "No active pod."
    session.terminate()
    return "Pod stopped."


def pod_status_text():
    if session.pod_id is None:
        return "⬛ **No active pod** -- the first click below will deploy one (~10-16 min)."
    return (f"\U0001f7e2 **Pod running:** `{session.pod_id}` ({session.gpu_id} @ "
            f"${session.gpu_price}/hr) -- session cost so far: "
            f"**~${session.session_cost_so_far():.2f}**")


CSS = """
.section-card { border: 1px solid var(--border-color-primary); border-radius: 12px;
                padding: 14px 16px; margin-bottom: 10px; background: var(--background-fill-secondary); }
.status-card { border-radius: 10px; padding: 10px 14px; background: var(--background-fill-secondary);
               border: 1px solid var(--border-color-primary); }
#generate-btn { min-height: 46px; font-size: 1.05em; }
"""

with gr.Blocks(title="AI Avatar Video", theme=gr.themes.Soft(primary_hue="blue"), css=CSS) as demo:
    seen_token_state = gr.State(value=None)  # per-browser display cursor, see refresh_from_state

    gr.Markdown("# \U0001f3ac AI Avatar Video")
    gr.Markdown(
        "Turn a photo + audio clip into a lip-synced talking-head video. "
        "Runs on a shared RunPod GPU account -- reload or reopen this page any time, "
        "an in-progress render keeps going and picks back up automatically."
    )

    with gr.Accordion("ℹ️ How this works & cost (click to expand)", open=False):
        gr.Markdown(
            "- **First click** on a fresh session deploys a GPU pod and takes **~10-16 min** "
            "(pod boot + a one-time model warmup). **Later clicks reuse the same pod** and are "
            "much faster (typically a few minutes).\n"
            "- **Cost**: shared account, an A100 80GB runs **~$1.40-1.50/hr**. A typical render "
            "is roughly **$0.25-0.40**. Trust the live session-cost figure below over any per-video "
            "estimate -- idle time between clicks bills too.\n"
            f"- The pod **auto-terminates after {IDLE_TIMEOUT_S // 60} min idle**, or immediately on "
            "**Stop pod**, or on any render error. Click **Stop pod** when you're done to be sure.\n"
            "- **Reloading or reopening this page is safe** -- progress and your last inputs are "
            "kept on this machine (not per-browser-tab) and reappear automatically.\n"
            "- If another pod is already active on the shared account (e.g. a colleague's), "
            "Generate is blocked with a warning until you explicitly confirm.\n"
            "- One-time machine setup (credentials, SSH key, Python deps) is **not** covered here -- "
            "see `docs/generate-video.md` in the repo if you're setting up a new machine.\n"
            "- Full technical background: `docs/decisions.md` (what's proven to work and why) and "
            "`docs/prompt-guide.md` (how to write a prompt that actually helps)."
        )

    status_md = gr.Markdown(pod_status_text(), elem_classes=["status-card"])
    foreign_pod_banner = gr.Markdown(visible=False, elem_classes=["status-card"])

    with gr.Row():
        with gr.Column(scale=1):
            with gr.Group(elem_classes=["section-card"]):
                gr.Markdown("### 1. Inputs")
                image_in = gr.Image(type="filepath", label="Reference photo")
                audio_in = gr.Audio(type="filepath", label="Driving audio")

            with gr.Group(elem_classes=["section-card"]):
                gr.Markdown("### 2. Prompt")
                prompt_in = gr.Textbox(
                    label="Prompt",
                    lines=3,
                    value=EXAMPLE_PROMPTS[0][0],
                    info="This has a real, direct effect on tone and expression -- it's not "
                         "decoration. Keep it to ONE consistent idea (e.g. \"calm and restrained\"); "
                         "mixing contradictory traits in one sentence (\"ironic but relaxed\") tends "
                         "to confuse the model into erratic mouth movement. See docs/prompt-guide.md.")
                with gr.Accordion("Example prompts (click one to use it)", open=False):
                    gr.Examples(examples=EXAMPLE_PROMPTS, inputs=[prompt_in])

            with gr.Group(elem_classes=["section-card"]):
                gr.Markdown("### 3. Options")
                resolution_in = gr.Radio(["480p", "720p"], value="480p", label="Resolution")
                with gr.Accordion("Advanced (rarely needed)", open=False):
                    end_trim_in = gr.Number(
                        label="End trim (seconds)", value=0, minimum=0,
                        info="The model tends to add a slight 'closing smile' at the very end of "
                             "the last segment. This crops that many seconds off the end -- but "
                             "only turn it on AFTER watching the untrimmed result once, since the "
                             "same crop can cut real trailing speech on a different clip. 0 = off.")
                    no_distill_in = gr.Checkbox(
                        label="Full 50-step sampler (--no-distill)",
                        value=False,
                        info="NOT RECOMMENDED: confirmed to distort facial geometry on this "
                             "checkpoint, and ~17x slower per segment. Kept only as an escape "
                             "hatch for experimentation -- use the prompt above to control "
                             "expression instead, it's the lever that actually works safely.")
                    override_in = gr.Checkbox(
                        label="Продовжити, попри активний под іншого користувача",
                        value=False, visible=False,
                        info="З'являється лише коли на спільному акаунті вже виявлено активний "
                             "под. Позначте, якщо точно знаєте, що робите.")

            with gr.Row():
                submit_btn = gr.Button("▶️  Generate", variant="primary", elem_id="generate-btn")
                stop_btn = gr.Button("⏹️  Stop pod")

        with gr.Column(scale=1):
            video_out = gr.Video(label="Result")
            status_out = gr.Textbox(label="Status / progress", lines=18, interactive=False)

    submit_btn.click(
        handle_submit,
        inputs=[image_in, audio_in, prompt_in, resolution_in, no_distill_in, end_trim_in,
                override_in, seen_token_state],
        outputs=[video_out, status_out, submit_btn, seen_token_state, override_in],
    )
    stop_btn.click(handle_stop, outputs=status_out)

    # Fast poll for render progress -- reads render_state, not any one
    # request's own data, so a reconnected/reloaded page catches up within
    # one tick instead of showing a dead page. concurrency_limit=None on
    # every Timer/load handler below: demo.queue()'s default concurrency
    # limit is 1 for the WHOLE process, and this app now has several cheap
    # pollers (this timer, the 10s status timer, demo.load()'s handlers)
    # that would otherwise serialize behind each other and behind
    # handle_submit -- exactly in the multi-connection scenario (phone +
    # desktop tab) this design targets.
    gr.Timer(2).tick(
        refresh_from_state, inputs=[seen_token_state],
        outputs=[video_out, status_out, submit_btn, seen_token_state],
        concurrency_limit=None)
    # Slower poll for pod cost/status and the foreign-pod banner -- both
    # are real RunPod API calls, kept off the fast 2s timer deliberately.
    gr.Timer(10).tick(pod_status_text, outputs=status_md, concurrency_limit=None)
    gr.Timer(10).tick(refresh_foreign_pod_banner, outputs=[foreign_pod_banner, override_in],
                       concurrency_limit=None)

    # Page load/reload: show current data immediately instead of waiting
    # for the first timer tick, and restore the last-used inputs on this
    # machine (browser reload/reconnect only -- an app_gradio.py process
    # restart still loses this, it's in-memory by design; see docs/decisions.md).
    demo.load(restore_inputs,
              outputs=[image_in, audio_in, prompt_in, resolution_in, no_distill_in, end_trim_in],
              concurrency_limit=None)
    demo.load(refresh_from_state, inputs=[seen_token_state],
              outputs=[video_out, status_out, submit_btn, seen_token_state],
              concurrency_limit=None)
    demo.load(pod_status_text, outputs=status_md, concurrency_limit=None)
    demo.load(refresh_foreign_pod_banner, outputs=[foreign_pod_banner, override_in],
              concurrency_limit=None)


atexit.register(session.terminate)  # covers Ctrl-C and other normal process
# exits -- Gradio's own block_thread() catches KeyboardInterrupt, does its
# own shutdown, and returns normally, so the interpreter reaches a normal
# exit and atexit fires. Deliberately NOT installing a custom SIGINT
# handler -- that would replace Python's default KeyboardInterrupt
# conversion and break Gradio's own shutdown path for no benefit.

# atexit alone does NOT cover "closing the terminal" (SIGHUP) or a process
# manager sending SIGTERM -- Python installs no default handler for either,
# atexit callbacks don't run on unhandled-signal termination, and a signal
# handler that returns without exiting just lets execution resume as if
# nothing happened. sys.exit(0) forces a normal unwind (atexit still fires
# too, harmlessly re-terminating an already-None pod_id).
def _handle_terminating_signal(signum, _frame):
    print(f"[app_gradio] caught signal {signum} -- terminating session pod")
    session.terminate()
    sys.exit(0)


signal.signal(signal.SIGTERM, _handle_terminating_signal)
if hasattr(signal, "SIGHUP"):  # not present on Windows
    signal.signal(signal.SIGHUP, _handle_terminating_signal)


def _parse_cli_args():
    p = argparse.ArgumentParser(
        description="AI Avatar Video -- opens the web UI by default. Pass "
                     "--image and --audio for a headless one-shot render "
                     "instead (no browser, no port bound), using the same "
                     "PodSession machinery as the UI.")
    p.add_argument("--image", help="Reference photo path -- triggers CLI mode together with --audio")
    p.add_argument("--audio", help="Driving audio path -- triggers CLI mode together with --image")
    p.add_argument("--prompt")
    p.add_argument("--resolution", choices=["480p", "720p"], default="480p")
    p.add_argument("--no-distill", action="store_true",
                    help="NOT RECOMMENDED -- see docs/decisions.md. Kept as an escape hatch.")
    p.add_argument("--end-trim-s", type=float,
                    help="Only set this after previewing the untrimmed render once -- see --help in "
                         "run_longcat_avatar.sh or docs/decisions.md for why there's no safe default.")
    p.add_argument("--output", help="Where to save the result (default: outputs/<job-id>.mp4)")
    p.add_argument("--override-foreign-pod", action="store_true",
                    help="Deploy even if another ai-avatar-video-* pod is already active "
                         "on the shared account (e.g. a colleague's). Off by default -- "
                         "without it, a foreign pod aborts the run before spending anything.")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_cli_args()

    if args.image or args.audio:
        if not (args.image and args.audio):
            sys.exit("--image and --audio must both be provided for CLI mode")
        t0 = time.monotonic()
        try:
            # No status_cb here: _make_loggers() (generate_avatar_video.py)
            # already unconditionally print()s every log/remote-log line
            # itself -- passing status_cb=print on top of that used to
            # double-print every line of a full render's output.
            local_path, _job_id = session.render(
                args.image, args.audio, prompt=args.prompt, resolution=args.resolution,
                no_distill=args.no_distill, end_trim_s=args.end_trim_s,
                output_path=args.output,
                pre_deploy_check=make_foreign_pod_check(override=args.override_foreign_pod))
        except Exception as e:
            print(f"[generate] FAILED: {e}", file=sys.stderr)
            session.terminate()
            sys.exit(1)
        elapsed_s = time.monotonic() - t0
        gpu_id, gpu_price, cost_so_far = session.snapshot_cost_info()
        print(f"[generate] video ready: {local_path} "
              f"({elapsed_s / 60:.1f} min, ~${cost_so_far:.2f} @ {gpu_id} ${gpu_price}/hr)")
        session.terminate()
        sys.exit(0)

    # demo.queue() is required: without it, generator/event handler updates
    # buffer and the browser only sees the final state -- for a ~10-16 min
    # job that reads as a frozen/broken UI. default_concurrency_limit=None:
    # this app's several Timer/load pollers are cheap in-memory reads and
    # must not serialize behind each other or behind handle_submit (see the
    # comment above the Timer/load wiring for the concrete scenario this
    # avoids).
    #
    # server_name defaults to 127.0.0.1 (localhost only). For remote access
    # over an existing Tailscale tailnet, prefer `tailscale serve --bg
    # http://localhost:7860` over changing this -- it keeps this process on
    # the safe localhost-only default and gets a proper HTTPS .ts.net URL
    # from Tailscale's own proxy. Only set BIND_ALL=1 (binds 0.0.0.0) if
    # `tailscale serve` isn't available on your tailnet yet -- reachability
    # is still tailnet-only either way (Tailscale's own network-level ACLs
    # gate who can even route to this machine's tailnet IP), you just lose
    # the HTTPS wrapper and the clean hostname. See docs/generate-video.md.
    bind_all = (pod_up.env("BIND_ALL", "0") or "0") == "1"
    demo.queue(default_concurrency_limit=None).launch(
        server_name="0.0.0.0" if bind_all else "127.0.0.1")
