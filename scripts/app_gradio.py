#!/usr/bin/env python3
"""Minimal local UI for scripts/generate_avatar_video.py: upload a photo and
audio clip, type a prompt, click Generate, watch progress, get a video back.

Runs on http://127.0.0.1:7860 -- localhost only, single machine, matches
this project's current phase (see docs/generate-video.md): each user (you or
a colleague) runs this locally on their own machine with their own copy of
the shared RunPod account key. There is no auth here and none is needed for
that reason; a public/hosted version is a deliberately separate later phase.

Unlike the CLI (one pod per invocation, always terminated), this UI keeps
ONE pod alive across a session via PodSession (scripts/generate_avatar_video.py)
so repeat "Generate" clicks skip the ~5 min torch.compile warmup cost of a
fresh pod. The pod is destroyed on: the Stop pod button, any render error,
an idle timeout (default 15 min, override with IDLE_TIMEOUT_S), or this
process exiting (Ctrl-C, SIGTERM, or the terminal closing -- best effort;
see docs/decisions.md#5-generation-ui for the accepted residual risk of a
SIGKILL/crash/sleep not being covered).

    pip install -r requirements.txt
    python3 scripts/app_gradio.py
"""
import atexit
import queue
import signal
import sys
import threading
import time
from pathlib import Path

import gradio as gr

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import pod_up  # noqa: E402
from generate_avatar_video import PodSession  # noqa: E402

MAX_LOG_LINES = 400
IDLE_TIMEOUT_S = int(pod_up.env("IDLE_TIMEOUT_S", "900") or "900")

session = PodSession(idle_timeout_s=IDLE_TIMEOUT_S)


def handle_submit(image, audio, prompt, resolution, no_distill, end_trim_s):
    if not image or not audio:
        yield None, "Please provide both a photo and an audio clip.", gr.update()
        return

    yield None, "Starting ...", gr.update(interactive=False)

    q: "queue.Queue" = queue.Queue()
    box = {}

    def worker():
        t0 = time.monotonic()
        try:
            local_path, job_id = session.render(
                image, audio, prompt=prompt or None, resolution=resolution,
                no_distill=no_distill, end_trim_s=end_trim_s or None,
                status_cb=q.put)
            box["result"] = (local_path, job_id, time.monotonic() - t0)
        except Exception as e:
            box["error"] = e
        finally:
            q.put(None)  # sentinel: done

    threading.Thread(target=worker, daemon=True).start()

    log_lines = []
    while True:
        line = q.get()
        if line is None:
            break
        log_lines.append(line)
        yield None, "\n".join(log_lines[-MAX_LOG_LINES:]), gr.update()

    if "error" in box:
        yield None, "\n".join(log_lines) + f"\n\nFAILED: {box['error']}", gr.update(interactive=True)
        return

    local_path, _job_id, elapsed_s = box["result"]
    summary = (f"\n\nThis render: {elapsed_s / 60:.1f} min. "
               f"Session so far: ~${session.session_cost_so_far():.2f} "
               f"({session.gpu_id} @ ${session.gpu_price}/hr).")
    yield local_path, "\n".join(log_lines) + summary, gr.update(interactive=True)


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


EXAMPLE_PROMPTS = [
    ["A person speaks calmly to the camera in a steady, natural tone, with a neutral, composed expression."],
    ["A tired woman speaks softly and slowly with minimal, subtle lip movement. Her face stays mostly "
     "still and relaxed between words, small restrained mouth motion, no exaggerated mouth opening, "
     "calm low-energy delivery, natural micro-expressions only."],
    ["A person speaks with warm, friendly energy, animated but controlled expression, natural hand-free "
     "conversational delivery, looking directly at the camera."],
]

CSS = """
.section-card { border: 1px solid var(--border-color-primary); border-radius: 12px;
                padding: 14px 16px; margin-bottom: 10px; background: var(--background-fill-secondary); }
.status-card { border-radius: 10px; padding: 10px 14px; background: var(--background-fill-secondary);
               border: 1px solid var(--border-color-primary); }
#generate-btn { min-height: 46px; font-size: 1.05em; }
"""

with gr.Blocks(title="AI Avatar Video", theme=gr.themes.Soft(primary_hue="blue"), css=CSS) as demo:
    gr.Markdown("# \U0001f3ac AI Avatar Video")
    gr.Markdown(
        "Turn a photo + audio clip into a lip-synced talking-head video. "
        "Runs on a shared RunPod GPU account -- this page only runs on your own machine."
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
            "- One-time machine setup (credentials, SSH key, Python deps) is **not** covered here -- "
            "see `docs/generate-video.md` in the repo if you're setting up a new machine.\n"
            "- Full technical background: `docs/decisions.md` (what's proven to work and why) and "
            "`docs/prompt-guide.md` (how to write a prompt that actually helps)."
        )

    status_md = gr.Markdown(pod_status_text(), elem_classes=["status-card"])

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

            with gr.Row():
                submit_btn = gr.Button("▶️  Generate", variant="primary", elem_id="generate-btn")
                stop_btn = gr.Button("⏹️  Stop pod")

        with gr.Column(scale=1):
            video_out = gr.Video(label="Result")
            status_out = gr.Textbox(label="Status / progress", lines=18, interactive=False)

    submit_btn.click(
        handle_submit,
        inputs=[image_in, audio_in, prompt_in, resolution_in, no_distill_in, end_trim_in],
        outputs=[video_out, status_out, submit_btn],
    )
    stop_btn.click(handle_stop, outputs=status_out)

    # Polls session state every 10s so an idle-timeout auto-termination
    # (which otherwise only prints to this process's own terminal, with
    # nothing to push a browser update outside an active request) becomes
    # visible without requiring another click.
    gr.Timer(10).tick(pod_status_text, outputs=status_md)


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


if __name__ == "__main__":
    # demo.queue() is required: without it, a generator-returning event
    # handler's yields all buffer and the browser only sees the final
    # state -- for a ~10-16 min job that reads as a frozen/broken UI.
    demo.queue().launch(server_name="127.0.0.1")
