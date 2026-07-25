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


def handle_submit(image, audio, prompt, resolution, no_distill, audio_gain_db):
    if not image or not audio:
        yield None, "Please provide both a photo and an audio clip."
        return

    q: "queue.Queue" = queue.Queue()
    box = {}

    def worker():
        t0 = time.monotonic()
        try:
            local_path, job_id = session.render(
                image, audio, prompt=prompt or None, resolution=resolution,
                no_distill=no_distill, audio_gain_db=audio_gain_db or None,
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
        yield None, "\n".join(log_lines[-MAX_LOG_LINES:])

    if "error" in box:
        yield None, "\n".join(log_lines) + f"\n\nFAILED: {box['error']}"
        return

    local_path, _job_id, elapsed_s = box["result"]
    summary = (f"\n\nThis render: {elapsed_s / 60:.1f} min. "
               f"Session so far: ~${session.session_cost_so_far():.2f} "
               f"({session.gpu_id} @ ${session.gpu_price}/hr).")
    yield local_path, "\n".join(log_lines) + summary


def handle_stop():
    if session.pod_id is None:
        return "No active pod."
    session.terminate()
    return "Pod stopped."


def pod_status_text():
    if session.pod_id is None:
        return "**Session pod:** none"
    return (f"**Session pod:** `{session.pod_id}` running {session.gpu_id} "
            f"@ ${session.gpu_price}/hr — session so far: "
            f"~${session.session_cost_so_far():.2f}")


with gr.Blocks(title="AI Avatar Video") as demo:
    gr.Markdown(
        "# AI Avatar Video\n"
        "Upload a photo and an audio clip, optionally add a prompt, and "
        "generate a lip-synced talking-head video. The first click deploys "
        "a GPU pod (~10-16 min); later clicks in the same session reuse it "
        "and are much faster. Click **Stop pod** when done, or it "
        f"auto-terminates after {IDLE_TIMEOUT_S // 60} min idle -- see "
        "`docs/generate-video.md` for setup and cost details."
    )
    status_md = gr.Markdown(pod_status_text())
    with gr.Row():
        with gr.Column():
            image_in = gr.Image(type="filepath", label="Photo")
            audio_in = gr.Audio(type="filepath", label="Driving audio")
            prompt_in = gr.Textbox(
                label="Prompt (optional)",
                placeholder="A person speaks calmly to the camera in a steady, natural tone.")
            resolution_in = gr.Radio(["480p", "720p"], value="480p", label="Resolution")
            with gr.Accordion("Advanced", open=False):
                no_distill_in = gr.Checkbox(
                    label="Disable distilled sampler (--no-distill)",
                    value=False,
                    info="Full 50-step sampler -- the only way the prompt/guidance "
                         "actually affects the result. ~6x longer render.")
                audio_gain_in = gr.Number(
                    label="Audio gain (dB)", value=0,
                    info="Negative = quieter. Quieter audio tends to reduce "
                         "exaggerated mouth articulation. 0 = unchanged.")
            with gr.Row():
                submit_btn = gr.Button("Generate", variant="primary")
                stop_btn = gr.Button("Stop pod")
        with gr.Column():
            video_out = gr.Video(label="Result")
            status_out = gr.Textbox(label="Status / progress", lines=18, interactive=False)

    submit_btn.click(
        handle_submit,
        inputs=[image_in, audio_in, prompt_in, resolution_in, no_distill_in, audio_gain_in],
        outputs=[video_out, status_out],
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
