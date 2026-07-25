#!/usr/bin/env python3
"""Minimal local UI for scripts/generate_avatar_video.py: upload a photo and
audio clip, type a prompt, click Generate, watch progress, get a video back.

Runs on http://127.0.0.1:7860 -- localhost only, single machine, matches
this project's current phase (see docs/generate-video.md): each user (you or
a colleague) runs this locally on their own machine with their own copy of
the shared RunPod account key. There is no auth here and none is needed for
that reason; a public/hosted version is a deliberately separate later phase.

    pip install -r requirements.txt
    python3 scripts/app_gradio.py
"""
import queue
import sys
import threading
from pathlib import Path

import gradio as gr

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from generate_avatar_video import generate_avatar_video  # noqa: E402

MAX_LOG_LINES = 400


def handle_submit(image, audio, prompt, resolution):
    if not image or not audio:
        yield None, "Please provide both a photo and an audio clip."
        return

    q: "queue.Queue" = queue.Queue()
    box = {}

    def worker():
        try:
            box["result"] = generate_avatar_video(
                image, audio, prompt=prompt or None, resolution=resolution,
                status_cb=q.put)
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

    result = box["result"]
    summary = (f"\n\nDone in {result.elapsed_s / 60:.1f} min, "
               f"~${result.est_cost_usd:.2f} "
               f"({result.gpu_id} @ ${result.gpu_price_per_hr}/hr)")
    yield result.local_path, "\n".join(log_lines) + summary


with gr.Blocks(title="AI Avatar Video") as demo:
    gr.Markdown(
        "# AI Avatar Video\n"
        "Upload a photo and an audio clip, optionally add a prompt, and "
        "generate a lip-synced talking-head video. Each run deploys a fresh "
        "GPU pod on request (~10-16 min total) and terminates it "
        "automatically when done -- see `docs/generate-video.md` for setup "
        "and expected cost."
    )
    with gr.Row():
        with gr.Column():
            image_in = gr.Image(type="filepath", label="Photo")
            audio_in = gr.Audio(type="filepath", label="Driving audio")
            prompt_in = gr.Textbox(
                label="Prompt (optional)",
                placeholder="A person speaks calmly to the camera in a steady, natural tone.")
            resolution_in = gr.Radio(["480p", "720p"], value="480p", label="Resolution")
            submit_btn = gr.Button("Generate", variant="primary")
        with gr.Column():
            video_out = gr.Video(label="Result")
            status_out = gr.Textbox(label="Status / progress", lines=18, interactive=False)

    submit_btn.click(
        handle_submit,
        inputs=[image_in, audio_in, prompt_in, resolution_in],
        outputs=[video_out, status_out],
    )


if __name__ == "__main__":
    # demo.queue() is required: without it, a generator-returning event
    # handler's yields all buffer and the browser only sees the final
    # state -- for a ~10-16 min job that reads as a frozen/broken UI.
    demo.queue().launch(server_name="127.0.0.1")
