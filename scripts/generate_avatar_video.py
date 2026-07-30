#!/usr/bin/env python3
"""On-demand avatar video generation: deploy a fresh LongCat pod, render,
retrieve the result, and terminate -- all in one call. Wraps scripts/pod_up.py
(GPU ranking/deploy/retry) and scripts/run_longcat_avatar.sh (the verified
--stage_1=ai2v invocation, see docs/decisions.md) so nobody has to repeat the
manual deploy -> SSH -> scp -> torchrun -> scp -> terminate loop by hand.

Usage:
    python3 scripts/generate_avatar_video.py --image face.jpg --audio voice.mp3
    python3 scripts/generate_avatar_video.py --image face.jpg --audio voice.mp3 \\
        --prompt "A person speaks warmly, gesturing with one hand." --resolution 720p
    python3 scripts/generate_avatar_video.py --image face.jpg --audio voice.mp3 --dry-run

See docs/generate-video.md for the full setup guide (credentials, SSH keys,
cost expectations) -- this docstring only covers the tool itself.

Cost safety: the pod is ALWAYS terminated in a finally block, including on
crash or Ctrl-C, and the remote render step has a hard timeout
(--timeout, default 1800s -- automatically 10800s when --no-distill is set,
since the full 50-step sampler is roughly 17x slower per segment than the
default 8-step distilled one, not just 6x as step-count alone would
suggest) so a stuck run can't bill indefinitely. If terminate() itself ever
fails, that failure is printed loudly rather than silently swallowed -- if
you ever see a "may STILL BE RUNNING AND BILLING" message, check the
RunPod console immediately. This guarantee is contingent on the LOCAL
process surviving long enough to run its finally/signal handlers -- a
SIGKILL, dead battery, or OS crash on your machine still leaves the pod
running with nothing to terminate it; scripts/check_balance.sh or the
RunPod console remain the backstop for that case.

Reliability: the render itself (render_on_pod/run_remote_detached) runs
DETACHED on the pod and is monitored via reconnecting polls, not one
long-lived SSH stream -- a transient LOCAL network blip (WiFi hiccup,
laptop sleep, VPN reconnect) used to kill the actual remote render via
SIGHUP when the one monitoring connection died (confirmed live, twice, real
GPU spend wasted). Now it just costs a few missed poll cycles, tolerated up
to 30 minutes of lost contact before giving up.

This module also exposes PodSession (used by scripts/app_gradio.py) for
keeping ONE pod alive across several renders instead of paying the ~5 min
torch.compile warmup cost on every video -- see docs/decisions.md#5-generation-ui
for why, and the class docstring below for the termination-safety contract.
The CLI above is unaffected by PodSession's existence: one pod per
invocation, always terminated, exactly as before.
"""
import argparse
import json
import re
import shlex
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from shutil import which

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))  # so `import pod_up` works regardless of caller's cwd
import pod_up  # noqa: E402

DEFAULT_IMAGE_REF = "ghcr.io/vasilypolyuhovich/ai-avatar-longcat:latest"

# Default hard timeout for the remote render step (--timeout). The
# --no-distill default is deliberately much larger than a naive "50/8 steps
# = ~6x" estimate would suggest: confirmed live 2026-07-25, a non-distilled
# step took ~38.5s vs a distilled step's ~14.5s (full classifier-free
# guidance runs TWO forward passes per step -- conditional + unconditional
# -- vs the distilled DMD LoRA's one), so the real slowdown per segment is
# closer to (50*38.5)/(8*14.5) =~ 17x, not 6x. The original 1800s default
# killed a real --no-distill render at 82% through segment 1 of 3 (of
# denoising alone) -- wasted GPU spend from an under-sized timeout, not an
# actual failure. Long audio (many segments) can still exceed even this
# larger default; pass --timeout explicitly for anything unusually long.
DEFAULT_RUN_TIMEOUT_S = 1800
DEFAULT_RUN_TIMEOUT_S_NO_DISTILL = 10800


def _resolve_run_timeout(run_timeout_s, no_distill):
    if run_timeout_s is not None:
        return run_timeout_s
    return DEFAULT_RUN_TIMEOUT_S_NO_DISTILL if no_distill else DEFAULT_RUN_TIMEOUT_S

# The exact line scripts/run_longcat_avatar.sh prints (to stderr) announcing
# the final output path -- the single source of truth for where the result
# will land, so this script doesn't need a second, driftable copy of the
# upstream --num_segments naming logic (ai2v_demo_1.mp4 vs video_continue_N.mp4).
FINAL_OUTPUT_RE = re.compile(r"^Final output will be: (\S+)")


@dataclass
class GenerationResult:
    local_path: str
    elapsed_s: float
    gpu_id: str
    gpu_price_per_hr: float
    est_cost_usd: float
    pod_id: str
    job_id: str


def _make_loggers(status_cb):
    """(log, remote_log) closures shared by the CLI wrapper and PodSession
    so the print+status_cb pattern lives in exactly one place."""
    def log(msg):
        print(f"[generate] {msg}")
        if status_cb:
            status_cb(msg)

    def remote_log(line):
        print(line, end="" if line.endswith("\n") else "\n")
        if status_cb:
            status_cb(line.rstrip("\n"))

    return log, remote_log


def validate_local_inputs(image_path, audio_path):
    image_path = Path(image_path)
    audio_path = Path(audio_path)
    if not image_path.is_file() or image_path.stat().st_size == 0:
        raise ValueError(f"Image not found or empty: {image_path}")
    if not audio_path.is_file() or audio_path.stat().st_size == 0:
        raise ValueError(f"Audio not found or empty: {audio_path}")
    # Best-effort local duration check -- purely informational, skipped
    # silently if ffprobe isn't installed locally (no new hard dependency).
    if which("ffprobe"):
        try:
            r = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "csv=p=0", str(audio_path)],
                capture_output=True, text=True, timeout=15, check=True)
            print(f"[generate] local audio duration: {float(r.stdout.strip()):.1f}s")
        except Exception:
            pass
    return image_path, audio_path


def compute_remote_paths(job_id, image_path, audio_path):
    # Everything lives on the persistent network volume under a per-job
    # directory, NOT the ephemeral /opt/LongCat-Video/outputs_avatar_single
    # default -- so even if the final scp back to the caller fails, the
    # render itself isn't lost (it can be pulled by hand later from the
    # same volume, before the next pod deploy reuses the space).
    base = f"/workspace/jobs/{job_id}"
    return {
        "job_dir": base,
        "input_dir": f"{base}/input",
        "output_dir": f"{base}/output",
        "image": f"{base}/input/{Path(image_path).name}",
        "audio": f"{base}/input/{Path(audio_path).name}",
        "run_script": f"{base}/run_longcat_avatar.sh",
    }


def build_remote_cmd(paths, prompt, resolution, num_segments, no_distill=False, end_trim_s=None):
    parts = [
        shlex.quote(paths["run_script"]),
        "--image", shlex.quote(paths["image"]),
        "--audio", shlex.quote(paths["audio"]),
        "--resolution", shlex.quote(resolution),
        "--output-dir", shlex.quote(paths["output_dir"]),
    ]
    if prompt:
        parts += ["--prompt", shlex.quote(prompt)]
    if num_segments:
        parts += ["--num-segments", shlex.quote(str(num_segments))]
    if no_distill:
        parts += ["--no-distill"]
    if end_trim_s:
        parts += ["--end-trim-s", shlex.quote(str(end_trim_s))]
    return " ".join(parts)


def run_ssh(ip, port, key_path, remote_cmd, timeout=60, text=True):
    cmd = ["ssh", "-p", str(port), *pod_up.ssh_flags(key_path),
           "-o", "ConnectTimeout=15", f"root@{ip}", remote_cmd]
    r = subprocess.run(cmd, capture_output=True, text=text, timeout=timeout)
    if r.returncode != 0:
        stderr = r.stderr if text else r.stderr.decode(errors="replace")
        raise RuntimeError(f"ssh command failed ({r.returncode}): {remote_cmd}\n{stderr}")
    return r.stdout


def scp_up(ip, port, key_path, local_path, remote_path, timeout=300):
    cmd = ["scp", "-P", str(port), *pod_up.ssh_flags(key_path), local_path, f"root@{ip}:{remote_path}"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"scp upload failed ({r.returncode}): {local_path} -> {remote_path}\n{r.stderr}")


def scp_down(ip, port, key_path, remote_path, local_path, timeout=300):
    cmd = ["scp", "-P", str(port), *pod_up.ssh_flags(key_path), f"root@{ip}:{remote_path}", local_path]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"scp download failed ({r.returncode}): {remote_path} -> {local_path}\n{r.stderr}")


def run_remote_detached(ip, port, key_path, remote_cmd, remote_job_dir, run_timeout_s, *,
                         poll_interval=10, max_disconnect_s=1800, on_line=None):
    """Launch remote_cmd DETACHED from the SSH session (setsid+nohup, stdio
    redirected to a file, backgrounded+disowned -- survives the SSH
    connection dying, unlike a foreground `ssh ... "cmd"` exec) and monitor
    it via repeated short-lived reconnecting polls instead of one fragile
    long-lived stream.

    This replaces an earlier design (one unbroken SSH stream for the whole
    render) that let a transient LOCAL network blip -- the client's WiFi,
    a laptop sleeping, a VPN reconnect, nothing to do with the pod itself --
    kill the actual remote torchrun process via SIGHUP on disconnect
    (confirmed live, twice, real GPU spend wasted both times: a foreground
    SSH exec with no nohup/setsid dies with its session). Now a blip just
    costs a few missed poll cycles (tolerated up to max_disconnect_s), not
    a dead render and a terminated pod.

    Raises TimeoutError if run_timeout_s (wall clock) elapses, RuntimeError
    if the remote command exits non-zero or reconnection fails for longer
    than max_disconnect_s. Returns the full accumulated remote log text on
    success -- render_on_pod scans THIS returned text (not the on_line
    stream, which only ever sees incremental deltas) for the "Final output
    will be: ..." line.

    Byte-accurate: polls fetch raw bytes (not text-decoded) so the local
    read offset tracked between polls exactly matches the remote file's
    real byte position, immune to a multi-byte UTF-8 boundary landing
    mid-character (render logs contain multi-byte progress-bar block
    characters) -- decoding for display only happens after the offset math
    is already done, with errors="replace" so a transient decode hiccup
    can't corrupt tracking.
    """
    log_path = f"{remote_job_dir}/render.log"
    exit_path = f"{remote_job_dir}/render.exit"
    marker = f"@@EXIT_{uuid.uuid4().hex}@@"
    launch = (
        f"rm -f {shlex.quote(exit_path)}; "
        f"setsid nohup bash -c {shlex.quote(remote_cmd + '; echo $? > ' + shlex.quote(exit_path))} "
        f"> {shlex.quote(log_path)} 2>&1 < /dev/null & disown"
    )
    run_ssh(ip, port, key_path, launch, timeout=30)

    chunks = []
    offset = 0
    deadline = time.time() + run_timeout_s
    last_success = time.time()
    while True:
        if time.time() > deadline:
            raise TimeoutError(f"remote command exceeded {run_timeout_s}s")
        poll_cmd = (
            f"tail -c +{offset + 1} {shlex.quote(log_path)} 2>/dev/null; "
            f"printf '%s' {shlex.quote(marker)}; "
            f"(test -f {shlex.quote(exit_path)} && cat {shlex.quote(exit_path)}) || echo RUNNING"
        )
        try:
            raw = run_ssh(ip, port, key_path, poll_cmd, timeout=30, text=False)
            last_success = time.time()
        except Exception as e:
            if time.time() - last_success > max_disconnect_s:
                raise RuntimeError(
                    f"lost contact with pod for over {max_disconnect_s}s: {e}")
            time.sleep(poll_interval)
            continue

        new_bytes, _, rest = raw.partition(marker.encode())
        if new_bytes:
            offset += len(new_bytes)  # exact remote byte count, no re-encoding round-trip
            chunks.append(new_bytes)
            if on_line:
                for line in new_bytes.decode(errors="replace").splitlines(keepends=True):
                    on_line(line)
        status = rest.strip().decode(errors="replace")
        if status != "RUNNING":
            exit_code = int(status) if status.lstrip("-").isdigit() else 1
            if exit_code != 0:
                raise RuntimeError(f"remote command exited with code {exit_code}")
            return b"".join(chunks).decode(errors="replace")
        time.sleep(poll_interval)


def deploy_pod(account_key, public_key, job_id, min_vram, max_price, gpu_match,
               network_volume_id, image_ref, start_timeout, log):
    log(f"ranking GPUs (>= {min_vram:g}GB, <= ${max_price:g}/hr, matching /{gpu_match}/) ...")
    ranked = pod_up.rank_gpus(account_key, min_vram, max_price, gpu_match)
    if not ranked:
        raise RuntimeError(
            f"No in-stock Secure GPU with >={min_vram:g}GB VRAM under "
            f"${max_price:g}/hr matching /{gpu_match}/ right now.")

    data_center_id = None
    if network_volume_id:
        data_center_id = pod_up.network_volume_dc(account_key, network_volume_id)

    # pod_name includes job_id (not pod_up.py's fixed default) so that on a
    # SHARED account -- this project's colleague-access model -- several
    # people's pods showing up in the same console at once are still
    # individually identifiable. No concurrency-control logic needed beyond
    # that; each run is fully independent.
    cfg = {
        "image": image_ref,
        "pod_name": f"{pod_up.POD_NAME_PREFIX}-{job_id}",
        "container_disk": 60,
        "ports": "22/tcp",  # LongCat has no HTTP server, unlike InfiniteTalk/ComfyUI
        "registry_auth_id": pod_up.DEFAULT_REGISTRY_AUTH_ID,
        "network_volume_id": network_volume_id,
        "data_center_id": data_center_id,
    }
    log(f"deploying pod {cfg['pod_name']} ...")
    pod_id, machine, gpu_id, gpu_price = pod_up.deploy_with_fallback(
        account_key, ranked, cfg, public_key, start_timeout)
    log(f"pod {pod_id} running on {gpu_id} @ ${gpu_price}/hr (host {machine})")
    return pod_id, gpu_id, gpu_price


def wait_for_ssh(account_key, pod_id, key_path, endpoint_timeout, ready_timeout, log):
    log("waiting for SSH port mapping to be published ...")
    deadline = time.time() + endpoint_timeout
    endpoint = None
    while time.time() < deadline:
        endpoint = pod_up.get_ssh_endpoint(account_key, pod_id)
        if endpoint:
            break
        time.sleep(5)
    if not endpoint:
        raise RuntimeError(f"Pod {pod_id} never published an SSH port mapping "
                            f"within {endpoint_timeout}s")
    ip, port = endpoint
    log(f"SSH endpoint {ip}:{port} -- waiting for sshd to accept connections ...")
    if not pod_up.wait_ssh_ready(ip, port, key_path, timeout=ready_timeout):
        raise RuntimeError(f"SSH never became ready at {ip}:{port} within {ready_timeout}s")
    log("SSH ready")
    return ip, port


def render_on_pod(ip, port, key_path, image_path, audio_path, *,
                   prompt=None, resolution="480p", num_segments=None,
                   output_path=None, run_timeout_s=None, job_id=None,
                   no_distill=False, audio_gain_db=None, end_trim_s=None,
                   log=print, remote_log=None):
    """Render on an already-SSH-ready pod. Raises on any failure, does not
    catch or terminate anything itself -- that's the caller's job (the CLI
    wrapper's finally-block, or PodSession.render's except-handler).

    If job_id is None, generates a fresh one -- PodSession's use case, where
    the pod's own name was already fixed at first deploy and each render on
    a reused pod still needs its own remote job directory. The CLI wrapper
    passes its own job_id through explicitly so pod_name/job-dir/output-
    filename correlation matches what it deployed with.

    no_distill: passed through to run_longcat_avatar.sh's --no-distill,
    which lets text_guidance_scale/audio_guidance_scale apply (the upstream
    script forces both to 1.0 whenever the distilled LoRA is active).
    NOT RECOMMENDED as a quality lever: confirmed live 2026-07-26 that
    guidance_scale > 1.0 produces badly distorted ("rubbery") facial
    geometry on this checkpoint, independent of step count -- the same
    distortion showed up at the full 50-step upstream defaults here AND
    when patched into the 8-step distilled sampler. See
    docs/decisions.md#2-generation-stack. Kept only as an escape hatch for
    future experimentation, not because it currently produces good output.
    Also costs roughly 17x the render time per segment (confirmed live:
    ~38.5s/step non-distilled vs ~14.5s/step distilled). run_timeout_s=None
    auto-scales to DEFAULT_RUN_TIMEOUT_S_NO_DISTILL for exactly this reason.

    audio_gain_db: if set, applies an ffmpeg volume filter to a LOCAL copy
    of the audio before upload (original file on disk is untouched).
    CONFIRMED NO-OP as of 2026-07-25: LongCat's own audio pipeline
    unconditionally normalizes the driving audio to a fixed -23 LUFS target
    before encoding, regardless of input loudness, so this value never
    reaches the model. Kept for backward compatibility, not because it
    does anything -- see docs/decisions.md#2-generation-stack. Prefer
    changing the prompt text instead (it fully drives the model's single
    conditional pass at guidance=1.0); see docs/prompt-guide.md.

    end_trim_s: passed through to run_longcat_avatar.sh's --end-trim-s.
    Chops a fixed duration off the very end of the final output, opt-in
    mitigation for a confirmed model behavior (a slight "closing" smile at
    the end of the last generated chunk). Not safe to set by default --
    the artifact's onset can overlap with genuine trailing speech, so a
    value that's safe for one clip can cut real words on another. Only
    set this after previewing the untrimmed render.
    """
    if remote_log is None:
        remote_log = log
    run_timeout_s = _resolve_run_timeout(run_timeout_s, no_distill)
    if job_id is None:
        job_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-" + uuid.uuid4().hex[:6]

    image_path, audio_path = validate_local_inputs(image_path, audio_path)
    paths = compute_remote_paths(job_id, image_path, audio_path)

    log(f"creating remote job dirs under {paths['job_dir']} ...")
    run_ssh(ip, port, key_path,
            f"mkdir -p {shlex.quote(paths['input_dir'])} {shlex.quote(paths['output_dir'])}")

    log("uploading image, audio, and run_longcat_avatar.sh ...")
    scp_up(ip, port, key_path, str(image_path), paths["image"])

    local_audio = audio_path
    gain_tmpdir = None
    if audio_gain_db:
        if not which("ffmpeg"):
            raise RuntimeError("--audio-gain-db requires ffmpeg installed locally")
        gain_tmpdir = tempfile.TemporaryDirectory()
        gained = Path(gain_tmpdir.name) / audio_path.name
        log(f"applying {audio_gain_db:+.1f}dB gain to a local copy of the audio before upload ...")
        r = subprocess.run(
            ["ffmpeg", "-y", "-i", str(audio_path), "-af", f"volume={audio_gain_db}dB", str(gained)],
            capture_output=True, text=True, timeout=60)
        if r.returncode != 0:
            gain_tmpdir.cleanup()
            raise RuntimeError(f"ffmpeg gain adjustment failed:\n{r.stderr}")
        local_audio = gained
    try:
        scp_up(ip, port, key_path, str(local_audio), paths["audio"])
    finally:
        if gain_tmpdir is not None:
            gain_tmpdir.cleanup()

    scp_up(ip, port, key_path, str(SCRIPT_DIR / "run_longcat_avatar.sh"), paths["run_script"])
    run_ssh(ip, port, key_path, f"chmod +x {shlex.quote(paths['run_script'])}")

    remote_cmd = build_remote_cmd(paths, prompt, resolution, num_segments,
                                   no_distill=no_distill, end_trim_s=end_trim_s)
    if no_distill:
        log(f"--no-distill: running the full 50-step sampler -- roughly 17x the "
            f"usual per-segment render time (timeout set to {run_timeout_s}s).")
    log("starting generation -- this takes roughly 10-16 minutes on a fresh pod "
        "(much less if the pod's torch.compile cache is already warm, much more "
        "with --no-distill), streaming remote progress below:")
    log_text = run_remote_detached(ip, port, key_path, remote_cmd, paths["job_dir"],
                                    run_timeout_s, on_line=remote_log)
    output_filename = None
    for line in log_text.splitlines():
        m = FINAL_OUTPUT_RE.match(line.strip())
        if m:
            output_filename = m.group(1)

    if not output_filename:
        raise RuntimeError(
            "run_longcat_avatar.sh finished but never printed "
            "'Final output will be: ...' -- can't locate the result")

    local_out = Path(output_path) if output_path else Path("outputs") / f"{job_id}.mp4"
    local_out.parent.mkdir(parents=True, exist_ok=True)
    log(f"downloading result to {local_out} ...")
    scp_down(ip, port, key_path, output_filename, str(local_out))

    return str(local_out), job_id


def _safe_terminate(account_key, pod_id, log):
    """A bare `finally: pod_up.terminate(...)` that itself raises would
    silently replace whatever original exception was propagating -- hiding
    the real failure AND leaving the pod up with no visible signal. Retry
    once, and if it still fails, shout rather than swallow. Stateless --
    safe to call from anywhere (CLI's finally, PodSession's several call
    sites) without any lock/instance-state involvement."""
    for attempt in (1, 2):
        try:
            pod_up.terminate(account_key, pod_id)
            log(f"pod {pod_id} terminated")
            return
        except Exception as e:
            log(f"CRITICAL: terminate attempt {attempt} failed for pod {pod_id}: {e}")
            time.sleep(5)
    log(f"CRITICAL: pod {pod_id} may STILL BE RUNNING AND BILLING -- "
        f"terminate manually now (RunPod console or scripts/check_balance.sh)")


class PodSession:
    """Keeps one LongCat pod alive across several render() calls instead of
    the CLI's deploy-render-terminate-per-call model -- for scripts/app_gradio.py,
    where redeploying for every "Generate" click means repaying the ~5 min
    one-time torch.compile warmup on every single video. See
    docs/decisions.md#5-generation-ui for the tradeoff this accepts.

    Termination safety contract (four triggers, all end up in
    _terminate_locked): an explicit terminate() call (e.g. a Stop button),
    any exception raised out of render(), an idle watchdog (no render
    activity for idle_timeout_s), and whatever the caller wires up for
    process-exit (see scripts/app_gradio.py's atexit/SIGTERM/SIGHUP
    handlers -- NOT this class's responsibility).

    Locking invariant, load-bearing, do not violate when touching this
    class: self._lock is an RLock. _terminate_locked() assumes the lock is
    ALREADY held and must never acquire it itself (the watchdog calls it
    directly from inside its own `with self._lock:` block -- routing that
    through the public terminate() would self-deadlock). ensure_ready()
    does NOT hold the lock across the actual deploy_pod()/wait_for_ssh()
    network calls (can take minutes) -- only for the quick reuse-check and
    for committing results afterward -- so Stop/signals/atexit stay
    responsive during a deploy instead of hanging for its full duration.
    render_on_pod() itself must NEVER be called while self._lock is held.
    """

    def __init__(self, *, min_vram=pod_up.DEFAULT_MIN_VRAM,
                 max_price=pod_up.DEFAULT_MAX_PRICE,
                 gpu_match=pod_up.DEFAULT_GPU_MATCH,
                 network_volume_id=pod_up.DEFAULT_NETWORK_VOLUME_ID,
                 image_ref=DEFAULT_IMAGE_REF, start_timeout=600,
                 ssh_endpoint_timeout=120, ssh_ready_timeout=180,
                 idle_timeout_s=900):
        self.account_key = pod_up.load_account_key()
        self.public_key = pod_up.load_public_key()
        self.key_path = pod_up.private_key_path()
        self.min_vram = min_vram
        self.max_price = max_price
        self.gpu_match = gpu_match
        self.network_volume_id = network_volume_id
        self.image_ref = image_ref
        self.start_timeout = start_timeout
        self.ssh_endpoint_timeout = ssh_endpoint_timeout
        self.ssh_ready_timeout = ssh_ready_timeout
        self.idle_timeout_s = idle_timeout_s

        self._lock = threading.RLock()
        self.pod_id = self.ip = self.port = self.gpu_id = self.gpu_price = None
        # deploy_pod()'s pod_name -- distinct from render_on_pod's per-render
        # job_id, fixed once at first deploy and reused for the pod's whole life.
        self._session_id = None
        self._session_started_at = None
        self._busy = False
        self._last_activity = time.monotonic()
        self._watchdog_thread = None
        self._watchdog_stop = threading.Event()

    def ensure_ready(self, status_cb=None, pre_deploy_check=None):
        log, _ = _make_loggers(status_cb)
        with self._lock:
            if self.pod_id is not None:
                self._last_activity = time.monotonic()
                return self.ip, self.port, self.key_path

        # NOT held under self._lock below: deploy_pod()+wait_for_ssh() can
        # take several minutes, and holding the lock that long would make
        # Stop/atexit/signal-handler termination attempts hang for the same
        # duration. self._busy (set by render() before calling this) is
        # what prevents a second concurrent deploy attempt, not this lock.

        # pre_deploy_check: an optional caller-supplied hook, called here
        # (outside the lock, immediately before the real deploy call) so it
        # can raise to abort -- e.g. app_gradio.py's foreign-pod check,
        # keeping "is another colleague's pod already up on the shared
        # account" policy out of PodSession itself (pod-lifecycle-only).
        # This is the real, race-resistant check; a page-load-time check
        # alone would leave a much longer window open. A residual gap still
        # exists between this call and the deploy call actually landing --
        # accepted, not solved, same as this project's other documented
        # SIGKILL/crash gaps (see docs/decisions.md).
        if pre_deploy_check is not None:
            pre_deploy_check()

        if self._session_id is None:
            self._session_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-" + uuid.uuid4().hex[:6]
        pod_id, gpu_id, gpu_price = deploy_pod(
            self.account_key, self.public_key, self._session_id,
            self.min_vram, self.max_price, self.gpu_match,
            self.network_volume_id, self.image_ref, self.start_timeout, log)
        # self.pod_id/gpu_id/gpu_price are set HERE, as soon as the pod
        # exists -- not after wait_for_ssh() below (which can take several
        # more minutes). A foreign-pod check (app_gradio.py's
        # _find_foreign_pod / pre_deploy_check) reads session.pod_id to
        # exclude "this session's own pod" -- leaving it None during the
        # whole SSH-boot window used to make this session's own
        # just-deployed pod look "foreign" to itself (and to periodic
        # polls) for that entire window. Safe to set early: ensure_ready()
        # has exactly one caller (render(), itself serialized by
        # self._busy), so no concurrent second call can observe pod_id set
        # while ip/port are still None.
        with self._lock:
            self.pod_id, self.gpu_id, self.gpu_price = pod_id, gpu_id, gpu_price
            self._session_started_at = time.monotonic()  # billing starts at pod creation, not SSH-ready
        try:
            ip, port = wait_for_ssh(self.account_key, pod_id, self.key_path,
                                     self.ssh_endpoint_timeout, self.ssh_ready_timeout, log)
        except Exception:
            with self._lock:
                self.pod_id = self.gpu_id = self.gpu_price = self._session_started_at = None
            _safe_terminate(self.account_key, pod_id, log)
            raise

        with self._lock:
            self.ip, self.port = ip, port
            self._last_activity = time.monotonic()
            if self._watchdog_thread is None or not self._watchdog_thread.is_alive():
                self._watchdog_stop.clear()
                self._watchdog_thread = threading.Thread(target=self._watchdog_loop, daemon=True)
                self._watchdog_thread.start()
        return self.ip, self.port, self.key_path

    def render(self, image_path, audio_path, *, prompt=None, resolution="480p",
               num_segments=None, output_path=None, run_timeout_s=None,
               no_distill=False, audio_gain_db=None, end_trim_s=None,
               pre_deploy_check=None, status_cb=None):
        log, remote_log = _make_loggers(status_cb)
        with self._lock:
            if self._busy:
                raise RuntimeError("a render is already in progress on this session")
            self._busy = True
        try:
            # ensure_ready()'s RETURN VALUE is used below, never self.ip/self.port
            # re-read afterward -- avoids a TOCTOU window where a concurrent
            # terminate() could null those fields between the two calls.
            # If pre_deploy_check raises (no pod deployed yet on this call),
            # that propagates straight past the inner try/except below --
            # deliberately: there's no pod of ours to terminate in that case.
            ip, port, key_path = self.ensure_ready(status_cb, pre_deploy_check=pre_deploy_check)
            try:
                local_path, job_id = render_on_pod(
                    ip, port, key_path, image_path, audio_path,
                    prompt=prompt, resolution=resolution, num_segments=num_segments,
                    output_path=output_path, run_timeout_s=run_timeout_s,
                    no_distill=no_distill, audio_gain_db=audio_gain_db, end_trim_s=end_trim_s,
                    log=log, remote_log=remote_log)  # job_id=None -> fresh, by design
            except Exception as e:
                log(f"render failed ({e}) -- terminating session pod so the "
                    f"next attempt starts clean")
                self.terminate(status_cb)
                raise
        finally:
            self._busy = False
            self._last_activity = time.monotonic()
        return local_path, job_id

    def session_cost_so_far(self):
        """Total spend since the session pod first came up, INCLUDING idle
        gaps between renders -- unlike a single render's own elapsed time,
        this is the number that actually matters for shared-account cost
        awareness."""
        if self._session_started_at is None or self.gpu_price is None:
            return 0.0
        return self.gpu_price * (time.monotonic() - self._session_started_at) / 3600

    def snapshot_cost_info(self):
        """Atomically read (gpu_id, gpu_price, session_cost_so_far()) under
        self._lock -- for callers building a display string right after
        render() returns, to avoid racing a concurrent terminate() (e.g. a
        Stop-pod click) that clears gpu_id/gpu_price mid-read."""
        with self._lock:
            return self.gpu_id, self.gpu_price, self.session_cost_so_far()

    def terminate(self, status_cb=None):
        with self._lock:
            pod_id, account_key = self._terminate_locked()
        if pod_id is not None:
            log, _ = _make_loggers(status_cb)
            _safe_terminate(account_key, pod_id, log)  # billing-critical --
            # deliberately OUTSIDE self._lock, see _terminate_locked's docstring.

    def _terminate_locked(self):
        """Caller must already hold self._lock; must never acquire it
        itself (the watchdog calls this directly from inside its own lock).
        Only clears local state and stops the watchdog thread -- does NOT
        call _safe_terminate()/status_cb itself. That's left to the caller,
        to run AFTER releasing self._lock: _safe_terminate is stateless (see
        its own docstring), and its log() calls invoke status_cb, which in
        the web UI (app_gradio.py) acquires RenderState._lock -- doing that
        while still holding self._lock would nest the two locks, exactly
        the hazard RenderState's own docstring says never happens. Returns
        (pod_id, account_key) to terminate, or (None, None) if no pod was up."""
        pod_id, account_key = self.pod_id, self.account_key
        if self._watchdog_thread is not None:
            self._watchdog_stop.set()
            # A thread can't join itself (RuntimeError) -- the watchdog's
            # own idle-fire path calls _terminate_locked from inside
            # _watchdog_loop, so this guard is required, not defensive fluff.
            if threading.current_thread() is not self._watchdog_thread:
                self._watchdog_thread.join(timeout=5)
            self._watchdog_thread = None
        self.pod_id = self.ip = self.port = self.gpu_id = self.gpu_price = None
        return pod_id, account_key

    def _watchdog_loop(self):
        while not self._watchdog_stop.wait(30):
            with self._lock:
                if self._busy or self.pod_id is None:
                    continue
                if time.monotonic() - self._last_activity <= self.idle_timeout_s:
                    continue
                print(f"[generate] idle for over {self.idle_timeout_s}s -- "
                      f"auto-terminating session pod {self.pod_id}")
                pod_id, account_key = self._terminate_locked()
            if pod_id is not None:
                _safe_terminate(account_key, pod_id, _make_loggers(None)[0])
            return


def generate_avatar_video(
    image_path, audio_path, *,
    prompt=None, resolution="480p", num_segments=None, output_path=None,
    no_distill=False, audio_gain_db=None, end_trim_s=None,
    min_vram=pod_up.DEFAULT_MIN_VRAM, max_price=pod_up.DEFAULT_MAX_PRICE,
    gpu_match=pod_up.DEFAULT_GPU_MATCH,
    network_volume_id=pod_up.DEFAULT_NETWORK_VOLUME_ID,
    image_ref=DEFAULT_IMAGE_REF,
    start_timeout=600, ssh_endpoint_timeout=120, ssh_ready_timeout=180,
    run_timeout_s=None, dry_run=False, pre_deploy_check=None, status_cb=None,
):
    """Deploy a fresh LongCat pod, render image+audio+prompt into a video,
    retrieve it, and terminate the pod -- always, even on error/Ctrl-C.
    Returns a GenerationResult, or None if dry_run=True. One pod per call --
    see PodSession above if you want to reuse a pod across several renders.

    pre_deploy_check: optional zero-arg callable, called right before the
    real deploy_pod() call below (mirrors PodSession.ensure_ready()'s hook
    of the same name) -- e.g. pod_up.make_foreign_pod_check(), so this
    standalone one-shot CLI path also gets the foreign-pod safety check,
    not just the PodSession-based web UI.
    """
    log, remote_log = _make_loggers(status_cb)

    t0 = time.monotonic()
    image_path, audio_path = validate_local_inputs(image_path, audio_path)

    job_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-" + uuid.uuid4().hex[:6]
    paths = compute_remote_paths(job_id, image_path, audio_path)

    account_key = pod_up.load_account_key()
    public_key = pod_up.load_public_key()
    key_path = pod_up.private_key_path()

    if dry_run:
        log(f"job_id={job_id}")
        ranked = pod_up.rank_gpus(account_key, min_vram, max_price, gpu_match)
        log(f"{len(ranked)} candidate GPU(s):")
        for g in ranked:
            log(f"  {g['id']:<42} {g['vram']:>4}G  ${g['price']:<6} stock={g['stock']}")
        log(f"would run: {build_remote_cmd(paths, prompt, resolution, num_segments, no_distill=no_distill, end_trim_s=end_trim_s)}")
        return None

    if pre_deploy_check is not None:
        pre_deploy_check()

    pod_id, gpu_id, gpu_price = deploy_pod(
        account_key, public_key, job_id, min_vram, max_price, gpu_match,
        network_volume_id, image_ref, start_timeout, log)

    try:
        ip, port = wait_for_ssh(account_key, pod_id, key_path,
                                 ssh_endpoint_timeout, ssh_ready_timeout, log)

        local_path, returned_job_id = render_on_pod(
            ip, port, key_path, image_path, audio_path,
            prompt=prompt, resolution=resolution, num_segments=num_segments,
            output_path=output_path, run_timeout_s=run_timeout_s, job_id=job_id,
            no_distill=no_distill, audio_gain_db=audio_gain_db, end_trim_s=end_trim_s,
            log=log, remote_log=remote_log)
        assert returned_job_id == job_id  # passed explicitly above, should echo back unchanged

        elapsed_s = time.monotonic() - t0
        est_cost_usd = gpu_price * elapsed_s / 3600
        log(f"done in {elapsed_s / 60:.1f} min, ~${est_cost_usd:.2f} "
            f"({gpu_id} @ ${gpu_price}/hr)")
        return GenerationResult(
            local_path=local_path, elapsed_s=elapsed_s, gpu_id=gpu_id,
            gpu_price_per_hr=gpu_price, est_cost_usd=est_cost_usd,
            pod_id=pod_id, job_id=job_id)
    finally:
        _safe_terminate(account_key, pod_id, log)


def _cli():
    p = argparse.ArgumentParser(
        description="Generate an avatar video via an on-demand RunPod LongCat pod.")
    p.add_argument("--image", required=True)
    p.add_argument("--audio", required=True)
    p.add_argument("--prompt")
    p.add_argument("--resolution", choices=["480p", "720p"], default="480p")
    p.add_argument("--num-segments", type=int)
    p.add_argument("--output")
    p.add_argument("--no-distill", action="store_true",
                    help="Disable the 8-step distilled sampler, letting guidance "
                         "scale apply. NOT RECOMMENDED: confirmed to produce "
                         "distorted facial geometry on this checkpoint, and costs "
                         "~17x longer render per segment. Kept as an escape hatch, "
                         "not a working quality lever -- see docs/decisions.md.")
    p.add_argument("--audio-gain-db", type=float,
                    help="Apply an ffmpeg volume filter (dB) to a local copy of "
                         "the audio before upload. CONFIRMED NO-OP: LongCat "
                         "normalizes loudness unconditionally before encoding, so "
                         "this never reaches the model. Kept for backward "
                         "compatibility only -- see docs/prompt-guide.md instead.")
    p.add_argument("--end-trim-s", type=float,
                    help="Trim this many seconds off the very end of the final "
                         "output -- opt-in mitigation for a slight 'closing smile' "
                         "the model tends toward at the end of the last generated "
                         "chunk. Only set this after previewing the untrimmed "
                         "render: the artifact's onset can overlap with genuine "
                         "trailing speech, so no value is safe for every clip.")
    p.add_argument("--max-price", type=float, default=pod_up.DEFAULT_MAX_PRICE)
    p.add_argument("--min-vram", type=float, default=pod_up.DEFAULT_MIN_VRAM)
    p.add_argument("--gpu-match", default=pod_up.DEFAULT_GPU_MATCH)
    p.add_argument("--timeout", type=int, default=None,
                    help="Hard timeout in seconds for the remote render step. "
                         "Default: 1800, or 10800 automatically with --no-distill "
                         "(the full sampler is ~17x slower per segment, not the "
                         "~6x step-count ratio alone would suggest).")
    p.add_argument("--dry-run", action="store_true",
                    help="Rank GPUs and print the command that would run -- no deploy, no spend")
    p.add_argument("--json", action="store_true",
                    help="Print the result as one JSON line (for scripting)")
    p.add_argument("--override-foreign-pod", action="store_true",
                    help="Deploy even if another ai-avatar-video-* pod is already active "
                         "on the shared account (e.g. a colleague's, or your own other "
                         "process). Off by default -- without it, a foreign pod aborts "
                         "the run before spending anything.")
    args = p.parse_args()

    account_key = pod_up.load_account_key()
    pre_deploy_check = pod_up.make_foreign_pod_check(account_key, override=args.override_foreign_pod)

    try:
        result = generate_avatar_video(
            args.image, args.audio,
            prompt=args.prompt, resolution=args.resolution, num_segments=args.num_segments,
            output_path=args.output, no_distill=args.no_distill, audio_gain_db=args.audio_gain_db,
            end_trim_s=args.end_trim_s,
            max_price=args.max_price, min_vram=args.min_vram,
            gpu_match=args.gpu_match, run_timeout_s=args.timeout, dry_run=args.dry_run,
            pre_deploy_check=pre_deploy_check,
        )
    except Exception as e:
        print(f"[generate] FAILED: {e}", file=sys.stderr)
        sys.exit(1)

    if args.dry_run:
        return
    assert result is not None  # only None when dry_run=True, handled above
    if args.json:
        print(json.dumps(asdict(result)))
    else:
        print(f"[generate] video ready: {result.local_path}")


if __name__ == "__main__":
    _cli()
