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
(--timeout, default 1800s) so a stuck run can't bill indefinitely. If
terminate() itself ever fails, that failure is printed loudly rather than
silently swallowed -- if you ever see a "may STILL BE RUNNING AND BILLING"
message, check the RunPod console immediately.

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


def build_remote_cmd(paths, prompt, resolution, num_segments, no_distill=False):
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
    return " ".join(parts)


def run_ssh(ip, port, key_path, remote_cmd, timeout=60):
    cmd = ["ssh", "-p", str(port), *pod_up.ssh_flags(key_path),
           "-o", "ConnectTimeout=15", f"root@{ip}", remote_cmd]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"ssh command failed ({r.returncode}): {remote_cmd}\n{r.stderr}")
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


def run_ssh_streaming(ip, port, key_path, remote_cmd, timeout_s, on_line=None):
    """Generator yielding each combined stdout+stderr line as it arrives.
    No -tt/PTY -- that was only ever needed for RunPod's ssh.runpod.io
    proxy; direct root@ip:port SSH (confirmed working all session) doesn't
    need it, and a PTY makes stdout/stderr merging behave worse, not better.
    """
    cmd = ["ssh", "-p", str(port), *pod_up.ssh_flags(key_path),
           "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=6",
           f"root@{ip}", remote_cmd]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, bufsize=1)
    timed_out = threading.Event()

    def _on_timeout():
        timed_out.set()
        proc.kill()

    timer = threading.Timer(timeout_s, _on_timeout)
    timer.start()
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            if on_line:
                on_line(line)
            yield line
    finally:
        rc = proc.wait()
        timer.cancel()
    if timed_out.is_set():
        raise TimeoutError(f"remote command exceeded {timeout_s}s -- killed")
    if rc != 0:
        raise subprocess.CalledProcessError(rc, remote_cmd)


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
        "pod_name": f"ai-avatar-video-{job_id}",
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
                   output_path=None, run_timeout_s=1800, job_id=None,
                   no_distill=False, audio_gain_db=None,
                   log=print, remote_log=None):
    """Render on an already-SSH-ready pod. Raises on any failure, does not
    catch or terminate anything itself -- that's the caller's job (the CLI
    wrapper's finally-block, or PodSession.render's except-handler).

    If job_id is None, generates a fresh one -- PodSession's use case, where
    the pod's own name was already fixed at first deploy and each render on
    a reused pod still needs its own remote job directory. The CLI wrapper
    passes its own job_id through explicitly so pod_name/job-dir/output-
    filename correlation matches what it deployed with.

    no_distill: passed through to run_longcat_avatar.sh's --no-distill --
    the ONLY way text_guidance_scale/audio_guidance_scale actually affect
    the output (the upstream script forces both to 1.0 whenever the
    distilled LoRA is active, unconditionally). Costs ~6x the render time.

    audio_gain_db: if set, applies an ffmpeg volume filter to a LOCAL copy
    of the audio before upload (original file on disk is untouched) --
    quieter/flatter input audio empirically produces less exaggerated
    mouth articulation, since the model's motion signal correlates with
    the driving audio's energy. Negative values attenuate (e.g. -6).
    """
    if remote_log is None:
        remote_log = log
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

    remote_cmd = build_remote_cmd(paths, prompt, resolution, num_segments, no_distill=no_distill)
    if no_distill:
        log("--no-distill: running the full 50-step sampler -- expect roughly "
            "6x the usual render time.")
    log("starting generation -- this takes roughly 10-16 minutes on a fresh pod "
        "(much less if the pod's torch.compile cache is already warm), "
        "streaming remote progress below:")
    output_filename = None
    for line in run_ssh_streaming(ip, port, key_path, remote_cmd, run_timeout_s, on_line=remote_log):
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

    def ensure_ready(self, status_cb=None):
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
        if self._session_id is None:
            self._session_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-" + uuid.uuid4().hex[:6]
        pod_id, gpu_id, gpu_price = deploy_pod(
            self.account_key, self.public_key, self._session_id,
            self.min_vram, self.max_price, self.gpu_match,
            self.network_volume_id, self.image_ref, self.start_timeout, log)
        try:
            ip, port = wait_for_ssh(self.account_key, pod_id, self.key_path,
                                     self.ssh_endpoint_timeout, self.ssh_ready_timeout, log)
        except Exception:
            # Instance state was never touched -- nothing to clean up but
            # the pod itself. Stateless call, no lock needed.
            _safe_terminate(self.account_key, pod_id, log)
            raise

        with self._lock:
            self.pod_id, self.ip, self.port = pod_id, ip, port
            self.gpu_id, self.gpu_price = gpu_id, gpu_price
            self._session_started_at = time.monotonic()
            self._last_activity = time.monotonic()
            if self._watchdog_thread is None or not self._watchdog_thread.is_alive():
                self._watchdog_stop.clear()
                self._watchdog_thread = threading.Thread(target=self._watchdog_loop, daemon=True)
                self._watchdog_thread.start()
        return self.ip, self.port, self.key_path

    def render(self, image_path, audio_path, *, prompt=None, resolution="480p",
               num_segments=None, output_path=None, run_timeout_s=1800,
               no_distill=False, audio_gain_db=None, status_cb=None):
        log, remote_log = _make_loggers(status_cb)
        with self._lock:
            if self._busy:
                raise RuntimeError("a render is already in progress on this session")
            self._busy = True
        try:
            # ensure_ready()'s RETURN VALUE is used below, never self.ip/self.port
            # re-read afterward -- avoids a TOCTOU window where a concurrent
            # terminate() could null those fields between the two calls.
            ip, port, key_path = self.ensure_ready(status_cb)
            try:
                local_path, job_id = render_on_pod(
                    ip, port, key_path, image_path, audio_path,
                    prompt=prompt, resolution=resolution, num_segments=num_segments,
                    output_path=output_path, run_timeout_s=run_timeout_s,
                    no_distill=no_distill, audio_gain_db=audio_gain_db,
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

    def terminate(self, status_cb=None):
        with self._lock:
            self._terminate_locked(status_cb)

    def _terminate_locked(self, status_cb=None):
        """Caller must already hold self._lock; must never acquire it
        itself (the watchdog calls this directly from inside its own lock)."""
        log, _ = _make_loggers(status_cb)
        if self.pod_id is not None:
            _safe_terminate(self.account_key, self.pod_id, log)  # billing-critical: first
        if self._watchdog_thread is not None:
            self._watchdog_stop.set()
            # A thread can't join itself (RuntimeError) -- the watchdog's
            # own idle-fire path calls _terminate_locked from inside
            # _watchdog_loop, so this guard is required, not defensive fluff.
            if threading.current_thread() is not self._watchdog_thread:
                self._watchdog_thread.join(timeout=5)
            self._watchdog_thread = None
        self.pod_id = self.ip = self.port = self.gpu_id = self.gpu_price = None

    def _watchdog_loop(self):
        while not self._watchdog_stop.wait(30):
            with self._lock:
                if self._busy or self.pod_id is None:
                    continue
                if time.monotonic() - self._last_activity <= self.idle_timeout_s:
                    continue
                log, _ = _make_loggers(None)  # no request's status_cb is live here
                log(f"idle for over {self.idle_timeout_s}s -- auto-terminating "
                    f"session pod {self.pod_id}")
                self._terminate_locked()
            return


def generate_avatar_video(
    image_path, audio_path, *,
    prompt=None, resolution="480p", num_segments=None, output_path=None,
    no_distill=False, audio_gain_db=None,
    min_vram=pod_up.DEFAULT_MIN_VRAM, max_price=pod_up.DEFAULT_MAX_PRICE,
    gpu_match=pod_up.DEFAULT_GPU_MATCH,
    network_volume_id=pod_up.DEFAULT_NETWORK_VOLUME_ID,
    image_ref=DEFAULT_IMAGE_REF,
    start_timeout=600, ssh_endpoint_timeout=120, ssh_ready_timeout=180,
    run_timeout_s=1800, dry_run=False, status_cb=None,
):
    """Deploy a fresh LongCat pod, render image+audio+prompt into a video,
    retrieve it, and terminate the pod -- always, even on error/Ctrl-C.
    Returns a GenerationResult, or None if dry_run=True. One pod per call --
    see PodSession above if you want to reuse a pod across several renders.
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
        log(f"would run: {build_remote_cmd(paths, prompt, resolution, num_segments, no_distill=no_distill)}")
        return None

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
            no_distill=no_distill, audio_gain_db=audio_gain_db,
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
                    help="Disable the 8-step distilled sampler -- the only way "
                         "--prompt/guidance scale actually affect the output. "
                         "~6x longer render.")
    p.add_argument("--audio-gain-db", type=float,
                    help="Apply an ffmpeg volume filter (dB, e.g. -6) to a local "
                         "copy of the audio before upload -- quieter audio tends "
                         "to reduce exaggerated mouth articulation.")
    p.add_argument("--max-price", type=float, default=pod_up.DEFAULT_MAX_PRICE)
    p.add_argument("--min-vram", type=float, default=pod_up.DEFAULT_MIN_VRAM)
    p.add_argument("--gpu-match", default=pod_up.DEFAULT_GPU_MATCH)
    p.add_argument("--timeout", type=int, default=1800,
                    help="Hard timeout in seconds for the remote render step (default 1800)")
    p.add_argument("--dry-run", action="store_true",
                    help="Rank GPUs and print the command that would run -- no deploy, no spend")
    p.add_argument("--json", action="store_true",
                    help="Print the result as one JSON line (for scripting)")
    args = p.parse_args()

    try:
        result = generate_avatar_video(
            args.image, args.audio,
            prompt=args.prompt, resolution=args.resolution, num_segments=args.num_segments,
            output_path=args.output, no_distill=args.no_distill, audio_gain_db=args.audio_gain_db,
            max_price=args.max_price, min_vram=args.min_vram,
            gpu_match=args.gpu_match, run_timeout_s=args.timeout, dry_run=args.dry_run,
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
