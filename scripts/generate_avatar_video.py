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
"""
import argparse
import json
import re
import shlex
import subprocess
import sys
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


def build_remote_cmd(paths, prompt, resolution, num_segments):
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


def _safe_terminate(account_key, pod_id, log):
    """A bare `finally: pod_up.terminate(...)` that itself raises would
    silently replace whatever original exception was propagating -- hiding
    the real failure AND leaving the pod up with no visible signal. Retry
    once, and if it still fails, shout rather than swallow."""
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


def generate_avatar_video(
    image_path, audio_path, *,
    prompt=None, resolution="480p", num_segments=None, output_path=None,
    min_vram=pod_up.DEFAULT_MIN_VRAM, max_price=pod_up.DEFAULT_MAX_PRICE,
    gpu_match=pod_up.DEFAULT_GPU_MATCH,
    network_volume_id=pod_up.DEFAULT_NETWORK_VOLUME_ID,
    image_ref=DEFAULT_IMAGE_REF,
    start_timeout=600, ssh_endpoint_timeout=120, ssh_ready_timeout=180,
    run_timeout_s=1800, dry_run=False, status_cb=None,
):
    """Deploy a fresh LongCat pod, render image+audio+prompt into a video,
    retrieve it, and terminate the pod -- always, even on error/Ctrl-C.
    Returns a GenerationResult, or None if dry_run=True.
    """
    def log(msg):
        print(f"[generate] {msg}")
        if status_cb:
            status_cb(msg)

    def remote_log(line):
        print(line, end="" if line.endswith("\n") else "\n")
        if status_cb:
            status_cb(line.rstrip("\n"))

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
        log(f"would run: {build_remote_cmd(paths, prompt, resolution, num_segments)}")
        return None

    pod_id, gpu_id, gpu_price = deploy_pod(
        account_key, public_key, job_id, min_vram, max_price, gpu_match,
        network_volume_id, image_ref, start_timeout, log)

    try:
        ip, port = wait_for_ssh(account_key, pod_id, key_path,
                                 ssh_endpoint_timeout, ssh_ready_timeout, log)

        log(f"creating remote job dirs under {paths['job_dir']} ...")
        run_ssh(ip, port, key_path,
                f"mkdir -p {shlex.quote(paths['input_dir'])} {shlex.quote(paths['output_dir'])}")

        log("uploading image, audio, and run_longcat_avatar.sh ...")
        scp_up(ip, port, key_path, str(image_path), paths["image"])
        scp_up(ip, port, key_path, str(audio_path), paths["audio"])
        scp_up(ip, port, key_path, str(SCRIPT_DIR / "run_longcat_avatar.sh"), paths["run_script"])
        run_ssh(ip, port, key_path, f"chmod +x {shlex.quote(paths['run_script'])}")

        remote_cmd = build_remote_cmd(paths, prompt, resolution, num_segments)
        log("starting generation -- this takes roughly 10-16 minutes total, "
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

        elapsed_s = time.monotonic() - t0
        est_cost_usd = gpu_price * elapsed_s / 3600
        log(f"done in {elapsed_s / 60:.1f} min, ~${est_cost_usd:.2f} "
            f"({gpu_id} @ ${gpu_price}/hr)")
        return GenerationResult(
            local_path=str(local_out), elapsed_s=elapsed_s, gpu_id=gpu_id,
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
            output_path=args.output, max_price=args.max_price, min_vram=args.min_vram,
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
