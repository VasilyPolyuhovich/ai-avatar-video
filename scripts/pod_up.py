#!/usr/bin/env python3
"""Bring an AI-Avatar-Video pod up on command: auto-pick the cheapest in-stock
Secure GPU with enough VRAM, attach the project's network volume, and deploy
from a prebuilt image (InfiniteTalk by default, LongCat via IMAGE=...).

Adapted from the runpod-pod-ops skill's bundled pod_up.py. Why it exists:
resuming a stopped Secure pod strands on "not enough free GPUs on the host
machine" whenever that one host is full, and pinning a single GPU type hits
the same wall. This queries live stock across all >=MIN_VRAM Secure GPUs,
ranks in-stock ones cheapest-first, deploys, and VERIFIES the container
actually starts (uptime>0) -- a broken host accepts the deploy but fails at
container init ("error creating device nodes"), leaving uptime at 0. On that
failure it terminates the pod, blocklists the machineId, and falls through to
the next host/candidate.

CAUTION: uptime staying at 0 has more than one cause, and this script can't
tell them apart by itself. A whole afternoon (2026-07-24) was spent chasing
"broken hosts" across three regions (EU-RO-1 -> EUR-IS-1 -> US-MD-1) -- every
single one of ~9 deploys actually failed the same way: RunPod's own pod-init
emails said `IMAGE_AUTH_ERROR: unauthorized`, because the GHCR packages
(ai-avatar-infinitetalk/-longcat) default to **private** visibility even
though the repo pushing them has `packages: write` -- GitHub's Actions token
can publish a package but can't flip its visibility, that's web-UI-only.
REGISTRY_AUTH_ID now defaults to a saved RunPod registry credential
(`saveRegistryAuth`, GHCR username must be lowercase) so this doesn't
silently recur. If uptime ever sits at 0 again, check the RunPod console's
pod-init error/email FIRST -- it names the real reason -- before assuming
host flakiness and burning time hopping regions.

Secrets: no app secret is required here (ComfyUI/LongCat have no built-in API
key). The RunPod account key is read from $ACCOUNT_KEY_FILE (default
~/.runpod-key-video) and never printed. The SSH public key is read from
$SSH_PUBKEY_FILE (default ~/.runpod/ssh/runpodctl-video-ssh-key.pub) and
passed as the pod's PUBLIC_KEY env so the images' entrypoint.sh can
authorize it.

Two RunPod accounts exist for this project (see docs/infra-notes.md): the
original account (`~/.runpod-key` / `~/.runpod/ssh/runpodctl-ssh-key.pub`,
its network volume `plk85ofiny` no longer exists -- 404, deleted) and the
funded one this project now uses by default (`~/.runpod-key-video` /
`~/.runpod/ssh/runpodctl-video-ssh-key.pub`, volume `fl7pl7z0sz`). Override
both ACCOUNT_KEY_FILE and SSH_PUBKEY_FILE together if you ever need to
target the old account -- they must point at the same account's key pair.

Usage:
    python3 scripts/pod_up.py                # deploy InfiniteTalk, print URL
    python3 scripts/pod_up.py --dry-run       # just rank GPUs by stock/price
    python3 scripts/pod_up.py --wait          # deploy then poll ComfyUI for 200
    IMAGE=ghcr.io/vasilypolyuhovich/ai-avatar-longcat:latest \\
        PORTS=22/tcp python3 scripts/pod_up.py   # deploy the LongCat image instead

Env knobs: IMAGE, MIN_VRAM (default 80), MAX_PRICE (default 2.50),
    GPU_MATCH (default "A100|H100" -- regex on the GPU type id; the LongCat
    image's flash-attn is only compiled for those two archs),
    CONTAINER_DISK_GB (default 60), PORTS (default "8188/http,22/tcp"),
    POD_NAME, NETWORK_VOLUME_ID (default fl7pl7z0sz), REGISTRY_AUTH_ID
    (default a saved GHCR credential -- see the CAUTION note above),
    ACCOUNT_KEY_FILE, SSH_PUBKEY_FILE, START_TIMEOUT (default 600s -- covers
    a cold image pull), MAX_TRIES_PER_GPU (default 2).
"""
import json
import os
import re
import subprocess
import sys
import time
import urllib.request

API = "https://api.runpod.io/graphql"
UA = "ai-avatar-pod-up/1.0"  # default python-urllib UA gets 403'd by the RunPod WAF

# Stock desirability: prefer plentiful hosts so the deploy actually lands.
STOCK_RANK = {"High": 0, "Medium": 1, "Low": 2}

# Defaults as module constants (not just buried in main()'s env(...) calls)
# so other scripts -- scripts/generate_avatar_video.py, and eventually a
# Telegram-bot phase -- can `import pod_up` and reuse the same single source
# of truth instead of re-hardcoding these strings a second time.
DEFAULT_MIN_VRAM = 80.0
DEFAULT_MAX_PRICE = 2.50
# Restricted to A100/H100: the LongCat image's flash-attn was compiled with
# TORCH_CUDA_ARCH_LIST="8.0;9.0" (those two families' compute capabilities)
# -- landing on other >=80GB hardware (e.g. an RTX PRO 6000 Blackwell) would
# have no matching kernel for it. Pass gpu_match="" to lift this.
DEFAULT_GPU_MATCH = "A100|H100"
DEFAULT_REGISTRY_AUTH_ID = "cmrzcm01x0079uy8y4v8536wo"
DEFAULT_NETWORK_VOLUME_ID = "fl7pl7z0sz"
DEFAULT_SSH_PUBKEY_FILE = "~/.runpod/ssh/runpodctl-video-ssh-key.pub"
DEFAULT_SSH_PRIVKEY_FILE = "~/.runpod/ssh/runpodctl-video-ssh-key"


def env(name, default=None):
    return os.environ.get(name, default)


def read_file(path):
    with open(os.path.expanduser(path)) as f:
        return f.read().strip()


def load_account_key():
    path = env("ACCOUNT_KEY_FILE", "~/.runpod-key-video")
    try:
        return read_file(path)
    except OSError as e:
        sys.exit(f"ERROR: cannot read RunPod account key from {path}: {e}")


def load_public_key():
    path = env("SSH_PUBKEY_FILE", DEFAULT_SSH_PUBKEY_FILE)
    try:
        return read_file(path)
    except OSError:
        print(f"[pod_up] WARNING: no SSH public key at {path} -- pod will "
              f"deploy without PUBLIC_KEY; SSH access won't be authorized.")
        return None


def private_key_path():
    """Local path to the matching private key for load_public_key()'s
    public key -- pod_up.py itself never opens an SSH connection (it only
    deploys via the GraphQL API), but callers that do (generate_avatar_video.py,
    wait_ssh_ready() below) need this same path."""
    return os.path.expanduser(env("SSH_PRIVKEY_FILE", DEFAULT_SSH_PRIVKEY_FILE) or DEFAULT_SSH_PRIVKEY_FILE)


def gql(account_key, query, variables=None):
    body = json.dumps({"query": query, "variables": variables or {}}).encode()
    req = urllib.request.Request(
        API,
        data=body,
        headers={
            "Authorization": f"Bearer {account_key}",
            "Content-Type": "application/json",
            "User-Agent": UA,
        },
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read())


def rank_gpus(account_key, min_vram, max_price, gpu_match):
    q = """query{gpuTypes{id memoryInGb secureCloud securePrice
      lowestPrice(input:{gpuCount:1,secureCloud:true}){stockStatus uninterruptablePrice}}}"""
    data = gql(account_key, q)
    gpus = (data.get("data") or {}).get("gpuTypes")
    if gpus is None:
        sys.exit(f"ERROR: gpuTypes query failed: {json.dumps(data)[:300]}")
    out = []
    for g in gpus:
        vram = g.get("memoryInGb") or 0
        price = g.get("securePrice")
        stock = (g.get("lowestPrice") or {}).get("stockStatus")
        if not g.get("secureCloud") or vram < min_vram:
            continue
        if gpu_match and not re.search(gpu_match, g["id"], re.IGNORECASE):
            continue
        if stock not in STOCK_RANK:  # None/unknown => not buyable right now
            continue
        if price is None or price > max_price:
            continue
        out.append({"id": g["id"], "vram": vram, "price": price, "stock": stock})
    out.sort(key=lambda x: (x["price"], STOCK_RANK[x["stock"]]))
    return out


def network_volume_dc(account_key, vol_id):
    data = gql(account_key, "query{myself{networkVolumes{id dataCenterId}}}")
    vols = ((data.get("data") or {}).get("myself") or {}).get("networkVolumes") or []
    for v in vols:
        if v["id"] == vol_id:
            return v["dataCenterId"]
    sys.exit(f"ERROR: network volume {vol_id} not found on this account")


def build_env_list(public_key):
    pairs = []
    if public_key:
        pairs.append(("PUBLIC_KEY", public_key))
    for k in ("STOP_UTC_HOUR",):  # pass through optional autostop knob
        v = env(k)
        if v:
            pairs.append((k, v))
    return [{"key": k, "value": v} for k, v in pairs]


def deploy(account_key, gpu_id, cfg, public_key):
    inp = {
        "cloudType": "SECURE",
        "gpuTypeId": gpu_id,
        "gpuCount": 1,
        "name": cfg["pod_name"],
        "imageName": cfg["image"],
        "containerDiskInGb": cfg["container_disk"],
        "volumeMountPath": "/workspace",
        "ports": cfg["ports"],
        "env": build_env_list(public_key),
    }
    if cfg["registry_auth_id"]:
        inp["containerRegistryAuthId"] = cfg["registry_auth_id"]
    if cfg["network_volume_id"]:
        inp["networkVolumeId"] = cfg["network_volume_id"]
        inp["dataCenterId"] = cfg["data_center_id"]  # network volumes are DC-locked
    else:
        inp["volumeInGb"] = int(env("VOLUME_GB") or "60")
    mut = ("mutation($input:PodFindAndDeployOnDemandInput!){"
           "podFindAndDeployOnDemand(input:$input){id imageName machineId}}")
    return gql(account_key, mut, {"input": inp})


def pod_status(account_key, pod_id):
    q = ("query{pod(input:{podId:%s}){desiredStatus machineId "
         "runtime{uptimeInSeconds}}}" % json.dumps(pod_id))
    p = (gql(account_key, q).get("data") or {}).get("pod") or {}
    rt = p.get("runtime") or {}
    return p.get("desiredStatus"), p.get("machineId"), (rt.get("uptimeInSeconds") or 0)


def terminate(account_key, pod_id):
    gql(account_key, "mutation{podTerminate(input:{podId:%s})}" % json.dumps(pod_id))


def wait_container_start(account_key, pod_id, machine, timeout):
    deadline = time.time() + timeout
    while time.time() < deadline:
        _, m, up = pod_status(account_key, pod_id)
        if m:
            machine = m
        if up and up > 0:
            return up, machine
        time.sleep(15)
    return 0, machine


def get_ssh_endpoint(account_key, pod_id):
    """(ip, port) for the pod's direct tcp/22 mapping, or None if
    runtime.ports isn't populated yet. Confirmed this session: deployed pods
    get a direct public IP:port TCP mapping for port 22, so plain
    `ssh -p <port> root@<ip>` works -- no need for the ssh.runpod.io proxy
    or a Connect-panel suffix (which changes on every recreate)."""
    q = ("query{pod(input:{podId:%s}){runtime{ports{ip isIpPublic "
         "publicPort privatePort type}}}}" % json.dumps(pod_id))
    p = (gql(account_key, q).get("data") or {}).get("pod") or {}
    for prt in ((p.get("runtime") or {}).get("ports")) or []:
        if prt.get("privatePort") == 22 and prt.get("type") == "tcp":
            return prt.get("ip"), prt.get("publicPort")
    return None


def ssh_flags(key_path):
    """Flags proven working all session for RunPod's pods: RunPod's proxy
    (and, empirically, some direct hosts too) only accepts legacy ssh-rsa,
    and host keys change every redeploy so checking them isn't useful here."""
    return ["-i", key_path,
            "-o", "PubkeyAcceptedAlgorithms=+ssh-rsa",
            "-o", "StrictHostKeyChecking=no",
            "-o", "BatchMode=yes"]


def wait_ssh_ready(ip, port, key_path, timeout=180, interval=5):
    """Poll with a real `ssh ... true` until it succeeds (sshd actually
    authenticating), not just a TCP connect -- container uptime>0 (what
    wait_container_start checks) does not mean sshd is accepting connections
    yet, there's a short additional gap."""
    deadline = time.time() + timeout
    cmd = ["ssh", "-p", str(port), *ssh_flags(key_path),
           "-o", "ConnectTimeout=10", f"root@{ip}", "true"]
    while time.time() < deadline:
        try:
            if subprocess.run(cmd, capture_output=True, timeout=15).returncode == 0:
                return True
        except subprocess.TimeoutExpired:
            pass
        time.sleep(interval)
    return False


def poll_ready(pod_id, port, minutes=20):
    # ComfyUI has no dedicated /health route -- treat any 200 on root as ready.
    # LongCat's image has no HTTP server at all; skip polling for it (caller
    # only invokes this for the InfiniteTalk/ComfyUI image).
    url = f"https://{pod_id}-{port}.proxy.runpod.net/"
    print(f"[pod_up] polling {url} (up to {minutes} min)")
    deadline = time.time() + minutes * 60
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=10) as r:
                if r.status == 200:
                    print("[pod_up] 200 OK -- ComfyUI is up")
                    return True
        except Exception:
            pass
        time.sleep(20)
    print("[pod_up] still not responding; check logs over SSH")
    return False


def deploy_with_fallback(account_key, ranked, cfg, public_key, start_timeout, max_tries=2):
    """Try each ranked GPU candidate; blocklist hosts that accept the deploy
    but never boot (uptime stays 0), retry the next candidate. Returns
    (pod_id, machine, gpu_id, gpu_price) for the first pod that actually
    starts. Raises RuntimeError if every candidate fails."""
    blocklist = set()
    for g in ranked:
        for attempt in range(1, max_tries + 1):
            print(f"[pod_up] trying {g['id']} @ ${g['price']}/hr "
                  f"(stock={g['stock']}, attempt {attempt}/{max_tries}) ...")
            res = deploy(account_key, g["id"], cfg, public_key)
            pod = (res.get("data") or {}).get("podFindAndDeployOnDemand")
            if not (pod and pod.get("id")):
                err = (res.get("errors") or [{}])[0].get("message", "unknown")
                print(f"[pod_up]   supply refused: {err}")
                break  # this GPU is out of stock -> next candidate
            pid = pod["id"]
            machine = pod.get("machineId")
            if machine and machine in blocklist:
                print(f"[pod_up]   landed on known-bad host {machine} again -> terminating, retry")
                terminate(account_key, pid)
                continue
            print(f"[pod_up] deployed {pid} on host {machine or '?'}; "
                  f"verifying container start (<= {start_timeout // 60} min) ...")
            up, machine = wait_container_start(account_key, pid, machine, start_timeout)
            if up > 0:
                print(f"[pod_up] CONTAINER STARTED (uptime={up}s, host {machine}) -- pod {pid}")
                return pid, machine, g["id"], g["price"]
            print(f"[pod_up]   host {machine or '?'} never started the container "
                  f"(broken host / device-node fault) -> terminating + blocklisting")
            terminate(account_key, pid)
            if machine:
                blocklist.add(machine)

    raise RuntimeError(
        "No candidate could start a container. Blocklisted hosts: "
        + (", ".join(sorted(blocklist)) or "none")
        + ". Supply may be short or hosts flaky -- retry later or raise MAX_PRICE.")


def main():
    args = set(sys.argv[1:])
    min_vram = float(env("MIN_VRAM") or str(DEFAULT_MIN_VRAM))
    max_price = float(env("MAX_PRICE") or str(DEFAULT_MAX_PRICE))
    gpu_match = env("GPU_MATCH", DEFAULT_GPU_MATCH)
    account_key = load_account_key()

    ranked = rank_gpus(account_key, min_vram, max_price, gpu_match)
    if not ranked:
        sys.exit(f"No in-stock Secure GPU with >={min_vram:g}GB VRAM under ${max_price:g}/hr "
                  f"matching /{gpu_match}/ right now.")

    print(f"Candidates (>= {min_vram:g}GB, Secure, in stock, <= ${max_price:g}/hr, matching /{gpu_match}/), best first:")
    for g in ranked:
        print(f"  {g['id']:<42} {g['vram']:>4}G  ${g['price']:<6} stock={g['stock']}")
    if "--dry-run" in args:
        return

    public_key = load_public_key()
    image = env("IMAGE", "ghcr.io/vasilypolyuhovich/ai-avatar-infinitetalk:latest")
    cfg = {
        "image": image,
        "pod_name": env("POD_NAME", "ai-avatar-video"),
        "container_disk": int(env("CONTAINER_DISK_GB") or "60"),
        "ports": env("PORTS", "8188/http,22/tcp"),
        "registry_auth_id": env("REGISTRY_AUTH_ID", DEFAULT_REGISTRY_AUTH_ID),
        "network_volume_id": env("NETWORK_VOLUME_ID", DEFAULT_NETWORK_VOLUME_ID),
        "data_center_id": None,
    }
    if cfg["network_volume_id"]:
        cfg["data_center_id"] = network_volume_dc(account_key, cfg["network_volume_id"])
        print(f"[pod_up] network volume {cfg['network_volume_id']} is in {cfg['data_center_id']} "
              f"-- deploy pinned to that DC (weights persist, no re-download)")

    start_timeout = int(env("START_TIMEOUT") or "600")
    max_tries = int(env("MAX_TRIES_PER_GPU") or "2")

    try:
        pid, machine, gpu_id, gpu_price = deploy_with_fallback(
            account_key, ranked, cfg, public_key, start_timeout, max_tries)
    except RuntimeError as e:
        sys.exit(str(e))
    print(f"[pod_up] pod {pid} running {gpu_id} @ ${gpu_price}/hr on host {machine}")

    if "8188/http" in cfg["ports"]:
        print(f"[pod_up] ComfyUI: https://{pid}-8188.proxy.runpod.net/ "
              f"(prefer an SSH tunnel over this public proxy)")
    endpoint = get_ssh_endpoint(account_key, pid)
    if endpoint:
        ip, port = endpoint
        print(f"[pod_up] SSH: ssh -p {port} -i {private_key_path()} "
              f"-o PubkeyAcceptedAlgorithms=+ssh-rsa -o StrictHostKeyChecking=no root@{ip}")
    else:
        print("[pod_up] SSH port mapping not published yet -- check the RunPod console's Connect panel")
    if "--wait" in args and "8188/http" in cfg["ports"]:
        poll_ready(pid, 8188)


if __name__ == "__main__":
    main()
