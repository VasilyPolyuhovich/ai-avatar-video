#!/usr/bin/env bash
set -uo pipefail
# Restart ComfyUI if it dies. Check the PROCESS, not the port: the port is
# unbound for 1-2 min while models load, so a port check would double-launch
# ComfyUI and OOM the GPU (same trap as the vLLM watchdog in the sibling skill).
cd /opt/ComfyUI
LOG=/workspace/logs/comfyui.log
mkdir -p /workspace/logs

while true; do
  if ! pgrep -f "python3 main.py" > /dev/null; then
    echo "$(date -u +%FT%TZ) starting ComfyUI" >> "$LOG"
    nohup python3 main.py --listen 0.0.0.0 --port 8188 \
      --output-directory /workspace/output --input-directory /workspace/input \
      >> "$LOG" 2>&1 &
  fi
  sleep 15
done
