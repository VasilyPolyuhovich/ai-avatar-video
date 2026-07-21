#!/usr/bin/env bash
set -euo pipefail

WORKSPACE=${WORKSPACE:-/workspace}
mkdir -p "$WORKSPACE/longcat-weights" "$WORKSPACE/longcat-outputs"
rm -rf /opt/LongCat-Video/weights /opt/LongCat-Video/outputs
ln -s "$WORKSPACE/longcat-weights" /opt/LongCat-Video/weights
ln -s "$WORKSPACE/longcat-outputs" /opt/LongCat-Video/outputs

if [ -n "${PUBLIC_KEY:-}" ]; then
  mkdir -p /root/.ssh
  echo "$PUBLIC_KEY" > /root/.ssh/authorized_keys
  chmod 700 /root/.ssh
  chmod 600 /root/.ssh/authorized_keys
fi
/usr/sbin/sshd

# Batch-job container: no server by default. Keep PID 1 alive over SSH and
# run_demo_avatar_*.py manually (docs/ab-test-plan.md), or pass a command to
# run it non-interactively (e.g. `docker run ... torchrun run_demo_avatar_...`).
if [ "$#" -eq 0 ]; then
  exec sleep infinity
else
  exec "$@"
fi
