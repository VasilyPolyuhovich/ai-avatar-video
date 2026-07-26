#!/usr/bin/env bash
set -euo pipefail

WORKSPACE=${WORKSPACE:-/workspace}
mkdir -p "$WORKSPACE"/models/diffusion_models "$WORKSPACE"/models/text_encoders \
         "$WORKSPACE"/models/vae "$WORKSPACE"/models/audio_encoders \
         "$WORKSPACE"/models/loras "$WORKSPACE"/output \
         "$WORKSPACE"/input "$WORKSPACE"/logs

# Weights/outputs live on the network volume, never baked into the image.
rm -rf /opt/ComfyUI/models /opt/ComfyUI/output /opt/ComfyUI/input
ln -s "$WORKSPACE/models" /opt/ComfyUI/models
ln -s "$WORKSPACE/output" /opt/ComfyUI/output
ln -s "$WORKSPACE/input"  /opt/ComfyUI/input

# Surface the bundled example workflow (baked into the image at
# /opt/workflows once Phase 3 produces a verified-working one, see the
# Dockerfile) in ComfyUI's own workflow browser, if present.
mkdir -p /opt/ComfyUI/user/default/workflows
if [ -f /opt/workflows/longcat_avatar_whisper_example.json ]; then
  ln -sf /opt/workflows/longcat_avatar_whisper_example.json \
    /opt/ComfyUI/user/default/workflows/longcat_avatar_whisper_example.json
fi

# SSH access via the RunPod proxy recipe (docs/infra-notes.md). PUBLIC_KEY is
# populated by scripts/pod_up.py from the local runpodctl-video-ssh-key.pub.
if [ -n "${PUBLIC_KEY:-}" ]; then
  mkdir -p /root/.ssh
  echo "$PUBLIC_KEY" > /root/.ssh/authorized_keys
  chmod 700 /root/.ssh
  chmod 600 /root/.ssh/authorized_keys
fi
/usr/sbin/sshd

exec /opt/supervisor.sh
