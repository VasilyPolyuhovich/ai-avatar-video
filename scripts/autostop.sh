#!/usr/bin/env bash
# Run ON the pod (daily self-stop). Needs the RunPod ACCOUNT key, not the
# pod's injected $RUNPOD_API_KEY (restricted -- can't stop pods). The human
# writes that key file directly on the pod (never over `ssh -tt`, which
# echoes it into the transcript) -- this script only reads it.
set -euo pipefail
STOP_UTC_HOUR=${STOP_UTC_HOUR:-22}
ACCOUNT_KEY_FILE=${ACCOUNT_KEY_FILE:-/workspace/.runpod-account-key}
POD_ID=${RUNPOD_POD_ID:?RUNPOD_POD_ID not set -- expected to be auto-injected by RunPod}

if [ ! -f "$ACCOUNT_KEY_FILE" ]; then
  echo "autostop: $ACCOUNT_KEY_FILE not found -- self-stop disabled until it's written" >&2
  exit 0
fi
KEY=$(cat "$ACCOUNT_KEY_FILE")
TARGET=$(printf '%02d' "$STOP_UTC_HOUR")

echo "autostop: will stop pod $POD_ID at ${TARGET}:00 UTC daily"
while true; do
  now_hour=$(date -u +%H)
  now_min=$(date -u +%M)
  if [ "$now_hour" = "$TARGET" ] && [ "$now_min" = "00" ]; then
    echo "autostop: stopping pod $POD_ID at $(date -u)"
    curl -s https://api.runpod.io/graphql \
      -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
      -H "User-Agent: ai-avatar-autostop/1.0" \
      -d "{\"query\":\"mutation{podStop(input:{podId:\\\"$POD_ID\\\"}){id desiredStatus}}\"}"
    exit 0
  fi
  sleep 30
done
