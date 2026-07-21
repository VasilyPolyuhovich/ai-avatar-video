#!/usr/bin/env bash
# Run LOCALLY (not on the pod). Prints balance + runway; never prints the key.
set -euo pipefail
KEY_FILE=${ACCOUNT_KEY_FILE:-~/.runpod-key}
KEY=$(tr -d '[:space:]' < "$(eval echo "$KEY_FILE")")

RESP=$(curl -s https://api.runpod.io/graphql \
  -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -H "User-Agent: ai-avatar-check/1.0" \
  -d '{"query":"query{myself{clientBalance currentSpendPerHr}}"}')

python3 -c "
import json, sys
d = json.loads('''$RESP''')['data']['myself']
bal, spend = d['clientBalance'], d['currentSpendPerHr']
print(f'balance:  \${bal:.2f}')
print(f'spend/hr: \${spend:.4f}')
if spend > 0:
    print(f'runway:   {bal/spend:.1f}h (~{bal/spend/24:.1f}d)')
else:
    print('runway:   no active spend (storage-only or no volume)')
"
