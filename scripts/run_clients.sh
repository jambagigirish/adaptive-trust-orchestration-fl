#!/usr/bin/env bash
set -euo pipefail
N="${1:-3}"
cd "$(dirname "$0")/../client"
if [[ ! -f .venv/bin/activate ]]; then python3 -m venv .venv; fi
source .venv/bin/activate
pip install -r requirements.txt
for i in $(seq -w 1 "$N"); do
  CID="edge-$i"
  echo "[spawn] $CID"
  (python edge_client.py --client-id "$CID" --server http://localhost:8080 &)
done
