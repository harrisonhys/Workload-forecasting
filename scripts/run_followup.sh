#!/usr/bin/env bash
# Follow-up after run_all.sh: neural baselines re-run on CPU (first pass silently used Apple MPS),
# then metrics + simulation for every dataset. Waits for run_all.sh to finish first.
set -uo pipefail
cd "$(dirname "$0")/.."
while pgrep -f "bash scripts/run_all.sh" >/dev/null; do sleep 60; done
step() { echo "[$(date +%H:%M:%S)] $*"; "$@" || echo "FAILED: $*"; }
ALL="huawei:5 alibaba2018:5 azure2021:5 azure2021:1 azure2019:5 azure2019:1"
for x in $ALL; do
  ds=${x%:*}; res=${x#*:}
  for m in nbeats lstm; do step uv run --project envs/train python -W ignore -m src.trained "$ds" "$res" "$m"; done
done
for x in $ALL; do step uv run python -W ignore -m src.metrics ${x%:*} ${x#*:}; done
for x in alibaba2018:5 azure2021:1 azure2019:1; do step uv run python -W ignore -m src.simulate ${x%:*} ${x#*:}; done
echo "[$(date +%H:%M:%S)] FOLLOWUP DONE"
