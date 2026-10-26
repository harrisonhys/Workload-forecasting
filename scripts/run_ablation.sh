#!/usr/bin/env bash
# Context-length ablation (SPEC §3.3). C is capped per model (fm.py) and at the first origin (run_fm.py).
set -uo pipefail
cd "$(dirname "$0")/.."
step() { echo "[$(date +%H:%M:%S)] $*"; "$@" || echo "FAILED: $*"; }
for c in 512 1024 4096 6912; do
  for m in chronos2 timesfm3 tirex; do step uv run --project envs/fm python -W ignore -m src.run_fm huawei 5 $m $c; done
  step uv run --project envs/toto python -W ignore -m src.run_fm huawei 5 toto $c
done
for c in 512 8192; do
  step uv run --project envs/fm python -W ignore -m src.run_fm azure2019 1 chronos2 $c
  step uv run --project envs/toto python -W ignore -m src.run_fm azure2019 1 toto $c
done
step uv run --project envs/fm python -W ignore -m src.run_fm azure2019 1 timesfm3 512
step uv run python -W ignore -m src.metrics huawei 5
step uv run python -W ignore -m src.metrics azure2019 1
echo "[$(date +%H:%M:%S)] ABLATION DONE"
