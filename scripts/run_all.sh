#!/usr/bin/env bash
# Full experiment queue, strictly sequential (parallel jobs fight over the CPU and distort timings).
# Every step skips outputs that already exist, so the script can be re-run after an interruption.
# Usage: bash scripts/run_all.sh > results/logs/run_all.log 2>&1
set -uo pipefail
cd "$(dirname "$0")/.."

FM_FM="chronos_bolt_small chronos2 timesfm25 timesfm3 tirex ttm_r2"
CHEAP="naive seasonal_naive swa3 ets croston tsb"

step() { echo "[$(date +%H:%M:%S)] $*"; "$@" || echo "FAILED: $*"; }

for x in "huawei 5" "alibaba2018 5" "azure2021 5" "azure2021 1" "azure2019 5" "azure2019 1"; do
  set -- $x; ds=$1; res=$2
  step uv run python -W ignore -m src.run_forecast "$ds" "$res" $CHEAP
  [ "$res" = 5 ] && step uv run python -W ignore -m src.run_forecast "$ds" "$res" mstl_ets
  for m in $FM_FM; do step uv run --project envs/fm python -W ignore -m src.run_fm "$ds" "$res" "$m"; done
  step uv run --project envs/toto python -W ignore -m src.run_fm "$ds" "$res" toto
  step uv run --project envs/moirai python -W ignore -m src.run_fm "$ds" "$res" moirai2_small
  for m in lightgbm nbeats lstm; do step uv run --project envs/train python -W ignore -m src.trained "$ds" "$res" "$m"; done
  step uv run python -W ignore -m src.metrics "$ds" "$res"
done

# expensive extras last
for ds in huawei alibaba2018; do step uv run --project envs/toto python -W ignore -m src.run_fm "$ds" 5 toto_s20; done
for ds in huawei alibaba2018 azure2021 azure2019; do step uv run python -W ignore -m src.run_forecast "$ds" 5 mstl_arima; done
for x in "huawei 5" "alibaba2018 5" "azure2021 5" "azure2021 1" "azure2019 5" "azure2019 1"; do step uv run python -W ignore -m src.metrics $x; done
echo "[$(date +%H:%M:%S)] ALL DONE"
