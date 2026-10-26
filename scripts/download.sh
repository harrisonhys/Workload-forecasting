#!/usr/bin/env bash
# Download raw traces into data/raw/<dataset>/. Idempotent; interrupted downloads resume from .part.
# Usage: bash scripts/download.sh [azure2019 azure2021 huawei alibaba2018]   (default: all)
# Google 2019 is BigQuery-only, see SPEC.md §2.5.
set -euo pipefail
cd "$(dirname "$0")/.."

get() {  # get <dir> <url> [outname]
  mkdir -p "data/raw/$1"
  local out="data/raw/$1/${3:-$(basename "$2")}"
  [ -s "$out" ] && { echo "== $out (exists)"; return; }
  echo ">> $out"
  curl -fL --retry 5 -C - -o "$out.part" "$2" && mv "$out.part" "$out"
}

azure2019() {
  get azure2019 https://github.com/Azure/AzurePublicDataset/releases/download/dataset-functions-2019/azurefunctions_dataset2019_azurefunctions-dataset2019.tar.xz azurefunctions-dataset2019.tar.xz
  tar -xf data/raw/azure2019/azurefunctions-dataset2019.tar.xz -C data/raw/azure2019
}

azure2021() {
  get azure2021 https://github.com/Azure/AzurePublicDataset/raw/master/data/AzureFunctionsInvocationTraceForTwoWeeksJan2021.rar
  # Archive is RAR7: bsdtar and 7-Zip fail, brew cask "rar" is disabled. Build unrar >= 7 from source:
  #   curl -O https://www.rarlab.com/rar/unrarsrc-7.3.1.tar.gz && tar xzf unrarsrc-*.tar.gz && make -C unrar
  # then put unrar/unrar on PATH (or set UNRAR=/path/to/unrar).
  "${UNRAR:-unrar}" x -o+ data/raw/azure2021/AzureFunctionsInvocationTraceForTwoWeeksJan2021.rar data/raw/azure2021/ >/dev/null
}

huawei() {  # Toner et al. 2025 curated subsets (5-min, 30 days)
  for i in 1 2 3 4; do
    get huawei "https://raw.githubusercontent.com/sir-lab/time-series-fm-dataset/main/data/datacentre_$i.zip"
    unzip -o -q "data/raw/huawei/datacentre_$i.zip" -d data/raw/huawei
  done
}

alibaba2018() {  # machine_usage only (1.8 GB). container_usage is 29 GB, skipped on purpose.
  get alibaba2018 http://aliopentrace.oss-cn-beijing.aliyuncs.com/v2018Traces/machine_usage.tar.gz
}

for ds in "${@:-azure2019 azure2021 huawei alibaba2018}"; do
  for d in $ds; do "$d"; done
done
