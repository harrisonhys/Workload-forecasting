# Zero-shot workload forecasting for serverless and middleware autoscaling

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23662411.svg)](https://doi.org/10.5281/zenodo.23662411)
(v0.5, the version used in the paper; all versions: https://doi.org/10.5281/zenodo.23662410)

Code, configuration and supporting outputs for the paper draft `paper/draft_paper_id.docx`
(Indonesian). Design and deviations: `SPEC.md`; result log: `RESULTS.md`.

## What is in the repository

| Path | Content |
|---|---|
| `scripts/download.sh` | downloads the public traces (Azure Functions 2019/2021, Alibaba 2018, Huawei Cloud) into `data/raw/` |
| `scripts/prepare.py` | raw traces -> `data/processed/<ds>.parquet` (series selection, binning, outage handling) |
| `configs/splits.json` | train/val/test boundaries, origins, horizons and sha256 of every processed panel |
| `configs/series/*.txt` | series ids in panel order for each dataset x resolution |
| `configs/models.json` | pinned Hugging Face revisions, baseline context lengths, metric settings |
| `src/` | forecasting (`run_forecast`, `run_fm`, `trained`), metrics, simulator, latency, analysis |
| `envs/{fm,toto,moirai,train}` | separate uv projects (with lock files) for libraries whose pins conflict |
| `results/metrics/*.parquet` | per-series test metrics for every model (input to all accuracy tables) |
| `results/sim/` | simulator outputs and policy frontiers |
| `results/latency.csv`, `results/replication/` | CPU latency benchmark, replication of Toner et al. (2025) |
| `paper/tables/*.csv`, `paper/figures/` | every number and figure used in the paper |
| `tests/` | `uv run pytest` (metrics, simulator, histogram policy) |

Per-origin forecasts (`results/forecasts/`, about 1 GB) are not in git. They are attached to the GitHub
release as `forecasts_<dataset>.zip` (checksums in `SHA256SUMS.txt`); unzip them in `results/` to skip
the forecasting steps, or regenerate them with the pipeline below.

## Reproducing

Tested on an Apple M3 Pro (11 cores, 18 GB), macOS, uv 0.11, Python 3.12, CPU only. On that machine
`run_all.sh` took about 14 hours and `run_followup.sh` about 2 hours.

```bash
bash scripts/download.sh                      # 4 GB; Azure 2021 needs unrar >= 7 (see script)
uv run scripts/prepare.py                     # data/processed/*.parquet
uv run python -m scripts.export_configs       # compare sha256 in configs/splits.json
bash scripts/run_all.sh > results/logs/run_all.log 2>&1
bash scripts/run_followup.sh > results/logs/followup.log 2>&1   # CPU neural baselines, metrics, simulation
bash scripts/run_ablation.sh > results/logs/ablation.log 2>&1   # context length
uv run --project envs/fm python -m src.latency chronos2         # one model per process, idle machine
uv run --project envs/fm python -m src.replicate_toner run chronos2
uv run python -m src.replicate_toner report
uv run python -m src.analysis && FIG_LANG=id uv run python -m src.analysis
uv run --with python-docx python paper/build_docx.py
```

Every step skips outputs that already exist, so an interrupted run can be restarted. LightGBM, N-BEATS
and LSTM use seed 42, but CPU thread scheduling can still change the last digits.

## Licences

Code: MIT (see `LICENSE`). The traces and model checkpoints keep the licences of their original
repositories; they are not redistributed here.
