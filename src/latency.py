"""CPU latency / memory benchmark (SPEC §5.2). Run on an otherwise idle machine, one model per process.

Usage: uv run --project envs/<venv> python -m src.latency <model>      (FM in its venv, or a baseline in the root env)
Appends one row per (batch, C) to results/latency.csv.
"""
import csv
import resource
import sys
import time

import numpy as np

from src.data import RESULTS, load

THREADS = 4
GRID = [(1, 512), (64, 512), (1, 2048), (64, 2048)]  # (batch, context)
H, WARMUP, REPS = 60, 5, 30
FIELDS = ["model", "batch", "context", "load_s", "p50_ms", "p95_ms", "peak_rss_mb", "threads"]


def contexts(batch: int, c: int) -> np.ndarray:
    """Real Azure 2019 1-min windows, fixed seed, so every model sees the same inputs."""
    p = load("azure2019", 1)
    rng = np.random.default_rng(0)
    rows = rng.choice(len(p.ids), batch, replace=False)
    return np.ascontiguousarray(p.y[rows, p.val_end - c : p.val_end])


def make(name: str, c: int):
    try:
        from src.fm import FMS
    except ImportError:  # root env has no torch; only baselines there
        FMS = {}
    if name in FMS:
        import torch

        torch.set_num_threads(THREADS)
        return FMS[name][2](c, H, 1)
    from src.baselines import BASELINES

    fn = BASELINES[name][0]
    return lambda ctx, h: (fn(ctx, h, 1440), None)


def main(name: str) -> None:
    rows = []
    for batch, c in GRID:
        t0 = time.perf_counter()
        predict = make(name, c)
        load_s = time.perf_counter() - t0
        x = contexts(batch, c)
        for _ in range(WARMUP):
            predict(x, H)
        ts = []
        for _ in range(REPS):
            t = time.perf_counter()
            predict(x, H)
            ts.append((time.perf_counter() - t) * 1000)
        rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20  # macOS reports bytes
        rows.append(dict(model=name, batch=batch, context=c, load_s=round(load_s, 2), p50_ms=round(np.median(ts), 2),
                         p95_ms=round(np.percentile(ts, 95), 2), peak_rss_mb=round(rss_mb), threads=THREADS))
        print(rows[-1])
    path = RESULTS / "latency.csv"
    new = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, FIELDS)
        if new:
            w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main(sys.argv[1])
