"""Replication of Toner et al. (2025) setting on the Huawei curated data: C = 520, H in {30, 96, 336}, 5-min.

Usage:
  uv run --project envs/<venv> python -m src.replicate_toner run <model> [...]   -> results/replication/<model>.npz
  uv run python -m src.replicate_toner report                                   (root env; adds naive baselines)
Origins: hourly over the test period (days 27-30) while the 336-step window still fits.
"""
import sys
import time

import numpy as np

from src.data import RESULTS, load

C, HS = 520, [30, 96, 336]
OUT = RESULTS / "replication"


def setup():
    p = load("huawei", 5)
    origins = np.arange(p.val_end, p.y.shape[1] - max(HS) + 1, 12)
    return p, origins


def run(name: str) -> None:
    import torch

    from src.fm import FMS

    torch.set_num_threads(3)  # shares the CPU with the main queue
    p, origins = setup()
    h = max(HS)
    if name == "ttm_r2":
        h = 96  # TTM-r2 heads stop at 96; Toner et al. rolled it out autoregressively, we report H<=96 only
    predict = FMS[name][2](C, h, 5)
    t0 = time.perf_counter()
    mean = np.stack([predict(np.ascontiguousarray(p.y[:, o - C : o]), h)[0] for o in origins], axis=1)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / f"{name}.npz", origins=origins, mean=np.maximum(mean, 0), seconds=time.perf_counter() - t0)
    print(f"{name}: {mean.shape} in {time.perf_counter() - t0:.0f}s")


def report() -> None:
    import pandas as pd

    from src.baselines import naive, seasonal_naive
    from src.metrics import mase_scale

    p, origins = setup()
    scale = mase_scale(p)
    y = p.y[:, origins[:, None] + np.arange(max(HS))]
    fcs = {"seasonal_naive": np.stack([seasonal_naive(p.y[:, o - C : o], max(HS), 288) for o in origins], 1),
           "naive": np.stack([naive(p.y[:, o - C : o], max(HS), 288) for o in origins], 1)}
    fcs |= {f.stem: np.load(f)["mean"] for f in sorted(OUT.glob("*.npz"))}
    rows = []
    for name, m in fcs.items():
        for h in HS:
            if m.shape[-1] < h:
                continue
            mase = np.abs(y[..., :h] - m[..., :h]).mean(axis=(1, 2)) / scale
            rows.append({"model": name, "H": h, "MASE_mean": mase.mean()})
    df = pd.DataFrame(rows).pivot(index="model", columns="H", values="MASE_mean")
    df.to_csv(OUT / "table.csv")
    print(df.round(3).sort_values(30).to_string())


if __name__ == "__main__":
    if sys.argv[1] == "report":
        report()
    else:
        for n in sys.argv[2:]:
            run(n)
