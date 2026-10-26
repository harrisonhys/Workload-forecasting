"""Rolling-origin forecasts -> results/forecasts/<ds>_<res>min/<model>.npz

Usage: uv run python -m src.run_forecast <ds> <res> [model ...]   (default: every baseline valid at <res>)
npz keys: ids [N], origins [O], mean [N, O, H] float32, (optional) quantiles [Q], q [N, O, H, Q], seconds
"""
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from src.baselines import BASELINES
from src.data import RESULTS, Panel, load

WORKERS = max(1, (os.cpu_count() or 2) - 2)


def _one_origin(args):
    name, ctx, h, season = args
    return BASELINES[name][0](ctx, h, season)


def forecast_baseline(p: Panel, name: str) -> tuple[np.ndarray, float]:
    _, days, _ = BASELINES[name]
    c = days * p.season
    origins = p.origins()
    jobs = [(name, p.y[:, max(0, o - c) : o], p.h_max, p.season) for o in origins]
    t0 = time.perf_counter()
    if name in ("zero", "naive", "seasonal_naive"):
        outs = [_one_origin(j) for j in jobs]
    else:
        with ProcessPoolExecutor(WORKERS) as ex:
            outs = list(ex.map(_one_origin, jobs))
    return np.stack(outs, axis=1).clip(min=0), time.perf_counter() - t0


def out_path(p: Panel, name: str):
    d = RESULTS / "forecasts" / f"{p.ds}_{p.res}min"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{name}.npz"


def main(ds: str, res: int, names: list[str]) -> None:
    p = load(ds, res)
    names = names or [n for n, (_, _, rs) in BASELINES.items() if res in rs]
    for name in names:
        path = out_path(p, name)
        if path.exists():
            print(f"skip {path.name} (exists)")
            continue
        mean, secs = forecast_baseline(p, name)
        np.savez_compressed(path, ids=p.ids, origins=p.origins(), mean=mean, seconds=secs)
        print(f"{ds} {res}min {name}: {mean.shape} in {secs:.0f}s")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), sys.argv[3:])
