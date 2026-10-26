"""Zero-shot FM forecasts -> results/forecasts/<ds>_<res>min/<model>_C<C>.npz (same layout as run_forecast).

Usage: uv run --project envs/<venv> python -m src.run_fm <ds> <res> <model> [C] [max_origins]
C defaults to 2048 (1440 for alibaba2018), capped at the model's max context and at the first origin.
max_origins (smoke tests only) keeps the first k origins and writes under results/smoke/.
"""
import sys
import time

import numpy as np
import torch

from src.data import RESULTS, load
from src.fm import FMS, QUANTILES

BATCH = 64
DEFAULT_C = {"alibaba2018": 1440}
torch.set_num_threads(8)


def main(ds: str, res: int, name: str, c: int | None = None, max_origins: int | None = None) -> None:
    p = load(ds, res)
    _, c_max, factory = FMS[name]
    origins = p.origins()[:max_origins]
    c = min(c or DEFAULT_C.get(ds, 2048), c_max, int(origins[0]))
    out = RESULTS / ("smoke" if max_origins else "forecasts") / f"{ds}_{res}min" / f"{name}_C{c}.npz"
    if out.exists():
        print(f"skip {out.name} (exists)")
        return
    out.parent.mkdir(parents=True, exist_ok=True)

    predict = factory(c, p.h_max, res)
    n, h = len(p.ids), p.h_max
    mean = np.empty((n, len(origins), h), np.float32)
    q = np.full((n, len(origins), h, len(QUANTILES)), np.nan, np.float32)
    t0 = time.perf_counter()
    for j, o in enumerate(origins):
        ctx = p.y[:, o - c : o]
        for b in range(0, n, BATCH):
            m_b, q_b = predict(np.ascontiguousarray(ctx[b : b + BATCH]), h)
            mean[b : b + BATCH, j] = m_b
            if q_b is not None:
                q[b : b + BATCH, j] = q_b
        print(f"  {name} origin {j + 1}/{len(origins)}  {time.perf_counter() - t0:.0f}s", end="\r", flush=True)
    secs = time.perf_counter() - t0
    extra = {} if np.isnan(q).all() else {"q": np.maximum(q, 0), "quantiles": np.array(QUANTILES)}
    np.savez_compressed(out, ids=p.ids, origins=origins, mean=np.maximum(mean, 0), seconds=secs, context=c, **extra)
    print(f"\n{ds} {res}min {name} C={c}: {mean.shape} in {secs:.0f}s")


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], int(a[1]), a[2], int(a[3]) if len(a) > 3 else None, int(a[4]) if len(a) > 4 else None)
