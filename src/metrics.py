"""Forecast npz -> per-series test metrics + summary (SPEC §5.1).

Usage: uv run python -m src.metrics <ds> <res>
Writes results/metrics/<ds>_<res>min.parquet (model, series_id, H, mae, mase, wql_num, abs_sum, cov90, sbc).
"""
import sys

import numpy as np
import pandas as pd

from scripts.prepare import sbc_stats
from src.data import HORIZONS, RESULTS, Panel, load

QUANTILES = np.round(np.arange(0.1, 1.0, 0.1), 1)
REFERENCE = "seasonal_naive"


def mase_scale(p: Panel) -> np.ndarray:
    """In-sample MAE of seasonal naive on train; naive lag-1 if that is 0; NaN if both are 0."""
    tr = p.y[:, : p.train_end].astype(np.float64)
    s_m = np.abs(tr[:, p.season :] - tr[:, : -p.season]).mean(axis=1)
    s_1 = np.abs(np.diff(tr, axis=1)).mean(axis=1)
    return np.where(s_m > 0, s_m, np.where(s_1 > 0, s_1, np.nan))


def actuals(p: Panel, origins: np.ndarray) -> np.ndarray:
    return p.y[:, origins[:, None] + np.arange(p.h_max)]  # [N, O, H]


def conformal(mean: np.ndarray, y: np.ndarray, is_val: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Split conformal: quantiles of scale-normalised val residuals, pooled over series, per step. -> [N, O, H, Q]"""
    s = np.nan_to_num(scale, nan=1.0)[:, None, None]
    r = ((y - mean) / s)[:, is_val]  # [N, Ov, H]
    qs = np.quantile(r.reshape(-1, r.shape[-1]), QUANTILES, axis=0)  # [Q, H]
    return np.maximum(mean[..., None] + qs.T[None, None] * s[..., None], 0)


def pinball(y: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Sum over quantiles of 2·pinball loss. y [..], q [.., Q] -> [..]"""
    d = y[..., None] - q
    return (2 * np.maximum(QUANTILES * d, (QUANTILES - 1) * d)).sum(axis=-1)


def series_metrics(p: Panel, model: str, f: dict, scale: np.ndarray, sbc: np.ndarray) -> pd.DataFrame:
    origins = f["origins"]
    y = actuals(p, origins)
    q = f["q"] if "q" in f else conformal(f["mean"], y, ~p.is_test(origins), scale)
    test = p.is_test(origins)
    y, mean, q = y[:, test], f["mean"][:, test], q[:, test]
    rows = []
    for h in HORIZONS[p.res]:
        yh, mh, qh = y[..., :h], mean[..., :h], q[..., :h, :]
        mae = np.abs(yh - mh).mean(axis=(1, 2))
        rows.append(pd.DataFrame({
            "model": model, "series_id": p.ids, "H": h, "mae": mae, "mase": mae / scale,
            "wql_num": pinball(yh, qh).sum(axis=(1, 2)) / len(QUANTILES), "abs_sum": np.abs(yh).sum(axis=(1, 2)),
            "cov90": (yh <= qh[..., -1]).mean(axis=(1, 2)), "sbc": sbc,
        }))
    return pd.concat(rows)


RATIO_FLOOR = 0.01  # per-series MAE ratio floored at 1%: a model with MAE = 0 on one series cannot drag the geometric mean


def log_ratio(mae, ref) -> np.ndarray:
    """log(max(MAE_model / MAE_ref, RATIO_FLOOR)); callers drop series with MAE_ref = 0 first."""
    return np.log(np.maximum(mae / ref, RATIO_FLOOR))


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """relMASE = geometric mean over series of max(MAE_model / MAE_reference, 1%), series with reference MAE 0 excluded.
    rMAE_sum = sum of MAE_model / sum of MAE_reference over the same series (no floor needed)."""
    ref = df[df.model == REFERENCE].set_index(["series_id", "H"]).mae.rename("ref")
    d = df.join(ref, on=["series_id", "H"])
    d = d[d.ref > 0]
    d["log_ratio"] = log_ratio(d.mae, d.ref)
    g = d.groupby(["model", "H"])
    return pd.DataFrame({
        "relMASE": np.exp(g.log_ratio.mean()),
        "rMAE_sum": g.mae.sum() / g.ref.sum(),
        "MASE_med": g.mase.median(),
        "WQL": g.wql_num.sum() / g.abs_sum.sum(),
        "cov90": g.cov90.mean(),
        "n": g.size(),
    })


def main(ds: str, res: int) -> pd.DataFrame:
    p = load(ds, res)
    scale = mase_scale(p)
    sbc = sbc_stats(p.y[:, : p.train_end]).sbc.to_numpy()
    files = sorted((RESULTS / "forecasts" / f"{ds}_{res}min").glob("*.npz"))
    df = pd.concat(series_metrics(p, f.stem, dict(np.load(f)), scale, sbc) for f in files)
    out = RESULTS / "metrics"
    out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out / f"{ds}_{res}min.parquet", index=False)
    s = summarize(df)
    print(f"\n== {ds} @ {res}min: {len(p.ids)} series, {int(np.isnan(scale).sum())} with zero MASE scale")
    print(s.relMASE.unstack("H").round(3).sort_values(max(HORIZONS[res])).to_string())
    print(s.xs(max(HORIZONS[res]), level="H")[["WQL", "cov90"]].round(3).to_string())
    return df


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]))
