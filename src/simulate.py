"""Trace-driven autoscaling simulation from saved forecasts (SPEC §5.3, revised in §9).

Predictive policies are hourly capacity plans: at each hourly test origin o, capacity for t in [o, o+60 min)
comes from that origin's forecast. Reactive baselines see actual demand with a one-step lag.

Serverless model (per application i, per minute t; all quantities are minute aggregates, not individual requests):
    need[i,t] = ceil(rate[i,t] * d_i / 60)      Little's law, d_i = mean duration from TRAIN days only, >= 1 if rate > 0
    cap[i,t]  = capacity the policy keeps warm during minute t
    cold-start proxy  = sum_{i,t} max(0, need - cap) / sum_{i,t} need      (pooled over apps, not averaged per app)
    idle per demand   = sum_{i,t} max(0, cap - need) / sum_{i,t} need
  Instances are not tracked across minute boundaries and in-flight invocations are not carried over: the proxy counts
  instance shortfalls per minute, not cold-started requests.
  Predictive: cap = ceil(gate(q_tau) * d_i / 60) with gate(x) = x if x >= theta else 0. (tau, theta) are chosen on the
  VALIDATION period (all validation-Pareto-optimal pairs are kept), then evaluated on test.

Usage: uv run python -m src.simulate <ds> <res>
  serverless (azure2019/azure2021 @ 1 min): cold-start rate vs idle instance-minutes
  vm         (alibaba2018/google2019 @ 5 min): under-provisioned fraction vs excess replicas
Writes results/sim/<ds>_<res>min.parquet (+ capacity_tau0.9.npz timelines for the energy paper).
"""
import sys

import numpy as np
import pandas as pd

from src.data import RESULTS, Panel, load, meta
from src.metrics import QUANTILES, actuals, conformal, mase_scale

TAUS = [0.5, 0.6, 0.7, 0.8, 0.9]
THETAS = [0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0]  # activation threshold, forecast invocations per minute
KEEP_ALIVE_MIN = 10  # keep-alive used inside the hybrid policy (Azure Functions-style)
KEEP_ALIVE_KNOB = [1, 2, 5, 10, 20, 30, 60]  # reactive serverless trade-off curve (minutes)
HPA_TARGET_KNOB = [0.4, 0.5, 0.6, 0.7, 0.8]  # reactive VM trade-off curve (provisioning target utilisation)
RISK_LEVELS = {"serverless": [0.01, 0.02, 0.05, 0.10], "vm": [0.02, 0.05, 0.10, 0.20]}
CONCURRENCY = 1  # requests per instance
R_MAX, U_TARGET = 20, 0.7  # VM: replicas at 100% CPU, target utilisation


def timeline(a: np.ndarray) -> np.ndarray:
    """[N, O, H] with stride == H -> contiguous [N, O*H]."""
    return a.reshape(len(a), -1)


def load_quantiles(p: Panel, path, scale: np.ndarray, split: str = "test") -> tuple[np.ndarray, np.ndarray]:
    """Quantile forecasts for the validation or test origins. ponytail: conformal quantiles of point models are
    calibrated on validation residuals, so their validation-period values are in-sample (used only for selection)."""
    f = dict(np.load(path))
    o = f["origins"]
    q = f["q"] if "q" in f else conformal(f["mean"], actuals(p, o), ~p.is_test(o), scale)
    sel = p.is_test(o) if split == "test" else ~p.is_test(o)
    return o[sel], q[:, sel]


def pareto_front(cands: list[tuple]) -> list[tuple]:
    """Keep (cfg..., risk, cost) tuples not dominated in (risk, cost)."""
    return [c for c in cands if not any(d[-2] <= c[-2] and d[-1] <= c[-1] and (d[-2] < c[-2] or d[-1] < c[-1]) for d in cands)]


# ---------- serverless ----------

def instances(rate_per_min: np.ndarray, dur_s: np.ndarray) -> np.ndarray:
    """Little's law: concurrency = rate/60 · duration; any demand needs >= 1 instance."""
    return np.ceil(np.maximum(rate_per_min, 0) * dur_s[:, None] / 60 / CONCURRENCY)


def keep_alive(need: np.ndarray, start: int, k: int) -> np.ndarray:
    """Warm instances at t = max demand over the previous k minutes (each instance lives k min after use)."""
    win = np.lib.stride_tricks.sliding_window_view(need[:, start - k : need.shape[1] - 1], k, axis=1)
    return win.max(axis=2)


HIST_RANGE_MIN = 240  # Shahrad et al. (ATC'20): 4-hour idle-time histogram, 1-minute bins
HIST_CV_MIN, HIST_MIN_SAMPLES, HIST_OOB_MAX = 2.0, 10, 0.5


def histogram_windows(active: np.ndarray) -> tuple[int, int]:
    """(pre-warm, keep-alive) minutes for one app from its idle-time history (5th/99th percentile, 10% margins).
    Falls back to (0, range) when the histogram is not representative. ponytail: the original switches to ARIMA
    when idle times often exceed the range; here those apps also fall back to a 4-hour keep-alive."""
    t = np.flatnonzero(active)
    it = np.diff(t)
    if len(it) < HIST_MIN_SAMPLES:
        return 0, HIST_RANGE_MIN
    inside = it[it <= HIST_RANGE_MIN]
    counts = np.bincount(inside, minlength=HIST_RANGE_MIN + 1)
    cv = counts.std() / counts.mean() if counts.mean() > 0 else 0.0
    if 1 - len(inside) / len(it) > HIST_OOB_MAX or cv < HIST_CV_MIN:
        return 0, HIST_RANGE_MIN
    pre = int(np.floor(0.9 * np.percentile(inside, 5)))
    keep = int(np.ceil(1.1 * np.percentile(inside, 99))) - pre
    return pre, max(keep, 1)


def hybrid_histogram(need: np.ndarray, t0: int) -> np.ndarray:
    """Capacity from t0 on. After the last invocation at minute u the app is unloaded until u + pre, then kept warm
    until u + pre + keep with capacity = peak demand of the preceding window. Windows learnt on history before t0."""
    cap = np.zeros((len(need), need.shape[1] - t0))
    for i, row in enumerate(need):
        pre, keep = histogram_windows(row[:t0] > 0)
        hist_active = np.flatnonzero(row[:t0] > 0)
        last = hist_active[-1] if len(hist_active) else -10**9
        for t in range(t0, need.shape[1]):
            if last + max(pre, 1) <= t < last + pre + keep + 1:
                cap[i, t - t0] = row[max(0, t - pre - keep) : t].max(initial=0)
            if row[t] > 0:
                last = t
    return cap


def serverless_cost(need: np.ndarray, cap: np.ndarray) -> dict:
    return {"cold_rate": np.maximum(need - cap, 0).sum() / need.sum(), "idle_per_demand": np.maximum(cap - need, 0).sum() / need.sum()}


def simulate_serverless(p: Panel, scale: np.ndarray) -> tuple[list[dict], dict]:
    dur = meta(p.ds)["mean_duration_ms"].reindex(p.ids).to_numpy() / 1000
    ok = np.isfinite(dur) & (dur > 0)
    need_all = instances(p.y[ok], dur[ok])
    t0 = p.val_end
    rows, caps = [], {}
    ka = keep_alive(need_all, t0, KEEP_ALIVE_MIN)  # hybrid policy's reactive part
    need = need_all[:, t0 : t0 + ka.shape[1]]
    for k in KEEP_ALIVE_KNOB:
        rows.append({"model": "reactive", "policy": "reactive", "knob": k, **serverless_cost(need, keep_alive(need_all, t0, k)[:, : need.shape[1]])})
    rows.append({"model": "oracle", "policy": "oracle", "knob": np.nan, **serverless_cost(need, need)})
    hh = hybrid_histogram(need_all, t0)[:, : need.shape[1]]
    rows.append({"model": "hybrid_histogram", "policy": "reactive_histogram", "knob": np.nan, **serverless_cost(need, hh)})
    caps["hybrid_histogram"] = hh.astype(np.int32)
    v0 = p.train_end
    need_val = need_all[:, v0:t0]
    ka_val = keep_alive(need_all, v0, KEEP_ALIVE_MIN)[:, : need_val.shape[1]]

    def plan(q: np.ndarray, tau: float, theta: float, n: int) -> np.ndarray:
        rate = timeline(q[ok][..., QUANTILES.tolist().index(tau)])
        return instances(np.where(rate >= theta, rate, 0.0), dur[ok])[:, :n]

    for path in sorted((RESULTS / "forecasts" / f"{p.ds}_{p.res}min").glob("*.npz")):
        o, q = load_quantiles(p, path, scale, "test")
        _, qv = load_quantiles(p, path, scale, "val")
        assert o[0] == t0
        for policy, floor_val, floor_test in [("predictive", 0, 0), ("hybrid", ka_val, ka)]:
            cands = []
            for tau in TAUS:
                for theta in THETAS:
                    c = serverless_cost(need_val, np.maximum(plan(qv, tau, theta, need_val.shape[1]), floor_val))
                    cands.append((tau, theta, c["cold_rate"], c["idle_per_demand"]))
            for tau, theta, _, _ in pareto_front(cands):  # chosen on validation, reported on test
                cap = np.maximum(plan(q, tau, theta, need.shape[1]), floor_test)
                rows.append({"model": path.stem, "policy": policy, "knob": tau, "theta": theta, **serverless_cost(need, cap)})
        caps[path.stem] = plan(q, 0.9, 0.0, need.shape[1]).astype(np.int32)
    caps["demand"] = need.astype(np.int32)
    return rows, caps


# ---------- VM / container ----------

def replicas(util_pct: np.ndarray, target: float = U_TARGET) -> np.ndarray:
    return np.ceil(np.maximum(util_pct, 0) / 100 * R_MAX / target)


def hpa(util_pct: np.ndarray, t0: int, target: float) -> np.ndarray:
    """Reactive HPA from t0 on: sees last step's demand, one-step scale-down stabilisation."""
    desired = replicas(util_pct[:, t0 - 2 : -1], target)
    return np.maximum(desired[:, 1:], desired[:, :-1])


def vm_cost(need: np.ndarray, cap: np.ndarray) -> dict:
    return {
        "under_frac": (cap < need).mean(),
        "deficit_per_demand": np.maximum(need - cap, 0).sum() / need.sum(),
        "excess_per_demand": np.maximum(cap - need, 0).sum() / need.sum(),
        "actions_per_day": (np.diff(cap, axis=1) != 0).sum(axis=1).mean() * (1440 / 5) / cap.shape[1],
    }


def simulate_vm(p: Panel, scale: np.ndarray) -> tuple[list[dict], dict]:
    t0 = p.val_end
    need_all = replicas(p.y)
    need = need_all[:, t0:]
    rows = [{"model": "reactive", "policy": "reactive", "knob": u, **vm_cost(need, hpa(p.y, t0, u))} for u in HPA_TARGET_KNOB]
    rows.append({"model": "oracle", "policy": "oracle", "knob": np.nan, **vm_cost(need, need)})
    hpa_default = hpa(p.y, t0, U_TARGET)
    caps = {}
    for path in sorted((RESULTS / "forecasts" / f"{p.ds}_{p.res}min").glob("*.npz")):
        o, q = load_quantiles(p, path, scale)
        for tau in TAUS:
            pred = replicas(timeline(q[..., QUANTILES.tolist().index(tau)]))[:, : need.shape[1]]
            n = need[:, : pred.shape[1]]
            rows.append({"model": path.stem, "policy": "predictive", "knob": tau, **vm_cost(n, pred)})
            rows.append({"model": path.stem, "policy": "hybrid", "knob": tau, **vm_cost(n, np.maximum(pred, hpa_default[:, : pred.shape[1]]))})
            if tau == 0.9:
                caps[path.stem] = pred.astype(np.int32)
    caps["demand"] = need.astype(np.int32)
    return rows, caps


def frontier(df: pd.DataFrame, risk: str, cost: str, levels: list[float]) -> pd.DataFrame:
    """Cost at matched risk: each (model, policy) curve over its knob, linearly interpolated; NaN outside its range."""
    rows = []
    for (m, pol), g in df[df.policy != "oracle"].groupby(["model", "policy"]):
        g = g.sort_values(risk)
        r, c = g[risk].to_numpy(), g[cost].to_numpy()
        rows.append({"model": m, "policy": pol, **{f"{cost}@{risk}={lv}": np.interp(lv, r, c) if r.min() <= lv <= r.max() else np.nan for lv in levels}})
    return pd.DataFrame(rows).sort_values(list(rows[0])[2:], na_position="last")


def main(ds: str, res: int) -> pd.DataFrame:
    p = load(ds, res)
    scale = mase_scale(p)
    rows, caps = (simulate_serverless if ds.startswith("azure") else simulate_vm)(p, scale)
    df = pd.DataFrame(rows)
    out = RESULTS / "sim"
    out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out / f"{ds}_{res}min.parquet", index=False)
    np.savez_compressed(out / f"{ds}_{res}min_capacity_tau0.9.npz", **caps)
    kind = "serverless" if ds.startswith("azure") else "vm"
    risk, cost = ("cold_rate", "idle_per_demand") if kind == "serverless" else ("under_frac", "excess_per_demand")
    f = frontier(df, risk, cost, RISK_LEVELS[kind])
    f.to_csv(out / f"{ds}_{res}min_frontier.csv", index=False)
    print(f.round(4).to_string(index=False))
    return df


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]))
