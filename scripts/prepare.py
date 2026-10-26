# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas>=2.2", "pyarrow>=16"]
# ///
"""Raw traces -> long-format parquet (series_id, ts, y) + per-series meta.

Usage: uv run scripts/prepare.py [azure2019 azure2021 alibaba2018 huawei google2019 selfcheck]
Out:   data/processed/<ds>.parquet       series_id, ts, y   (selected series only)
       data/processed/<ds>_meta.parquet  one row per candidate series, `selected` flag
See SPEC.md §2-3 for the selection rules implemented here.
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW, OUT = ROOT / "data/raw", ROOT / "data/processed"
EPOCH = pd.Timestamp("2000-01-01")  # traces are anonymised; only relative time matters
SEED = 42
TRAIN_DAYS = 10  # Azure train split (src/data.py); durations for the simulator use these days only
N_TOP = 50  # Azure 2019: always keep highest-volume apps
N_PER_CLASS = 125  # Azure 2019: random apps per Syntetos-Boylan class
N_MACHINES = 200  # Alibaba: random machines (+ cluster mean)
ALIBABA_BIN_S = 300  # Alibaba samples are irregular (median 60 s, max ~640 s) -> 5-min bins, not 1-min
MIN_COVERAGE = 0.90  # Alibaba: fraction of bins a machine must report (median machine: ~93%)
MAX_GAP_BINS = 3  # Alibaba: interpolate gaps up to this many bins


def sbc_stats(mat: np.ndarray) -> pd.DataFrame:
    """Per-row intermittency stats and Syntetos-Boylan class (cut-offs ADI 1.32, CV² 0.49)."""
    mat = mat.astype(np.float64)  # ponytail: full copy, ~3 GB for Azure 2019; chunk rows if RAM bites
    n_nz = (mat > 0).sum(1)
    k = np.maximum(n_nz, 1)
    mean_nz = mat.sum(1) / k
    with np.errstate(divide="ignore", invalid="ignore"):
        cv2 = ((mat**2).sum(1) / k - mean_nz**2) / mean_nz**2
    adi = np.where(n_nz > 0, mat.shape[1] / k, np.inf)
    cls = np.select(
        [(adi < 1.32) & (cv2 < 0.49), adi < 1.32, cv2 < 0.49],
        ["smooth", "erratic", "intermittent"],
        "lumpy",
    )
    return pd.DataFrame({"total": mat.sum(1), "zero_frac": 1 - n_nz / mat.shape[1], "adi": adi, "cv2": cv2, "sbc": cls})


def bin_invocations(df: pd.DataFrame) -> pd.DataFrame:
    """Azure 2021 per-invocation rows -> app x minute counts (arrival = end - duration)."""
    minute = ((df.end_timestamp - df.duration) // 60).astype(np.int64)
    counts = df.assign(minute=minute - minute.min()).groupby(["app", "minute"]).size()
    mat = counts.unstack(fill_value=0)
    return mat.reindex(columns=range(mat.columns.max() + 1), fill_value=0)


def write(name: str, mat: pd.DataFrame, freq: str, meta: pd.DataFrame) -> None:
    """mat: index=series_id, columns=int step. Writes only rows flagged selected in meta."""
    OUT.mkdir(parents=True, exist_ok=True)
    sel = mat.loc[meta.index[meta.selected]]
    long = sel.stack().rename("y").reset_index()
    long.columns = ["series_id", "step", "y"]
    long["ts"] = EPOCH + long.step.to_numpy() * pd.Timedelta(freq)
    long = long.astype({"series_id": "category", "y": "float32"})[["series_id", "ts", "y"]]
    long.to_parquet(OUT / f"{name}.parquet", index=False)
    meta.assign(freq=freq, length=mat.shape[1]).to_parquet(OUT / f"{name}_meta.parquet")
    print(f"{name}: {meta.selected.sum()}/{len(meta)} series x {mat.shape[1]} steps @ {freq}")
    print(meta[meta.selected].sbc.value_counts().to_string() if "sbc" in meta else "")


def azure2019() -> None:
    src = RAW / "azure2019"
    days, dur, trigs = [], [], []
    for d in range(1, 15):
        inv = pd.read_csv(src / f"invocations_per_function_md.anon.d{d:02d}.csv")
        mins = [str(i) for i in range(1, 1441)]
        app = inv.groupby("HashApp")[mins].sum().astype(np.int32)
        app.columns = range((d - 1) * 1440, d * 1440)
        days.append(app)
        if d <= TRAIN_DAYS:  # durations feed the autoscaling simulator: train days only, no test-period information
            dur.append(pd.read_csv(src / f"function_durations_percentiles.anon.d{d:02d}.csv", usecols=["HashApp", "Average", "Count"]))
        trigs.append(inv[mins].sum(axis=1).groupby([inv.HashApp, inv.Trigger]).sum())
    # ponytail: keep apps present on all 14 days (long-lived); absent day = no invocations that day
    mat = pd.concat(days, axis=1, join="inner")
    meta = sbc_stats(mat.to_numpy()).set_index(mat.index)

    dur = pd.concat(dur)
    dur["w"] = dur.Average * dur.Count
    g = dur.groupby("HashApp")[["w", "Count"]].sum()
    meta["mean_duration_ms"] = (g.w / g.Count).reindex(meta.index)
    trig = pd.concat(trigs).groupby(level=[0, 1]).sum()
    meta["trigger"] = trig.unstack().idxmax(1).reindex(meta.index)

    top = meta.total.nlargest(N_TOP).index
    rest = meta.drop(top)
    sampled = rest.groupby("sbc", group_keys=False).apply(lambda x: x.sample(min(len(x), N_PER_CLASS), random_state=SEED)).index
    meta["selected"] = meta.index.isin(top.union(sampled))
    write("azure2019", mat, "1min", meta)


def azure2021() -> None:
    src = next((RAW / "azure2021").glob("*.txt"), None) or next((RAW / "azure2021").glob("*.csv"))
    df = pd.read_csv(src)
    mat = bin_invocations(df)
    meta = sbc_stats(mat.to_numpy()).set_index(mat.index)
    arrival = df.end_timestamp - df.duration
    train = df[arrival < arrival.min() + TRAIN_DAYS * 86400]  # train days only, as for azure2019
    meta["mean_duration_ms"] = train.groupby("app").duration.mean().reindex(meta.index) * 1000
    meta["selected"] = True  # small trace, keep every app
    write("azure2021", mat, "1min", meta)


def alibaba2018() -> None:
    cols = ["machine_id", "time_stamp", "cpu", "mem"]
    parts = []
    # stream straight from the 1.8 GB tar.gz; uncompressed CSV never touches disk
    tar = subprocess.Popen(["tar", "-xzOf", RAW / "alibaba2018/machine_usage.tar.gz"], stdout=subprocess.PIPE)
    for ch in pd.read_csv(tar.stdout, header=None, usecols=[0, 1, 2, 3], names=cols, chunksize=5_000_000):
        ch = ch[ch.cpu.between(0, 100) & ch.mem.between(0, 100)]
        key = ch.machine_id.str[2:].astype(np.int64) * 100_000 + (ch.time_stamp // ALIBABA_BIN_S).astype(np.int64)
        parts.append(ch[["cpu"]].assign(n=1).groupby(key.to_numpy()).sum())
        print(f"  alibaba chunk {len(parts)}", end="\r")
    if tar.wait():
        raise RuntimeError("tar failed reading machine_usage.tar.gz")
    agg = pd.concat(parts).groupby(level=0).sum()
    mid, step = divmod(agg.index.to_numpy(), 100_000)
    cpu = pd.Series(agg.cpu.to_numpy() / agg.n.to_numpy(), index=pd.MultiIndex.from_arrays([mid, step - step.min()]))
    mat = cpu.unstack()
    mat = mat.reindex(columns=range(mat.columns.max() + 1))
    # trace-wide outage (bins 470-534, ~5.4 h on day 2): excluded from coverage, filled from 1 day earlier
    outage = mat.columns[mat.notna().mean() < 0.5]
    coverage = mat.drop(columns=outage).notna().mean(axis=1)
    mat = mat[coverage >= MIN_COVERAGE]
    mat[outage] = mat[outage - 86400 // ALIBABA_BIN_S].to_numpy()
    print(f"  outage bins seasonally filled: {outage.min()}-{outage.max()} ({len(outage)})")
    mat = mat.interpolate(axis=1, limit=MAX_GAP_BINS, limit_area="inside").bfill(axis=1).ffill(axis=1)
    mat = mat.dropna()
    mat = mat[mat.mean(axis=1) >= 1.0]  # drop idle/dead machines
    mat.index = [f"m_{i}" for i in mat.index]
    mat.loc["cluster_mean"] = mat.mean()

    meta = pd.DataFrame(index=mat.index)
    meta["coverage"] = coverage.rename(lambda i: f"m_{i}").reindex(meta.index).fillna(1.0)
    meta["mean_cpu"] = mat.mean(axis=1)
    rng = np.random.default_rng(SEED)
    machines = mat.index.drop("cluster_mean")
    picked = rng.choice(machines, size=min(N_MACHINES, len(machines)), replace=False)
    meta["selected"] = meta.index.isin(list(picked) + ["cluster_mean"])
    print(f"  coverage quantiles: {coverage.quantile([.05, .25, .5, .75]).round(3).to_dict()}")
    write("alibaba2018", mat, "5min", meta)


def huawei() -> None:
    frames = []
    for f in sorted((RAW / "huawei").rglob("datacentre_*.csv")):
        df = pd.read_csv(f, index_col=0)
        df.columns = [f"{f.stem}/{c}" for c in df.columns]
        frames.append(df.reset_index(drop=True).T)
    mat = pd.concat(frames)
    meta = sbc_stats(mat.to_numpy()).set_index(mat.index)
    meta["selected"] = True
    write("huawei", mat, "5min", meta)


def google2019() -> None:
    """Expects BigQuery export(s) from SPEC.md §2.5 with columns series_id, t5, y."""
    df = pd.concat(pd.read_csv(f) for f in (RAW / "google2019").glob("*.csv"))
    df["t5"] -= df.t5.min()
    mat = df.pivot_table(index="series_id", columns="t5", values="y", aggfunc="sum")
    mat = mat.reindex(columns=range(mat.columns.max() + 1)).fillna(0)
    meta = sbc_stats(mat.to_numpy()).set_index(mat.index)
    meta["selected"] = True
    write("google2019", mat, "5min", meta)


def selfcheck() -> None:
    s = sbc_stats(np.array([[5, 5, 5, 5], [0, 0, 0, 9], [1, 9, 1, 9], [0, 0, 0, 0]]))
    assert list(s.sbc[:3]) == ["smooth", "intermittent", "erratic"], s
    assert s.adi[1] == 4 and s.zero_frac[3] == 1
    inv = pd.DataFrame({"app": ["a", "a", "b"], "end_timestamp": [61.0, 200.0, 125.0], "duration": [1.0, 10.0, 5.0]})
    m = bin_invocations(inv)  # arrivals 60, 190, 120 -> minutes 1, 3, 2 -> offsets 0, 2, 1
    assert m.loc["a"].tolist() == [1, 0, 1] and m.loc["b"].tolist() == [0, 1, 0], m
    print("selfcheck ok")


if __name__ == "__main__":
    for name in sys.argv[1:] or ["azure2019", "azure2021", "alibaba2018", "huawei"]:
        globals()[name]()
