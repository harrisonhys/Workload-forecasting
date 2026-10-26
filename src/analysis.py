"""Paper tables and figures from results/ (SPEC §5): bootstrap CIs, Friedman + Nemenyi CD diagrams,
accuracy-latency Pareto, autoscaling trade-off curves, context ablation.

Usage: uv run python -m src.analysis      -> paper/tables/*.csv|.tex, paper/figures/*.pdf
"""
import os
import re

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, studentized_range

from src.data import RESULTS, ROOT
from src.metrics import log_ratio

OUT_T, OUT_F = ROOT / "paper/tables", ROOT / "paper/figures"
SEED, N_BOOT = 42, 1000
MAIN_C = {"2048", "1440", "1536"}  # main-run contexts; other C values are ablation runs
DATASETS = ["huawei_5min", "alibaba2018_5min", "azure2019_5min", "azure2019_1min", "azure2021_5min", "azure2021_1min"]
DENSE = ["huawei_5min", "alibaba2018_5min", "azure2019_5min", "azure2019_1min"]
NAMES = {
    "timesfm3": "TimesFM-3", "chronos2": "Chronos-2", "toto": "Toto", "toto_s20": "Toto (20 samples)",
    "timesfm25": "TimesFM-2.5", "tirex": "TiRex", "moirai2_small": "Moirai-2-S", "chronos_bolt_small": "Chronos-Bolt-S",
    "ttm_r2": "TTM-r2", "lightgbm": "LightGBM", "nbeats": "N-BEATS", "lstm": "LSTM", "mstl_arima": "MSTL-ARIMA",
    "mstl_ets": "MSTL-ETS", "ets": "ETS", "naive": "Naive",
    "zero": "Selalu nol" if os.environ.get("FIG_LANG", "en") == "id" else "Always zero", "seasonal_naive": "Seasonal naive", "swa3": "SWA-3",
    "croston": "Croston", "tsb": "TSB",
}
FM = {"timesfm3", "chronos2", "toto", "toto_s20", "timesfm25", "tirex", "moirai2_small", "chronos_bolt_small", "ttm_r2"}
TRAINED = {"lightgbm", "nbeats", "lstm"}
# reference palette slots 1-3 (validated all-pairs for scatter) + neutral ink for the reactive reference
COLOR = {"fm": "#2a78d6", "trained": "#eb6834", "stat": "#1baf7a", "ref": "#52514e"}
INK, MUTED = "#0b0b0b", "#52514e"
plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2,
                     "axes.grid": True, "grid.color": "#e6e5e1", "grid.linewidth": 0.6, "pdf.fonttype": 42})


LANG = os.environ.get("FIG_LANG", "en")  # FIG_LANG=id -> Indonesian labels, files suffixed _id
ID = {
    "CPU latency p50, batch 64, C = 2048 (ms, log)": "Latensi CPU p50, batch 64, C = 2048 (ms, log)",
    "relMASE, geo-mean of 4 dense datasets (lower is better)": "relMASE, rerata geometrik 4 dataset padat (lebih kecil lebih baik)",
    "Foundation model": "Model fondasi (TSFM)", "Statistical": "Statistik",
    "Trained (dotted: accuracy only)": "Terlatih (garis titik: akurasi saja)",
    "Azure 2021 (serverless, sparse)": "Azure 2021 (serverless, jarang)", "cold-start proxy": "proksi rasio cold start",
    "idle instance-min per demanded (proxy)": "proksi instance-menit idle per kebutuhan",
    "fraction of intervals under-provisioned": "fraksi interval kurang kapasitas",
    "excess replicas per needed": "replika berlebih per kebutuhan", "Reactive (knob)": "Reaktif",
    "predictive": "prediktif", "hybrid": "hibrida", "Adapted hybrid histogram (no ARIMA)": "Adaptasi histogram hibrida (tanpa ARIMA)", "context length C (steps, log2)": "panjang konteks C (langkah, log2)",
    "relMASE at H max": "relMASE pada H maks", "steps": "langkah",
}


def tr(text: str) -> str:
    return ID.get(text, text) if LANG == "id" else text


def save(fig, path) -> None:
    """PDF for LaTeX, 300-dpi PNG for the Word draft."""
    if LANG != "en":
        path = path.with_name(f"{path.stem}_{LANG}{path.suffix}")
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight", dpi=300)


def base(name: str) -> str:
    return re.sub(r"_C\d+$", "", name)


def family(m: str) -> str:
    return "fm" if m in FM else "trained" if m in TRAINED else "stat"


def load(ds: str) -> pd.DataFrame:
    df = pd.read_parquet(RESULTS / "metrics" / f"{ds}.parquet")
    c = df.model.str.extract(r"_C(\d+)$")[0]
    df = df[c.isna() | c.isin(MAIN_C)].copy()
    df["model"] = df.model.map(base)
    return df


def mae_matrix(df: pd.DataFrame, h: int) -> pd.DataFrame:
    """series x model MAE at horizon h; series with seasonal-naive MAE 0 dropped (relMASE undefined there)."""
    m = df[df.H == h].pivot(index="series_id", columns="model", values="mae")
    return m[m["seasonal_naive"] > 0]


def log_ratio_matrix(df: pd.DataFrame, h: int) -> pd.DataFrame:
    """series x model matrix of log(max(MAE_model / MAE_seasonal_naive, 1%)) (see metrics.log_ratio)."""
    m = mae_matrix(df, h)
    return pd.DataFrame(log_ratio(m.to_numpy(), m[["seasonal_naive"]].to_numpy()), index=m.index, columns=m.columns)


def bootstrap(lr: pd.DataFrame) -> pd.DataFrame:
    """relMASE (geometric mean) with 95% percentile CI, resampling series."""
    rng = np.random.default_rng(SEED)
    x = lr.to_numpy()
    idx = rng.integers(0, len(x), size=(N_BOOT, len(x)))
    boots = np.exp(x[idx].mean(axis=1))  # [B, models]
    return pd.DataFrame({"rel": np.exp(x.mean(axis=0)), "lo": np.percentile(boots, 2.5, axis=0),
                         "hi": np.percentile(boots, 97.5, axis=0)}, index=lr.columns)


def friedman(m: pd.DataFrame) -> dict:
    """Friedman test on a series x model matrix (scipy, tie-corrected). log10 p via mpmath: doubles underflow to 0."""
    import mpmath

    stat = friedmanchisquare(*m.to_numpy().T).statistic
    k, n = m.shape[1], m.shape[0]
    p = mpmath.gammainc((k - 1) / 2, stat / 2, mpmath.inf, regularized=True)  # chi-square survival function
    return {"n": n, "k": k, "chi2": float(stat), "log10_p": float(mpmath.log10(p))}


def nemenyi(m: pd.DataFrame, alpha: float = 0.05) -> tuple[pd.Series, float, dict]:
    """Average ranks of per-series MAE (1 = best), Nemenyi critical difference, Friedman statistics."""
    ranks = m.rank(axis=1).mean().sort_values()
    k, n = m.shape[1], m.shape[0]
    cd = studentized_range.ppf(1 - alpha, k, np.inf) / np.sqrt(2) * np.sqrt(k * (k + 1) / (6 * n))
    return ranks, cd, friedman(m)


# ---------------- tables ----------------

def tex(df: pd.DataFrame, path, caption: str) -> None:
    cols = " & ".join(["Model", *df.columns])
    rows = "\n".join(" & ".join([i, *map(str, r)]) + r" \\" for i, r in zip(df.index, df.to_numpy()))
    path.write_text(f"% {caption}\n\\begin{{tabular}}{{l{'r' * len(df.columns)}}}\n\\toprule\n{cols} \\\\\n\\midrule\n{rows}\n\\bottomrule\n\\end{{tabular}}\n")


def accuracy_table() -> pd.DataFrame:
    cells, best = {}, {}
    for ds in DATASETS:
        df = load(ds)
        b = bootstrap(log_ratio_matrix(df, df.H.max()))
        cells[ds] = b.apply(lambda r: f"{r.rel:.3f} [{r.lo:.2f}, {r.hi:.2f}]", axis=1)
        best[ds] = b.rel.idxmin()
        b.to_csv(OUT_T / f"relmase_ci_{ds}.csv")
    t = pd.DataFrame(cells)
    t = t.loc[[m for m in NAMES if m in t.index]].fillna("–")
    for ds, m in best.items():
        t.loc[m, ds] = r"\textbf{" + t.loc[m, ds] + "}"
    t.index = [NAMES[m] for m in t.index]
    t.to_csv(OUT_T / "accuracy.csv")
    tex(t, OUT_T / "accuracy.tex", "relMASE at the longest horizon (60 min), geometric mean with 95% bootstrap CI over series")
    return t


def dataset_table() -> pd.DataFrame:
    rows = {}
    for ds in DATASETS:
        df = load(ds)
        n = df.series_id.nunique()
        cls = df[(df.model == "naive") & (df.H == df.H.max())].sbc.value_counts()
        rows[ds] = {"series": n, "horizon_steps": df.H.max(),
                    **{c: int(cls.get(c, 0)) for c in ["smooth", "erratic", "intermittent", "lumpy"]}}
    t = pd.DataFrame(rows).T
    t.to_csv(OUT_T / "datasets.csv")
    tex(t, OUT_T / "datasets.tex", "Series per dataset and Syntetos-Boylan class (train split, evaluation resolution)")
    return t


def stats_table() -> pd.DataFrame:
    rows = {}
    for ds in DATASETS:
        df = load(ds)
        for h in sorted(df.H.unique())[:: max(1, len(df.H.unique()) - 1)]:  # shortest and longest horizon
            ranks, cd, fr = nemenyi(mae_matrix(df, h))
            rows[(ds, h)] = {"n": fr["n"], "k": fr["k"], "chi2": round(fr["chi2"], 1), "log10_p": round(fr["log10_p"], 1),
                             "CD": round(cd, 2), "best": NAMES[ranks.index[0]],
                             "rank_best": round(ranks.iloc[0], 2), "second": NAMES[ranks.index[1]],
                             "rank_second": round(ranks.iloc[1], 2),
                             "best_vs_second_significant": ranks.iloc[1] - ranks.iloc[0] > cd}
    t = pd.DataFrame(rows).T
    t.to_csv(OUT_T / "friedman_nemenyi.csv")
    return t


def validation_relmase(ds: str) -> pd.Series:
    """relMASE of every main-context model on the VALIDATION origins (longest horizon), same floor as the test metric."""
    from src.data import load as load_panel
    from src.metrics import actuals

    name, res = ds.rsplit("_", 1)
    p = load_panel(name, int(res.replace("min", "")))
    out = {}
    for f in sorted((RESULTS / "forecasts" / ds).glob("*.npz")):
        c = re.search(r"_C(\d+)$", f.stem)
        if c and c.group(1) not in MAIN_C:
            continue
        z = np.load(f)
        o = z["origins"]
        val = ~p.is_test(o)
        y = actuals(p, o)[:, val]
        out[base(f.stem)] = np.abs(y - z["mean"][:, val]).mean(axis=(1, 2))
    m = pd.DataFrame(out, index=p.ids)
    m = m[m["seasonal_naive"] > 0]
    return np.exp(pd.DataFrame(log_ratio(m.to_numpy(), m[["seasonal_naive"]].to_numpy()), columns=m.columns).mean())


def paired_table() -> pd.DataFrame:
    """Best TSFM vs best baseline per dataset (both chosen by VALIDATION relMASE): Wilcoxon signed-rank on per-series floored
    log-ratios, paired bootstrap CI of the geometric-mean ratio, Holm correction over the six datasets."""
    from scipy.stats import wilcoxon

    rng = np.random.default_rng(SEED)
    rows = []
    for ds in DATASETS:
        lr = log_ratio_matrix(load(ds), load(ds).H.max())
        rel = validation_relmase(ds)  # pair chosen on validation, tested on test: no winner's-curse selection
        fm = rel[[m for m in rel.index if m in FM and m != "toto_s20" and m in lr]].idxmin()
        bl = rel[[m for m in rel.index if m not in FM and m != "seasonal_naive" and m in lr]].idxmin()
        d = (lr[fm] - lr[bl]).to_numpy()
        idx = rng.integers(0, len(d), size=(N_BOOT, len(d)))
        boot = np.exp(d[idx].mean(axis=1))
        rows.append({"dataset": ds, "tsfm": fm, "baseline": bl, "n": len(d), "ratio": float(np.exp(d.mean())),
                     "lo": float(np.percentile(boot, 2.5)), "hi": float(np.percentile(boot, 97.5)),
                     "p_wilcoxon": float(wilcoxon(d).pvalue)})
    t = pd.DataFrame(rows)
    order = t.p_wilcoxon.argsort().to_numpy()  # Holm step-down
    adj = np.empty(len(t))
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(t) - rank) * t.p_wilcoxon.iloc[i]))
        adj[i] = running
    t["p_holm"] = adj
    t.to_csv(OUT_T / "paired_tests.csv", index=False)
    return t


def floor_sensitivity_table() -> pd.DataFrame:
    """Best TSFM vs best baseline under alternative treatments of zero errors."""
    rows = []
    for ds in DATASETS:
        m = mae_matrix(load(ds), load(ds).H.max())
        ratio = m.div(m["seasonal_naive"], axis=0).drop(columns=["seasonal_naive"])
        fm = [c for c in ratio if c in FM and c != "toto_s20"]
        bl = [c for c in ratio if c not in FM]
        metrics = {f"floor {fl:g}": np.exp(np.log(np.maximum(ratio, fl)).mean()) for fl in (1e-9, 0.001, 0.01, 0.1)}
        metrics["ratio of sums"] = m.drop(columns=["seasonal_naive"]).sum() / m["seasonal_naive"].sum()
        metrics["median ratio"] = ratio.median()
        for name, v in metrics.items():
            rows.append({"dataset": ds, "metric": name, "tsfm": v[fm].idxmin(), "tsfm_value": v[fm].min(),
                         "baseline": v[bl].idxmin(), "baseline_value": v[bl].min()})
    t = pd.DataFrame(rows)
    t.to_csv(OUT_T / "floor_sensitivity.csv", index=False)
    return t


def sparse_split_table() -> pd.DataFrame:
    """Azure 2021: apps with >= 1 invocation in the test period (relMASE, rMAE_sum) vs apps silent in test
    (mean absolute over-forecast, invocations per step, since the actual is 0 throughout)."""
    from src.data import load as load_panel

    rows = []
    for ds, res in [("azure2021_1min", 1), ("azure2021_5min", 5)]:
        p = load_panel("azure2021", res)
        active = pd.Series(p.y[:, p.val_end:].sum(axis=1) > 0, index=p.ids)
        df = load(ds)
        h = df.H.max()
        m = df[df.H == h].pivot(index="series_id", columns="model", values="mae")
        act, idle = m[active.reindex(m.index).to_numpy()], m[~active.reindex(m.index).to_numpy()]
        act_def = act[act["seasonal_naive"] > 0]
        lr = pd.DataFrame(log_ratio(act_def.to_numpy(), act_def[["seasonal_naive"]].to_numpy()), columns=act_def.columns)
        for mdl in [x for x in ["timesfm3", "chronos2", "toto", "tirex", "lightgbm", "lstm", "nbeats", "naive", "tsb", "zero"] if x in m]:
            rows.append({"dataset": ds, "model": NAMES.get(mdl, "Selalu nol" if mdl == "zero" else mdl),
                         "n_active": len(act_def), "relMASE_active": np.exp(lr[mdl].mean()),
                         "rMAE_sum_active": act_def[mdl].sum() / act_def["seasonal_naive"].sum(),
                         "n_silent": len(idle), "overforecast_silent": idle[mdl].mean()})
    t = pd.DataFrame(rows)
    t.to_csv(OUT_T / "azure2021_active_silent.csv", index=False)
    return t


# ---------------- figures ----------------

def cd_diagram(ranks: pd.Series, cd: float, title: str, path) -> None:
    """Demšar-style critical-difference diagram; bars join models whose ranks differ by less than CD."""
    k = len(ranks)
    fig, ax = plt.subplots(figsize=(7, 0.9 + 0.17 * k))
    ax.set_xlim(k + 0.5, 0.5)  # rank 1 (best) on the right
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.plot([1, k], [0.92, 0.92], color=MUTED, lw=1)
    for r in range(1, k + 1):
        ax.plot([r, r], [0.92, 0.95], color=MUTED, lw=1)
        ax.text(r, 0.97, str(r), ha="center", va="bottom", color=MUTED, fontsize=7)
    half = (k + 1) // 2
    for i, (m, r) in enumerate(ranks.items()):
        right = i < half  # best half labelled on the right
        y = 0.85 - (i if right else k - 1 - i) * (0.8 / half)
        x_lab = 0.4 if right else k + 0.6
        ax.plot([r, r, x_lab], [0.92, y, y], color=COLOR[family(m)], lw=1)
        ax.text(x_lab, y, f"{NAMES[m]} ({r:.2f})", ha="left" if right else "right", va="center", fontsize=7, color=INK)
    # cliques: maximal runs of consecutive models within CD of the run's first member
    cliques, last_end = [], -1
    for i in range(k):
        j = max(jj for jj in range(i, k) if ranks.iloc[jj] - ranks.iloc[i] < cd)
        if j > i and j > last_end:
            cliques.append((i, j))
            last_end = j
    for n, (i, j) in enumerate(cliques):
        y = 0.88 - 0.035 * (n % 4)
        ax.plot([ranks.iloc[i] - 0.05, ranks.iloc[j] + 0.05], [y, y], color=INK, lw=2.5, solid_capstyle="round")
    ax.plot([k, k - cd], [0.06, 0.06], color=INK, lw=1)
    ax.text(k - cd / 2, 0.08, f"CD = {cd:.2f}", ha="center", va="bottom", fontsize=7, color=INK)
    ax.set_title(title, fontsize=8, color=INK, loc="left")
    save(fig, path)
    plt.close(fig)


def fig_cd() -> None:
    for ds in DATASETS:
        df = load(ds)
        h = df.H.max()
        ranks, cd, fr = nemenyi(mae_matrix(df, h))
        exp = str(round(fr["log10_p"])).translate(str.maketrans("-0123456789", "\u207b\u2070\u00b9\u00b2\u00b3\u2074\u2075\u2076\u2077\u2078\u2079"))
        title = f"{ds}, H = {h} {tr('steps')}  (Friedman \u03c7\u00b2 = {fr['chi2']:.0f}, p \u2248 10{exp}, n = {fr['n']}, k = {fr['k']})"
        cd_diagram(ranks, cd, title, OUT_F / f"cd_{ds}.pdf")


PARETO_OFFSETS = {"toto": (-5, 4), "chronos2": (-5, -9), "timesfm3": (5, -8), "timesfm25": (5, 2), "tirex": (5, 3),
                  "naive": (5, -8), "croston": (5, -8), "seasonal_naive": (5, 3)}


def fig_pareto() -> None:
    lat = pd.read_csv(RESULTS / "latency.csv")
    lat = lat[(lat.batch == 64) & (lat.context == 2048)].set_index("model").p50_ms
    acc = pd.concat([bootstrap(log_ratio_matrix(load(ds), load(ds).H.max())).rel for ds in DENSE], axis=1)
    acc = np.exp(np.log(acc).mean(axis=1)).dropna()  # geometric mean over the dense datasets
    acc = acc[acc.index.isin(lat.index) | acc.index.isin(TRAINED)]
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    for m, a in acc.items():
        if m not in lat.index:
            continue  # trained models: inference latency not benchmarked (see training-time table)
        x = max(lat[m], 0.01)  # naive models round to 0 ms; floor them onto the log axis
        ax.scatter(x, a, s=36, color=COLOR[family(m)], edgecolor="white", linewidth=1.2, zorder=3)
        ax.annotate(NAMES[m], (x, a), xytext=PARETO_OFFSETS.get(m, (4, 3)), textcoords="offset points", fontsize=6.5,
                    color=INK, ha="right" if PARETO_OFFSETS.get(m, (4, 3))[0] < 0 else "left")
    for m in TRAINED & set(acc.index):
        ax.axhline(acc[m], color=COLOR["trained"], lw=1, ls=":", zorder=1)
        ax.text(0.02, acc[m], f" {NAMES[m]}", transform=ax.get_yaxis_transform(),
                fontsize=6.5, color=MUTED, va="bottom")
    ax.set_xscale("log")
    ax.set_xlabel(tr("CPU latency p50, batch 64, C = 2048 (ms, log)"))
    ax.set_ylabel(tr("relMASE, geo-mean of 4 dense datasets (lower is better)"))
    handles = [plt.Line2D([], [], marker="o", ls="", color=COLOR[f], label=tr(l))
               for f, l in [("fm", "Foundation model"), ("stat", "Statistical"), ("trained", "Trained (dotted: accuracy only)")]]
    ax.legend(handles=handles, fontsize=6.5, frameon=False, loc="lower left")
    save(fig, OUT_F / "pareto_accuracy_latency.pdf")
    plt.close(fig)


def fig_tradeoff() -> None:
    panels = [("azure2019_1min", "cold_rate", "idle_per_demand", "Azure 2019 (serverless)", "cold-start proxy", "idle instance-min per demanded (proxy)"),
              ("azure2021_1min", "cold_rate", "idle_per_demand", "Azure 2021 (serverless, sparse)", "cold-start proxy", "idle instance-min per demanded (proxy)"),
              ("alibaba2018_5min", "under_frac", "excess_per_demand", "Alibaba 2018 (VM)", "fraction of intervals under-provisioned", "excess replicas per needed")]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5))
    for ax, (ds, x, y, title, xl, yl) in zip(axes, panels):
        s = pd.read_parquet(RESULTS / "sim" / f"{ds}.parquet")
        s["base"] = s.model.map(lambda m: base(m) if m not in ("reactive", "oracle") else m)
        s = s[~s.model.str.contains(r"_C\d+$") | s.model.str.extract(r"_C(\d+)$")[0].isin(MAIN_C)]
        acc = bootstrap(log_ratio_matrix(load(ds), load(ds).H.max())).rel
        best_fm = acc[[m for m in acc.index if m in FM]].idxmin()
        r = s[s.policy == "reactive"].sort_values(x)
        ax.plot(r[x], r[y], color=COLOR["ref"], marker="s", ms=4, label=tr("Reactive (knob)"))
        hh = s[s.policy == "reactive_histogram"]
        if len(hh):  # Shahrad et al. hybrid histogram: one operating point
            ax.scatter(hh[x], hh[y], color=COLOR["ref"], marker="D", s=40, facecolor="white", linewidth=1.5, zorder=4,
                       label=tr("Adapted hybrid histogram (no ARIMA)"))
        for m, col in [(best_fm, COLOR["fm"]), ("lightgbm", COLOR["trained"])]:
            for pol, ls, mk in [("predictive", "-", "o"), ("hybrid", "--", "^")]:
                g = s[(s.base == m) & (s.policy == pol)].sort_values(x)
                ax.plot(g[x], g[y], color=col, ls=ls, marker=mk, ms=4, label=f"{NAMES[m]}, {tr(pol)}")
        ax.set_xscale("log")
        ax.set_yscale("log")
        for axis in (ax.xaxis, ax.yaxis):
            axis.set_minor_formatter(matplotlib.ticker.NullFormatter())
            axis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.set_title(tr(title), fontsize=8, loc="left", color=INK)
        ax.set_xlabel(tr(xl), fontsize=7)
        ax.set_ylabel(tr(yl), fontsize=7)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, fontsize=6.5, frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, OUT_F / "tradeoff_autoscaling.pdf")
    plt.close(fig)


def fig_context() -> pd.DataFrame | None:
    rows = []
    for ds in ["huawei_5min", "azure2019_1min"]:
        df = pd.read_parquet(RESULTS / "metrics" / f"{ds}.parquet")
        h = df.H.max()
        m = df[df.H == h].pivot(index="series_id", columns="model", values="mae")
        m = m[m["seasonal_naive"] > 0]
        for col in m.columns:
            c = re.search(r"_C(\d+)$", col)
            if c and base(col) in {"chronos2", "timesfm3", "toto", "tirex"}:
                rows.append({"ds": ds, "model": base(col), "C": int(c.group(1)),
                             "rel": float(np.exp(log_ratio(m[col], m["seasonal_naive"]).mean()))})
    if not rows:
        return None
    t = pd.DataFrame(rows).sort_values(["ds", "model", "C"])
    t.to_csv(OUT_T / "context_ablation.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(6, 2.4))
    hues = {"chronos2": "#2a78d6", "timesfm3": "#eb6834", "toto": "#1baf7a", "tirex": "#eda100"}  # slots 1-4, adjacent-validated
    for ax, (ds, g) in zip(axes, t.groupby("ds")):
        for m, gm in g.groupby("model"):
            ax.plot(gm.C, gm.rel, marker="o", ms=4, color=hues[m])
            ax.annotate(NAMES[m], (gm.C.iloc[-1], gm.rel.iloc[-1]), xytext=(4, 0), textcoords="offset points", fontsize=6.5, va="center")
        ax.set_xscale("log", base=2)
        ax.set_title(ds, fontsize=8, loc="left")
        ax.set_xlabel(tr("context length C (steps, log2)"))
        ax.set_ylabel(tr("relMASE at H max"))
    fig.tight_layout()
    save(fig, OUT_F / "context_ablation.pdf")
    plt.close(fig)
    return t


def main() -> None:
    OUT_T.mkdir(parents=True, exist_ok=True)
    OUT_F.mkdir(parents=True, exist_ok=True)
    print(dataset_table().to_string(), "\n")
    print(accuracy_table().to_string(), "\n")
    print(stats_table().to_string(), "\n")
    print(sparse_split_table().round(3).to_string(index=False), "\n")
    print(paired_table().to_string(index=False), "\n")
    print(floor_sensitivity_table().round(3).to_string(index=False), "\n")
    fig_cd()
    fig_pareto()
    fig_tradeoff()
    t = fig_context()
    print(t.to_string(index=False) if t is not None else "no ablation runs yet")
    print(f"\nwrote {OUT_T} and {OUT_F}")


if __name__ == "__main__":
    main()
