"""Trained global baselines (SPEC §4.3): one model per (dataset, resolution), fit on train, early-stopped on val.

Usage: uv run --project envs/train python -m src.trained <ds> <res> <lightgbm|nbeats|lstm>
Output: results/forecasts/<ds>_<res>min/<model>.npz with mean [N, O, H], seconds (inference), train_seconds.
"""
import sys
import time

import numpy as np

from src.data import RESULTS, Panel, load

SEED = 42
SAMPLES_PER_SERIES = 2000  # LightGBM training rows per series and split


# ---------- LightGBM: recursive one-step model on scale-normalised lag features ----------

def lgb_lags(season: int) -> list[int]:
    return list(range(1, 16)) + [30, 60, season - 1, season, season + 1, 2 * season]


def lgb_features(win: np.ndarray, season: int) -> np.ndarray:
    """win [B, L] (already normalised), predicting the step right after it -> [B, F]."""
    lags = np.stack([win[:, -l] for l in lgb_lags(season)], axis=1)
    rolls = np.stack([win[:, -k:].mean(axis=1) for k in (15, 60, 240)], axis=1)
    yday = win[:, -season - 5 : -season + 6].mean(axis=1, keepdims=True)  # same time yesterday, ±5 steps
    return np.concatenate([lags, rolls, yday], axis=1)


def lgb_scale(win: np.ndarray, season: int) -> np.ndarray:
    return np.abs(win[:, -season:]).mean(axis=1, keepdims=True) + 1e-3


def lgb_rows(y: np.ndarray, lo: int, hi: int, season: int, rng) -> tuple[np.ndarray, np.ndarray]:
    """Sample (features, target) at random positions t in [lo, hi) of every series."""
    need = 2 * season + 1
    t = rng.integers(max(lo, need), hi, size=(len(y), SAMPLES_PER_SERIES))
    rows = np.repeat(np.arange(len(y)), SAMPLES_PER_SERIES)
    win = y[rows[:, None], t.reshape(-1, 1) - need + np.arange(need)]  # [B, need]
    s = lgb_scale(win, season)
    return lgb_features(win / s, season), y[rows, t.ravel()] / s[:, 0]


def lightgbm(p: Panel):
    import lightgbm as lgb

    rng = np.random.default_rng(SEED)
    xtr, ytr = lgb_rows(p.y, 0, p.train_end, p.season, rng)
    xva, yva = lgb_rows(p.y, p.train_end, p.val_end, p.season, rng)
    model = lgb.LGBMRegressor(objective="l1", n_estimators=2000, learning_rate=0.05, num_leaves=63,
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=SEED, verbose=-1)
    model.fit(xtr, ytr, eval_set=[(xva, yva)], callbacks=[lgb.early_stopping(50, verbose=False)])
    need = 2 * p.season + 1

    def predict(ctx: np.ndarray, h: int) -> np.ndarray:
        s = lgb_scale(ctx, p.season)
        win = ctx[:, -need:] / s
        out = np.empty((len(ctx), h), np.float32)
        for k in range(h):  # recursive: feed each prediction back as the newest lag
            out[:, k] = model.predict(lgb_features(win, p.season))
            win = np.concatenate([win[:, 1:], out[:, k : k + 1]], axis=1)
        return out * s

    return predict, need


# ---------- N-BEATS / LSTM via neuralforecast ----------

def to_long(y: np.ndarray, ids: list[str]):
    import pandas as pd

    n, t = y.shape
    return pd.DataFrame({"unique_id": np.repeat(ids, t), "ds": np.tile(np.arange(t), n), "y": y.ravel()})


def neural(p: Panel, kind: str, scaler: str | None = None, input_size: int | None = None):
    from neuralforecast import NeuralForecast
    from neuralforecast.losses.pytorch import MAE
    from neuralforecast.models import LSTM, NBEATS

    input_size = input_size or min(1440, 3 * p.season)
    # N-BEATS: "robust" scaling divides by a ~0 IQR on sparse series (val relMASE 1.40 vs 0.69 with "standard" on
    # azure2019 5-min validation, 2026-10-25). Chosen on validation only, then applied to every dataset.
    scaler = scaler or ("standard" if kind == "nbeats" else "robust")
    common = dict(h=p.h_max, input_size=input_size, loss=MAE(), max_steps=1000, val_check_steps=100,
                  early_stop_patience_steps=5, scaler_type=scaler, random_seed=SEED, batch_size=64,
                  enable_progress_bar=False, logger=False, accelerator="cpu")  # Lightning silently picks Apple MPS otherwise
    if kind == "lstm":  # ponytail: recurrence over 864+ steps x 1024 windows took 3.5 h/1000 steps on CPU
        input_size = min(input_size, 2 * p.h_max)  # short window (2 h), as in classic LSTM autoscaling baselines
        common.update(input_size=input_size, windows_batch_size=256, max_steps=500)
    model = NBEATS(**common) if kind == "nbeats" else LSTM(encoder_hidden_size=128, **common)
    nf = NeuralForecast(models=[model], freq=1)
    nf.fit(to_long(p.y[:, : p.val_end], list(p.ids)), val_size=p.val_end - p.train_end)
    col = type(model).__name__

    def predict(ctx: np.ndarray, h: int) -> np.ndarray:
        ids = [str(i) for i in range(len(ctx))]
        fc = nf.predict(df=to_long(ctx[:, -input_size:], ids))
        fc["unique_id"] = fc.unique_id.astype(int)
        return fc.sort_values(["unique_id", "ds"])[col].to_numpy(np.float32).reshape(len(ctx), -1)[:, :h]

    return predict, input_size


def main(ds: str, res: int, kind: str) -> None:
    p = load(ds, res)
    out = RESULTS / "forecasts" / f"{ds}_{res}min" / f"{kind}.npz"
    if out.exists():
        print(f"skip {out.name} (exists)")
        return
    t0 = time.perf_counter()
    predict, c = lightgbm(p) if kind == "lightgbm" else neural(p, kind)
    train_s = time.perf_counter() - t0
    origins = p.origins()
    t0 = time.perf_counter()
    batch = 8  # origins per predict call: neuralforecast overhead is per call
    chunks = []
    for i in range(0, len(origins), batch):
        os_ = origins[i : i + batch]
        ctx = np.concatenate([p.y[:, o - c : o] for o in os_])  # [len(os_) * N, c], origin-major
        chunks.append(predict(ctx, p.h_max).reshape(len(os_), len(p.ids), -1))
    mean = np.concatenate(chunks).transpose(1, 0, 2)  # [N, O, H]
    secs = time.perf_counter() - t0
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, ids=p.ids, origins=origins, mean=np.maximum(mean, 0), seconds=secs, train_seconds=train_s)
    print(f"{ds} {res}min {kind}: train {train_s:.0f}s, predict {secs:.0f}s, {mean.shape}")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), sys.argv[3])
