"""Statistical baselines. Shared signature: fn(ctx [B, C] float, h, season) -> mean forecast [B, h]."""
import warnings

import numpy as np
from statsforecast.models import TSB, AutoARIMA, AutoETS, CrostonOptimized, MSTL, SeasonalWindowAverage


def zero(ctx: np.ndarray, h: int, season: int) -> np.ndarray:
    """Always forecasts 0: the reference point for applications that are almost never invoked."""
    return np.zeros((len(ctx), h))


def naive(ctx: np.ndarray, h: int, season: int) -> np.ndarray:
    return np.repeat(ctx[:, -1:], h, axis=1)


def seasonal_naive(ctx: np.ndarray, h: int, season: int) -> np.ndarray:
    return ctx[:, -season:][:, np.arange(h) % season]


def _per_series(make_model):
    """Wrap a statsforecast model as a batched fn. Failed fits fall back to seasonal naive, with a warning."""

    def fn(ctx: np.ndarray, h: int, season: int) -> np.ndarray:
        out = np.empty((len(ctx), h), np.float32)
        failed = 0
        for i, y in enumerate(ctx.astype(np.float64)):
            try:
                out[i] = make_model(season).forecast(y=y, h=h)["mean"]
            except Exception:  # constant/all-zero series break some fits
                out[i] = seasonal_naive(y[None], h, season)[0]
                failed += 1
        if failed:
            warnings.warn(f"{failed}/{len(ctx)} fits fell back to seasonal naive")
        return out

    return fn


swa3 = _per_series(lambda s: SeasonalWindowAverage(season_length=s, window_size=3))
ets = _per_series(lambda s: AutoETS(model="ZZN"))
mstl_ets = _per_series(lambda s: MSTL(season_length=s, trend_forecaster=AutoETS(model="ZZN")))
mstl_arima = _per_series(lambda s: MSTL(season_length=s, trend_forecaster=AutoARIMA()))
croston = _per_series(lambda s: CrostonOptimized())
tsb = _per_series(lambda s: TSB(alpha_d=0.1, alpha_p=0.1))

STAT_CONTEXT_DAYS = 7
# name -> (fn, context length in days, resolutions it runs at). ponytail: MSTL is 4-9 s/fit at 1-min, so 5-min only.
BASELINES = {
    "zero": (zero, 1, (1, 5)),
    "naive": (naive, STAT_CONTEXT_DAYS, (1, 5)),
    "seasonal_naive": (seasonal_naive, STAT_CONTEXT_DAYS, (1, 5)),
    "swa3": (swa3, 3, (1, 5)),
    "ets": (ets, STAT_CONTEXT_DAYS, (1, 5)),
    "mstl_ets": (mstl_ets, STAT_CONTEXT_DAYS, (5,)),
    "mstl_arima": (mstl_arima, STAT_CONTEXT_DAYS, (5,)),
    "croston": (croston, STAT_CONTEXT_DAYS, (1, 5)),
    "tsb": (tsb, STAT_CONTEXT_DAYS, (1, 5)),
}
