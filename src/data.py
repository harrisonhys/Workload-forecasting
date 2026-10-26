"""Processed parquet -> dense [N, T] panel at a chosen resolution, plus rolling origins (SPEC §3)."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data/processed"
RESULTS = ROOT / "results"


@dataclass(frozen=True)
class DatasetCfg:
    native_min: int  # native resolution (minutes)
    agg: str  # resample: "sum" for counts, "mean" for utilisation
    train_days: int  # train = [0, train_days)
    val_days: int  # val = [train_days, val_days), test = [val_days, end)


DATASETS = {
    "azure2019": DatasetCfg(1, "sum", 10, 12),
    "azure2021": DatasetCfg(1, "sum", 10, 12),
    "alibaba2018": DatasetCfg(5, "mean", 5, 6),
    "huawei": DatasetCfg(5, "sum", 24, 26),
    "google2019": DatasetCfg(5, "sum", 23, 26),
}
HORIZONS = {1: [1, 5, 15, 60], 5: [1, 3, 12]}  # evaluated horizons in steps, keyed by resolution (min)
STRIDE_MIN = 60  # one origin per hour


@dataclass(frozen=True)
class Panel:
    ds: str
    res: int  # minutes per step
    ids: np.ndarray  # [N] str
    y: np.ndarray  # [N, T] float32
    train_end: int
    val_end: int

    @property
    def season(self) -> int:
        return 1440 // self.res

    @property
    def h_max(self) -> int:
        return max(HORIZONS[self.res])

    def origins(self) -> np.ndarray:
        """Val + test origins: forecast y[:, o:o+h] from y[:, :o]."""
        return np.arange(self.train_end, self.y.shape[1] - self.h_max + 1, STRIDE_MIN // self.res)

    def is_test(self, origins: np.ndarray) -> np.ndarray:
        return origins >= self.val_end


def resample(y: np.ndarray, k: int, agg: str) -> np.ndarray:
    """Aggregate k consecutive steps; a trailing partial block is dropped."""
    if k == 1:
        return y
    blocks = y[:, : y.shape[1] // k * k].reshape(len(y), -1, k)
    return blocks.sum(axis=2) if agg == "sum" else blocks.mean(axis=2)


def load(ds: str, res: int) -> Panel:
    cfg = DATASETS[ds]
    if res % cfg.native_min:
        raise ValueError(f"{ds} is native {cfg.native_min}-min; cannot load at {res}-min")
    wide = pd.read_parquet(PROCESSED / f"{ds}.parquet").pivot(index="series_id", columns="ts", values="y")
    y = resample(wide.to_numpy(np.float32), res // cfg.native_min, cfg.agg)
    spd = 1440 // res
    return Panel(ds, res, np.asarray(wide.index.astype(str), dtype=str), y, cfg.train_days * spd, cfg.val_days * spd)


def meta(ds: str) -> pd.DataFrame:
    m = pd.read_parquet(PROCESSED / f"{ds}_meta.parquet")
    m.index = m.index.astype(str)
    return m
