import numpy as np
import pandas as pd

from src.baselines import naive, seasonal_naive
from src.data import Panel, resample
from src.metrics import QUANTILES, conformal, pinball, summarize


def panel(y, train_end, val_end, res=5):
    return Panel("t", res, np.array([f"s{i}" for i in range(len(y))]), np.asarray(y, np.float32), train_end, val_end)


def test_seasonal_naive_repeats_last_season():
    ctx = np.array([[1, 2, 3, 4, 5, 6]], float)
    assert seasonal_naive(ctx, 5, 3).tolist() == [[4, 5, 6, 4, 5]]
    assert naive(ctx, 2, 3).tolist() == [[6, 6]]


def test_resample_sum_and_mean_drop_partial_block():
    y = np.arange(7, dtype=float)[None]
    assert resample(y, 3, "sum").tolist() == [[3, 12]]
    assert resample(y, 3, "mean").tolist() == [[1, 4]]


def test_origins_are_hourly_and_split_at_val_end():
    p = panel(np.zeros((1, 288 * 3)), train_end=288, val_end=576)  # 5-min, 3 days
    o = p.origins()
    assert o[0] == 288 and np.all(np.diff(o) == 12) and o[-1] == 288 * 3 - 12
    assert p.is_test(o).sum() == 24  # one day of hourly test origins


def test_conformal_quantiles_are_ordered_and_nonnegative():
    rng = np.random.default_rng(0)
    y = rng.poisson(5, (3, 10, 4)).astype(float)
    q = conformal(np.full_like(y, 5.0), y, np.arange(10) < 5, np.ones(3))
    assert q.shape == (3, 10, 4, len(QUANTILES)) and (q >= 0).all() and (np.diff(q, axis=-1) >= 0).all()


def test_pinball_is_zero_for_perfect_quantiles():
    y = np.array([2.0])
    assert pinball(y, np.full((1, len(QUANTILES)), 2.0))[0] == 0


def test_reference_has_relmase_one():
    df = pd.DataFrame({
        "model": ["seasonal_naive", "seasonal_naive", "m", "m"], "series_id": ["a", "b", "a", "b"], "H": 1,
        "mae": [2.0, 4.0, 1.0, 8.0], "mase": 1.0, "wql_num": 1.0, "abs_sum": 1.0, "cov90": 0.9,
    })
    s = summarize(df)
    assert np.isclose(s.loc[("seasonal_naive", 1), "relMASE"], 1)
    assert np.isclose(s.loc[("m", 1), "relMASE"], 1)  # geomean(0.5, 2) = 1


def test_keep_alive_holds_peak_for_k_minutes():
    from src.simulate import keep_alive, serverless_cost

    need = np.array([[0, 2, 0, 0, 0, 1, 0]], float)
    ka = keep_alive(need, start=3, k=2)  # warm at t=3..6 = max(need[t-2:t])
    assert ka.tolist() == [[2, 0, 0, 1]]
    c = serverless_cost(need[:, 3:], ka)
    # demand [0, 0, 1, 0]: the call at t=5 finds nothing warm (cold); 3 idle instance-minutes per 1 demanded
    assert c["cold_rate"] == 1 and c["idle_per_demand"] == 3


def test_instances_little_law_and_floor_of_one():
    from src.simulate import instances

    assert instances(np.array([[0.0, 1.0, 120.0]]), np.array([30.0])).tolist() == [[0, 1, 60]]


def test_histogram_windows_periodic_app_gets_prewarm():
    from src.simulate import histogram_windows

    active = np.zeros(2000, bool)
    active[::30] = True  # invoked exactly every 30 minutes
    pre, keep = histogram_windows(active)
    assert pre == 27 and keep == 33 - 27  # 0.9*30 and ceil(1.1*30) - pre


def test_hybrid_histogram_equals_keep_alive_without_prewarm():
    from src.simulate import hybrid_histogram, keep_alive

    rng = np.random.default_rng(0)
    need = (rng.random((3, 400)) < 0.02).astype(float)  # sparse, non-representative -> 4 h keep-alive
    assert np.array_equal(hybrid_histogram(need, 300), keep_alive(need, 300, 240))


def test_pareto_front_drops_dominated_configs():
    from src.simulate import pareto_front

    cands = [("a", 0.1, 1.0), ("b", 0.2, 0.5), ("c", 0.2, 0.9), ("d", 0.05, 2.0)]  # c is dominated by b
    assert [c[0] for c in pareto_front(cands)] == ["a", "b", "d"]


def test_zero_mae_cannot_dominate_geometric_mean():
    # one series with MAE 0 used to pull the geometric mean towards 0 (epsilon 1e-9); the 1% floor bounds it
    df = pd.DataFrame({
        "model": ["seasonal_naive"] * 2 + ["m"] * 2, "series_id": ["a", "b"] * 2, "H": 1,
        "mae": [1.0, 1.0, 0.0, 1.0], "mase": 1.0, "wql_num": 1.0, "abs_sum": 1.0, "cov90": 0.9,
    })
    s = summarize(df).loc[("m", 1)]
    assert np.isclose(s.relMASE, np.sqrt(0.01 * 1.0))  # geomean(max(0, 0.01), 1) = 0.1
    assert np.isclose(s.rMAE_sum, 0.5)
