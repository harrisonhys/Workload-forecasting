"""Zero-shot foundation-model wrappers (SPEC §4).

Each factory loads one pinned checkpoint and returns predict(ctx [B, C] float32, h) -> (mean [B, h], q [B, h, 9] | None).
Imports are inside the factories: each model family lives in its own venv (envs/fm, envs/toto, envs/moirai).
"""
import numpy as np
import torch

QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def chronos(repo: str, rev: str):
    from chronos import BaseChronosPipeline

    pipe = BaseChronosPipeline.from_pretrained(repo, revision=rev, device_map="cpu", torch_dtype=torch.float32)

    is_v2 = type(pipe).__name__ == "Chronos2Pipeline"  # wants [series, variates, time], returns per-series lists

    def predict(ctx, h):
        x = torch.from_numpy(ctx[:, None, :] if is_v2 else ctx)
        q, mean = pipe.predict_quantiles(x, prediction_length=h, quantile_levels=QUANTILES)
        if is_v2:
            q, mean = torch.stack(q)[:, 0], torch.stack(mean)[:, 0]
        return mean.numpy(), q.numpy()

    return predict


def timesfm25(repo: str, rev: str, max_context: int, h_max: int = 128):
    import timesfm

    m = timesfm.TimesFM_2p5_200M_torch.from_pretrained(repo, revision=rev)
    m.compile(timesfm.ForecastConfig(
        max_context=max_context, max_horizon=max(128, -(-h_max // 128) * 128), normalize_inputs=True, per_core_batch_size=64,
        use_continuous_quantile_head=True, fix_quantile_crossing=True, infer_is_positive=True,
    ))

    def predict(ctx, h):
        point, quant = m.forecast(horizon=h, inputs=list(ctx))
        return point, quant[..., 1:]  # quant[..., 0] is the mean, 1..9 are deciles

    return predict


def timesfm3(repo: str, rev: str):
    import timesfm

    m = timesfm.TimesFM3Forecaster.from_pretrained(repo, device="cpu", revision=rev)

    def predict(ctx, h):
        outs = list(m.predict_batch(list(ctx), horizon=h, return_quantiles=True, make_positive=True))
        return np.stack([o.forecast for o in outs]), np.stack([o.quantiles for o in outs])

    return predict


def tirex(repo: str, rev: str):
    from tirex import load_model

    m = load_model(repo, device="cpu", backend="torch", hf_kwargs={"revision": rev})

    def predict(ctx, h):
        q, mean = m.forecast(context=torch.from_numpy(ctx), prediction_length=h)
        return mean.numpy(), q.numpy()

    return predict


def ttm(repo: str, rev: str, context: int, h_max: int):
    """Point forecasts only; quantiles come from split conformal in metrics.py."""
    from tsfm_public.toolkit.get_model import get_model

    # TTM keeps one branch per (context, horizon) variant; get_model picks it, so `rev` (main) is not passed.
    m = get_model(repo, context_length=context, prediction_length=h_max).eval()
    cl = m.config.context_length

    def predict(ctx, h):
        with torch.no_grad():
            out = m(past_values=torch.from_numpy(ctx[:, -cl:, None])).prediction_outputs
        return out[:, :h, 0].numpy(), None

    return predict


def toto(repo: str, rev: str, res_min: int, num_samples: int | None = None):
    """num_samples=None: analytic mean only (0.6 s/origin/50 series); 20 samples is ~30x slower on CPU."""
    from toto.data.util.dataset import MaskedTimeseries
    from toto.inference.forecaster import TotoForecaster
    from toto.model.toto import Toto

    f = TotoForecaster(Toto.from_pretrained(repo, revision=rev).model.eval())

    def predict(ctx, h):
        x = torch.from_numpy(ctx)[:, None, :]  # independent univariate series: [B, variates=1, C]
        b = len(x)
        inp = MaskedTimeseries(
            series=x, padding_mask=torch.ones_like(x, dtype=torch.bool), id_mask=torch.zeros_like(x, dtype=torch.int),  # full-length: broadcast (b,1,1) breaks short contexts
            timestamp_seconds=torch.zeros_like(x, dtype=torch.int), time_interval_seconds=torch.full((b, 1), res_min * 60),
        )
        with torch.no_grad():
            fc = f.forecast(inp, prediction_length=h, num_samples=num_samples, samples_per_batch=10)  # 100 at once OOMs on 18 GB
        if num_samples is None:
            return fc.mean[:, 0].numpy(), None
        s = fc.samples[:, 0]  # [B, h, S]
        return s.mean(-1).numpy(), torch.quantile(s, torch.tensor(QUANTILES), dim=-1).permute(1, 2, 0).numpy()

    return predict


def moirai2(repo: str, rev: str, context: int, h_max: int):
    from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module

    m = Moirai2Forecast(
        module=Moirai2Module.from_pretrained(repo, revision=rev), prediction_length=h_max, context_length=context,
        target_dim=1, feat_dynamic_real_dim=0, past_feat_dynamic_real_dim=0,
    )
    levels = list(m.module.quantile_levels)
    idx = [levels.index(q) for q in QUANTILES]

    def predict(ctx, h):
        q = m.predict([c[:, None] for c in ctx])  # [B, Q_model, h_max]
        q = q[:, idx, :h].transpose(0, 2, 1)
        return q[..., QUANTILES.index(0.5)], q  # ponytail: median as point forecast (no mean head)

    return predict


R = {  # pinned Hugging Face revisions (2026-10-22)
    "amazon/chronos-bolt-small": "772f3d25d38aec6d914c8949dab4462e2d46f5d8",
    "amazon/chronos-2": "29ec3766d36d6f73f0696f85560a422f50e8498c",
    "google/timesfm-2.5-200m-pytorch": "1d952420fba87f3c6dee4f240de0f1a0fbc790e3",
    "google/timesfm-3.0-pytorch": "43046b85ec22d584a13f8098c2ed39c889e129c2",
    "Salesforce/moirai-2.0-R-small": "30f43ff08c8494f4943ae1521e9d4e94a0fbb389",
    "Datadog/Toto-Open-Base-1.0": "0411ceb27bdf7fc3e4892e99edc8ad08192dc3c5",
    "NX-AI/TiRex": "63c740922493f5fbe60b277609ec62babfba2762",
    "ibm-granite/granite-timeseries-ttm-r2": "d6a79570cac0f33d526601cd3a0fc7c80a8f9a2f",
}

# name -> (venv, max context, factory(C, h_max, res_min))
FMS = {
    "chronos_bolt_small": ("fm", 2048, lambda c, h, r: chronos("amazon/chronos-bolt-small", R["amazon/chronos-bolt-small"])),
    "chronos2": ("fm", 8192, lambda c, h, r: chronos("amazon/chronos-2", R["amazon/chronos-2"])),
    "timesfm25": ("fm", 16384, lambda c, h, r: timesfm25("google/timesfm-2.5-200m-pytorch", R["google/timesfm-2.5-200m-pytorch"], c, h)),
    "timesfm3": ("fm", 16384, lambda c, h, r: timesfm3("google/timesfm-3.0-pytorch", R["google/timesfm-3.0-pytorch"])),
    "tirex": ("fm", 16384, lambda c, h, r: tirex("NX-AI/TiRex", R["NX-AI/TiRex"])),
    "ttm_r2": ("fm", 1536, lambda c, h, r: ttm("ibm-granite/granite-timeseries-ttm-r2", R["ibm-granite/granite-timeseries-ttm-r2"], c, h)),
    "toto": ("toto", 4096, lambda c, h, r: toto("Datadog/Toto-Open-Base-1.0", R["Datadog/Toto-Open-Base-1.0"], r)),
    "toto_s20": ("toto", 4096, lambda c, h, r: toto("Datadog/Toto-Open-Base-1.0", R["Datadog/Toto-Open-Base-1.0"], r, 20)),
    "moirai2_small": ("moirai", 4096, lambda c, h, r: moirai2("Salesforce/moirai-2.0-R-small", R["Salesforce/moirai-2.0-R-small"], c, h)),
}
