"""Dump everything needed to re-run the paper's experiments into configs/ (small, committed files).

Usage: uv run python scripts/export_configs.py
Writes configs/splits.json (per dataset x resolution: steps, train/val/test boundaries, origins, horizons, sha256 of the
processed parquet), configs/series/<ds>_<res>min.txt (series ids in panel order) and configs/models.json (pinned
checkpoints, baseline settings). Checkpoint revisions are parsed from src/fm.py so torch is not needed here.
"""
import ast
import hashlib
import json

from src.baselines import BASELINES
from src.data import DATASETS, HORIZONS, PROCESSED, ROOT, STRIDE_MIN, load
from src.metrics import QUANTILES, RATIO_FLOOR, REFERENCE

RUNS = [("huawei", 5), ("alibaba2018", 5), ("azure2019", 5), ("azure2019", 1), ("azure2021", 5), ("azure2021", 1)]
OUT = ROOT / "configs"


def sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fm_revisions() -> dict:
    tree = ast.parse((ROOT / "src/fm.py").read_text())
    node = next(n for n in tree.body if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", None) == "R")
    return ast.literal_eval(node.value)


def main() -> None:
    (OUT / "series").mkdir(parents=True, exist_ok=True)
    splits = {}
    for ds, res in RUNS:
        p, cfg = load(ds, res), DATASETS[ds]
        o = p.origins()
        key = f"{ds}_{res}min"
        splits[key] = {
            "n_series": len(p.ids), "n_steps": int(p.y.shape[1]), "step_minutes": res,
            "train_days": cfg.train_days, "val_days": cfg.val_days - cfg.train_days,
            "test_days": round(p.y.shape[1] * res / 1440) - cfg.val_days,
            "train_end_step": p.train_end, "val_end_step": p.val_end,
            "origins_val": int((~p.is_test(o)).sum()), "origins_test": int(p.is_test(o).sum()),
            "origin_stride_minutes": STRIDE_MIN, "horizons_steps": HORIZONS[res],
            "processed_parquet": f"data/processed/{ds}.parquet", "sha256": sha256(PROCESSED / f"{ds}.parquet"),
        }
        (OUT / "series" / f"{key}.txt").write_text("\n".join(map(str, p.ids)) + "\n")
    (OUT / "splits.json").write_text(json.dumps(splits, indent=2) + "\n")
    models = {
        "foundation_model_revisions": fm_revisions(),
        "baselines": {k: {"context_days": d, "resolutions_min": list(r)} for k, (_, d, r) in BASELINES.items()},
        "trained": "see src/trained.py: LightGBM recursive, N-BEATS (standard scaler), LSTM (robust scaler, input 2*h_max)",
        "metrics": {"reference": REFERENCE, "ratio_floor": RATIO_FLOOR, "quantiles": QUANTILES.tolist()},
    }
    (OUT / "models.json").write_text(json.dumps(models, indent=2) + "\n")
    print(f"wrote {OUT}: {len(splits)} splits")


if __name__ == "__main__":
    main()
