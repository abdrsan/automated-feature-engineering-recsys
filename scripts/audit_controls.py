import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
import psutil

from src.data import load_movielens, save_json
from src.datasets import load_kion, load_yambda
from src.feature_engine import generate_features
from src.feature_spec import FeatureSpec
from src.metrics import evaluate
from src.model import base_matrix
from src.split import temporal_split


def audit_window_controls(df, split, root):
    root = Path(root)
    # Источник каждого control выбираем по search, до чтения его audit score.
    lock = json.loads((root / "locked_selection.json").read_text())
    manifest = json.loads((root / "run_manifest.json").read_text())
    rows = pl.read_csv(root / "experiments.csv")
    source_rows = rows.filter((pl.col("split") == "validation_search") & (pl.col("seed") == 42))
    threshold = manifest["auto_config"]["keep_min_logloss_gain"]
    names = []
    for prefix in ["auto_fixed_windows", "auto_adaptive"]:
        pool = source_rows.filter(pl.col("experiment_name").str.starts_with(prefix))
        best = pool["logloss"].min()
        selected = pool.filter(pl.col("logloss") <= best + threshold).sort("n_features").row(0, named=True)
        names.append(selected["experiment_name"])
    names.append("shared_windows_control")
    save_json(root / "audit_controls_lock.json", {"sources": names, "selection_data": "validation_search",
                                                 "not_used_to_change_main_lock": True})
    # Для этой проверки достаточно development-префикса; test сюда не передаем.
    dev = df.head(split.validation_end)
    all_specs = {spec.name: spec for spec in [FeatureSpec.from_dict(s) for s in lock["all_specs"]]}
    configs = {name: json.loads((root / "specs" / f"{name}_seed42.json").read_text()) for name in names}
    needed = list(dict.fromkeys(n for config in configs.values() for n in config["features"]))
    features, profile = generate_features(dev, [all_specs[n] for n in needed])
    base, _, _ = base_matrix(dev, json.loads((root / "encoder_development.json").read_text()))
    results = []
    for name, config in configs.items():
        booster = lgb.Booster(model_file=str(root / "models" / f"{name}_seed42.txt"))
        x = np.column_stack([base[split.audit], features.select(config["features"])[split.audit].to_numpy()])
        pred = booster.predict(x, num_threads=manifest["config"]["threads"])
        results.append({"experiment_name": name, "seed": 42, "split": "validation_audit_control",
                        "n_features": len(config["base_names"]) + len(config["features"]),
                        "training_time": 0., "feature_generation_time": profile["seconds"] if not results else 0.,
                        "cache_hit": bool(results), "memory_usage": psutil.Process().memory_info().rss / 2**20,
                        "best_iteration": config["iterations"], "n_train": split.train_end,
                        **evaluate(dev[split.audit], pred)})
    pl.DataFrame(results).write_csv(root / "audit_controls.csv")
    print(pl.DataFrame(results).select("experiment_name", "logloss", "auc", "n_features"))


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="artifacts")
    parser.add_argument("--user-modulus", type=int, default=1)
    args = parser.parse_args()
    root = Path(args.output)
    manifest = json.loads((root / "run_manifest.json").read_text())
    name = manifest["config"].get("dataset_name", "movielens")
    df = load_kion(user_modulus=args.user_modulus) if name == "kion" else load_yambda() if name == "yambda" else load_movielens()
    df = df.head(manifest["split"]["n_rows"])
    split = temporal_split(df, tuple(manifest["config"]["split_fractions"]))
    audit_window_controls(df, split, root)


if __name__ == "__main__":
    main()
