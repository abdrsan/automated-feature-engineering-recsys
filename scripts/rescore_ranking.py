import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

from src.data import save_json
from src.datasets import load_kion, load_yambda
from src.metrics import ranking_metrics
from src.model import base_matrix, matrix_from_features
from src.representations import temporal_svd_features
from src.split import temporal_split


def clean_query_rows(df):
    keys = ["user_id", "timestamp"]
    groups = df.group_by(keys).agg(pl.len().alias("n"), pl.col("item_id").n_unique().alias("items"),
                                   pl.col("label").sum().alias("p"))
    groups = groups.filter((pl.col("n") == pl.col("items")) & (pl.col("n") >= 2) &
                            (pl.col("p") > 0) & (pl.col("p") < pl.col("n")))
    return df.join(groups.select(keys), on=keys, how="inner").sort("row_id")


def rescore(name):
    root = Path("artifacts") / name
    manifest = json.loads((root / "run_manifest.json").read_text())
    lock = json.loads((root / "locked_selection.json").read_text())
    final = json.loads((root / "final_results.json").read_text())
    df = load_kion() if name == "kion" else load_yambda()
    split = temporal_split(df, tuple(manifest["config"]["split_fractions"]))
    rows = pl.read_csv(root / "experiments.csv")
    if (root / "audit_controls.csv").exists():
        rows = pl.concat([rows.filter(pl.col("split") != "validation_audit_control"),
                          pl.read_csv(root / "audit_controls.csv")], how="diagonal_relaxed")
    parts = {"validation_search": df[split.search], "validation_audit": df[split.audit],
             "validation_audit_control": df[split.audit], "test": df[split.test]}
    queries = {part: clean_query_rows(value) for part, value in parts.items()}
    query_ids = pl.concat([value.select("row_id") for value in queries.values()]).unique().sort("row_id")
    features = query_ids
    # Сначала берем итоговые признаки, рассчитанные и для test.
    # Из development cache добавляем только недостающие колонки для ранних экспериментов.
    paths = sorted((root / "cache").glob("*.parquet"), key=lambda p: (not p.name.startswith("final_features"), p.name))
    for path in paths:
        schema = pl.read_parquet_schema(path)
        names = [c for c in schema if c not in features.columns]
        if names:
            block = pl.scan_parquet(path).select("row_id", *names).filter(
                pl.col("row_id").is_in(query_ids["row_id"].implode())).collect()
            features = features.join(block, on="row_id", how="left", validate="1:1")
    if lock["warmup_end"] is not None:
        embedding, _ = temporal_svd_features(df, lock["warmup_end"])
        features = features.join(embedding, on="row_id", how="left", validate="1:1")
    encoders = {"dev": json.loads((root / "encoder_development.json").read_text()),
                "test": json.loads((root / "encoder_final.json").read_text())}
    primary_before = rows.select("experiment_name", "seed", "split", "auc", "logloss").to_dicts()
    output = []
    for row in rows.to_dicts():
        part = row["split"]
        if part == "test":
            key = f"final_{row['experiment_name']}_seed{row['seed']}"
        else:
            source = row.get("source_experiment") or row["experiment_name"]
            key = f"{source}_seed{row['seed']}"
        query = queries[part]
        booster = lgb.Booster(model_file=str(root / "models" / f"{key}.txt"))
        base, base_names, _ = base_matrix(query, encoders["test" if part == "test" else "dev"])
        names = booster.feature_name()[len(base_names):]
        selected = query.select("row_id").join(features, on="row_id", how="left", validate="1:1")
        matrix = matrix_from_features(base, selected, names)
        prediction = booster.predict(matrix, num_threads=manifest["config"]["threads"])
        metrics = ranking_metrics(query, prediction)
        # После фильтрации query счетчик дублей был бы нулевым.
        # Поэтому число исключенных групп считаем на полной исходной части выборки.
        full = parts[part].group_by("user_id", "timestamp").agg(pl.len().alias("n"),
            pl.col("item_id").n_unique().alias("items"), pl.col("label").sum().alias("p"))
        metrics["ranking_groups_excluded_duplicates"] = full.filter(
            (pl.col("n") > pl.col("items")) & (pl.col("p") > 0) & (pl.col("p") < pl.col("n"))).height
        output.append({**row, **metrics})
    updated = pl.DataFrame(output, infer_schema_length=None)
    assert updated.select("experiment_name", "seed", "split", "auc", "logloss").to_dicts() == primary_before
    updated.write_csv(root / "experiments.csv")
    updated.filter(pl.col("split") == "validation_audit_control").write_csv(root / "audit_controls.csv")
    lookup = {(row["experiment_name"], row["seed"], row["split"]): row for row in output}
    final["rows"] = [lookup[(row["experiment_name"], row["seed"], row["split"])] for row in final["rows"]]
    final["ranking_protocol"] = "user x timestamp, mixed labels, >=2 DISTINCT items; duplicate-candidate groups excluded"
    save_json(root / "final_results.json", final)
    save_json(root / "ranking_audit.json", {"primary_metrics_unchanged": True,
        "model_selection_unchanged": True, "models_retrained": False,
        "test_groups": queries["test"].select("user_id", "timestamp").unique().height,
        "excluded_duplicate_test_groups": next(row["ranking_groups_excluded_duplicates"]
            for row in output if row["split"] == "test")})
    print(name, "ranking corrected; AUC/LogLoss unchanged", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["kion", "yambda"], required=True)
    rescore(parser.parse_args().dataset)
