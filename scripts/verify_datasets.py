import ast
import json
from pathlib import Path

import nbformat
import numpy as np
import polars as pl

from src.dataset_sources import file_sha256
from src.data import save_json
from src.provenance import STUDY_PROTOCOL_VERSION


def verify_dataset(name):
    root = Path("artifacts") / name
    load = lambda filename: json.loads((root / filename).read_text())
    manifest, lock, final = [load(filename) for filename in ["run_manifest.json", "locked_selection.json", "final_results.json"]]
    rows = pl.read_csv(root / "experiments.csv")
    assert rows.height == rows.unique(["experiment_name", "seed", "split"]).height
    assert np.isfinite(rows["auc"].to_numpy()).all() and np.isfinite(rows["logloss"].to_numpy()).all()
    for row in rows.to_dicts():
        for metric in ["ndcg_3", "ndcg_5", "ndcg_10", "map_10", "recall_10"]:
            if row["ranking_groups"]:
                assert math_isfinite(row[metric]) and 0 <= row[metric] <= 1 + 1e-6
    selected = pl.read_csv(root / "selected_features.csv")
    assert set(selected["feature"]) == set(lock["auto_features"])
    assert lock["locked_before_audit_and_test"] and final["test_used_for_selection"] is False
    assert manifest["config"]["dataset_name"] == name
    assert manifest["protocol_version"] == STUDY_PROTOCOL_VERSION
    assert final["evaluation_status"].startswith("post-audit")
    pool = rows.filter((pl.col("split") == "validation_search") & (pl.col("seed") == 42)
                       & pl.col("experiment_name").is_in(lock["manual_selection_pool"]))
    best_manual = pool.sort(["logloss", "n_features", "experiment_name"]).row(0, named=True)
    assert lock["manual_name"] == best_manual["experiment_name"]
    assert lock["manual_selection_data"] == "validation_search"
    candidates = pl.read_csv(root / "candidate_features.csv")
    # Агент добавляет specs позже. Они не должны задним числом менять исходный pool controls.
    candidate_specs = [json.loads(s) for s in candidates["spec_json"]]
    fixed_allowed = {row["feature"] for row, spec in zip(candidates.to_dicts(), candidate_specs)
                     if spec["window_days"] in [None, 7, 30]
                     and (spec["aggregation"] != "trend" or spec["long_window_days"] == 30)}
    initial_count = load("generation_profile.json")["n_candidates"]
    initial_names = set(candidates.head(initial_count)["feature"])
    fixed_allowed &= initial_names
    fixed_profile = load("fixed_pool_profile.json")
    fixed_kept = set(load("specs/auto_fixed_windows_all_seed42.json")["features"])
    fixed_removed = set(pl.read_csv(root / "fixed_pruned_features.csv")["feature"])
    assert fixed_profile["pruning_scope"] == "fixed pool only, before cross-window comparisons"
    assert fixed_kept.isdisjoint(fixed_removed)
    assert fixed_kept | fixed_removed == fixed_allowed
    assert len(fixed_kept) == fixed_profile["after_pruning"]
    shared_allowed = {row["feature"] for row, spec in zip(candidates.to_dicts(), candidate_specs)
                      if spec["window_days"] is None or
                      (spec["window_days"] in lock["shared_windows"] and spec["aggregation"] != "trend")}
    shared_allowed &= initial_names
    shared_kept = set(load("specs/shared_windows_pool_seed42.json")["features"])
    shared_removed = set(pl.read_csv(root / "shared_pruned_features.csv")["feature"])
    assert shared_kept.isdisjoint(shared_removed)
    assert shared_kept | shared_removed == shared_allowed
    for arm in ["manual", "automatic"]:
        seeds = rows.filter((pl.col("split") == "test") & (pl.col("experiment_name") == arm))["seed"].sort().to_list()
        assert seeds == sorted(manifest["config"]["seeds"])
    for source in load("source_manifest.json")["files"]:
        path = Path("data/raw") / name / source["name"]
        assert path.stat().st_size == source["size"]
        assert file_sha256(path) == source["sha256"]
    trials = {arm: pl.read_csv(root / f"{arm}_tuning.csv").filter(pl.col("state") == "COMPLETE").height for arm in ["manual", "auto"]}
    assert set(trials.values()) == {manifest["config"]["hyperparameter_trials"]}
    assert len(list((root / "plots").glob("*.png"))) == 8
    ranking_audit = load("ranking_audit.json")
    assert ranking_audit["primary_metrics_unchanged"] and ranking_audit["model_selection_unchanged"]
    assert ranking_audit["models_retrained"] is False
    assert rows.filter(pl.col("split") == "test")["ranking_groups_excluded_duplicates"].unique().to_list() == [ranking_audit["excluded_duplicate_test_groups"]]
    result = {"status": "passed", "dataset": name, "rows": manifest["split"]["n_rows"],
              "experiments": rows.height, "selected_specs": selected.height, "hpo_trials": trials,
              "source_checksums_verified": True, "ranking_nan_allowed_only_without_groups": True,
              "test_used_for_selection": False, "manual_reference_verified_against_all_stages": True,
              "fixed_shared_pools_pruned_independently": True,
              "protocol_version": STUDY_PROTOCOL_VERSION}
    save_json(root / "verification.json", result)
    return result


def math_isfinite(value):
    return value is not None and bool(np.isfinite(value))


def main():
    results = [verify_dataset(name) for name in ["kion", "yambda"]]
    notebook = nbformat.read("notebooks/00_full_research.ipynb", as_version=4)
    nbformat.validate(notebook)
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    assert all(cell.execution_count is not None for cell in code_cells)
    for cell in code_cells:
        for node in ast.walk(ast.parse(cell.source)):
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith(("src", "scripts", "tests"))
            if isinstance(node, ast.Import):
                assert all(not alias.name.startswith(("src", "scripts", "tests")) for alias in node.names)
        assert not any(output.output_type == "error" for output in cell.outputs)
    summary = {"datasets": results, "standalone_notebook": "passed", "executed_code_cells": len(code_cells)}
    save_json("artifacts/datasets_verification.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
