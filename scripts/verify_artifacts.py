import hashlib
import json
from pathlib import Path
import re

import nbformat
import numpy as np
import polars as pl

from src.data import save_json


def main():
    root = Path("artifacts")
    experiments = pl.read_csv(root / "experiments.csv")
    lock = json.loads((root / "locked_selection.json").read_text())
    selected = pl.read_csv(root / "selected_features.csv")
    assert experiments.height == experiments.unique(["experiment_name", "seed", "split"]).height
    for metric in ["auc", "logloss", "ndcg_3", "ndcg_5", "ndcg_10", "recall_10", "map_10"]:
        values = experiments[metric].to_numpy()
        assert np.isfinite(values).all(), metric
        assert (values >= 0).all(), metric
        if metric != "logloss":
            assert (values <= 1 + 1e-6).all(), metric
    assert set(selected["feature"].to_list()) == set(lock["auto_features"])
    assert lock["locked_before_audit_and_test"]
    assert hashlib.md5(Path("data/raw/ml-1m.zip").read_bytes()).hexdigest() == "c4d9eecfca2ab87c1945afe126590906"
    for name in ["README.md", "REPORT.md"]:
        for link in re.findall(r"\]\(([^)]+)\)", Path(name).read_text()):
            if not link.startswith(("http", "app:", "codex:")):
                assert Path(link).exists(), link
    executed = []
    for path in sorted(Path("notebooks").glob("*.ipynb")):
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
        assert all(cell.execution_count is not None for cell in cells), path
        assert not any(output.output_type == "error" for cell in cells for output in cell.outputs), path
        executed.append(path.name)
    plots = list((root / "plots").glob("*.png"))
    assert len(plots) == 8 and all(path.stat().st_size > 1000 for path in plots)
    summary = {"status": "passed", "experiment_rows": experiments.height,
               "unique_experiment_identities": True, "finite_metrics": True,
               "selected_feature_count": selected.height, "notebooks_executed": executed,
               "plots": len(plots), "dataset_md5_verified": True,
               "hpo_trials": {name: pl.read_csv(root / f"{name}_tuning.csv").height for name in ["manual", "auto"]},
               "main_test_used_for_feature_selection": False}
    save_json(root / "verification.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
