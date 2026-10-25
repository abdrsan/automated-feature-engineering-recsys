"""Check published predictions, metrics and executed notebooks without model fitting."""
import ast
import hashlib
import json
from pathlib import Path

import nbformat
import numpy as np
import polars as pl

from scipy.stats import rankdata


def independent_scores(labels, probabilities):
    labels = np.asarray(labels, dtype=float)
    positives = labels.sum()
    ranks = rankdata(probabilities)
    auc = (ranks[labels == 1].sum()-positives*(positives+1)/2)/(positives*(len(labels)-positives))
    p = np.clip(probabilities, np.finfo(float).eps, 1-np.finfo(float).eps)
    loss = np.mean(-labels*np.log(p)-(1-labels)*np.log1p(-p))
    order = np.argsort(-probabilities, kind="stable")
    scores, y = probabilities[order], labels[order]
    ends = np.r_[np.flatnonzero(np.diff(scores)), len(y)-1]
    hits = np.cumsum(y)[ends]
    ap = np.sum(hits/(ends+1)*np.diff(np.r_[0., hits/positives]))
    return {"auc": float(auc), "logloss": float(loss), "average_precision": float(ap)}


def main():
    root = Path(__file__).resolve().parents[1]
    comparisons = 0
    for dataset in ["kion", "yambda"]:
        folder = root / "results/full" / dataset
        lock = json.loads((folder / "locked_protocol.json").read_text())
        assert lock["mode"] == "full" and lock["selection_data"] == "search only"
        assert lock["seed"] == 42
        expected_ids = np.arange(lock["split"]["validation_end"], lock["split"]["n_rows"])
        table = pl.read_csv(folder / "comparison.csv")
        labels = None
        for row in table.to_dicts():
            source = row["shared_with"] or row["method"]
            predictions = pl.read_parquet(folder / ("predictions_" + source + ".parquet"))
            np.testing.assert_array_equal(predictions["row_id"].to_numpy(), expected_ids)
            if labels is None:
                labels = predictions["label"].to_numpy()
            np.testing.assert_array_equal(labels, predictions["label"].to_numpy())
            for metric, value in independent_scores(labels, predictions["prediction"].to_numpy()).items():
                assert abs(value - row[metric]) < 1e-12, (dataset, row["method"], metric)
                comparisons += 1
        checksums = json.loads((folder / "files_sha256.json").read_text())
        assert all(hashlib.sha256((folder/name).read_bytes()).hexdigest() == digest
                   for name, digest in checksums.items())
    notebooks = []
    for name, expected in [("FINAL_SUBMISSION/feature_engineering_comparison.ipynb", 21)]:
        notebook = nbformat.read(root/name, as_version=4)
        nbformat.validate(notebook)
        cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
        assert len(cells) == expected
        assert [cell.execution_count for cell in cells] == list(range(1, expected+1))
        assert not any(output.output_type == "error" for cell in cells for output in cell.outputs)
        assert 'run_line_magic("pip"' in cells[0].source
        for cell in cells:
            ast.parse(cell.source)
        notebooks.append({"path": name, "executed_cells": len(cells), "errors": 0})
    print(json.dumps({"status": "PASS", "independent_metric_comparisons": comparisons,
                      "notebooks": notebooks}, indent=2))


if __name__ == "__main__":
    main()
