import ast
from pathlib import Path

import nbformat
import numpy as np
import pytest
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score

from audit.verify_comparison_submission import independent_scores


@pytest.mark.parametrize("labels,probabilities", [
    ([0, 1, 1, 0], [.9, .9, .4, .1]),
    ([0, 1, 0, 1], [.5, .5, .5, .5]),
    ([1, 0, 0, 1], [.8, .1, .3, .6]),
])
def test_independent_classification_formulas_include_score_ties(labels, probabilities):
    p = np.array(probabilities)
    actual = independent_scores(labels, p)
    assert actual["auc"] == pytest.approx(roc_auc_score(labels, p), abs=1e-15)
    assert actual["logloss"] == pytest.approx(log_loss(labels, p), abs=1e-15)
    assert actual["average_precision"] == pytest.approx(average_precision_score(labels, p), abs=1e-15)


def test_submission_reuses_original_implementations_without_project_imports():
    root = Path(__file__).resolve().parents[1]
    notebook = nbformat.read(root/"FINAL_SUBMISSION/feature_engineering_comparison.ipynb", as_version=4)
    verified = 0
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        tree = ast.parse(cell.source)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith(("src", "scripts", "audit"))
            elif isinstance(node, ast.Import):
                assert not any(n.name.startswith(("src", "scripts", "audit")) for n in node.names)
        source = cell.metadata.get("original_source")
        if source:
            original = {n.name: ast.dump(n, include_attributes=False)
                        for n in ast.parse((root/source).read_text()).body
                        if isinstance(n, (ast.ClassDef, ast.FunctionDef))}
            for node in tree.body:
                if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
                    assert ast.dump(node, include_attributes=False) == original[node.name]
                    verified += 1
    assert verified == 27


@pytest.mark.parametrize("missing_numpy", [False, True])
def test_notebook_installs_only_missing_packages_in_current_kernel(monkeypatch, missing_numpy):
    from importlib import metadata
    root = Path(__file__).resolve().parents[1]
    notebook = nbformat.read(root/"FINAL_SUBMISSION/feature_engineering_comparison.ipynb", as_version=4)
    bootstrap = next(cell.source for cell in notebook.cells if cell.cell_type == "code")
    requirements = (root/"FINAL_SUBMISSION/requirements.txt").read_text().splitlines()
    versions = dict(requirement.split("==") for requirement in requirements)
    if missing_numpy:
        versions.pop("numpy")
    def version(package):
        if package not in versions:
            raise metadata.PackageNotFoundError(package)
        return versions[package]
    calls = []
    class Kernel:
        def run_line_magic(self, magic, command):
            calls.append((magic, command))
            versions["numpy"] = "2.3.2"
    monkeypatch.setattr(metadata, "version", version)
    exec(bootstrap, {"get_ipython": lambda: Kernel()})
    assert len(calls) == int(missing_numpy)
    if calls:
        assert calls[0] == ("pip", "install --disable-pip-version-check numpy==2.3.2")
