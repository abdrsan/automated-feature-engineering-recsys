import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

os.environ.setdefault("JUPYTER_RUNTIME_DIR", "/tmp/fe-final-jupyter")
os.environ.setdefault("IPYTHONDIR", "/tmp/fe-final-ipython")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/fe-comparison-mpl")
os.environ.setdefault("POLARS_MAX_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("OMP_NUM_THREADS", "4")

import nbformat
from nbclient import NotebookClient


def execute(path, mode, datasets, kernel_root):
    kernel = kernel_root / "kernels/fe-comparison"
    kernel.mkdir(parents=True)
    (kernel / "kernel.json").write_text(json.dumps({"argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
        "display_name": "Feature comparison audit", "language": "python"}))
    os.environ["JUPYTER_PATH"] = str(kernel_root)
    os.environ["FE_MODE"] = mode
    os.environ["FE_DATASETS"] = datasets
    notebook = nbformat.read(path, as_version=4)
    def progress(cell, cell_index, **kwargs):
        print("Executed cell", cell_index, flush=True)
    NotebookClient(notebook, timeout=3600, startup_timeout=90, kernel_name="fe-comparison",
        resources={"metadata": {"path": str(path.parent)}}, on_cell_executed=progress).execute()
    code = [cell for cell in notebook.cells if cell.cell_type == "code"]
    assert all(cell.execution_count is not None for cell in code)
    assert not any(out.output_type == "error" for cell in code for out in cell.outputs)
    nbformat.write(notebook, path)
    return len(code)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["demo", "full"], default="demo")
    parser.add_argument("--datasets", default="kion,yambda")
    parser.add_argument("--isolated", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    source = root / "FINAL_SUBMISSION/feature_engineering_comparison.ipynb"
    evidence = root / "artifacts/final_comparison_audit_20261025"
    evidence.mkdir(exist_ok=True)
    os.environ["FE_RAW_DIR"] = str(root / "data/raw")
    with tempfile.TemporaryDirectory(prefix="fe-final-kernel-") as kernels:
        if args.isolated:
            with tempfile.TemporaryDirectory(prefix="fe-final-isolated-") as directory:
                folder = Path(directory)
                for name in ["feature_engineering_comparison.ipynb", "README.md", "requirements.txt", ".gitignore"]:
                    shutil.copy2(source.parent / name, folder / name)
                assert not (folder / "src").exists() and not (folder / "scripts").exists()
                count = execute(folder / source.name, args.mode, args.datasets, Path(kernels))
                target = evidence / (args.mode + "_isolated")
                shutil.copytree(folder / ".runs", target, dirs_exist_ok=True)
                shutil.copy2(folder / source.name, evidence / (args.mode + "_executed.ipynb"))
        else:
            count = execute(source, args.mode, args.datasets, Path(kernels))
    result = {"status": "PASS", "mode": args.mode, "datasets": args.datasets.split(","),
        "isolated": args.isolated, "executed_code_cells": count, "python": sys.executable,
        "data_access": "external pinned raw files; no project module imports"}
    path = evidence / (args.mode + "_execution.json")
    path.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
