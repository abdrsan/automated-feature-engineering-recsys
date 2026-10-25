import argparse
import os
from pathlib import Path
import shutil
import tempfile

os.environ.setdefault("JUPYTER_RUNTIME_DIR", "/tmp/yandex-jupyter")
os.environ.setdefault("IPYTHONDIR", "/tmp/yandex-ipython")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/yandex-cache")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/yandex-notebook-mpl")
os.environ.setdefault("POLARS_MAX_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("OMP_NUM_THREADS", "4")

import nbformat
from nbclient import NotebookClient


def execute(path, cwd):
    notebook = nbformat.read(path, as_version=4)
    NotebookClient(notebook, timeout=1800, kernel_name="python3",
                   resources={"metadata": {"path": str(cwd)}}).execute()
    assert all(cell.execution_count is not None for cell in notebook.cells if cell.cell_type == "code")
    assert not any(output.output_type == "error" for cell in notebook.cells
                   if cell.cell_type == "code" for output in cell.outputs)
    nbformat.write(notebook, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--isolated-smoke", action="store_true")
    parser.add_argument("--dataset", choices=["kion", "yambda", "movielens"], default="kion")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    source = root / "notebooks" / "00_full_research.ipynb"
    os.environ["TEMPORAL_NOTEBOOK_DATASET"] = args.dataset
    if args.isolated_smoke:
        with tempfile.TemporaryDirectory(prefix="temporal-notebook-") as folder:
            isolated = Path(folder)
            notebook_path = isolated / source.name
            shutil.copy2(source, notebook_path)
            raw = isolated / "data" / "raw"
            raw.mkdir(parents=True)
            if args.dataset == "movielens":
                shutil.copy2(root / "data" / "raw" / "ml-1m.zip", raw / "ml-1m.zip")
            else:
                shutil.copytree(root / "data" / "raw" / args.dataset, raw / args.dataset)
            os.environ["TEMPORAL_NOTEBOOK_MODE"] = "smoke"
            assert not (isolated / "src").exists() and not (isolated / "scripts").exists()
            print("Running full notebook in isolated directory without project modules...", flush=True)
            execute(notebook_path, isolated)
            output = root / "artifacts" / (args.dataset + "_notebook_validation")
            output.mkdir(exist_ok=True)
            artifacts = isolated / "artifacts" / (args.dataset + "_notebook_smoke")
            for path in artifacts.glob("*.csv"):
                shutil.copy2(path, output / path.name)
            for path in artifacts.glob("*.json"):
                shutil.copy2(path, output / path.name)
            print("PASS isolated smoke: all inline implementation and training/evaluation cells executed", flush=True)
    else:
        os.environ["TEMPORAL_NOTEBOOK_MODE"] = "saved"
        execute(source, root)
        print("PASS saved mode: complete notebook executed and outputs embedded", flush=True)


if __name__ == "__main__":
    main()
