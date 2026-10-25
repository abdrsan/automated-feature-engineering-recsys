import os
from pathlib import Path

os.environ.setdefault("JUPYTER_RUNTIME_DIR", "/tmp/yandex-jupyter")
os.environ.setdefault("IPYTHONDIR", "/tmp/yandex-ipython")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/yandex-cache")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/yandex-mpl")
os.environ.setdefault("POLARS_MAX_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

import nbformat
from nbclient import NotebookClient


def main():
    root = Path.cwd()
    for path in sorted((root / "notebooks").glob("*.ipynb")):
        notebook = nbformat.read(path, as_version=4)
        client = NotebookClient(notebook, timeout=60, kernel_name="python3",
                                resources={"metadata": {"path": str(root)}})
        client.execute()
        nbformat.write(notebook, path)
        print("Executed", path.name, flush=True)


if __name__ == "__main__":
    main()
