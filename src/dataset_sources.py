from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time
import urllib.request


KION_REVISION = "f69775be31fa5779907cf0a92ddedb70037fb5ae"
YAMBDA_REVISION = "dd6f3a19eef5866e346c3270e098baa641a44948"


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_verified(url, path, expected_size=None, sha256=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    def valid():
        return (path.exists() and (expected_size is None or path.stat().st_size == expected_size)
                and (sha256 is None or file_sha256(path) == sha256))
    if valid():
        return path
    partial = path.with_suffix(path.suffix + ".part")
    for attempt in range(8):
        offset = partial.stat().st_size if partial.exists() else 0
        request = urllib.request.Request(url, headers={"Range": f"bytes={offset}-"} if offset else {})
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                resume = offset and response.status == 206
                if resume and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                    raise RuntimeError("Wrong resume byte range")
                with partial.open("ab" if resume else "wb") as out:
                    for chunk in iter(lambda: response.read(1024 * 256), b""):
                        out.write(chunk)
            if expected_size is not None and partial.stat().st_size != expected_size:
                raise OSError("Incomplete download")
            if sha256 and file_sha256(partial) != sha256:
                partial.unlink()
                raise OSError("SHA256 mismatch")
            partial.replace(path)
            print(f"Downloaded and verified {path.name}: {path.stat().st_size / 2**20:.1f} MiB", flush=True)
            return path
        except OSError as error:
            print(f"Retry {path.name} #{attempt + 1}: {error}", flush=True)
            time.sleep(min(attempt + 1, 5))
    raise RuntimeError(f"Cannot download {url}")


def fetch_sources(dataset, raw_dir="data/raw"):
    raw = Path(raw_dir) / dataset
    raw.mkdir(parents=True, exist_ok=True)
    manifest_path = raw / "source_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    elif dataset == "kion":
        with urllib.request.urlopen(f"https://api.github.com/repos/irsafilo/KION_DATASET/git/trees/{KION_REVISION}?recursive=1", timeout=60) as r:
            tree = json.load(r)["tree"]
        wanted = {"interactions.csv.zip", "data_original/users.csv", "data_original/items.csv"}
        manifest = {"dataset": "kion", "revision": KION_REVISION, "files": [
            {"name": Path(row["path"]).name, "size": row["size"],
             "url": f"https://raw.githubusercontent.com/irsafilo/KION_DATASET/{KION_REVISION}/{row['path']}"}
            for row in tree if row["path"] in wanted]}
    elif dataset == "yambda":
        revision = YAMBDA_REVISION
        with urllib.request.urlopen(f"https://huggingface.co/api/datasets/yandex/yambda/tree/{revision}?recursive=true&limit=1000", timeout=60) as r:
            tree = json.load(r)
        wanted = {"flat/50m/likes.parquet", "flat/50m/dislikes.parquet", "artist_item_mapping.parquet", "album_item_mapping.parquet"}
        manifest = {"dataset": "yambda", "revision": revision, "scale": "50m", "files": [
            {"name": Path(row["path"]).name, "size": row["size"], "sha256": row["lfs"]["oid"],
             "url": f"https://huggingface.co/datasets/yandex/yambda/resolve/{revision}/{row['path']}?download=true"}
            for row in tree if row["path"] in wanted]}
    else:
        raise ValueError("Supported sources: kion, yambda")
    expected_count = 3 if dataset == "kion" else 4
    if len(manifest["files"]) != expected_count:
        raise ValueError("Official source inventory differs from expected files")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    def fetch(entry):
        path = download_verified(entry["url"], raw / entry["name"], entry["size"], entry.get("sha256"))
        return {**entry, "sha256": file_sha256(path)}
    with ThreadPoolExecutor(max_workers=4) as pool:
        manifest["files"] = list(pool.map(fetch, manifest["files"]))
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
