import hashlib
from pathlib import Path
import time
import urllib.request


URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
EXPECTED_MD5 = "c4d9eecfca2ab87c1945afe126590906"


def download():
    path = Path("data/raw/ml-1m.zip")
    path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(20):
        if path.exists() and hashlib.md5(path.read_bytes()).hexdigest() == EXPECTED_MD5:
            print("MovieLens archive verified", flush=True)
            return
        offset = path.stat().st_size if path.exists() else 0
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        try:
            request = urllib.request.Request(URL, headers=headers)
            with urllib.request.urlopen(request, timeout=60) as response:
                if offset and response.status != 206:
                    raise RuntimeError("Server did not honor resume range")
                if offset and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                    raise RuntimeError("Incorrect resume range")
                with path.open("ab" if offset else "wb") as out:
                    last_log = time.monotonic()
                    while True:
                        chunk = response.read(16384)
                        if not chunk:
                            break
                        out.write(chunk)
                        if time.monotonic() - last_log >= 30:
                            out.flush()
                            print(f"Downloaded {out.tell() / 2**20:.2f} MiB", flush=True)
                            last_log = time.monotonic()
            if hashlib.md5(path.read_bytes()).hexdigest() != EXPECTED_MD5:
                raise RuntimeError("Archive checksum mismatch")
        except (OSError, TimeoutError) as error:
            print(f"Download retry {attempt + 1}: {error}", flush=True)
            time.sleep(min(attempt + 1, 5))
            continue
    raise RuntimeError("Download failed after 20 resumptions")


if __name__ == "__main__":
    download()
