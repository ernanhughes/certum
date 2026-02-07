from pathlib import Path
import requests


def download_dataset(url: str, dest: Path, *, overwrite: bool = False) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and not overwrite:
        return dest

    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

    return dest

path = download_dataset(
    url="https://raw.githubusercontent.com/hover-nlp/hover/refs/heads/main/data/hover/hover_dev_release_v1.1.json",
    dest=Path("datasets/hover/hover_dev_release_v1.1.json"),
)
print(f"Downloaded to {path}")



