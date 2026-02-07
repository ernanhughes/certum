import tarfile
import io
import requests
from pathlib import Path

URL = "https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz"

def main(out_dir="datasets/scifact_raw"):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("Downloading:", URL)
    r = requests.get(URL, timeout=300)
    r.raise_for_status()

    with tarfile.open(fileobj=io.BytesIO(r.content), mode="r:gz") as tf:
        tf.extractall(out)

    print("Extracted to:", out.resolve())

if __name__ == "__main__":
    main()
