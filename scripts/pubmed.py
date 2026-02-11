from datasets import load_dataset
from pathlib import Path
import json
import argparse


def normalize_row(row, split_name):
    claim = row.get("question") or row.get("claim")
    label = row.get("final_decision")

    # PubMedQA evidence structure
    contexts = []
    if isinstance(row.get("context"), dict):
        contexts = row["context"].get("contexts", [])
    elif isinstance(row.get("context"), list):
        contexts = row["context"]

    contexts = [c.strip() for c in contexts if isinstance(c, str) and c.strip()]

    if not claim or not contexts:
        return None

    return {
        "dataset": "pubmedqa",
        "split": split_name,
        "id": str(row.get("id", "")),
        "claim": claim.strip(),
        "label": label,
        "evidence_texts": contexts,
        "meta": {}
    }


def main(out_dir: str):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("qiaojin/PubMedQA", 'pqa_unlabeled')

    for split_name, dataset_split in ds.items():
        out_file = out_path / f"pubmedqa_{split_name}.jsonl"

        count = 0
        with out_file.open("w", encoding="utf-8") as f:
            for row in dataset_split:
                converted = normalize_row(row, split_name)
                if converted:
                    f.write(json.dumps(converted, ensure_ascii=False) + "\n")
                    count += 1

        print(f"Wrote {count} rows -> {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="datasets/pubmedqa")
    args = parser.parse_args()
    main(args.out_dir)
