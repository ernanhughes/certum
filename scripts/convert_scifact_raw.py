from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


# ------------------------------------------------------------
# Main conversion
# ------------------------------------------------------------

def convert_scifact(
    dataset_dir: Path,
    split: str = "dev",           # train | dev | test
    mode: str = "rationale",      # rationale | abstract
    out_path: Path | None = None,
):
    """
    Convert raw SciFact files into DPGSS-ready JSONL format.

    dataset_dir should contain:
        corpus.jsonl
        claims_{split}.jsonl
    """

    corpus_path = dataset_dir / "corpus.jsonl"
    claims_path = dataset_dir / f"claims_{split}.jsonl"

    if not corpus_path.exists():
        raise FileNotFoundError(f"Missing corpus: {corpus_path}")
    if not claims_path.exists():
        raise FileNotFoundError(f"Missing claims file: {claims_path}")

    print(f"Loading corpus from: {corpus_path}")

    # ------------------------------------------------------------
    # Build corpus index: doc_id -> abstract sentences
    # ------------------------------------------------------------
    corpus: Dict[int, List[str]] = {}

    for row in read_jsonl(corpus_path):
        doc_id = int(row["doc_id"])
        abstract = row.get("abstract", [])
        if not isinstance(abstract, list):
            abstract = []
        corpus[doc_id] = [str(s).strip() for s in abstract if str(s).strip()]

    print(f"Loaded {len(corpus)} documents")

    # ------------------------------------------------------------
    # Convert claims
    # ------------------------------------------------------------
    rows = []

    for ex in read_jsonl(claims_path):
        claim_id = ex.get("id")
        claim = ex["claim"]

        evidence = ex.get("evidence") or {}

        for doc_id_str, ev_list in evidence.items():
            doc_id = int(doc_id_str)
            abstract = corpus.get(doc_id, [])

            if not abstract:
                continue

            for ev in ev_list:
                label = ev.get("label", "")
                sent_ids = ev.get("sentences") or []

                if mode == "rationale":
                    evidence_texts = [
                        abstract[i]
                        for i in sent_ids
                        if 0 <= i < len(abstract)
                    ]
                elif mode == "abstract":
                    evidence_texts = list(abstract)
                else:
                    raise ValueError("mode must be 'rationale' or 'abstract'")

                if not evidence_texts:
                    continue

                rows.append({
                    "dataset": "scifact",
                    "split": split,
                    "id": str(claim_id) if claim_id is not None else None,
                    "claim": claim,
                    "label": label,
                    "evidence_texts": evidence_texts,
                    "meta": {
                        "doc_id": doc_id,
                        "sent_ids": sent_ids,
                    },
                })

    # ------------------------------------------------------------
    # Write output
    # ------------------------------------------------------------
    if out_path is None:
        out_path = dataset_dir / f"scifact_{split}_{mode}.jsonl"

    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(rows)} rows -> {out_path}")


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=str, required=True,
                    help="Path to folder containing corpus.jsonl + claims_*.jsonl")
    ap.add_argument("--split", type=str, default="dev",
                    choices=["train", "dev", "test"])
    ap.add_argument("--mode", type=str, default="rationale",
                    choices=["rationale", "abstract"])
    ap.add_argument("--out", type=str, default=None)

    args = ap.parse_args()

    convert_scifact(
        dataset_dir=Path(args.dir),
        split=args.split,
        mode=args.mode,
        out_path=Path(args.out) if args.out else None,
    )
