import json
from pathlib import Path
from datasets import load_dataset

def main(split="validation", mode="rationale", out_path=None):
    # Load SciFact
    claims = load_dataset("allenai/scifact", "claims", split=split)
    corpus = load_dataset("allenai/scifact", "corpus", split="train")

    # doc_id -> abstract sentences
    doc_abs = {int(d["doc_id"]): d["abstract"] for d in corpus}

    rows = []
    for ex in claims:
        claim = ex["claim"]
        doc_id = int(ex["evidence_doc_id"]) if ex.get("evidence_doc_id") else None
        sent_ids = ex.get("evidence_sentences") or []
        label = ex.get("evidence_label") or ""

        if doc_id is None or doc_id not in doc_abs:
            continue

        abstract = doc_abs[doc_id]
        if mode == "rationale":
            evidence_texts = [abstract[i] for i in sent_ids if 0 <= i < len(abstract)]
        elif mode == "abstract":
            evidence_texts = list(abstract)
        else:
            raise ValueError("mode must be 'rationale' or 'abstract'")

        if not evidence_texts:
            continue

        rows.append({
            "dataset": "scifact",
            "split": split,
            "id": str(ex["id"]),
            "claim": claim,
            "label": label,
            "evidence_texts": evidence_texts,
            "meta": {"doc_id": doc_id, "sent_ids": sent_ids},
        })

    if out_path is None:
        out_path = f"datasets/scifact/scifact_{split}_{mode}.jsonl"

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    print(f"Wrote {len(rows)} rows -> {out}")

if __name__ == "__main__":
    main()
