import json
from pathlib import Path

def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def main(split="dev", mode="rationale",
         raw_dir="datasets/scifact_raw/data",
         out_path=None):

    raw = Path(raw_dir)
    corpus_path = raw / "corpus.jsonl"
    claims_path = raw / f"claims_{split}.jsonl"

    # doc_id -> abstract sentences (list[str])
    corpus = {}
    for d in read_jsonl(corpus_path):
        corpus[int(d["doc_id"])] = d["abstract"]

    rows = []
    for ex in read_jsonl(claims_path):
        claim_id = ex.get("id")
        claim = ex["claim"]

        # evidence is a mapping: doc_id -> list[ {label, sentences:[idx]} ]
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
                    "id": str(claim_id) if claim_id is not None else None,
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
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Wrote {len(rows)} rows -> {out}")

if __name__ == "__main__":
    main()
