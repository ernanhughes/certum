\
from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from tqdm import tqdm


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def fetch_flags(conn: sqlite3.Connection, element_ids: List[str], model: str) -> Tuple[int, int, int]:
    """Return (n_total, n_resolved_ok, n_has_embedding) for a list of element_ids."""
    if not element_ids:
        return (0, 0, 0)

    q_marks = ",".join(["?"] * len(element_ids))

    # resolved.ok
    cur = conn.execute(
        f"SELECT element_id, ok FROM resolved WHERE element_id IN ({q_marks})",
        element_ids,
    )
    ok_map = {eid: int(ok) for eid, ok in cur.fetchall()}
    n_ok = sum(1 for eid in element_ids if ok_map.get(eid, 0) == 1)

    # embeddings presence for model
    cur = conn.execute(
        f"SELECT element_id FROM embeddings WHERE model=? AND element_id IN ({q_marks})",
        [model, *element_ids],
    )
    emb_set = {eid for (eid,) in cur.fetchall()}
    n_emb = sum(1 for eid in element_ids if eid in emb_set)

    return (len(element_ids), n_ok, n_emb)


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate a scored_dump.jsonl against the FEVEROUS cache DB.")
    ap.add_argument("--dump", type=Path, required=True, help="Path to scored_dump.jsonl")
    ap.add_argument("--cache_db", type=Path, required=True, help="Path to feverous_cache.sqlite (resolved + embeddings tables)")
    ap.add_argument("--model", type=str, default="sentence-transformers/all-MiniLM-L6-v2", help="Embedding model key used in cache")
    ap.add_argument("--max_rows", type=int, default=0, help="Optional cap for quick checks (0 = no cap)")
    ap.add_argument("--show_examples", type=int, default=5, help="Print a few failing examples")
    args = ap.parse_args()

    if not args.dump.exists():
        raise SystemExit(f"Missing dump: {args.dump}")
    if not args.cache_db.exists():
        raise SystemExit(f"Missing cache DB: {args.cache_db}")

    conn = sqlite3.connect(str(args.cache_db))
    conn.row_factory = sqlite3.Row

    totals = Counter()
    by_split = defaultdict(Counter)
    failing_rows: List[Dict[str, Any]] = []

    for i, row in enumerate(tqdm(iter_jsonl(args.dump), desc="Rows")):
        if args.max_rows and i >= args.max_rows:
            break

        split = row.get("split", "unknown")
        evidence_ids = row.get("evidence_ids") or []
        if not isinstance(evidence_ids, list):
            evidence_ids = list(evidence_ids) if evidence_ids is not None else []

        totals["rows"] += 1
        by_split[split]["rows"] += 1

        # Basic integrity
        if row.get("id") is None:
            totals["missing_id"] += 1
            by_split[split]["missing_id"] += 1

        # Duplicate IDs inside evidence
        dup = len(evidence_ids) - len(set(evidence_ids))
        if dup:
            totals["evidence_dups"] += 1
            by_split[split]["evidence_dups"] += 1

        n_total, n_ok, n_emb = fetch_flags(conn, evidence_ids, args.model)
        totals["evidence_total"] += n_total
        totals["resolved_ok"] += n_ok
        totals["has_embedding"] += n_emb

        by_split[split]["evidence_total"] += n_total
        by_split[split]["resolved_ok"] += n_ok
        by_split[split]["has_embedding"] += n_emb

        # Mark row as failing if anything missing
        if n_ok != n_total or n_emb != n_total:
            totals["rows_with_missing"] += 1
            by_split[split]["rows_with_missing"] += 1
            if len(failing_rows) < args.show_examples:
                failing_rows.append({
                    "id": row.get("id"),
                    "set_idx": row.get("set_idx"),
                    "split": split,
                    "label": row.get("label"),
                    "neg_mode": row.get("neg_mode"),
                    "energy": row.get("energy"),
                    "evidence_total": n_total,
                    "resolved_ok": n_ok,
                    "has_embedding": n_emb,
                })

    def pct(a: int, b: int) -> float:
        return (100.0 * a / b) if b else 0.0

    print("\n================================================================================")
    print("CACHE VALIDATION SUMMARY")
    print("================================================================================")
    print(f"dump:     {args.dump}")
    print(f"cache_db: {args.cache_db}")
    print(f"model:    {args.model}\n")

    print(f"Rows: {totals['rows']}")
    if totals["missing_id"]:
        print(f"⚠️  Rows missing id: {totals['missing_id']}")
    if totals["evidence_dups"]:
        print(f"⚠️  Rows with duplicate evidence_ids: {totals['evidence_dups']}")

    print("\nEvidence coverage:")
    print(f"- total evidence ids:   {totals['evidence_total']}")
    print(f"- resolved.ok == 1:     {totals['resolved_ok']}  ({pct(totals['resolved_ok'], totals['evidence_total']):.2f}%)")
    print(f"- has embedding vector: {totals['has_embedding']}  ({pct(totals['has_embedding'], totals['evidence_total']):.2f}%)")
    print(f"- rows with any missing: {totals['rows_with_missing']}  ({pct(totals['rows_with_missing'], totals['rows']):.2f}%)")

    print("\nBy split:")
    for split, c in by_split.items():
        print(f"- {split}: rows={c['rows']}, missing_rows={c['rows_with_missing']} ({pct(c['rows_with_missing'], c['rows']):.2f}%), "
              f"resolved_ok={pct(c['resolved_ok'], c['evidence_total']):.2f}%, embeddings={pct(c['has_embedding'], c['evidence_total']):.2f}%")

    if failing_rows:
        print("\nExample failing rows:")
        for r in failing_rows:
            print(json.dumps(r, ensure_ascii=False))

    conn.close()


if __name__ == "__main__":
    main()
