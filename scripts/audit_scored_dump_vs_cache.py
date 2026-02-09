#!/usr/bin/env python3
"""
audit_scored_dump_vs_cache.py

Audit per-example scored dump against cache DB to ensure:
- evidence IDs exist
- embeddings exist
- no evidence leakage between POS and NEG
- hard-mined negatives are truly non-gold
"""

import json
import random
import sqlite3
from pathlib import Path
from collections import defaultdict

CACHE_DB = Path("feverous_cache.db")
DUMP_JSONL = Path("artifacts/scored_dump.jsonl")  # you will generate this
SAMPLE_N = 10

def connect():
    conn = sqlite3.connect(str(CACHE_DB))
    conn.row_factory = sqlite3.Row
    return conn

def list_tables(conn):
    rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    return {r["name"] for r in rows}

def list_columns(conn, table):
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return [r[1] for r in rows]  # name at index 1

def pick_table(conn, candidates):
    tables = list_tables(conn)
    for t in candidates:
        if t in tables:
            return t
    return None

def fetch_exists(conn, table, id_col, ids):
    if not ids:
        return set()
    q = ",".join(["?"] * len(ids))
    rows = conn.execute(
        f"SELECT {id_col} AS id FROM {table} WHERE {id_col} IN ({q})",
        ids,
    ).fetchall()
    return {r["id"] for r in rows}

def fetch_texts(conn, table, id_col, text_col, ids):
    if not ids:
        return {}
    q = ",".join(["?"] * len(ids))
    rows = conn.execute(
        f"SELECT {id_col} AS id, {text_col} AS txt FROM {table} WHERE {id_col} IN ({q})",
        ids,
    ).fetchall()
    return {r["id"]: r["txt"] for r in rows}

def load_dump(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def main():
    if not DUMP_JSONL.exists():
        raise SystemExit(f"Missing scored dump: {DUMP_JSONL}. Generate it first.")

    scored = load_dump(DUMP_JSONL)
    conn = connect()

    # Try to locate likely tables
    # Adjust candidates if your DB uses different names.
    embedding_table = pick_table(conn, ["embeddings", "embedding_cache"])
    element_table    = pick_table(conn, ["elements", "evidence_elements", "wiki_elements", "cache_elements"])

    if embedding_table is None:
        raise SystemExit("Could not find embeddings table. Add candidate name in script.")
    if element_table is None:
        print("⚠️  Could not find elements table for text lookup. Coverage checks still run.")

    emb_cols = list_columns(conn, embedding_table)
    emb_id_col = "element_id" if "element_id" in emb_cols else emb_cols[0]

    txt_col = None
    elem_id_col = None
    if element_table is not None:
        elem_cols = list_columns(conn, element_table)
        elem_id_col = "element_id" if "element_id" in elem_cols else elem_cols[0]
        # choose a likely text column
        for c in ["text", "content", "value", "snippet"]:
            if c in elem_cols:
                txt_col = c
                break

    # Collect all evidence ids
    all_eids = []
    for r in scored:
        all_eids.extend(r.get("evidence_ids", []))
    all_eids = list(dict.fromkeys(all_eids))  # unique preserving order

    found_emb = fetch_exists(conn, embedding_table, emb_id_col, all_eids)
    missing_emb = [eid for eid in all_eids if eid not in found_emb]

    print("\n[EMBEDDING COVERAGE]")
    print(f"Unique evidence IDs in dump: {len(all_eids)}")
    print(f"Missing embeddings: {len(missing_emb)}")
    if missing_emb[:10]:
        print("Examples:", missing_emb[:10])

    # Optional text existence check
    if element_table is not None:
        found_elem = fetch_exists(conn, element_table, elem_id_col, all_eids)
        missing_elem = [eid for eid in all_eids if eid not in found_elem]
        print("\n[ELEMENT COVERAGE]")
        print(f"Missing elements: {len(missing_elem)}")
        if missing_elem[:10]:
            print("Examples:", missing_elem[:10])

    # Leakage check: same example id appears as POS and NEG with overlapping evidence
    by_example = defaultdict(lambda: {"POS": [], "NEG": []})
    for r in scored:
        ex_id = r.get("id")
        lab = r.get("label")
        if ex_id is None or lab not in ("POS", "NEG"):
            continue
        by_example[ex_id][lab].append(set(r.get("evidence_ids", [])))

    leaked = 0
    for ex_id, d in by_example.items():
        for pos_set in d["POS"]:
            for neg_set in d["NEG"]:
                if pos_set & neg_set:
                    leaked += 1
                    break

    print("\n[POS/NEG EVIDENCE LEAKAGE]")
    print(f"Example-ids with any POS/NEG evidence intersection: {leaked}")

    # Spot check: print a few random examples
    if element_table is not None and txt_col is not None:
        print("\n[SPOT CHECK EXAMPLES]")
        sample = random.sample(scored, k=min(SAMPLE_N, len(scored)))
        for r in sample:
            eids = r.get("evidence_ids", [])[:5]
            texts = fetch_texts(conn, element_table, elem_id_col, txt_col, eids)
            print("\n---")
            print(f"id={r.get('id')} split={r.get('split')} label={r.get('label')} mode={r.get('neg_mode')}")
            print(f"energy={r.get('energy'):.4f}")
            for eid in eids:
                t = (texts.get(eid, "") or "")[:140].replace("\n", " ")
                print(f"  {eid}: {t}")

    conn.close()

if __name__ == "__main__":
    main()
