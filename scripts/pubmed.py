#!/usr/bin/env python3
"""
PubMedQA (parquet) -> DPGSS jsonl converter

Outputs JSONL rows compatible with your DPGSS "jsonl" loader:
  {
    "dataset": "pubmedqa",
    "split": "...",
    "id": "...",
    "claim": "...",              # we use the PubMedQA question by default
    "label": "...",              # "yes" | "no" | "maybe" (if available)
    "evidence_texts": [...],     # list[str] (abstract sentences)
    "meta": {...}                # extra fields for audit/debug
  }

Examples:
  py scripts/pubmed.py --in_dir datasets/pubmedqa_parquet --out_dir datasets/pubmedqa --split all
  py scripts/convert_pubmedqa_parquet_to_jsonl.py --in_dir datasets/pubmedqa_parquet --split train --mode abstract
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
import pandas as pd

from datasets import load_dataset

ds = load_dataset("qiaojin/PubMedQA")
print(ds)

def _stable_unique_strs(xs: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for x in xs:
        x = str(x).strip()
        if not x:
            continue
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _infer_split_from_path(p: Path) -> str:
    s = p.as_posix().lower()
    # Common HF patterns
    for key in ("train", "test", "validation", "valid", "dev"):
        if re.search(rf"(^|/|_)({key})(/|_|\.|$)", s):
            return "dev" if key in ("validation", "valid") else key
    return "unknown"


def _read_parquet_files(parquet_files: List[Path]) -> pd.DataFrame:
    # pandas can read a list of parquet paths and concatenate
    frames = []
    ds = load_dataset("parquet", data_files={"train": "datasets/pubmedqa_parquet/*.parquet"})
    for pf in ds["train"]:
        df = pd.read_parquet(pf, engine="pyarrow")
        df["_source_file"] = str(pf)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _extract_context_sentences(row: Dict[str, Any]) -> List[str]:
    """
    PubMedQA variants commonly include one of:
      - row["context"] as dict with keys like "contexts"/"abstract" or "sentences"
      - row["context"] as list[str]
      - row["abstract"] as list[str] or str
    We normalize to list[str].
    """
    # 1) direct list fields
    for k in ("evidence_texts", "abstract", "sentences", "context_sentences"):
        v = row.get(k)
        if isinstance(v, list):
            return _stable_unique_strs(v)

    # 2) direct string fields
    for k in ("abstract", "context", "evidence", "text"):
        v = row.get(k)
        if isinstance(v, str) and v.strip():
            # try sentence-ish split only if it's clearly a blob
            blob = v.strip()
            # keep as 1 evidence span (DPGSS supports list[str])
            return [blob]

    # 3) nested context dict
    ctx = row.get("context")
    if isinstance(ctx, dict):
        # HuggingFace PubMedQA often: {"contexts":[...], "labels":[...], "meshes":[...]}
        for ck in ("contexts", "abstract", "sentences", "context", "texts"):
            cv = ctx.get(ck)
            if isinstance(cv, list):
                return _stable_unique_strs(cv)
            if isinstance(cv, str) and cv.strip():
                return [cv.strip()]

    # 4) some dumps use "long_answer" containing abstract text
    la = row.get("long_answer")
    if isinstance(la, str) and la.strip():
        return [la.strip()]

    return []


def _pick_claim_text(row: Dict[str, Any]) -> Optional[str]:
    for k in ("claim", "question", "query", "prompt"):
        v = row.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def _pick_label(row: Dict[str, Any]) -> Optional[str]:
    # HF PubMedQA: final_decision is usually "yes"/"no"/"maybe"
    for k in ("final_decision", "label", "answer", "decision"):
        v = row.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    # sometimes nested
    ctx = row.get("context")
    if isinstance(ctx, dict):
        v = ctx.get("final_decision")
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def convert_df_to_rows(df: pd.DataFrame, split: str, mode: str) -> List[Dict[str, Any]]:
    """
    mode:
      - "abstract": evidence_texts = abstract sentences
      - "abstract_plus_question": prepend question as first evidence sentence (optional)
    """
    out: List[Dict[str, Any]] = []
    for i, r in enumerate(df.to_dict(orient="records")):
        claim = _pick_claim_text(r)
        if not claim:
            continue

        ev = _extract_context_sentences(r)
        if not ev:
            continue

        if mode == "abstract_plus_question":
            ev = [claim] + ev

        label = _pick_label(r)

        rid = r.get("id", r.get("pubid", None))
        if rid is None:
            # deterministic-ish id: file + row index
            rid = f"{Path(str(r.get('_source_file','unknown'))).stem}:{i}"

        out.append(
            {
                "dataset": "pubmedqa",
                "split": split,
                "id": str(rid),
                "claim": claim,
                "label": None if label is None else str(label),
                "evidence_texts": ev,
                "meta": {
                    "source_file": r.get("_source_file"),
                    "mode": mode,
                    # keep extra fields if they exist (helpful for audit)
                    "pmid": r.get("pubmed_id", r.get("pmid")),
                    "question_id": r.get("question_id"),
                },
            }
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", type=Path, required=True, help="Folder containing PubMedQA parquet files (recursively).")
    ap.add_argument("--out_dir", type=Path, default=Path("datasets/pubmedqa"), help="Output folder for jsonl.")
    ap.add_argument(
        "--split",
        type=str,
        default="all",
        choices=["all", "train", "dev", "test", "unknown"],
        help="Which split to export. Split is inferred from file paths.",
    )
    ap.add_argument(
        "--mode",
        type=str,
        default="abstract",
        choices=["abstract", "abstract_plus_question"],
        help="How to build evidence_texts.",
    )
    ap.add_argument("--max_rows", type=int, default=0, help="Optional cap per split (0 = no cap).")
    args = ap.parse_args()

    parquet_files = sorted(args.in_dir.rglob("*.parquet"))
    if not parquet_files:
        raise SystemExit(f"No .parquet files found under: {args.in_dir}")

    # group by inferred split
    by_split: Dict[str, List[Path]] = {}
    for pf in parquet_files:
        sp = _infer_split_from_path(pf)
        by_split.setdefault(sp, []).append(pf)

    # which splits to export
    splits: List[str]
    if args.split == "all":
        splits = sorted(by_split.keys())
    else:
        splits = [args.split]

    args.out_dir.mkdir(parents=True, exist_ok=True)

    total_written = 0
    for sp in splits:
        files = by_split.get(sp, [])
        if not files:
            print(f"Skipping split '{sp}' (no files).")
            continue

        df = _read_parquet_files(files)
        if df.empty:
            print(f"Skipping split '{sp}' (empty parquet read).")
            continue

        rows = convert_df_to_rows(df, split=sp, mode=args.mode)

        if args.max_rows and len(rows) > args.max_rows:
            rows = rows[: args.max_rows]

        out_path = args.out_dir / f"pubmedqa_{sp}_{args.mode}.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        print(f"Wrote {len(rows)} rows -> {out_path}")
        total_written += len(rows)

    print(f"DONE. Total rows written: {total_written}")


if __name__ == "__main__":
    main()
