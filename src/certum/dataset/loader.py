# src/certum/dataset.py
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Tuple, Dict, Any, Iterator, Optional
import json
import random

from certum.protocols.evidence_store import EvidenceStore


def load_examples(
    kind: str,
    path: Path,
    n: int,
    seed: int,
    *,
    evidence_store: Optional[EvidenceStore] = None,
    model: str = "",
    include_context: bool = True,
    require_complete: bool = True,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Load (claim, evidence-set) pairs.

    For FEVEROUS, if `evidence_store` is provided, evidence strings are resolved
    via the evidence store and evidence embeddings are pulled from it.
    """
    rng = random.Random(seed)
    out: List[Dict[str, Any]] = []
    stats: Dict[str, Any] = {}

    if kind == "feverous":
        rows, stats = load_feverous_pairs(
            path,
            evidence_store=evidence_store,
            include_context=include_context,
            require_complete=require_complete,
        )

        rng.shuffle(rows)
        for r in rows:
            claim = r.get("claim")
            ev = r.get("evidence")
            if not isinstance(claim, str) or not claim.strip():
                continue
            if not isinstance(ev, list) or not ev:
                continue

            ev = stable_unique([str(x).strip() for x in ev if str(x).strip()])
            if not ev:
                continue

            ex: Dict[str, Any] = {
                "claim": claim.strip(),
                "evidence": ev,
                "label": r.get("label"),
                "id": r.get("id"),
                "evidence_set": r.get("evidence_set"),
            }

            out.append(ex)
            if len(out) >= n:
                break

        return out, stats

    if kind == "jsonl":
        rows = list(iter_jsonl(path))
        rng.shuffle(rows)

        claim_keys = ["claim", "claim_text", "text"]
        evidence_keys = [
            "evidence_texts",
            "evidence",
            "rationale",
            "rationale_texts",
            "evidence_sentence_texts",
            "evidence_text",
        ]

        for r in rows:
            claim = _pick_first_str(r, claim_keys)
            ev = _pick_evidence_list(r, evidence_keys)
            if not claim or not ev:
                continue

            ev = stable_unique([str(x).strip() for x in ev if str(x).strip()])
            if not ev:
                continue

            out.append({"claim": claim, "evidence": ev, "label": r.get("label")})
            if len(out) >= n:
                break

        return out, stats

    raise ValueError("kind must be: feverous | jsonl")


def load_feverous_pairs(
    path: Path,
    evidence_store: Optional[EvidenceStore],
    include_context: bool,
    require_complete: bool,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Build claim↔evidence-set pairs.

    Each FEVEROUS example can have multiple evidence sets; we treat each set as a separate
    (claim, evidence) pair to preserve the correct mapping.
    """
    pairs: List[Dict[str, Any]] = []
    stats: Dict[str, Any] = {
        "claims_seen": 0,
        "evidence_sets_seen": 0,
        "evidence_sets_kept": 0,
        "evidence_sets_dropped": 0,
        "missing_ids_total": 0,
    }

    for ex in load_feverous(path):
        stats["claims_seen"] += 1
        claim = ex.get("claim", "")
        label = ex.get("label", "")
        ex_id = ex.get("id", None)

        for set_idx, eset in enumerate(_iter_evidence_sets(ex)):
            stats["evidence_sets_seen"] += 1

            ids = required_ids_for_evidence_set(eset, include_context)

            if evidence_store is None:
                # Fallback: raw ids as evidence (not recommended)
                pairs.append(
                    {
                        "id": ex_id,
                        "set_idx": set_idx,
                        "label": label,
                        "claim": claim,
                        "evidence": ids,
                        "evidence_ids": ids,
                        "evidence_vecs": None,
                    }
                )
                stats["evidence_sets_kept"] += 1
                continue

            texts, missing = evidence_store.get_texts(ids)

            if missing:
                stats["missing_ids_total"] += len(missing)
                if require_complete:
                    stats["evidence_sets_dropped"] += 1
                    continue

            pairs.append(
                {
                    "id": ex_id,
                    "set_idx": set_idx,
                    "label": label,
                    "claim": claim,
                    "evidence": texts,       # resolved strings
                    "evidence_ids": ids,     # original ids
                    "evidence_vecs": None,   # (n,d) float32
                    "evidence_set": eset,    # optional: keep for audit/debug
                }
            )
            stats["evidence_sets_kept"] += 1

    return pairs, stats


def load_feverous(path: Path) -> Iterator[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _iter_evidence_sets(example: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    ev = example.get("evidence", [])
    if isinstance(ev, dict):
        yield ev
    elif isinstance(ev, list):
        for e in ev:
            if isinstance(e, dict):
                yield e


def required_ids_for_evidence_set(evidence_set: Dict[str, Any], include_context: bool) -> List[str]:
    content_ids = list(evidence_set.get("content", []) or [])
    if not include_context:
        return stable_unique([str(x) for x in content_ids])

    ctx = evidence_set.get("context", {}) or {}
    ctx_ids: List[str] = []
    for cid in content_ids:
        ctx_ids.extend(ctx.get(cid, []) or [])

    all_ids = [str(x) for x in content_ids] + [str(x) for x in ctx_ids]
    return stable_unique(all_ids)


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def stable_unique(xs: Iterable[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for x in xs:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _pick_first_str(row: dict, keys: List[str]) -> Optional[str]:
    for k in keys:
        v = row.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def _pick_evidence_list(row: dict, keys: List[str]) -> Optional[List[str]]:
    for k in keys:
        v = row.get(k)
        if isinstance(v, list) and v:
            out = [str(x).strip() for x in v if str(x).strip()]
            if out:
                return out
        if isinstance(v, str) and v.strip():
            return [v.strip()]
    return None
