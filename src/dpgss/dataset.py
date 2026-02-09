# src/dpgss/dataset.py
from pathlib import Path
from typing import Iterable, List, Tuple, Dict, Any, Iterator, Optional
import json
import random

from .cache import FeverousCache  # Circular import guard - place in separate module


def load_feverous_samples(
    in_path: Path,
    cache_db: Path,
    n: int,
    seed: int,
    model: str = "sentence-transformers/all-MiniLM-L6-v2",
    include_context: bool = True,
    require_complete: bool = True,
) -> List[Tuple[str, List[str]]]:
    """
    Production-grade loader using FeverousCache for complete evidence resolution.
    
    Use this when you need fully resolved evidence text (not just context snippets).
    Requires pre-built cache DB from feverous_cache_build.py.
    """
    cache = FeverousCache(cache_db)
    try:
        # Reuse your existing load_feverous_pairs logic here
        pairs, _ = load_feverous_pairs(
            in_path,
            cache=cache,
            model=model,
            include_context=include_context,
            require_complete=require_complete,
        )
        
        # Convert to (claim, evidence) tuples
        samples = [
            (p["claim"].strip(), p["evidence"])
            for p in pairs
            if p.get("claim") and p.get("evidence")
        ]
        
        random.Random(seed).shuffle(samples)
        return samples[:min(n, len(samples))]
    finally:
        cache.close()

def load_feverous_pairs(
    path: Path,
    cache: Optional[FeverousCache],
    model: str,
    include_context: bool,
    require_complete: bool,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Build claim↔evidence-set pairs.

    Each FEVEROUS example can have multiple evidence sets; we treat each set as a separate
    (claim, evidence) pair to preserve the correct mapping.
    """
    pairs: List[Dict[str, Any]] = []
    stats = {
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

        for j, eset in enumerate(_iter_evidence_sets(ex)):
            stats["evidence_sets_seen"] += 1
            ids = required_ids_for_evidence_set(eset, include_context)

            if cache is None:
                # Fallback: use the raw ids as "evidence" (not recommended).
                pairs.append({
                    "id": ex_id,
                    "set_idx": j,
                    "label": label,
                    "claim": claim,
                    "evidence": ids,
                    "evidence_ids": ids,
                    "evidence_vecs": None,
                })
                stats["evidence_sets_kept"] += 1
                continue

            # Validate completeness and fetch texts+vecs.
            texts, vecs, missing = cache.get_texts_and_vecs(ids, model)
            if missing:
                stats["missing_ids_total"] += len(missing)
                if require_complete:
                    stats["evidence_sets_dropped"] += 1
                    continue

            # If not requiring complete, we keep what we have.
            if not texts or vecs.size == 0:
                stats["evidence_sets_dropped"] += 1
                continue

            pairs.append({
                "id": ex_id,
                "set_idx": j,
                "label": label,
                "claim": claim,
                "evidence": texts,
                "evidence_ids": ids,
                "evidence_vecs": vecs,
            })
            stats["evidence_sets_kept"] += 1

    return pairs, stats

def _iter_evidence_sets(example: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    ev = example.get("evidence", [])
    if isinstance(ev, dict):
        yield ev
    elif isinstance(ev, list):
        for e in ev:
            if isinstance(e, dict):
                yield e


def required_ids_for_evidence_set(evidence_set: Dict[str, Any], include_context: bool) -> List[str]:
    """Return the *element_ids* needed to consider an evidence set complete.

    FEVEROUS evidence sets contain:
      - content: ["Page_sentence_0", "Page_cell_0_1_1", ...]
      - context: { content_id: ["Page_title", "Page_section_4", ...], ... }

    If include_context=True we require both the content ids and their linked context ids.
    """
    content_ids = list(evidence_set.get("content", []) or [])
    if not include_context:
        return _stable_unique([str(x) for x in content_ids])

    ctx = evidence_set.get("context", {}) or {}
    ctx_ids: List[str] = []
    for cid in content_ids:
        ctx_ids.extend(ctx.get(cid, []) or [])
    all_ids = [str(x) for x in content_ids] + [str(x) for x in ctx_ids]
    return _stable_unique(all_ids)

def _stable_unique(xs: Iterable[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for x in xs:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def load_feverous(path: Path) -> Iterator[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

