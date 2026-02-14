# src/certum/utils/id_utils.py
from __future__ import annotations

import hashlib
from typing import Iterable, Tuple


def _norm_text(s: str) -> str:
    # stable: strip + collapse whitespace
    return " ".join((s or "").strip().split())


def _norm_evidence(evidence: Iterable[str]) -> str:
    # preserve order (do NOT sort) — evidence order is part of the pair definition
    parts = [_norm_text(x) for x in (evidence or []) if _norm_text(x)]
    return "\n".join(parts)


def sha1_hex(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def compute_ids(claim: str, evidence: Iterable[str]) -> Tuple[str, str, str]:
    """
    Returns (pair_id, claim_id, evidence_id) as hex strings.
    Deterministic across runs as long as text content is the same.
    """
    c = _norm_text(claim)
    e = _norm_evidence(evidence)

    claim_id = sha1_hex(c)
    evidence_id = sha1_hex(e)
    pair_id = sha1_hex(c + "\n---EVIDENCE---\n" + e)

    return pair_id, claim_id, evidence_id
