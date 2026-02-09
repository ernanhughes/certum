#!/usr/bin/env python3
"""
gate_suite.py — Hallucination-energy gating with negative-control calibration (v2)

Core idea:
- Energy = 1 - ||U_r^T c||^2  where U_r is evidence subspace basis (SVD on top-k evidence vectors)
- Decision: accept if energy <= tau
- Calibrate tau on shuffled/deranged negatives to bound FAR (false-accept rate)
- Report metrics on a holdout split to avoid "tuned on test"

This evaluates evidence-conditioned groundedness (curated evidence), not raw-document hallucination detection.
"""

from __future__ import annotations

import argparse
import json
import random
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# src/verity_gate/dataset.py
from typing import Iterator

def load_feverous(path: Path) -> Iterator[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def _stable_unique(xs: Iterable[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for x in xs:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


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


class FeverousCache:
    """Read-only helper for feverous_cache.db.

    We use this to:
      1) validate an evidence set is *complete* (every required element_id resolved+embedded)
      2) retrieve the resolved text + cached embedding vectors for those element_ids
    """

    def __init__(self, cache_db: Path):
        self.cache_db = Path(cache_db)
        self.conn = sqlite3.connect(str(self.cache_db))
        self.conn.row_factory = sqlite3.Row

        # Small speedups; safe for read-only usage
        cur = self.conn.cursor()
        cur.execute("PRAGMA journal_mode=WAL")
        cur.execute("PRAGMA synchronous=NORMAL")
        cur.execute("PRAGMA temp_store=MEMORY")
        cur.close()

    def close(self) -> None:
        try:
            self.conn.close()
        except Exception:
            pass

    def has_ok_embedding(self, element_id: str, model: str) -> bool:
        cur = self.conn.cursor()
        cur.execute(
            """
            SELECT 1
            FROM resolved r
            JOIN embeddings e
              ON e.element_id = r.element_id
            WHERE r.element_id = ?
              AND r.ok = 1
              AND e.model = ?
            LIMIT 1
            """,
            (element_id, model),
        )
        row = cur.fetchone()
        cur.close()
        return row is not None

    def get_texts_and_vecs(self, element_ids: List[str], model: str) -> Tuple[List[str], np.ndarray, List[str]]:
        """Return (texts, vecs, missing_ids) for requested element_ids.

        Cache DB schema expectation:
          - resolved.ok = 1 indicates resolved text is valid
          - embeddings has (element_id, model, dim, vec) where vec is raw float32 bytes (v.tobytes()).
        """
        if not element_ids:
            return [], np.zeros((0, 0), dtype=np.float32), []

        # SQLite has a variable limit; chunk to be safe
        rows: Dict[str, sqlite3.Row] = {}
        chunk = 900
        for i in range(0, len(element_ids), chunk):
            sub = element_ids[i : i + chunk]
            q_marks = ",".join(["?"] * len(sub))
            sql = f"""
                SELECT r.element_id, r.text, e.vec, e.dim
                FROM resolved r
                JOIN embeddings e
                  ON e.element_id = r.element_id
                WHERE r.ok = 1
                  AND e.model = ?
                  AND r.element_id IN ({q_marks})
            """
            cur = self.conn.cursor()
            cur.execute(sql, [model, *sub])
            for r in cur.fetchall():
                rows[str(r["element_id"])] = r
            cur.close()

        texts: List[str] = []
        vecs_list: List[np.ndarray] = []
        missing: List[str] = []
        dim_expected: Optional[int] = None

        for eid in element_ids:
            r = rows.get(eid)
            if r is None:
                missing.append(eid)
                continue

            txt = r["text"]
            vec_blob = r["vec"]
            dim = int(r["dim"])

            if dim_expected is None:
                dim_expected = dim
            if dim_expected != dim:
                missing.append(eid)
                continue

            v = np.frombuffer(vec_blob, dtype=np.float32)
            if v.ndim != 1 or v.shape[0] != dim:
                missing.append(eid)
                continue

            texts.append(str(txt))
            vecs_list.append(v)

        if not vecs_list:
            return [], np.zeros((0, 0), dtype=np.float32), missing

        vecs = np.stack(vecs_list, axis=0).astype(np.float32, copy=False)
        return texts, vecs, missing


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


class HFEmbedder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        )


# -----------------------------------------------------------------------------
# Fixed (editorial) policies: your "hard gate" knobs
# -----------------------------------------------------------------------------
POLICIES = {
    "editorial": 0.55,
    "standard": 0.45,
    "strict": 0.30,
}


def apply_policy(energy: float, regime: str, delta: float = 0.10) -> str:
    tau = float(POLICIES[regime])
    if energy <= tau:
        return "accept"
    if energy <= tau + float(delta):
        return "review"
    return "reject"


# -----------------------------------------------------------------------------
# Hallucination Energy (SVD residual)
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class EnergyResult:
    energy: float           # in [0,1]
    explained: float        # ||U_r^T c||^2
    identity_error: float   # |1 - (explained + energy)|
    topk: int
    rank_r: int
    effective_rank: int
    used_count: int


def _unit_norm(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = float(np.linalg.norm(x))
    if n < eps:
        return x * 0.0
    return x / n


def _unit_norm_rows(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms < eps, 1.0, norms)
    return X / norms


def cosine_scores(c: np.ndarray, E: np.ndarray) -> np.ndarray:
    # c: (d,) unit, E: (n,d) unit rows
    return (E @ c).astype(np.float32)


def build_evidence_basis(
    c: np.ndarray,
    E: np.ndarray,
    *,
    top_k: int,
    rank_r: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (U_r, S)
      - U_r: (d, r) orthonormal basis vectors (columns)
      - S: singular values
    """
    if E.size == 0:
        return np.zeros((c.shape[0], 0), dtype=np.float32), np.array([], dtype=np.float32)

    top_k = max(1, int(top_k))
    rank_r = max(1, int(rank_r))

    scores = cosine_scores(c, E)
    k = min(top_k, E.shape[0])
    idx = np.argsort(-scores)[:k]
    E_k = E[idx]  # (k,d)

    # Thin SVD: E_k = U * diag(S) * Vt
    _, S, Vt = np.linalg.svd(E_k, full_matrices=False)

    r_full = Vt.shape[0]
    r = min(rank_r, r_full)
    U_r = Vt[:r].T  # (d,r)

    return U_r.astype(np.float32), S.astype(np.float32)


def hallucination_energy_svd(
    claim_vec: np.ndarray,
    evidence_vecs: np.ndarray,
    *,
    top_k: int = 12,
    rank_r: int = 8,
) -> EnergyResult:
    if claim_vec is None or evidence_vecs is None:
        return EnergyResult(
            energy=1.0,
            explained=0.0,
            identity_error=1.0,
            topk=int(top_k),
            rank_r=int(rank_r),
            effective_rank=0,
            used_count=0,
        )

    c = _unit_norm(np.asarray(claim_vec, dtype=np.float32))

    E = np.asarray(evidence_vecs, dtype=np.float32)
    if E.ndim != 2 or E.shape[0] == 0:
        return EnergyResult(
            energy=1.0,
            explained=0.0,
            identity_error=1.0,
            topk=int(top_k),
            rank_r=int(rank_r),
            effective_rank=0,
            used_count=0,
        )

    E = _unit_norm_rows(E)

    U_r, S = build_evidence_basis(c, E, top_k=top_k, rank_r=rank_r)
    if U_r.shape[1] == 0:
        return EnergyResult(
            energy=1.0,
            explained=0.0,
            identity_error=1.0,
            topk=int(top_k),
            rank_r=int(rank_r),
            effective_rank=0,
            used_count=int(E.shape[0]),
        )

    proj_coords = U_r.T @ c
    explained = float(np.dot(proj_coords, proj_coords))
    energy = 1.0 - explained

    # clamp
    explained = max(0.0, min(1.0, explained))
    energy = max(0.0, min(1.0, energy))

    identity_error = abs(1.0 - (explained + energy))
    effective_rank = int(np.sum(S > 1e-6))

    return EnergyResult(
        energy=energy,
        explained=explained,
        identity_error=identity_error,
        topk=int(top_k),
        rank_r=int(rank_r),
        effective_rank=effective_rank,
        used_count=int(E.shape[0]),
    )


def evaluate_claim(
    claim_vec: np.ndarray,
    evidence_vecs: np.ndarray,
    *,
    regime: str,
    top_k: int,
    rank_r: int,
) -> Tuple[EnergyResult, str, List[float]]:
    base = hallucination_energy_svd(claim_vec, evidence_vecs, top_k=top_k, rank_r=rank_r)

    # Robustness probe (same idea you had)
    probe = []
    for k in (8, 12, 20):
        kk = max(1, min(int(k), int(evidence_vecs.shape[0])))
        r = hallucination_energy_svd(claim_vec, evidence_vecs, top_k=kk, rank_r=rank_r)
        probe.append(float(r.energy))

    decision_fixed = apply_policy(base.energy, regime)
    return base, decision_fixed, probe


# -----------------------------------------------------------------------------
# IO + dataset adapters
# -----------------------------------------------------------------------------
def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def stable_unique(xs: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in xs:
        if x not in seen:
            out.append(x)
            seen.add(x)
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


def load_examples(
    kind: str,
    path: Path,
    n: int,
    seed: int,
    *,
    cache: Optional[FeverousCache] = None,
    model: str = "",
    include_context: bool = True,
    require_complete: bool = True,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Load (claim, evidence-set) pairs.

    For FEVEROUS, if `cache` is provided, evidence strings are resolved from the cache DB
    and evidence embeddings are pulled from the cache (so energy is computed on real text,
    not on element-id strings).
    """
    rng = random.Random(seed)
    out: List[Dict[str, Any]] = []
    stats: Dict[str, Any] = {}

    if kind == "feverous":
        rows, st = load_feverous_pairs(
            path,
            cache=cache,
            model=model,
            include_context=include_context,
            require_complete=require_complete,
        )
        stats = st

        rng.shuffle(rows)
        for r in rows:
            claim = r.get("claim")
            ev = r.get("evidence")
            if not isinstance(claim, str) or not claim.strip() or not isinstance(ev, list) or not ev:
                continue
            ev = stable_unique([str(x).strip() for x in ev if str(x).strip()])
            if not ev:
                continue
            ex = {
                "claim": claim.strip(),
                "evidence": ev,
                "label": r.get("label"),
                "id": r.get("id"),
                "evidence_set": r.get("evidence_set"),
            }
            if "evidence_vecs" in r:
                ex["evidence_vecs"] = r["evidence_vecs"]
            out.append(ex)
            if len(out) >= n:
                break
        return out, stats

    if kind == "jsonl":
        rows = list(iter_jsonl(path))
        rng.shuffle(rows)

        claim_keys = ["claim", "claim_text", "text"]
        evidence_keys = ["evidence_texts", "rationale", "rationale_texts", "evidence_sentence_texts", "evidence_text"]

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


# -----------------------------------------------------------------------------
# Negatives
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class DerangementResult:
    perm: List[int]
    fixed_points: int
    tries: int
    method: str


def derangement_indices(
    n: int,
    rng: random.Random,
    *,
    method: str = "uniform",   # "uniform" (publish-safe) or "sattolo"
    max_tries: int = 10_000,   # only used for uniform
) -> DerangementResult:
    """
    Return a permutation p of [0..n-1] such that p[i] != i for all i (a derangement).

    method="uniform":
        Uniform over ALL derangements (exact): sample a uniform permutation and reject
        if any fixed points exist. Expected ~e tries.

    method="sattolo":
        Sattolo's algorithm: uniform over n-cycles. Always a derangement for n>1,
        but NOT uniform over all derangements.

    Notes:
      - For n=0: returns empty.
      - For n=1: no derangement exists; returns [0] with fixed_points=1 (caller should handle).
    """
    if n < 0:
        raise ValueError("n must be >= 0")

    if n == 0:
        return DerangementResult([], fixed_points=0, tries=1, method=method)

    if n == 1:
        # No derangement exists; make this explicit and let caller decide.
        return DerangementResult([0], fixed_points=1, tries=1, method=method)

    if method == "sattolo":
        p = list(range(n))
        # Sattolo: produce a single cycle => no fixed points for n>1.
        for i in range(n - 1, 0, -1):
            j = rng.randrange(i)  # 0 <= j < i
            p[i], p[j] = p[j], p[i]
        # sanity
        fp = sum(1 for i, v in enumerate(p) if i == v)
        assert fp == 0
        return DerangementResult(p, fixed_points=0, tries=1, method="sattolo")

    if method == "uniform":
        # Exact uniform derangement via rejection sampling.
        for t in range(1, max_tries + 1):
            p = list(range(n))
            rng.shuffle(p)
            fp = 0
            for i, v in enumerate(p):
                if i == v:
                    fp += 1
                    break
            if fp == 0:
                return DerangementResult(p, fixed_points=0, tries=t, method="uniform")

        # Extremely unlikely, but we fail closed and return a guaranteed derangement.
        # (If you prefer, raise instead.)
        fallback = derangement_indices(n, rng, method="sattolo")
        return DerangementResult(fallback.perm, fixed_points=0, tries=max_tries, method="uniform->sattolo")

    raise ValueError(f"Unknown method: {method!r}")


def assert_is_derangement(p: List[int]) -> None:
    n = len(p)
    if sorted(p) != list(range(n)):
        raise AssertionError("Not a permutation of 0..n-1")
    for i, v in enumerate(p):
        if i == v:
            raise AssertionError(f"Fixed point at i={i}")


def make_negatives(
    pairs: List[Dict[str, Any]],
    mode: str,
    seed: int = 0,
    offset: int = 1,
    embedder: Optional["HFEmbedder"] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    n = len(pairs)
    if n == 0:
        return [], {"mode": mode, "n": 0}

    # IMPORTANT: cannot form negatives when only 1 example exists
    if n == 1:
        return [], {"mode": mode, "n": 1, "error": "cannot_make_negatives_for_n=1"}

    def _neg_from(i: int, j: int) -> Dict[str, Any]:
        src = pairs[j]
        out = {
            "id": pairs[i].get("id", i),
            "claim": pairs[i]["claim"],
            "evidence": src.get("evidence", []),
            "evidence_texts": src.get("evidence_texts", []),
            "evidence_vecs": src.get("evidence_vecs", None),
            "label": "NEG",
        }
        return out

    # ------------------------------------------------------------
    # hard mined
    # ------------------------------------------------------------
    if mode == "hard_mined":
        if embedder is None:
            raise ValueError("neg_mode=hard_mined requires an embedder instance")

        valid = []
        centroids = []
        for idx, ex in enumerate(pairs):
            ev = ex.get("evidence_vecs", None)
            if ev is None:
                continue
            ev = np.asarray(ev, dtype=np.float32)
            if ev.ndim != 2 or ev.shape[0] == 0:
                continue
            ev_u = _unit_norm_rows(ev)
            c = _unit_norm(ev_u.mean(axis=0))
            if np.isfinite(c).all():
                valid.append(idx)
                centroids.append(c)

        if len(valid) < 2:
            # Fallback: use publish-safe derangement below
            mode = "deranged"
        else:
            claim_vecs = embedder.embed([ex["claim"] for ex in pairs]).astype(np.float32, copy=False)
            claim_vecs = _unit_norm_rows(claim_vecs)

            centroid_mat = np.stack(centroids, axis=0).astype(np.float32, copy=False)
            centroid_mat = _unit_norm_rows(centroid_mat)

            sim = claim_vecs @ centroid_mat.T

            vpos = {orig: pos for pos, orig in enumerate(valid)}
            for i in range(n):
                p = vpos.get(i, None)
                if p is not None:
                    sim[i, p] = -np.inf

            best_pos = np.argmax(sim, axis=1)
            best_j = [valid[int(p)] for p in best_pos]

            negs = [_neg_from(i, best_j[i]) for i in range(n)]
            meta = {
                "mode": "hard_mined",
                "n": n,
                "candidates": len(valid),
                "method": "argmax cosine(claim, evidence_centroid)",
            }
            return negs, meta

    # ------------------------------------------------------------
    # existing simple modes
    # ------------------------------------------------------------
    rng = random.Random(seed)
    idx = list(range(n))

    if mode == "cyclic":
        perm = [(i + 1) % n for i in idx]
        negs = [_neg_from(i, perm[i]) for i in idx]
        return negs, {"mode": mode, "n": n, "fixed_points": 0}

    if mode == "offset":
        off = offset % n
        perm = [(i + off) % n for i in idx]
        fixed = sum(1 for i in idx if perm[i] == i)
        negs = [_neg_from(i, perm[i]) for i in idx]
        return negs, {"mode": mode, "n": n, "offset": off, "fixed_points": fixed}

    if mode == "permute":
        perm = idx[:]
        rng.shuffle(perm)
        fixed = sum(1 for i in idx if perm[i] == i)
        negs = [_neg_from(i, perm[i]) for i in idx]
        return negs, {"mode": mode, "n": n, "fixed_points": fixed}

    # ------------------------------------------------------------
    # deranged (publish-safe)
    # ------------------------------------------------------------
    # Use your canonical derangement sampler
    dr = derangement_indices(n, rng, method="uniform")
    perm = dr.perm

    # Fail closed: if anything is wrong, crash loudly.
    # (Optional but recommended while you iterate.)
    assert_is_derangement(perm)

    negs = [_neg_from(i, perm[i]) for i in idx]
    meta = {
        "mode": "deranged",
        "n": n,
        "fixed_points": dr.fixed_points,  # should be 0 for n>1
        "tries": dr.tries,
        "method": dr.method,              # "uniform" or "uniform->sattolo"
    }
    return negs, meta

def summarize(x: np.ndarray) -> Dict[str, float]:
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return {"count": 0.0}
    return {
        "count": float(x.size),
        "mean": float(x.mean()),
        "std": float(x.std()),
        "p50": float(np.percentile(x, 50)),
        "p90": float(np.percentile(x, 90)),
        "p95": float(np.percentile(x, 95)),
        "p99": float(np.percentile(x, 99)),
        "min": float(x.min()),
        "max": float(x.max()),
    }


def _rankdata_average_ties(x: np.ndarray) -> np.ndarray:
    """
    Rank data with average ranks for ties. Ranks are 1..N.
    """
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x, kind="mergesort")  # stable
    xs = x[order]
    ranks = np.empty_like(xs, dtype=np.float64)

    n = len(xs)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and xs[j + 1] == xs[i]:
            j += 1
        # average rank in [i..j] with 1-indexing
        avg_rank = 0.5 * (i + j) + 1.0
        ranks[i : j + 1] = avg_rank
        i = j + 1

    out = np.empty_like(ranks)
    out[order] = ranks
    return out


def auc_lower_is_positive(pos: np.ndarray, neg: np.ndarray) -> float:
    """
    AUC where LOWER energy indicates positive (better grounded).
    Equivalent to AUC on score = -energy where HIGHER indicates positive.
    """
    pos = np.asarray(pos, dtype=np.float64)
    neg = np.asarray(neg, dtype=np.float64)
    if pos.size == 0 or neg.size == 0:
        return float("nan")

    scores = np.concatenate([-pos, -neg])  # higher = more positive
    labels = np.concatenate([np.ones_like(pos, dtype=np.int32), np.zeros_like(neg, dtype=np.int32)])

    ranks = _rankdata_average_ties(scores)
    pos_ranks = ranks[labels == 1]

    n_pos = float(len(pos))
    n_neg = float(len(neg))
    auc = (pos_ranks.sum() - n_pos * (n_pos + 1.0) / 2.0) / (n_pos * n_neg)
    return float(auc)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Negative-control calibration for hallucination-energy policy gates (v2)")
    ap.add_argument("--kind", required=True, choices=["feverous", "jsonl"])
    ap.add_argument("--in_path", required=True, type=Path)

    ap.add_argument(
        "--model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformer model name used to embed claims and (if needed) evidence.",
    )

    # FEVEROUS cache: resolved text + embeddings (built via feverous_cache_build.py)
    ap.add_argument(
        "--cache_db",
        type=Path,
        default=None,
        help="Optional SQLite cache DB with resolved evidence text + embeddings. Strongly recommended for kind=feverous.",
    )
    ap.add_argument(
        "--no_context",
        action="store_true",
        help="For kind=feverous, do NOT include contextual elements (titles/sections/neighbor cells) when building evidence sets.",
    )
    ap.add_argument(
        "--allow_incomplete",
        action="store_true",
        help="For kind=feverous, allow evidence sets with missing cache entries (otherwise dropped).",
    )

    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=1337)

    ap.add_argument("--regime", type=str, default="standard", choices=["standard", "strict", "editorial"])
    ap.add_argument("--delta", type=float, default=0.10)

    ap.add_argument("--top_k", type=int, default=12)
    ap.add_argument("--rank_r", type=int, default=8)

    ap.add_argument("--far", type=float, default=0.01, help="Target FAR on negatives for rule: accept if energy <= tau")
    ap.add_argument("--cal_frac", type=float, default=0.5, help="Fraction of examples used to calibrate tau; rest is holdout eval")

    ap.add_argument("--neg_mode", type=str, default="deranged", choices=["deranged", "offset", "cyclic", "permute", "hard_mined"])
    ap.add_argument("--neg_offset", type=int, default=37)

    ap.add_argument("--out_report", type=Path, default=Path("artifacts/negcal_report.json"))
    ap.add_argument("--out_pos_scored", type=Path, default=None, help="Optional JSONL of scored positives (holdout split)")
    ap.add_argument("--out_neg_scored", type=Path, default=None, help="Optional JSONL of scored negatives (holdout split)")
    ap.add_argument("--plot_png", type=Path, default=None, help="Optional PNG histogram (pos vs neg, holdout split)")
    args = ap.parse_args()

    # -----------------------------
    # Load + split
    # -----------------------------
    cache: Optional[FeverousCache] = None
    if args.kind == "feverous" and args.cache_db is not None:
        cache = FeverousCache(args.cache_db)

    pairs, load_stats = load_examples(
        args.kind,
        args.in_path,
        args.n,
        args.seed,
        cache=cache,
        model=args.model,
        include_context=not args.no_context,
        require_complete=not args.allow_incomplete,
    )
    if len(pairs) < 50:
        raise RuntimeError(f"Too few usable examples ({len(pairs)}). Check input format/evidence extraction.")

    rng = random.Random(args.seed)
    rng.shuffle(pairs)

    cal_frac = float(args.cal_frac)
    cal_frac = max(0.1, min(0.9, cal_frac))
    n_cal = max(1, int(len(pairs) * cal_frac))

    cal_pos = pairs[:n_cal]
    ev_pos = pairs[n_cal:]

    # -----------------------------
    # Embedder
    # -----------------------------
    embedder = HFEmbedder(model_name=args.model)

    # -----------------------------
    # Negatives (cal + eval)
    # -----------------------------
    cal_neg, neg_meta_cal = make_negatives(
        cal_pos,
        mode=args.neg_mode,
        seed=args.seed + 100,
        offset=args.neg_offset,
        embedder=embedder if args.neg_mode == "hard_mined" else None,
    )
    ev_neg, neg_meta_ev = make_negatives(
        ev_pos if ev_pos else cal_pos,  # if no holdout (shouldn't happen), reuse
        mode=args.neg_mode,
        seed=args.seed + 200,
        offset=args.neg_offset,
        embedder=embedder if args.neg_mode == "hard_mined" else None,
    )
    claim_cache: Dict[str, np.ndarray] = {}

    def compute_energy_for_pair(ex: Dict[str, Any]) -> Tuple[float, float, int, int, List[float], str]:
        c = ex["claim"]
        claim_vec = claim_cache.get(c)
        if claim_vec is None:
            claim_vec = embedder.embed([c])[0]
            claim_cache[c] = claim_vec
        if "evidence_vecs" in ex and ex["evidence_vecs"] is not None:
            ev_vecs = ex["evidence_vecs"]
        else:
            ev_vecs = embedder.embed(ex["evidence"])
        base, decision_fixed, probe = evaluate_claim(
            claim_vec,
            ev_vecs,
            regime=args.regime,
            top_k=min(int(args.top_k), len(ex["evidence"])),
            rank_r=int(args.rank_r),
        )
        return (
            float(base.energy),
            float(base.explained),
            int(base.effective_rank),
            int(len(ex["evidence"])),
            [float(x) for x in probe],
            decision_fixed,
        )

    # -----------------------------
    # Compute energies
    # -----------------------------
    cal_pos_e = []
    cal_neg_e = []

    for ex in tqdm(cal_pos, desc="Compute CAL POS energies"):
        e, *_ = compute_energy_for_pair(ex)
        cal_pos_e.append(e)

    for ex in tqdm(cal_neg, desc="Compute CAL NEG energies"):
        e, *_ = compute_energy_for_pair(ex)
        cal_neg_e.append(e)

    cal_pos_e = np.asarray(cal_pos_e, dtype=np.float32)
    cal_neg_e = np.asarray(cal_neg_e, dtype=np.float32)

    # -----------------------------
    # Calibrate tau on CAL negatives
    # -----------------------------
    far = float(args.far)
    far = max(0.0, min(0.5, far))  # sane range
    tau_cal = float(np.percentile(cal_neg_e, far * 100.0))  # accept if energy <= tau

    cal_FAR = float((cal_neg_e <= tau_cal).mean())
    cal_TPR = float((cal_pos_e <= tau_cal).mean())
    cal_AUC = auc_lower_is_positive(cal_pos_e, cal_neg_e)

    # -----------------------------
    # Holdout eval energies
    # -----------------------------
    if ev_pos:
        ev_pos_e = []
        ev_neg_e = []

        for ex in tqdm(ev_pos, desc="Compute EVAL POS energies"):
            e, *_ = compute_energy_for_pair(ex)
            ev_pos_e.append(e)

        for ex in tqdm(ev_neg, desc="Compute EVAL NEG energies"):
            e, *_ = compute_energy_for_pair(ex)
            ev_neg_e.append(e)

        ev_pos_e = np.asarray(ev_pos_e, dtype=np.float32)
        ev_neg_e = np.asarray(ev_neg_e, dtype=np.float32)

        ev_FAR = float((ev_neg_e <= tau_cal).mean())
        ev_TPR = float((ev_pos_e <= tau_cal).mean())
        ev_AUC = auc_lower_is_positive(ev_pos_e, ev_neg_e)
    else:
        ev_pos_e = np.asarray([], dtype=np.float32)
        ev_neg_e = np.asarray([], dtype=np.float32)
        ev_FAR, ev_TPR, ev_AUC = float("nan"), float("nan"), float("nan")

    # -----------------------------
    # Hardest negatives (lowest energy mismatches)
    # -----------------------------
    hardest = []
    if ev_pos:
        # score a small sample of negatives with text for audit
        scored = []
        for i, ex in enumerate(ev_neg[: min(200, len(ev_neg))]):
            e, _, _, _, _, _ = compute_energy_for_pair(ex)
            scored.append((e, ex))
        scored.sort(key=lambda t: t[0])  # lowest energy = most dangerous negative
        for e, ex in scored[:10]:
            hardest.append({
                "energy": float(e),
                "claim": ex["claim"][:200],
                "evidence_0": ex["evidence"][0][:200] if ex["evidence"] else None,
                "evidence_len": len(ex["evidence"]),
            })

    # -----------------------------
    # Report
    # -----------------------------
    report = {
        "kind": args.kind,
        "in_path": str(args.in_path),
        "n_total": int(len(pairs)),
        "split": {
            "cal_frac": cal_frac,
            "n_cal": int(len(cal_pos)),
            "n_eval": int(len(ev_pos)),
        },
        "params": {
            "regime": args.regime,
            "delta": float(args.delta),
            "top_k": int(args.top_k),
            "rank_r": int(args.rank_r),
            "far_target": float(args.far),
            "tau_cal": float(tau_cal),
            "seed": int(args.seed),
            "neg_mode": args.neg_mode,
            "neg_offset": int(args.neg_offset),
            "neg_meta_cal": neg_meta_cal,
            "neg_meta_eval": neg_meta_ev,
        },
        "stats": {
            "cal": {
                "pos": summarize(cal_pos_e),
                "neg": summarize(cal_neg_e),
                "FAR": cal_FAR,
                "TPR": cal_TPR,
                "AUC": cal_AUC,
            },
            "eval": {
                "pos": summarize(ev_pos_e),
                "neg": summarize(ev_neg_e),
                "FAR": ev_FAR,
                "TPR": ev_TPR,
                "AUC": ev_AUC,
            },
        },
        "hardest_negatives_preview": hardest,
        "interpretation": {
            "accept_rule": "accept if energy <= tau",
            "calibration": "tau is set to the FAR-percentile of CAL negative energies",
            "what_this_tests": "evidence-conditioned groundedness (curated evidence + adversarial mismatches)",
            "what_this_does_not_test": "raw-document hallucination detection (uncurated documents)",
        },
    }

    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    args.out_report.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n" + "=" * 80)
    print("✅ Negative-control calibration (v2, NO ORACLE)")
    print("=" * 80)
    print(f"tau_cal (accept if energy<=tau): {tau_cal:.4f}")
    print(f"CAL  FAR/TPR/AUC: {cal_FAR:.3%} / {cal_TPR:.3%} / {cal_AUC:.3f}")
    if len(ev_pos) > 0:
        print(f"EVAL FAR/TPR/AUC: {ev_FAR:.3%} / {ev_TPR:.3%} / {ev_AUC:.3f}")
    print(f"Wrote report -> {args.out_report}")

    # -----------------------------
    # Optional: write scored JSONLs (eval split)
    # -----------------------------
    def write_scored(path: Path, data: List[Dict[str, Any]], tau: float, tag: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for ex in tqdm(data, desc=f"Write {tag} scored"):
                e, cov, erank, evlen, probe, decision_fixed = compute_energy_for_pair(ex)
                decision_tau = "accept" if e <= tau else ("review" if e <= tau + float(args.delta) else "reject")
                f.write(json.dumps({
                    "claim": ex["claim"],
                    "label": ex.get("label"),
                    "energy": float(e),
                    "coverage": float(cov),
                    "effective_rank": int(erank),
                    "evidence_len": int(evlen),
                    "probe": probe,
                    "decision_fixed": decision_fixed,
                    "decision_tau": decision_tau,
                    "tau_cal": float(tau),
                }, ensure_ascii=False) + "\n")

    if args.out_pos_scored is not None:
        write_scored(args.out_pos_scored, ev_pos if len(ev_pos) > 0 else cal_pos, tau_cal, "POS")

    if args.out_neg_scored is not None:
        write_scored(args.out_neg_scored, ev_neg if len(ev_neg) > 0 else cal_neg, tau_cal, "NEG")

    # -----------------------------
    # Optional: plot
    # -----------------------------
    if args.plot_png is not None:
        try:
            import matplotlib.pyplot as plt

            pos_plot = ev_pos_e if len(ev_pos_e) > 0 else cal_pos_e
            neg_plot = ev_neg_e if len(ev_neg_e) > 0 else cal_neg_e

            args.plot_png.parent.mkdir(parents=True, exist_ok=True)
            plt.figure()
            plt.hist(pos_plot, bins=40, alpha=0.7, label="pos (real evidence)")
            plt.hist(neg_plot, bins=40, alpha=0.7, label=f"neg ({args.neg_mode})")
            plt.axvline(tau_cal, linestyle="--", linewidth=2, label=f"tau_cal={tau_cal:.3f}")
            plt.xlabel("Hallucination Energy")
            plt.ylabel("Count")
            plt.legend()
            plt.title("Energy separation: real vs negatives (holdout eval)")
            plt.savefig(args.plot_png, dpi=160)
            plt.close()
            print(f"Wrote plot -> {args.plot_png}")
        except Exception as e:
            print(f"Plot skipped: {e}")

    if cache is not None:
        cache.close()


if __name__ == "__main__":
    main()
