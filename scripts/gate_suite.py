#!/usr/bin/env python3
"""
gate_suite.py — Hallucination-energy gating with negative-control calibration (v2)

Core idea:
- Energy = 1 - ||U_r^T c||^2  where U_r is an evidence-subspace basis (SVD on top-k evidence vectors)
- Decision: accept/review/reject using calibrated thresholds (tau_accept, tau_review)
- Calibrate thresholds on negative controls (mismatched evidence) to bound FAR(s)
- Optional: hard-mined negatives (hard_mined) and distractor sentences (signal-in-noise)
- Report metrics on a holdout split to avoid “tuned on test”

This evaluates evidence-conditioned groundedness (curated evidence), not raw-document hallucination detection.
"""

import argparse
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

# Optional plotting
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# Embedding backend (HuggingFace / sentence-transformers)
# -----------------------------------------------------------------------------
# The project uses a local embedder wrapper; keep it simple here to remain 1-file.
from sentence_transformers import SentenceTransformer


class HFEmbedder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        if SentenceTransformer is None:
            raise RuntimeError(
                "sentence-transformers is required. Install with: pip install sentence-transformers"
            )
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: List[str]) -> np.ndarray:
        # returns float32 vectors
        vecs = self.model.encode(texts, normalize_embeddings=False, show_progress_bar=False)
        return np.asarray(vecs, dtype=np.float32)


# -----------------------------------------------------------------------------
# Dataset adapters (kept minimal and robust)
# -----------------------------------------------------------------------------
def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def save_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def stable_unique(seq: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for x in seq:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def extract_pairs_kind(rows: List[Dict[str, Any]], kind: str) -> List[Dict[str, Any]]:
    """
    Normalizes different input formats into:
    {
      "claim": str,
      "evidence": [str, ...],
      "label": optional (kept if present)
    }
    Supported kinds:
    - feverous: expects {"claim":..., "evidence":[...]} already or nested evidence
    - scifact: supports {"claim":..., "evidence":[...]} or {"claim":..., "sentences":[...]}
    - generic: expects claim/evidence fields directly
    """
    pairs: List[Dict[str, Any]] = []

    if kind == "generic":
        for r in rows:
            claim = r.get("claim")
            evidence = r.get("evidence") or r.get("sentences") or r.get("evidence_sentences")
            if not claim or not evidence:
                continue
            if isinstance(evidence, str):
                evidence = [evidence]
            evidence = [str(x).strip() for x in evidence if str(x).strip()]
            if not evidence:
                continue
            pairs.append({"claim": str(claim).strip(), "evidence": stable_unique(evidence), "label": r.get("label")})
        return pairs

    if kind == "scifact":
        for r in rows:
            claim = r.get("claim") or r.get("query") or r.get("text")
            if not claim:
                continue
            evidence = r.get("evidence") or r.get("sentences") or r.get("abstract") or r.get("abstract_sentences")
            if evidence is None:
                continue
            if isinstance(evidence, str):
                evidence = [evidence]
            evidence = [str(x).strip() for x in evidence if str(x).strip()]
            if not evidence:
                continue
            pairs.append({"claim": str(claim).strip(), "evidence": stable_unique(evidence), "label": r.get("label")})
        return pairs

    if kind == "feverous":
        # FEVEROUS can be messy; many pipelines pre-extract evidence as a list of strings already.
        for r in rows:
            claim = r.get("claim")
            if not claim:
                continue
            evidence = r.get("evidence")
            if evidence is None:
                # fallback: try "evidence_text" fields
                evidence = r.get("evidence_text") or r.get("sentences")
            if evidence is None:
                continue
            if isinstance(evidence, str):
                evidence = [evidence]
            # Flatten nested evidence if needed
            flat: List[str] = []
            for x in evidence:
                if isinstance(x, str):
                    s = x.strip()
                    if s:
                        flat.append(s)
                elif isinstance(x, dict):
                    # FEVEROUS dev_challenges style: {"content":[ids...], "context":{id:[...], ...}}
                    content = x.get("content")
                    if isinstance(content, list):
                        for y in content:
                            if isinstance(y, str) and y.strip():
                                flat.append(y.strip())
                    ctx = x.get("context")
                    if isinstance(ctx, dict):
                        for v in ctx.values():
                            if isinstance(v, list):
                                for y in v:
                                    if isinstance(y, str) and y.strip():
                                        flat.append(y.strip())
                            elif isinstance(v, str) and v.strip():
                                flat.append(v.strip())
                    # common: {"text": "..."} or {"sentence": "..."}
                    for k in ("text", "sentence", "value"):
                        if k in x and isinstance(x[k], str) and x[k].strip():
                            flat.append(x[k].strip())
                elif isinstance(x, (list, tuple)):
                    for y in x:
                        if isinstance(y, str) and y.strip():
                            flat.append(y.strip())
            flat = [s for s in flat if s]
            if not flat:
                continue
            pairs.append({"claim": str(claim).strip(), "evidence": stable_unique(flat), "label": r.get("label")})
        return pairs

    raise ValueError(f"Unknown kind: {kind}")


# -----------------------------------------------------------------------------
# Hallucination energy via SVD (evidence subspace)
# -----------------------------------------------------------------------------
@dataclass
class EnergyResult:
    energy: float
    explained: float
    effective_rank: int
    idx: np.ndarray


def _unit_norm(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    n = float(np.linalg.norm(v) + eps)
    return v / n


def _unit_norm_rows(M: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    M = np.asarray(M, dtype=np.float32)
    n = np.linalg.norm(M, axis=1, keepdims=True) + eps
    return M / n


def build_evidence_basis(
    claim_vec: np.ndarray,
    evidence_vecs: np.ndarray,
    *,
    top_k: int,
    rank_r: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Select top_k evidence vectors by cosine similarity to claim_vec,
    then compute SVD basis U on that evidence matrix (evidence vectors are rows).

    Returns:
      U_r: (d, r_eff) basis columns
      s: singular values
      idx: selected evidence indices (into original evidence list)
    """
    c = _unit_norm(claim_vec)
    E = _unit_norm_rows(evidence_vecs)

    if E.size == 0:
        return (
            np.zeros((c.shape[0], 0), dtype=np.float32),
            np.array([], dtype=np.float32),
            np.array([], dtype=np.int64),
        )

    sims = E @ c
    k = min(int(top_k), E.shape[0])
    idx = np.argsort(-sims)[:k]

    A = E[idx]  # (k, d)
    if A.shape[0] == 0:
        return (
            np.zeros((c.shape[0], 0), dtype=np.float32),
            np.array([], dtype=np.float32),
            np.array([], dtype=np.int64),
        )

    # SVD on (k, d) -> Vh is (d, d) if full; using full_matrices=False gives Vh (min(k,d), d)
    try:
        U, s, Vh = np.linalg.svd(A, full_matrices=False)
    except np.linalg.LinAlgError:
        # fallback: no basis
        return (
            np.zeros((c.shape[0], 0), dtype=np.float32),
            np.array([], dtype=np.float32),
            idx.astype(np.int64),
        )

    # Basis in embedding space: rows of Vh are principal directions; take transpose to get columns.
    # If Vh is (m, d), then V = Vh.T is (d, m).
    V = Vh.T.astype(np.float32)

    r_eff = min(int(rank_r), V.shape[1])
    if r_eff <= 0:
        return (
            np.zeros((c.shape[0], 0), dtype=np.float32),
            s.astype(np.float32),
            idx.astype(np.int64),
        )

    U_r = V[:, :r_eff]  # (d, r_eff)
    return U_r, s.astype(np.float32), idx.astype(np.int64)


def hallucination_energy_svd(
    claim_vec: np.ndarray,
    evidence_vecs: np.ndarray,
    *,
    top_k: int = 6,
    rank_r: int = 3,
) -> EnergyResult:
    c = _unit_norm(claim_vec)
    E = np.asarray(evidence_vecs, dtype=np.float32)
    U_r, s, idx = build_evidence_basis(c, E, top_k=top_k, rank_r=rank_r)

    if U_r.shape[1] == 0:
        # no basis => energy high (unknown)
        return EnergyResult(energy=1.0, explained=0.0, effective_rank=0, idx=idx)

    proj = U_r.T @ c  # (r,)
    captured = float(np.sum(proj * proj))
    energy = float(1.0 - captured)

    # explained variance proxy from singular values
    if s.size > 0:
        tot = float(np.sum(s * s) + 1e-12)
        r_eff = U_r.shape[1]
        explained = float(np.sum((s[:r_eff] * s[:r_eff])) / tot)
    else:
        explained = 0.0

    return EnergyResult(energy=energy, explained=explained, effective_rank=U_r.shape[1], idx=idx)


# -----------------------------------------------------------------------------
# Policy / decision rules
# -----------------------------------------------------------------------------
def apply_policy(energy: float, regime: str, *, delta: float = 0.05) -> str:
    """
    Fixed policy regime (not calibrated):
      - strict: accept if e <= 0.35 else reject
      - standard: accept if e <= 0.45, review if e <= 0.50, else reject
      - relaxed: accept if e <= 0.55, review if e <= 0.60, else reject
      - delta: accept if e <= 0.45, review if e <= 0.45+delta else reject
    """
    e = float(energy)
    if regime == "strict":
        return "accept" if e <= 0.35 else "reject"
    if regime == "standard":
        if e <= 0.45:
            return "accept"
        if e <= 0.50:
            return "review"
        return "reject"
    if regime == "relaxed":
        if e <= 0.55:
            return "accept"
        if e <= 0.60:
            return "review"
        return "reject"
    if regime == "delta":
        if e <= 0.45:
            return "accept"
        if e <= 0.45 + float(delta):
            return "review"
        return "reject"
    raise ValueError(f"Unknown regime: {regime}")


def probe_energies(
    claim_vec: np.ndarray,
    evidence_vecs: np.ndarray,
    *,
    top_k: int,
    rank_r: int,
) -> List[float]:
    """
    Probe energies by removing each selected evidence sentence (one at a time)
    and recomputing energy. Returns list aligned to selected idx order.
    """
    base = hallucination_energy_svd(claim_vec, evidence_vecs, top_k=top_k, rank_r=rank_r)
    idx = list(base.idx.tolist())
    out: List[float] = []
    for j in idx:
        mask = [i for i in range(evidence_vecs.shape[0]) if i != j]
        if not mask:
            out.append(float("nan"))
            continue
        r = hallucination_energy_svd(claim_vec, evidence_vecs[mask], top_k=top_k, rank_r=rank_r)
        out.append(float(r.energy))
    return out


def evaluate_claim(
    claim_vec: np.ndarray,
    evidence_vecs: np.ndarray,
    *,
    regime: str,
    top_k: int,
    rank_r: int,
    delta: float,
) -> Tuple[EnergyResult, str, List[float]]:
    """
    Returns:
      base: EnergyResult
      decision_fixed: apply_policy(base.energy, regime)
      probe: list of leave-one-out energies for selected evidence
    """
    base = hallucination_energy_svd(claim_vec, evidence_vecs, top_k=top_k, rank_r=rank_r)
    decision_fixed = apply_policy(base.energy, regime, delta=float(delta))
    probe = probe_energies(claim_vec, evidence_vecs, top_k=top_k, rank_r=rank_r)
    return base, decision_fixed, probe


# -----------------------------------------------------------------------------
# Negative-control generation
# -----------------------------------------------------------------------------
def _derangement_indices(n: int, rng: random.Random) -> List[int]:
    """
    Random derangement for n>1; returns list p such that p[i] != i.
    For n<=1 returns [0].
    """
    if n <= 1:
        return [0]
    while True:
        p = list(range(n))
        rng.shuffle(p)
        if all(p[i] != i for i in range(n)):
            return p


def make_negatives(
    pairs: List[Dict[str, Any]],
    *,
    neg_mode: str,
    seed: int,
    offset: int = 37,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Generates negative controls by mismatching evidence pools to claims.
    Modes:
      - deranged: random permutation with no fixed points
      - offset: match claim i with evidence (i+offset) mod n
      - cyclic: match claim i with evidence (i+1) mod n
      - permute: random shuffle (may include fixed points)
      - hard_mined: (handled in main with vectors)
    """
    rng = random.Random(seed)
    n = len(pairs)
    if n == 0:
        return [], {"mode": neg_mode, "warning": "no pairs"}

    if neg_mode == "offset":
        off = int(offset) % max(1, n)
        if n > 1 and off == 0:
            off = 1
        neg = []
        for i in range(n):
            j = (i + off) % n
            neg.append({"claim": pairs[i]["claim"], "evidence": pairs[j]["evidence"], "label": pairs[i].get("label")})
        return neg, {"mode": "offset", "offset": off}

    if neg_mode == "cyclic":
        neg = []
        for i in range(n):
            j = (i + 1) % n if n > 1 else i
            neg.append({"claim": pairs[i]["claim"], "evidence": pairs[j]["evidence"], "label": pairs[i].get("label")})
        return neg, {"mode": "cyclic"}

    if neg_mode == "permute":
        idx = list(range(n))
        rng.shuffle(idx)
        neg = []
        for i in range(n):
            neg.append({"claim": pairs[i]["claim"], "evidence": pairs[idx[i]]["evidence"], "label": pairs[i].get("label")})
        return neg, {"mode": "permute", "warning": "may include fixed points"}

    if neg_mode == "deranged":
        p = _derangement_indices(n, rng)
        neg = []
        for i in range(n):
            neg.append({"claim": pairs[i]["claim"], "evidence": pairs[p[i]]["evidence"], "label": pairs[i].get("label")})
        return neg, {"mode": "deranged", "guarantees": "no fixed points (n>1)"}

    raise ValueError("neg_mode must be: deranged | offset | cyclic | permute | hard_mined")


# -----------------------------------------------------------------------------
# Metrics helpers
# -----------------------------------------------------------------------------
def summarize(x: np.ndarray) -> Dict[str, float]:
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return {"n": 0.0, "min": float("nan"), "p50": float("nan"), "p90": float("nan"), "max": float("nan")}
    return {
        "n": float(x.size),
        "min": float(np.min(x)),
        "p50": float(np.percentile(x, 50)),
        "p90": float(np.percentile(x, 90)),
        "max": float(np.max(x)),
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
    }


def _rankdata_average_ties(x: np.ndarray) -> np.ndarray:
    """
    Rank data with average ranks for ties (1..n).
    """
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x)
    ranks = np.empty_like(order, dtype=np.float64)
    n = len(x)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and x[order[j + 1]] == x[order[i]]:
            j += 1
        avg_rank = 0.5 * (i + j) + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    return ranks


def auc_lower_is_positive(pos: np.ndarray, neg: np.ndarray) -> float:
    """AUC where *lower* energy implies positive (better evidence match).

    Implemented by negating energies and computing a rank-based AUC using
    average ranks for ties (Mann–Whitney U statistic).
    """
    pos = np.asarray(pos, dtype=np.float64)
    neg = np.asarray(neg, dtype=np.float64)
    if pos.size == 0 or neg.size == 0:
        return float("nan")

    scores = np.concatenate([-pos, -neg])  # higher = more positive
    labels = np.concatenate([np.ones_like(pos, dtype=np.int32), np.zeros_like(neg, dtype=np.int32)])

    ranks = _rankdata_average_ties(scores)
    pos_ranks = ranks[labels == 1]

    n_pos = float(pos.size)
    n_neg = float(neg.size)
    auc = (pos_ranks.sum() - n_pos * (n_pos + 1.0) / 2.0) / (n_pos * n_neg)
    return float(auc)


def _sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _try_git_commit(cwd: Optional[Path] = None) -> Optional[str]:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(cwd or Path.cwd()),
            stderr=subprocess.DEVNULL,
        )
        s = out.decode("utf-8", errors="ignore").strip()
        return s if s else None
    except Exception:
        return None


def _env_manifest(in_path: Path) -> Dict[str, Any]:
    st_ver = None
    try:
        import sentence_transformers as _st  # type: ignore
        st_ver = getattr(_st, "__version__", None)
    except Exception:
        pass

    torch_ver = None
    try:
        import torch  # type: ignore
        torch_ver = getattr(torch, "__version__", None)
    except Exception:
        pass

    tfm_ver = None
    try:
        import transformers as _tfm  # type: ignore
        tfm_ver = getattr(_tfm, "__version__", None)
    except Exception:
        pass

    return {
        "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "numpy": getattr(np, "__version__", None),
        "sentence_transformers": st_ver,
        "transformers": tfm_ver,
        "torch": torch_ver,
        "git_commit": _try_git_commit(),
        "input": {
            "path": str(in_path),
            "sha256": _sha256_file(in_path) if in_path.exists() else None,
            "bytes": int(in_path.stat().st_size) if in_path.exists() else None,
        },
    }


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_path", type=Path, required=True)
    ap.add_argument("--kind", type=str, default="generic", choices=["generic", "feverous", "scifact"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cal_frac", type=float, default=0.5, help="Fraction for calibration split")
    ap.add_argument("--regime", type=str, default="delta", choices=["strict", "standard", "relaxed", "delta"])
    ap.add_argument("--top_k", type=int, default=6)
    ap.add_argument("--rank_r", type=int, default=3)
    ap.add_argument("--delta", type=float, default=0.05, help="Used only for fixed-policy (regime=delta)")
    ap.add_argument("--neg_mode", type=str, default="deranged", choices=["deranged", "offset", "cyclic", "permute", "hard_mined"])
    ap.add_argument("--neg_offset", type=int, default=37)
    ap.add_argument("--neg_mine_k", type=int, default=20, help="For hard_mined: candidates per claim (larger => harder negatives)")
    ap.add_argument("--distractors", type=int, default=0, help="Add N random distractor sentences to each evidence pool (signal-in-noise test)")
    ap.add_argument("--far", type=float, default=0.01, help="Target FAR for ACCEPT (negatives <= tau_accept)")
    ap.add_argument("--far_review", type=float, default=0.05, help="Target FAR for REVIEW boundary (negatives <= tau_review)")
    ap.add_argument("--out_report", type=Path, default=Path("artifacts/report.json"))
    ap.add_argument("--out_plot", type=Path, default=Path("artifacts/sep.png"))
    ap.add_argument("--out_pos_scored", type=Path, default=Path("artifacts/pos_scored.jsonl"))
    ap.add_argument("--out_neg_scored", type=Path, default=Path("artifacts/neg_scored.jsonl"))
    args = ap.parse_args()

    rows = load_jsonl(args.in_path)
    pairs = extract_pairs_kind(rows, args.kind)

    if len(pairs) == 0:
        raise SystemExit("No usable pairs found in input file.")

    rng = random.Random(args.seed)
    rng.shuffle(pairs)

    n_cal = int(max(1, min(len(pairs) - 1, round(len(pairs) * float(args.cal_frac)))))
    cal_pos = pairs[:n_cal]
    ev_pos = pairs[n_cal:]

    # -----------------------------
    # Optional: add distractors (signal-in-noise)
    # -----------------------------
    distract_n = max(0, int(args.distractors))
    if distract_n > 0:
        pool = stable_unique([s for ex in pairs for s in ex["evidence"] if isinstance(s, str) and s.strip()])
        if len(pool) < 10:
            print("⚠️  Not enough evidence sentences to build distractor pool; skipping distractors.")
        else:
            for i, ex in enumerate(pairs):
                rng_ex = random.Random(args.seed + 900_000 + i)
                have = set(ex["evidence"])
                added: List[str] = []
                tries = 0
                while len(added) < distract_n and tries < distract_n * 50:
                    tries += 1
                    cand = pool[rng_ex.randrange(len(pool))]
                    if cand not in have:
                        added.append(cand)
                        have.add(cand)
                if added:
                    ex["evidence"] = stable_unique(ex["evidence"] + added)

    # -----------------------------
    # Embedder
    # -----------------------------
    embedder = HFEmbedder()

    # Embed once per example (enables hard-mining + speeds everything up)
    for ex in tqdm(pairs, desc="Embed all pairs"):
        ex["claim_vec"] = embedder.embed([ex["claim"]])[0]
        ex["evidence_vecs"] = embedder.embed(ex["evidence"])

    # Refresh split views (pairs were mutated in place)
    cal_pos = pairs[:n_cal]
    ev_pos = pairs[n_cal:]

    # -----------------------------
    # Negatives (cal + eval), including hard-mined option
    # -----------------------------
    def _make_negatives_with_vectors(
        pos_pairs: List[Dict[str, Any]],
        *,
        mode: str,
        seed: int,
        offset: int,
        mine_k: int,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        rng_local = random.Random(seed)
        n = len(pos_pairs)

        def mk(i: int, j: int, mined: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
            out = {
                "claim": pos_pairs[i]["claim"],
                "evidence": pos_pairs[j]["evidence"],
                "label": pos_pairs[i].get("label"),
                "claim_vec": pos_pairs[i]["claim_vec"],
                "evidence_vecs": pos_pairs[j]["evidence_vecs"],
            }
            if mined is not None:
                out["mined"] = mined
            return out

        if mode == "offset":
            off = int(offset) % max(1, n)
            if n > 1 and off == 0:
                off = 1
            neg = [mk(i, (i + off) % n) for i in range(n)]
            return neg, {"mode": "offset", "offset": off, "vectors": True}

        if mode == "cyclic":
            if n <= 1:
                neg = [mk(i, i) for i in range(n)]
            else:
                neg = [mk(i, (i + 1) % n) for i in range(n)]
            return neg, {"mode": "cyclic", "vectors": True}

        if mode == "permute":
            idxs = list(range(n))
            rng_local.shuffle(idxs)
            neg = [mk(i, idxs[i]) for i in range(n)]
            return neg, {"mode": "permute", "warning": "may include fixed points", "vectors": True}

        if mode == "deranged":
            p = _derangement_indices(n, rng_local)
            neg = [mk(i, p[i]) for i in range(n)]
            return neg, {"mode": "deranged", "guarantees": "no fixed points (n>1)", "vectors": True}

        if mode == "hard_mined":
            mine_k = max(1, int(mine_k))
            neg: List[Dict[str, Any]] = []
            energies: List[float] = []
            for i in tqdm(range(n), desc="Mining hard negatives"):
                if n <= 1:
                    neg.append(mk(i, i, mined={"from": i, "energy": None, "candidates": 0}))
                    continue

                cand = set()
                while len(cand) < min(mine_k, n - 1):
                    j = rng_local.randrange(n)
                    if j != i:
                        cand.add(j)

                best_j = None
                best_e = float("inf")
                for j in cand:
                    r = hallucination_energy_svd(
                        pos_pairs[i]["claim_vec"],
                        pos_pairs[j]["evidence_vecs"],
                        top_k=min(int(args.top_k), len(pos_pairs[j]["evidence"])),
                        rank_r=int(args.rank_r),
                    )
                    if r.energy < best_e:
                        best_e = float(r.energy)
                        best_j = j

                assert best_j is not None
                energies.append(best_e)
                neg.append(
                    mk(
                        i,
                        best_j,
                        mined={"from": int(best_j), "energy": float(best_e), "candidates": int(len(cand))},
                    )
                )

            return neg, {
                "mode": "hard_mined",
                "mine_k": int(mine_k),
                "summary": summarize(np.asarray(energies, dtype=np.float32)),
                "vectors": True,
            }

        raise ValueError("neg_mode must be: deranged | offset | cyclic | permute | hard_mined")

    cal_neg, neg_meta_cal = _make_negatives_with_vectors(
        cal_pos,
        mode=args.neg_mode,
        seed=args.seed + 100,
        offset=args.neg_offset,
        mine_k=args.neg_mine_k,
    )

    base_for_eval = ev_pos if ev_pos else cal_pos
    ev_neg, neg_meta_ev = _make_negatives_with_vectors(
        base_for_eval,
        mode=args.neg_mode,
        seed=args.seed + 200,
        offset=args.neg_offset,
        mine_k=args.neg_mine_k,
    )

    def compute_energy_for_pair(ex: Dict[str, Any]) -> Tuple[float, float, int, int, List[float], str]:
        # Prefer pre-embedded vectors; fall back to embedding if missing.
        claim_vec = ex.get("claim_vec")
        ev_vecs = ex.get("evidence_vecs")
        if claim_vec is None:
            claim_vec = embedder.embed([ex["claim"]])[0]
        if ev_vecs is None:
            ev_vecs = embedder.embed(ex["evidence"])

        base, decision_fixed, probe = evaluate_claim(
            claim_vec,
            ev_vecs,
            regime=args.regime,
            top_k=min(int(args.top_k), len(ex["evidence"])),
            rank_r=int(args.rank_r),
            delta=float(args.delta),
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
    def score_set(data: List[Dict[str, Any]], tag: str) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        energies: List[float] = []
        scored: List[Dict[str, Any]] = []
        for ex in tqdm(data, desc=f"Scoring {tag}"):
            e, expl, erank, n_evs, probe, dec_fixed = compute_energy_for_pair(ex)
            energies.append(e)
            scored.append(
                {
                    "claim": ex["claim"],
                    "evidence": ex["evidence"],
                    "label": ex.get("label"),
                    "energy": float(e),
                    "explained": float(expl),
                    "effective_rank": int(erank),
                    "n_evidence": int(n_evs),
                    "probe_energies": probe,
                    "decision_fixed": dec_fixed,
                    "mined": ex.get("mined"),
                }
            )
        return np.asarray(energies, dtype=np.float32), scored

    cal_pos_e, cal_pos_scored = score_set(cal_pos, "CAL POS")
    cal_neg_e, cal_neg_scored = score_set(cal_neg, "CAL NEG")

    # -----------------------------
    # Calibrate tau_accept / tau_review on CAL negatives
    # -----------------------------
    far_accept = float(args.far)
    far_accept = max(0.0, min(0.5, far_accept))  # sane range
    far_review = float(args.far_review)
    far_review = max(far_accept, min(0.5, far_review))

    tau_accept = float(np.percentile(cal_neg_e, far_accept * 100.0))  # accept if energy <= tau_accept
    tau_review = float(np.percentile(cal_neg_e, far_review * 100.0))  # review if energy <= tau_review
    tau_review = max(tau_review, tau_accept)

    cal_FAR_accept = float((cal_neg_e <= tau_accept).mean())
    cal_TPR_accept = float((cal_pos_e <= tau_accept).mean())
    cal_FAR_review = float((cal_neg_e <= tau_review).mean())
    cal_TPR_review = float((cal_pos_e <= tau_review).mean())
    cal_AUC = auc_lower_is_positive(cal_pos_e, cal_neg_e)

    # -----------------------------
    # Holdout eval energies
    # -----------------------------
    if len(ev_pos) > 0:
        ev_pos_e, ev_pos_scored = score_set(ev_pos, "EVAL POS")
        ev_neg_e, ev_neg_scored = score_set(ev_neg, "EVAL NEG")

        ev_FAR_accept = float((ev_neg_e <= tau_accept).mean())
        ev_TPR_accept = float((ev_pos_e <= tau_accept).mean())
        ev_FAR_review = float((ev_neg_e <= tau_review).mean())
        ev_TPR_review = float((ev_pos_e <= tau_review).mean())
        ev_AUC = auc_lower_is_positive(ev_pos_e, ev_neg_e)
    else:
        ev_pos_e = np.asarray([], dtype=np.float32)
        ev_neg_e = np.asarray([], dtype=np.float32)
        ev_pos_scored = []
        ev_neg_scored = []
        ev_FAR_accept = ev_TPR_accept = ev_FAR_review = ev_TPR_review = float("nan")
        ev_AUC = float("nan")

    # -----------------------------
    # Hardest negatives preview (EVAL)
    # -----------------------------
    hardest_preview: List[Dict[str, Any]] = []
    if len(ev_neg_scored) > 0:
        # take 10 lowest-energy negatives (hardest)
        idxs = np.argsort(ev_neg_e)[:10].tolist()
        for j in idxs:
            ex = ev_neg_scored[j]
            hardest_preview.append(
                {
                    "energy": float(ex["energy"]),
                    "claim": ex["claim"][:200],
                    "evidence0": (ex["evidence"][0][:200] if ex["evidence"] else ""),
                    "mined": ex.get("mined"),
                }
            )

    # -----------------------------
    # Build report
    # -----------------------------
    report = {
        "manifest": _env_manifest(args.in_path),
        "kind": args.kind,
        "input_path": str(args.in_path),
        "n_pairs": len(pairs),
        "splits": {"n_cal": len(cal_pos), "n_eval": len(ev_pos)},
        "params": {
            "model_name": getattr(embedder, "model_name", None),
            "regime_fixed": args.regime,
            "delta": float(args.delta),
            "top_k": int(args.top_k),
            "rank_r": int(args.rank_r),
            "distractors": int(distract_n),
            "far_accept": float(far_accept),
            "far_review": float(far_review),
            "tau_accept": float(tau_accept),
            "tau_review": float(tau_review),
            "seed": int(args.seed),
            "neg_mode": args.neg_mode,
            "neg_offset": int(args.neg_offset),
            "neg_mine_k": int(args.neg_mine_k),
            "neg_meta_cal": neg_meta_cal,
            "neg_meta_eval": neg_meta_ev,
        },
        "stats": {
            "cal": {
                "pos": summarize(cal_pos_e),
                "neg": summarize(cal_neg_e),
                "FAR_accept": cal_FAR_accept,
                "TPR_accept": cal_TPR_accept,
                "FAR_review": cal_FAR_review,
                "TPR_review": cal_TPR_review,
                "AUC": float(cal_AUC),
            },
            "eval": {
                "pos": summarize(ev_pos_e),
                "neg": summarize(ev_neg_e),
                "FAR_accept": ev_FAR_accept,
                "TPR_accept": ev_TPR_accept,
                "FAR_review": ev_FAR_review,
                "TPR_review": ev_TPR_review,
                "AUC": float(ev_AUC),
            },
        },
        "hardest_negatives_preview": hardest_preview,
        "interpretation": {
            "accept_rule": "accept if e<=tau_accept; review if e<=tau_review; else reject",
            "calibration": "taus are set to FAR-percentiles of CAL negative energies (negative controls)",
            "note": "AUC is computed such that lower energy corresponds to positive.",
        },
    }

    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    args.out_report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("✅ Negative-control calibration (v2, NO ORACLE)")
    print("=" * 80)
    print(f"tau_accept (FAR={far_accept:.2%}): {tau_accept:.4f}")
    print(f"tau_review  (FAR={far_review:.2%}): {tau_review:.4f}")
    print(
        f"CAL  FAR_a/TPR_a | FAR_r/TPR_r | AUC: "
        f"{cal_FAR_accept:.3%}/{cal_TPR_accept:.3%} | {cal_FAR_review:.3%}/{cal_TPR_review:.3%} | {cal_AUC:.3f}"
    )
    if len(ev_pos) > 0:
        print(
            f"EVAL FAR_a/TPR_a | FAR_r/TPR_r | AUC: "
            f"{ev_FAR_accept:.3%}/{ev_TPR_accept:.3%} | {ev_FAR_review:.3%}/{ev_TPR_review:.3%} | {ev_AUC:.3f}"
        )
    print(f"Wrote report -> {args.out_report}")

    # -----------------------------
    # Write scored outputs
    # -----------------------------
    def write_scored(path: Path, data: List[Dict[str, Any]], tag: str) -> None:
        out_rows = []
        for ex in data:
            e = float(ex["energy"])
            decision_tau = "accept" if e <= tau_accept else ("review" if e <= tau_review else "reject")
            out_rows.append(
                {
                    "claim": ex["claim"],
                    "evidence": ex["evidence"],
                    "label": ex.get("label"),
                    "energy": float(e),
                    "decision_tau": decision_tau,
                    "decision_fixed": ex.get("decision_fixed"),
                    "tau_accept": float(tau_accept),
                    "tau_review": float(tau_review),
                    "top_k": int(args.top_k),
                    "rank_r": int(args.rank_r),
                    "probe_energies": ex.get("probe_energies"),
                    "explained": float(ex.get("explained", 0.0)),
                    "effective_rank": int(ex.get("effective_rank", 0)),
                    "n_evidence": int(ex.get("n_evidence", 0)),
                    "mined": ex.get("mined"),
                }
            )
        save_jsonl(path, out_rows)
        print(f"Write {tag} scored: {len(out_rows)} rows -> {path}")

    # Prefer EVAL sets if available; otherwise CAL
    if args.out_pos_scored:
        write_scored(args.out_pos_scored, ev_pos_scored if len(ev_pos_scored) > 0 else cal_pos_scored, "POS")
    if args.out_neg_scored:
        write_scored(args.out_neg_scored, ev_neg_scored if len(ev_neg_scored) > 0 else cal_neg_scored, "NEG")

    # -----------------------------
    # Plot distributions
    # -----------------------------
    if plt is not None and args.out_plot:
        args.out_plot.parent.mkdir(parents=True, exist_ok=True)
        use_pos = ev_pos_e if ev_pos_e.size > 0 else cal_pos_e
        use_neg = ev_neg_e if ev_neg_e.size > 0 else cal_neg_e

        plt.figure(figsize=(8, 4))
        plt.hist(use_pos, bins=40, alpha=0.6, label="POS (matched)")
        plt.hist(use_neg, bins=40, alpha=0.6, label="NEG (mismatched)")
        plt.axvline(tau_accept, linestyle="--", linewidth=2, label=f"tau_accept={tau_accept:.3f}")
        plt.axvline(tau_review, linestyle="--", linewidth=2, label=f"tau_review={tau_review:.3f}")
        plt.title("Energy separation: POS vs NEG")
        plt.xlabel("Energy (lower = better match)")
        plt.ylabel("Count")
        plt.legend()
        plt.tight_layout()
        plt.savefig(args.out_plot)
        print(f"Wrote plot -> {args.out_plot}")
    elif args.out_plot:
        print("Skipping plot (matplotlib not available).")


if __name__ == "__main__":
    main()
