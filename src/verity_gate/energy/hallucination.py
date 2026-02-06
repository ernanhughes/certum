# src/verity_gate/energy/hallucination.py
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class EnergyResult:
    energy: float  # H_E in [0,1]
    explained: float  # ||U_r^T c||^2
    identity_error: float  # |1 - (explained + energy)|
    topk: int
    rank_r: int
    effective_rank: int  # rank used by SVD
    sv: Optional[List[float]]
    topk_scores: Optional[List[float]]
    used_count: int


def build_evidence_basis(
    c: np.ndarray,
    E: np.ndarray,
    *,
    top_k: int,
    rank_r: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (U_r, S, idx) where:
      - U_r: (d, r) orthonormal basis vectors
      - S: singular values (r_full,)
      - idx: indices of selected evidence vectors
    """
    if E.size == 0:
        return (
            np.zeros((c.shape[0], 0), dtype=np.float32),
            np.array([], dtype=np.float32),
            np.array([], dtype=np.int64),
        )

    # Rank selection guard
    top_k = max(1, int(top_k))
    rank_r = max(1, int(rank_r))

    # 1) select top-k by cosine to claim
    scores = cosine_scores(c, E)
    k = min(top_k, E.shape[0])
    idx = np.argsort(-scores)[:k]  # descending
    E_k = E[idx]  # (k,d)

    # 2) SVD on (k,d) -> basis in embedding space is V^T rows (or equivalently U of E_k^T)
    # Use thin SVD for stability.
    # E_k = U_e * diag(S) * Vt, where Vt is (r_full, d)
    U_e, S, Vt = np.linalg.svd(E_k, full_matrices=False)

    # 3) Orthonormal basis vectors in embedding space = rows of Vt => columns of Vt.T
    r_full = Vt.shape[0]
    r = min(rank_r, r_full)
    U_r = Vt[:r].T  # (d, r)

    return U_r.astype(np.float32), S.astype(np.float32), idx.astype(np.int64)


def hallucination_energy_svd(
    claim_vec: np.ndarray,
    evidence_vecs: np.ndarray,
    *,
    top_k: int = 12,
    rank_r: int = 8,
    return_debug: bool = True,
) -> EnergyResult:
    """
    Paper-safe hallucination energy:
      H_E = 1 - ||U_r^T c||^2, with unit-normalized claim and evidence vectors.

    claim_vec: (d,)
    evidence_vecs: (n,d)
    """
    if claim_vec is None or evidence_vecs is None:
        return EnergyResult(
            energy=1.0,
            explained=0.0,
            topk=top_k,
            rank_r=rank_r,
            sv=None,
            topk_scores=None,
            used_count=0,
        )

    c = _unit_norm(np.asarray(claim_vec, dtype=np.float32))

    E = np.asarray(evidence_vecs, dtype=np.float32)
    if E.ndim != 2 or E.shape[0] == 0:
        return EnergyResult(
            energy=1.0,
            explained=0.0,
            topk=top_k,
            rank_r=rank_r,
            sv=None,
            topk_scores=None,
            used_count=0,
        )

    E = _unit_norm_rows(E)

    # Build basis from top-k most relevant evidence sentences
    U_r, S, idx = build_evidence_basis(c, E, top_k=top_k, rank_r=rank_r)

    if U_r.shape[1] == 0:
        return EnergyResult(
            energy=1.0,
            explained=0.0,
            topk=top_k,
            rank_r=rank_r,
            sv=None,
            topk_scores=None,
            used_count=0,
        )

    # Explained energy = ||U_r^T c||^2
    proj_coords = U_r.T @ c  # (r,)
    explained = float(np.dot(proj_coords, proj_coords))
    energy = 1.0 - explained

    # Numerical safety
    explained = max(0.0, min(1.0, explained))
    energy = max(0.0, min(1.0, energy))

    identity_error = abs(1.0 - (explained + energy))
    effective_rank = int(np.sum(S > 1e-6))

    if not return_debug:
        return EnergyResult(
            energy=energy,
            explained=explained,
            identity_error=identity_error,
            effective_rank=effective_rank,
            topk=top_k,
            rank_r=rank_r,
            sv=None,
            topk_scores=None,
            used_count=int(len(idx)),
        )

    # Debug: topk scores against claim
    scores = cosine_scores(c, E)
    topk_scores = scores[idx].tolist() if idx.size else []
    sv_list = S.tolist() if S.size else []

    return EnergyResult(
        energy=energy,
        explained=explained,
        identity_error=identity_error,
        topk=top_k,
        rank_r=rank_r,
        effective_rank=effective_rank,
        sv=sv_list,
        topk_scores=topk_scores,
        used_count=int(len(idx)),
    )


def _unit_norm(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x)
    if n < eps:
        return x * 0.0
    return x / n


def _unit_norm_rows(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms < eps, 1.0, norms)
    return X / norms


def cosine_scores(c: np.ndarray, E: np.ndarray) -> np.ndarray:
    """
    c: (d,) unit
    E: (n,d) unit rows
    returns (n,)
    """
    return (E @ c).astype(np.float32)
