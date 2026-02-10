from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Dict, Any
import random
import numpy as np

# ---------------------------------------------------------------------
# Base interface
# ---------------------------------------------------------------------

class AdversarialPairGenerator(ABC):
    """
    Generates adversarial (claim, evidence) PAIRS.
    This is the ONLY valid way to generate hallucination negatives.
    """

    @abstractmethod
    def generate(
        self,
        pairs: List[Dict[str, Any]],
        *,
        seed: int,
        embedder: Optional[Any] = None,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass


# ---------------------------------------------------------------------
# Utilities (ported verbatim from gate_suite)
# ---------------------------------------------------------------------

def _unit_norm(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x)
    return x / max(n, eps)

def _unit_norm_rows(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms < eps, 1.0, norms)
    return X / norms


def derangement_indices(
    n: int,
    rng: random.Random,
    *,
    method: str = "uniform",
    max_tries: int = 10_000,
) -> Tuple[List[int], Dict[str, Any]]:
    if n <= 1:
        return list(range(n)), {"fixed_points": n, "tries": 1, "method": method}

    if method == "sattolo":
        p = list(range(n))
        for i in range(n - 1, 0, -1):
            j = rng.randrange(i)
            p[i], p[j] = p[j], p[i]
        return p, {"fixed_points": 0, "tries": 1, "method": "sattolo"}

    for t in range(1, max_tries + 1):
        p = list(range(n))
        rng.shuffle(p)
        if all(i != p[i] for i in range(n)):
            return p, {"fixed_points": 0, "tries": t, "method": "uniform"}

    p, meta = derangement_indices(n, rng, method="sattolo")
    meta["method"] = "uniform->sattolo"
    meta["tries"] = max_tries
    return p, meta


def _neg_from(pairs: List[Dict[str, Any]], i: int, j: int) -> Dict[str, Any]:
    src = pairs[j]
    return {
        "id": pairs[i].get("id", i),
        "claim": pairs[i]["claim"],
        "evidence": src.get("evidence", []),
        "evidence_vecs": src.get("evidence_vecs", None),
        "label": "NEG",
    }


# ---------------------------------------------------------------------
# Pair generators
# ---------------------------------------------------------------------

class DerangedPairGenerator(AdversarialPairGenerator):
    @property
    def name(self) -> str:
        return "deranged"

    def generate(
        self,
        pairs: List[Dict[str, Any]],
        *,
        seed: int,
        embedder: Optional[Any] = None,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        n = len(pairs)
        if n <= 1:
            # cannot derange, fallback to cyclic/offset/no-op
            negs = [_neg_from(pairs, 0, 0)] if n == 1 else []
            meta = {"mode": "deranged", "n": n, "fixed_points": n, "note": "degenerate"}
            return negs, meta
        rng = random.Random(seed)
        perm, meta = derangement_indices(n, rng)

        assert all(i != perm[i] for i in range(n)), "Derangement failed"

        negs = [_neg_from(pairs, i, perm[i]) for i in range(n)]
        meta.update({"mode": "deranged", "n": n})
        return negs, meta


class CyclicPairGenerator(AdversarialPairGenerator):
    @property
    def name(self) -> str:
        return "cyclic"

    def generate(self, pairs, *, seed: int, embedder=None):
        n = len(pairs)
        if n == 0:
            return [], {"mode": "cyclic", "n": 0, "fixed_points": 0, "note": "empty"}
        if n == 1:
            negs = [_neg_from(pairs, 0, 0)]
            return negs, {"mode": "cyclic", "n": 1, "fixed_points": 1, "note": "degenerate"}

        perm = [(i + 1) % n for i in range(n)]
        negs = [_neg_from(pairs, i, perm[i]) for i in range(n)]
        return negs, {"mode": "cyclic", "n": n, "fixed_points": 0}


class OffsetPairGenerator(AdversarialPairGenerator):
    def __init__(self, offset: int = 1):
        self.offset = offset

    @property
    def name(self) -> str:
        return f"offset_{self.offset}"

    def generate(self, pairs, *, seed: int, embedder=None):
        n = len(pairs)
        if n == 0:
            return [], {"mode": "offset", "n": 0, "requested_offset": self.offset, "effective_offset": 0, "fixed_points": 0, "note": "empty"}

        off = self.offset % n
        adjusted = False
        if n > 1 and off == 0:
            off = 1
            adjusted = True

        perm = [(i + off) % n for i in range(n)]
        fixed = sum(1 for i in range(n) if perm[i] == i)
        negs = [_neg_from(pairs, i, perm[i]) for i in range(n)]
        meta = {
            "mode": "offset",
            "n": n,
            "requested_offset": self.offset,
            "effective_offset": off,
            "fixed_points": fixed,
        }
        if adjusted:
            meta["note"] = "offset% n == 0 adjusted to 1"
        return negs, meta


class PermutePairGenerator(AdversarialPairGenerator):
    @property
    def name(self) -> str:
        return "permute"

    def generate(
        self,
        pairs: List[Dict[str, Any]],
        *,
        seed: int,
        embedder: Optional[Any] = None,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        n = len(pairs)
        rng = random.Random(seed)
        perm = list(range(n))
        rng.shuffle(perm)
        fixed = sum(1 for i in range(n) if perm[i] == i)
        negs = [_neg_from(pairs, i, perm[i]) for i in range(n)]
        return negs, {"mode": "permute", "n": n, "fixed_points": fixed}


class HardMinedPairGenerator(AdversarialPairGenerator):
    @property
    def name(self) -> str:
        return "hard_mined"

    def generate(
        self,
        pairs: List[Dict[str, Any]],
        *,
        seed: int,
        embedder: Optional[Any] = None,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        if embedder is None:
            raise ValueError("hard_mined requires embedder")

        valid = []
        centroids = []

        for idx, ex in enumerate(pairs):
            ev = ex.get("evidence_vecs")
            if ev is None:
                continue
            ev = np.asarray(ev, dtype=np.float32)
            if ev.ndim != 2 or ev.shape[0] == 0:
                continue
            c = _unit_norm(_unit_norm_rows(ev).mean(axis=0))
            if np.isfinite(c).all():
                valid.append(idx)
                centroids.append(c)

        if len(valid) < 2:
            # fallback
            return DerangedPairGenerator().generate(pairs, seed=seed)

        claim_vecs = embedder.embed([ex["claim"] for ex in pairs])
        claim_vecs = _unit_norm_rows(np.asarray(claim_vecs, dtype=np.float32))
        centroid_mat = _unit_norm_rows(np.stack(centroids))

        sim = claim_vecs @ centroid_mat.T

        vpos = {orig: pos for pos, orig in enumerate(valid)}
        for i, pos in vpos.items():
            sim[i, pos] = -np.inf

        best = np.argmax(sim, axis=1)
        best_j = [valid[int(p)] for p in best]

        negs = [_neg_from(pairs, i, best_j[i]) for i in range(len(pairs))]
        meta = {
            "mode": "hard_mined",
            "n": len(pairs),
            "candidates": len(valid),
            "method": "argmax cosine(claim, evidence_centroid)",
        }
        return negs, meta
