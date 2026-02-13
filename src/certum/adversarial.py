from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Dict, Any
import random
import numpy as np
import logging

logger = logging.getLogger(__name__)

# ============================================================
# Base Interface
# ============================================================

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
        energy_computer: Optional[Any] = None,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass


# ============================================================
# Utilities
# ============================================================

def _unit_norm(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    n = np.linalg.norm(x)
    return x / max(n, eps)


def _unit_norm_rows(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms < eps, 1.0, norms)
    return X / norms


# ============================================================
# Derangement Utility
# ============================================================

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

    logger.warning("Uniform derangement failed; falling back to Sattolo.")
    return derangement_indices(n, rng, method="sattolo")


def _neg_from(pairs: List[Dict[str, Any]], i: int, j: int) -> Dict[str, Any]:
    src = pairs[j]
    return {
        "id": pairs[i].get("id", i),
        "claim": pairs[i]["claim"],
        "evidence": src.get("evidence", []),
        "evidence_vecs": src.get("evidence_vecs"),
        "label": "NEG",
    }


# ============================================================
# Generators
# ============================================================

class DerangedPairGenerator(AdversarialPairGenerator):

    @property
    def name(self) -> str:
        return "deranged"

    def generate(self, pairs, *, seed, embedder=None, energy_computer=None):
        n = len(pairs)
        rng = random.Random(seed)

        if n <= 1:
            logger.warning("Degenerate derangement case.")
            negs = [_neg_from(pairs, 0, 0)] if n == 1 else []
            return negs, {"mode": "deranged", "n": n, "note": "degenerate"}

        perm, meta = derangement_indices(n, rng)
        negs = [_neg_from(pairs, i, perm[i]) for i in range(n)]

        meta.update({"mode": "deranged", "n": n})
        return negs, meta


# ------------------------------------------------------------

class CyclicPairGenerator(AdversarialPairGenerator):

    @property
    def name(self) -> str:
        return "cyclic"

    def generate(self, pairs, *, seed, embedder=None, energy_computer=None):
        n = len(pairs)

        if n == 0:
            return [], {"mode": "cyclic", "n": 0}

        perm = [(i + 1) % n for i in range(n)]
        negs = [_neg_from(pairs, i, perm[i]) for i in range(n)]

        return negs, {"mode": "cyclic", "n": n}


# ------------------------------------------------------------

class OffsetPairGenerator(AdversarialPairGenerator):

    def __init__(self, offset: int = 1):
        self.offset = offset

    @property
    def name(self) -> str:
        return f"offset_{self.offset}"

    def generate(self, pairs, *, seed, embedder=None, energy_computer=None):
        n = len(pairs)

        if n == 0:
            return [], {"mode": "offset", "n": 0}

        off = self.offset % n
        if n > 1 and off == 0:
            off = 1

        perm = [(i + off) % n for i in range(n)]
        negs = [_neg_from(pairs, i, perm[i]) for i in range(n)]

        return negs, {
            "mode": "offset",
            "n": n,
            "effective_offset": off,
        }


# ------------------------------------------------------------

class PermutePairGenerator(AdversarialPairGenerator):

    @property
    def name(self) -> str:
        return "permute"

    def generate(self, pairs, *, seed, embedder=None, energy_computer=None):
        n = len(pairs)
        rng = random.Random(seed)

        perm = list(range(n))
        rng.shuffle(perm)

        negs = [_neg_from(pairs, i, perm[i]) for i in range(n)]
        fixed = sum(1 for i in range(n) if perm[i] == i)

        return negs, {
            "mode": "permute",
            "n": n,
            "fixed_points": fixed,
        }


# ------------------------------------------------------------
# Hard Mined (Centroid Similarity)
# ------------------------------------------------------------

class HardMinedPairGenerator(AdversarialPairGenerator):

    @property
    def name(self) -> str:
        return "hard_mined"

    def generate(self, pairs, *, seed, embedder=None, energy_computer=None):

        if embedder is None:
            raise ValueError("hard_mined requires embedder")

        logger.debug("Running hard_mined negative generation.")

        valid_indices = []
        centroids = []

        for i, p in enumerate(pairs):
            ev = p.get("evidence_vecs")
            if ev is None:
                continue

            ev = np.asarray(ev, dtype=np.float32)
            if ev.ndim != 2 or ev.shape[0] == 0:
                continue

            ev_norm = _unit_norm_rows(ev)
            centroid = _unit_norm(ev_norm.mean(axis=0))

            if np.isfinite(centroid).all():
                valid_indices.append(i)
                centroids.append(centroid)

        if len(valid_indices) < 2:
            logger.warning("Insufficient valid evidence for hard_mined.")
            return DerangedPairGenerator().generate(pairs, seed=seed)

        claim_vecs = _unit_norm_rows(
            np.asarray(embedder.embed([p["claim"] for p in pairs]), dtype=np.float32)
        )

        centroid_mat = _unit_norm_rows(np.stack(centroids))
        sim = claim_vecs @ centroid_mat.T

        vpos = {orig: pos for pos, orig in enumerate(valid_indices)}
        for i, pos in vpos.items():
            sim[i, pos] = -np.inf

        best = np.argmax(sim, axis=1)
        best_j = [valid_indices[int(p)] for p in best]

        negs = [_neg_from(pairs, i, best_j[i]) for i in range(len(pairs))]

        return negs, {
            "mode": "hard_mined",
            "n": len(pairs),
            "candidates": len(valid_indices),
        }


# ------------------------------------------------------------
# Hard Mined V2 (Energy-Minimizing)
# ------------------------------------------------------------

class HardMinedPairGeneratorV2(AdversarialPairGenerator):

    def __init__(self, top_candidates: int = 16):
        self.top_candidates = top_candidates

    @property
    def name(self) -> str:
        return "hard_mined_v2"

    def generate(self, pairs, *, seed, embedder=None, energy_computer=None):

        if embedder is None or energy_computer is None:
            raise ValueError("hard_mined_v2 requires embedder and energy_computer")

        logger.debug("Running hard_mined_v2 adversarial search.")

        n = len(pairs)
        if n < 2:
            return [], {"mode": "hard_mined_v2", "n": n}

        claim_vecs = _unit_norm_rows(
            np.asarray(embedder.embed([p["claim"] for p in pairs]), dtype=np.float32)
        )

        valid_indices = []
        ev_vecs_list = []
        centroids = []

        for i, p in enumerate(pairs):
            ev = p.get("evidence_vecs")
            if ev is None:
                continue

            ev = np.asarray(ev, dtype=np.float32)
            if ev.ndim != 2 or ev.shape[0] == 0:
                continue

            ev_norm = _unit_norm_rows(ev)
            centroid = _unit_norm(ev_norm.mean(axis=0))

            if np.isfinite(centroid).all():
                valid_indices.append(i)
                ev_vecs_list.append(ev_norm)
                centroids.append(centroid)

        if len(valid_indices) < 2:
            logger.warning("Fallback to deranged from hard_mined_v2.")
            return DerangedPairGenerator().generate(pairs, seed=seed)

        centroid_mat = _unit_norm_rows(np.stack(centroids))
        sim = claim_vecs @ centroid_mat.T

        K = min(self.top_candidates, len(valid_indices))
        rng = np.random.default_rng(seed)

        chosen_js = []
        chosen_energies = []

        for i in range(n):

            idx = np.argpartition(-sim[i], K - 1)[:K]

            best_j = None
            best_e = float("inf")

            for cand_pos in idx:
                j_idx = valid_indices[int(cand_pos)]
                if j_idx == i:
                    continue

                e = energy_computer.compute(
                    claim_vecs[i],
                    ev_vecs_list[valid_indices.index(j_idx)]
                ).energy

                if e < best_e:
                    best_e = e
                    best_j = j_idx

            if best_j is None:
                best_j = valid_indices[int(rng.integers(0, len(valid_indices)))]
                best_e = 1.0

            chosen_js.append(best_j)
            chosen_energies.append(best_e)

        negs = [
            {
                "id": pairs[i].get("id", i),
                "claim": pairs[i]["claim"],
                "evidence": pairs[chosen_js[i]].get("evidence", []),
                "evidence_vecs": pairs[chosen_js[i]].get("evidence_vecs"),
                "label": "NEG_HARD_V2",
            }
            for i in range(n)
        ]

        return negs, {
            "mode": "hard_mined_v2",
            "n": n,
            "mean_energy": float(np.mean(chosen_energies)),
            "min_energy": float(np.min(chosen_energies)),
            "max_energy": float(np.max(chosen_energies)),
        }


# ============================================================
# Factory
# ============================================================

def get_adversarial_generator(mode: str, **kwargs) -> AdversarialPairGenerator:

    off = kwargs.get("neg_offset", 37)
    off = 37 if off is None else int(off)

    if mode == "deranged":
        return DerangedPairGenerator()
    if mode == "offset":
        return OffsetPairGenerator(offset=off)
    if mode == "cyclic":
        return CyclicPairGenerator()
    if mode == "permute":
        return PermutePairGenerator()
    if mode == "hard_mined":
        return HardMinedPairGenerator()
    if mode == "hard_mined_v2":
        return HardMinedPairGeneratorV2()

    raise ValueError(f"Unknown neg_mode: {mode}")
