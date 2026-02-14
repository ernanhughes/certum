import logging
import random
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

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
        "evidence_ids": src.get("evidence_ids"),
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

        # IMPORTANT: for "NEG" generation we must avoid fixed points,
        # otherwise some "NEG" pairs are actually supported positives.
        perm, meta = derangement_indices(n, rng)

        negs = [_neg_from(pairs, i, perm[i]) for i in range(n)]
        meta.update({"mode": "permute", "n": n})
        return negs, meta

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


# --- helpers ---------------------------------------------------------------

def _page_from_element_id(eid: str) -> str:
    """Best-effort FEVEROUS element-id → page key.
    Works for *_sentence_*, *_cell_*, *_section_*, *_table_*, *_list_* and *_title.
    """
    s = str(eid)
    for key in ("_sentence_", "_cell_", "_section_", "_table_", "_list_"):
        if key in s:
            return s.split(key, 1)[0]
    # context ids like Page_title (no trailing underscore)
    if s.endswith("_title"):
        return s[: -len("_title")]
    # fallback: avoid splitting page titles that contain underscores too aggressively
    # take everything up to the last two underscores as "page-ish"
    parts = s.split("_")
    return "_".join(parts[:-2]) if len(parts) > 2 else s

def _pages_from_ids(evidence_ids: Optional[List[str]]) -> set:
    if not evidence_ids:
        return set()
    return {_page_from_element_id(eid) for eid in evidence_ids if isinstance(eid, str) and eid}

def _ids_set(evidence_ids: Optional[List[str]]) -> set:
    if not evidence_ids:
        return set()
    return {str(eid) for eid in evidence_ids if isinstance(eid, str) and eid}

# --- generator -------------------------------------------------------------

class HardMinedPairGeneratorV2(AdversarialPairGenerator):
    """
    Two modes:
      - rerank_by_energy=False => "hard_mined_v2"  (similarity-only hard negatives)
      - rerank_by_energy=True  => "hardest_energy_mined" (adaptive adversary vs energy)
    """

    def __init__(
        self,
        top_candidates: int = 16,
        *,
        rerank_by_energy: bool = False,
        require_page_mismatch: bool = True,
        require_disjoint_ids: bool = True,
    ):
        self.top_candidates = top_candidates
        self.rerank_by_energy = rerank_by_energy
        self.require_page_mismatch = require_page_mismatch
        self.require_disjoint_ids = require_disjoint_ids

    @property
    def name(self) -> str:
        return "hardest_energy_mined" if self.rerank_by_energy else "hard_mined_v2"

    def generate(self, pairs, *, seed, embedder=None, energy_computer=None):
        if embedder is None or energy_computer is None:
            raise ValueError("hard_mined_v2 requires embedder and energy_computer")

        logger.debug("Running %s adversarial search.", self.name)

        n = len(pairs)
        if n < 2:
            return [], {"mode": self.name, "n": n}

        # Embed claims (unit norm)
        claim_vecs = _unit_norm_rows(
            np.asarray(embedder.embed([p["claim"] for p in pairs]), dtype=np.float32)
        )

        # Build candidate pool from examples with evidence_vecs
        valid_indices: List[int] = []
        centroids: List[np.ndarray] = []
        ev_pages: List[set] = []
        ev_ids: List[set] = []

        for i, p in enumerate(pairs):
            ev = p.get("evidence_vecs")
            if ev is None:
                continue

            ev = np.asarray(ev, dtype=np.float32)
            if ev.ndim != 2 or ev.shape[0] == 0:
                continue

            # centroid for similarity search only (normalize evidence rows first)
            ev_norm = _unit_norm_rows(ev)
            c = _unit_norm(ev_norm.mean(axis=0))
            if not np.isfinite(c).all():
                continue

            valid_indices.append(i)
            centroids.append(c)

            # leakage guards (best effort)
            ev_pages.append(_pages_from_ids(p.get("evidence_ids")))
            ev_ids.append(_ids_set(p.get("evidence_ids")))

        if len(valid_indices) < 2:
            logger.warning("Fallback to deranged from %s (insufficient valid evidence_vecs).", self.name)
            return DerangedPairGenerator().generate(pairs, seed=seed)

        centroid_mat = _unit_norm_rows(np.stack(centroids, axis=0))
        sim = claim_vecs @ centroid_mat.T  # (n, m)

        K = min(self.top_candidates, len(valid_indices))
        rng = np.random.default_rng(seed)

        chosen_js: List[int] = []
        chosen_energies: List[float] = []

        # Precompute claim-side pages/ids if present (for stronger mismatch checks)
        claim_pages = [_pages_from_ids(p.get("evidence_ids")) for p in pairs]
        claim_ids = [_ids_set(p.get("evidence_ids")) for p in pairs]

        for i in range(n):
            # Top-K centroid-similar candidate positions (in [0..m))
            topk_pos = np.argpartition(-sim[i], K - 1)[:K]
            # Sort those K by similarity descending for deterministic selection
            topk_pos = topk_pos[np.argsort(-sim[i, topk_pos])]

            best_j: Optional[int] = None
            best_e: float = float("inf")

            # Try strict filters first; if nothing found, relax deterministically.
            # Relax order: drop page mismatch, then drop id disjointness, then random fallback.
            relax_steps = [
                (self.require_page_mismatch, self.require_disjoint_ids),
                (False, self.require_disjoint_ids),
                (False, False),
            ]

            for req_page_mismatch, req_disjoint_ids in relax_steps:
                for cand_pos in topk_pos:
                    j_idx = valid_indices[int(cand_pos)]
                    if j_idx == i:
                        continue

                    if req_page_mismatch:
                        cp = claim_pages[i]
                        jp = ev_pages[int(cand_pos)]
                        if cp and jp and (cp & jp):
                            continue

                    if req_disjoint_ids:
                        ci = claim_ids[i]
                        ji = ev_ids[int(cand_pos)]
                        if ci and ji and not ci.isdisjoint(ji):
                            continue

                    if self.rerank_by_energy:
                        # Adaptive: choose lowest-energy among filtered candidates
                        ev_vecs = pairs[j_idx].get("evidence_vecs")
                        if ev_vecs is None:
                            continue
                        e = float(energy_computer.compute(claim_vecs[i], np.asarray(ev_vecs, dtype=np.float32)).energy)
                        if e < best_e:
                            best_e = e
                            best_j = j_idx
                    else:
                        # Non-adaptive: choose the first (highest similarity) candidate that passes filters
                        best_j = j_idx
                        best_e = float(energy_computer.compute(
                            claim_vecs[i],
                            np.asarray(pairs[j_idx].get("evidence_vecs"), dtype=np.float32),
                        ).energy)
                        break

                if best_j is not None:
                    break

            if best_j is None:
                # last resort (should be rare)
                cand_pos = int(rng.integers(0, len(valid_indices)))
                best_j = valid_indices[cand_pos]
                best_e = float(energy_computer.compute(
                    claim_vecs[i],
                    np.asarray(pairs[best_j].get("evidence_vecs"), dtype=np.float32),
                ).energy)

            chosen_js.append(best_j)
            chosen_energies.append(best_e)

        negs = [
            {
                "id": pairs[i].get("id", i),
                "claim": pairs[i]["claim"],
                "evidence": pairs[chosen_js[i]].get("evidence", []),
                "evidence_ids": pairs[chosen_js[i]].get("evidence_ids"),
                "evidence_vecs": pairs[chosen_js[i]].get("evidence_vecs"),
                "label": "NEG_HARD_V2" if not self.rerank_by_energy else "NEG_HARDEST",
            }
            for i in range(n)
        ]

        return negs, {
            "mode": self.name,
            "n": n,
            "top_candidates": int(K),
            "rerank_by_energy": bool(self.rerank_by_energy),
            "require_page_mismatch": bool(self.require_page_mismatch),
            "require_disjoint_ids": bool(self.require_disjoint_ids),
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
        return HardMinedPairGeneratorV2(
            top_candidates=int(kwargs.get("top_candidates", 16)),
            rerank_by_energy=bool(kwargs.get("rerank_by_energy", False)),
            require_page_mismatch=bool(kwargs.get("require_page_mismatch", True)),
            require_disjoint_ids=bool(kwargs.get("require_disjoint_ids", True)),
        )
    if mode == "hardest_energy_mined":
        return HardMinedPairGeneratorV2(
            top_candidates=int(kwargs.get("top_candidates", 16)),
            rerank_by_energy=True,
            require_page_mismatch=bool(kwargs.get("require_page_mismatch", True)),
            require_disjoint_ids=bool(kwargs.get("require_disjoint_ids", True)),
        )
    raise ValueError(f"Unknown neg_mode: {mode}")
