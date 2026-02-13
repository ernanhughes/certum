# src/certum/calibration.py
from typing import List, Tuple, Dict, Any, Optional
import random
import numpy as np
import logging

from certum.energy import HallucinationEnergyComputer
from certum.protocols.embedder import Embedder
from certum.gate import VerifiabilityGate

logger = logging.getLogger(__name__)

class AdaptiveCalibrator:
    """
    Learns percentile thresholds from data distribution.
    Replaces hand-tuned thresholds with data-calibrated policy.
    """
    
    def __init__(self, gate: 'VerifiabilityGate', embedder: Embedder):
        self.gate = gate
        self.embedder = embedder
    
    def run_sweep(
        self,
        claims: List[str],
        evidence_sets: List[List[str]],
        evidence_vecs: List[np.ndarray],
        claim_vec_cache: Dict[str, np.ndarray],
        pos_coverage: float = 95.0,
        percentiles: List[int] = [1, 5, 10, 20, 30],
        neg_mode: str = "deranged",
        neg_offset: int = 37,
        seed: int = 1337,
    ) -> Dict[str, Any]:
        # 1. Compute energies/sensitivity/participation_ratio values for POSITIVE
        pos_energies = []
        pos_pr_values = []
        pos_sensitivity_values = []
        for claim, ev in zip(claims, evidence_vecs):
            if claim in claim_vec_cache:
                claim_vec = claim_vec_cache[claim]
            else:
                claim_vec = self.embedder.embed([claim])[0]
                claim_vec_cache[claim] = claim_vec

            res = self.gate.energy_computer.compute(
                claim_vec=claim_vec,
                evidence_vecs=ev,
            )
            energy = res.energy
            pos_pr_values.append(res.geometry.participation_ratio)
            pos_sensitivity_values.append(res.geometry.sensitivity)
            pos_energies.append(energy)
        

        # 2. Compute NEGATIVE energies (easy baseline: deranged)
        neg_energies_deranged = self._compute_neg_energies(
            claims=claims,
            evidence_sets=evidence_sets,
            evidence_vecs=evidence_vecs,
            neg_mode="deranged",
            neg_offset=neg_offset,
            seed=seed,
            claim_vec_cache=claim_vec_cache,
        )

        # 3. Compute NEGATIVE energies (hard baseline)
        neg_energies_hard = self._compute_neg_energies(
            claims=claims,
            evidence_sets=evidence_sets,
            evidence_vecs=evidence_vecs,
            neg_mode="hard_mined_v2",
            neg_offset=neg_offset,
            seed=seed,
            claim_vec_cache=claim_vec_cache,
        )
        
        if len(neg_energies_deranged) < 10 or len(neg_energies_hard) < 10:
            raise ValueError(
                "Insufficient negative samples for hard-negative gap calibration."
            )
        

        # Condition on energy passing threshold
        tau_energy = float(np.percentile(neg_energies_hard, percentiles[0]))

        mask = np.array(pos_energies) <= tau_energy

        pos_pr_cond = np.array(pos_pr_values)[mask]
        pos_sens_cond = np.array(pos_sensitivity_values)[mask]

        if len(pos_pr_cond) < 10:
            logger.warning("Too few positives under energy threshold for PR calibration.")
            pos_pr_cond = np.array(pos_pr_values)
            pos_sens_cond = np.array(pos_sensitivity_values)

        tau_pr = float(np.percentile(pos_pr_cond, 90))
        tau_sensitivity = float(np.percentile(pos_sens_cond, 90))

        logger.info(
            f"[Calibration] PR τ @ {pos_coverage}% = {tau_pr:.4f}, "
            f"Sensitivity τ @ {pos_coverage}% = {tau_sensitivity:.4f}"
        )


        tau_pr_by_percentile = {
            p: float(np.percentile(pos_pr_values, 100 - p))
            for p in percentiles
        }

        hard_negative_gap = self._compute_hard_negative_gap(
            neg_energies_deranged,
            neg_energies_hard,
        )

        std_ref = float(np.std(neg_energies_deranged))
        if std_ref < 1e-6:
            hard_negative_gap_norm = 0.0
        else:
            hard_negative_gap_norm = hard_negative_gap / std_ref


        
        # 4-5. Calibrate thresholds + compute separation (unchanged)
        tau_by_percentile = {p: float(np.percentile(neg_energies_hard, p)) for p in percentiles}
        separation_delta = np.mean(neg_energies_hard) - np.mean(pos_energies) if pos_energies and neg_energies_hard else 0.0

        logger.info(
            f"[Calibration] mean_deranged={np.mean(neg_energies_deranged):.4f} "
            f"mean_hard={np.mean(neg_energies_hard):.4f} "
            f"gap={hard_negative_gap:.4f} "
            f"gap_norm={hard_negative_gap_norm:.4f}"
        )
        p = percentiles[0]
        tau_energy = float(np.percentile(neg_energies_hard, p))


        return {
            "tau_energy": tau_energy,
            "tau_sensitivity": tau_sensitivity,
            "tau_pr": tau_pr,
            "tau_by_percentile": tau_by_percentile,
            "tau_energy_by_percentile": tau_by_percentile,
            "tau_pr_by_percentile": tau_pr_by_percentile,
            "pos_energies": pos_energies,
            "neg_energies": neg_energies_hard,
            "separation_delta": separation_delta,
            "sample_count": len(pos_energies),
            "neg_sample_count": len(neg_energies_hard),
            "neg_mode": neg_mode,
            "hard_negative_gap": hard_negative_gap,
            "hard_negative_gap_norm": hard_negative_gap_norm,
            "pos_pr_values": pos_pr_values,
            "pos_sensitivity_values": pos_sensitivity_values,
        }    

    def compute_effectiveness_curve(d_values, accepted_flags, bins=10):
        d_values = np.array(d_values)
        accepted_flags = np.array(accepted_flags)

        edges = np.linspace(0, 1, bins + 1)
        results = []

        for i in range(bins):
            mask = (d_values >= edges[i]) & (d_values < edges[i+1])
            if mask.sum() == 0:
                continue

            acc_rate = accepted_flags[mask].mean()
            results.append((0.5*(edges[i]+edges[i+1]), acc_rate))

        return results


    def _generate_negatives(
        self,
        claims: List[str],
        evidence_sets: List[List[str]],
        evidence_vecs_list: List[np.ndarray],  # ← NEW: precomputed evidence vectors
        mode: str = "deranged",
        offset: int = 37,
        seed: int = 1337,
        energy_computer: Optional[HallucinationEnergyComputer] = None,  # ← NEW
    ) -> List[Tuple[str, List[str], np.ndarray]]:  # ← Returns (claim, evidence_texts, evidence_vecs)
        """
        Generate adversarial negative samples.
        
        Returns:
            List of (claim, mismatched_evidence_texts, mismatched_evidence_vecs) tuples.
        """
        n = len(claims)
        if n == 0:
            return []
        if n == 1:
            # Cannot create meaningful negative with single sample
            return [(claims[0], evidence_sets[0], evidence_vecs_list[0])]
        
        rng = random.Random(seed)
        idx = list(range(n))
        
        # ------------------------------------------------------------
        # Simple permutation modes (deranged/offset/cyclic/permute)
        # ------------------------------------------------------------
        if mode in ("deranged", "offset", "cyclic", "permute"):
            if mode == "cyclic":
                perm = [(i + 1) % n for i in idx]
            elif mode == "offset":
                off = offset % n
                perm = [(i + off) % n for i in idx]
            elif mode == "permute":
                perm = idx[:]
                rng.shuffle(perm)
            else:  # "deranged" (default)
                perm = self._derangement_indices(n, rng)
            
            negatives = [
                (claims[i], evidence_sets[perm[i]], evidence_vecs_list[perm[i]])
                for i in idx
                if i != perm[i]
            ]
            return negatives
        
        # ------------------------------------------------------------
        # Hard-mined v1: centroid cosine similarity (existing behavior)
        # ------------------------------------------------------------
        elif mode == "hard_mined":
            # Compute claim embeddings
            claim_vecs = self.embedder.embed(claims)
            claim_vecs = self._unit_norm_rows(claim_vecs)
            
            # Compute evidence centroids
            centroids = []
            valid_indices = []
            for i, ev_vecs in enumerate(evidence_vecs_list):
                if ev_vecs.size == 0:
                    continue
                ev_norm = self._unit_norm_rows(ev_vecs)
                centroid = self._unit_norm(ev_norm.mean(axis=0))
                if np.isfinite(centroid).all():
                    centroids.append(centroid)
                    valid_indices.append(i)
            
            if len(centroids) < 2:
                return self._generate_negatives(
                    claims, evidence_sets, evidence_vecs_list,
                    mode="deranged", seed=seed
                )
            
            centroid_mat = np.stack(centroids)
            centroid_mat = self._unit_norm_rows(centroid_mat)
            sim = claim_vecs @ centroid_mat.T
            
            negatives = []
            for i in range(n):
                if i not in valid_indices:
                    continue
                
                # Exclude self-similarity
                self_pos = valid_indices.index(i) if i in valid_indices else -1
                if self_pos >= 0:
                    sim[i, self_pos] = -np.inf
                
                best_idx = int(np.argmax(sim[i]))
                best_evidence_idx = valid_indices[best_idx]
                negatives.append((
                    claims[i],
                    evidence_sets[best_evidence_idx],
                    evidence_vecs_list[best_evidence_idx]
                ))
            
            return negatives
        
        # ------------------------------------------------------------
        # Hard-mined v2: ENERGY-AWARE MINING (NEW)
        # Select mismatched evidence that MINIMIZES hallucination energy
        # ------------------------------------------------------------
        elif mode == "hard_mined_v2":
            if energy_computer is None:
                raise ValueError(
                    "hard_mined_v2 requires energy_computer parameter. "
                    "Pass gate.energy_computer to _generate_negatives()."
                )
            
            # Precompute claim vectors
            claim_vecs = self._unit_norm_rows(np.asarray(
                self.embedder.embed(claims), dtype=np.float32
            ))
            
            # Build evidence index (skip empty evidence)
            valid_indices = []
            ev_vecs_norm = []
            for i, ev in enumerate(evidence_vecs_list):
                if ev.size == 0:
                    continue
                ev_norm = self._unit_norm_rows(np.asarray(ev, dtype=np.float32))
                if ev_norm.shape[0] > 0:
                    valid_indices.append(i)
                    ev_vecs_norm.append(ev_norm)
            
            if len(valid_indices) < 2:
                return self._generate_negatives(
                    claims, evidence_sets, evidence_vecs_list,
                    mode="deranged", seed=seed
                )
            
            # Shortlist candidates by centroid similarity (fast filter)
            centroids = [self._unit_norm(ev.mean(axis=0)) for ev in ev_vecs_norm]
            centroid_mat = self._unit_norm_rows(np.stack(centroids))
            sim = claim_vecs @ centroid_mat.T  # (n, m)
            
            negatives = []
            K = min(16, len(valid_indices))  # Top-16 candidates
            
            for i in range(n):
                # Shortlist top-K candidates by centroid similarity
                idx = np.argpartition(-sim[i], K - 1)[:K]
                
                # Find candidate with MINIMUM energy (true adversarial)
                best_j = None
                best_e = float("inf")
                
                for cand_pos in idx:
                    j_idx = valid_indices[int(cand_pos)]
                    if j_idx == i:  # Skip self
                        continue
                    
                    # Compute ACTUAL energy with mismatched evidence
                    e = energy_computer.compute(
                        claim_vec=claim_vecs[i],
                        evidence_vecs=ev_vecs_norm[valid_indices.index(j_idx)]
                    ).energy
                    
                    if e < best_e:
                        best_e = e
                        best_j = j_idx
                
                # Fallback if no candidate found (shouldn't happen with K>=2)
                if best_j is None:
                    best_j = valid_indices[int(rng.randint(0, len(valid_indices) - 1))]
                    best_e = 1.0
                
                negatives.append((
                    claims[i],
                    evidence_sets[best_j],
                    evidence_vecs_list[best_j]
                ))
            
            return negatives
        
        else:
            raise ValueError(
                f"Unknown neg_mode: '{mode}'. "
                "Valid modes: deranged, offset, cyclic, permute, hard_mined, hard_mined_v2"
            )    

    def _derangement_indices(self, n: int, rng: random.Random) -> List[int]:
        """
        Generate a true derangement (permutation with zero fixed points).
        Uses rejection sampling with Sattolo fallback for robustness.
        """
        if n <= 1:
            return list(range(n))
        
        # Rejection sampling (uniform over all derangements)
        for _ in range(1000):  # Practical upper bound
            perm = list(range(n))
            rng.shuffle(perm)
            if all(perm[i] != i for i in range(n)):
                return perm
        
        # Fallback: Sattolo's algorithm (guaranteed derangement for n>1)
        perm = list(range(n))
        for i in range(n - 1, 0, -1):
            j = rng.randrange(i)  # 0 <= j < i
            perm[i], perm[j] = perm[j], perm[i]
        return perm
    
    def _unit_norm(self, x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        norm = np.linalg.norm(x)
        return x / max(norm, eps)
    
    def _unit_norm_rows(self, X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms = np.where(norms < eps, 1.0, norms)
        return X / norms

    def _compute_neg_energies(
        self,
        claims: List[str],
        evidence_sets: List[List[str]],
        evidence_vecs: List[np.ndarray],
        *,
        neg_mode: str,
        neg_offset: int,
        seed: int,
        claim_vec_cache: Dict[str, np.ndarray],
    ) -> List[float]:
        neg_samples = self._generate_negatives(
            claims=claims,
            evidence_sets=evidence_sets,
            evidence_vecs_list=evidence_vecs,
            mode=neg_mode,
            offset=neg_offset,
            seed=seed,
            energy_computer=self.gate.energy_computer if neg_mode == "hard_mined_v2" else None,
        )

        energies = []
        for claim_text, _, ev_vecs_neg in neg_samples:
            if ev_vecs_neg.size == 0:
                continue

            if claim_text in claim_vec_cache:
                claim_vec = claim_vec_cache[claim_text]
            else:
                claim_vec = self.embedder.embed([claim_text])[0]
                claim_vec_cache[claim_text] = claim_vec

            e = compute_energy_from_vectors(
                claim_vec=claim_vec,
                evidence_vecs=ev_vecs_neg,
                energy_computer=self.gate.energy_computer,
            )
            energies.append(float(e))

        return energies

    def _compute_hard_negative_gap(
        self,
        deranged_energies: List[float],
        hard_energies: List[float],
    ) -> float:
        if len(deranged_energies) == 0 or len(hard_energies) == 0:
            return 0.0
        
        mean_deranged = float(np.mean(deranged_energies))
        mean_hard = float(np.mean(hard_energies))
        
        return mean_hard - mean_deranged

def compute_energy_from_vectors(
    claim_vec: np.ndarray,
    evidence_vecs: np.ndarray,
    energy_computer: HallucinationEnergyComputer,
) -> float:
    res = energy_computer.compute(
        claim_vec=claim_vec,
        evidence_vecs=evidence_vecs,
    )
    return res.energy

