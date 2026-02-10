# src/dpgss/gate.py
from typing import List, Optional

from dpgss.difficulty import DifficultyMetrics
from .custom_types import EnergyResult, EvaluationResult
from .embedder import Embedder
from .energy import HallucinationEnergyComputer
from .policy import Policy
import numpy as np

class VerifiabilityGate:
    def __init__(
        self,
        embedder: Embedder,
        energy_computer: HallucinationEnergyComputer,
    ):
        self.embedder = embedder
        self.energy_computer = energy_computer
    
    def compute_energy(
        self,
        claim: str,
        evidence_texts: List[str],
        *,
        claim_vec: Optional[np.ndarray] = None,    
        evidence_vecs: Optional[np.ndarray] = None,
    ) -> EnergyResult:
        """
        Compute hallucination energy WITHOUT policy decision.
        Used for calibration/sweeping — no dummy policy needed.
        """
        if claim_vec is None:
            claim_vec_raw = self.embedder.embed([claim])
            claim_vec = claim_vec_raw[0] if claim_vec_raw.shape[0] == 1 else claim_vec_raw
        
        if evidence_vecs is None:
            evidence_vecs = self.embedder.embed(evidence_texts)
        
        return self.energy_computer.compute(claim_vec, evidence_vecs)
    
    def evaluate(
        self,
        claim: str,
        evidence_texts: List[str],
        policy: Policy,
        *,
        claim_vec: Optional[np.ndarray] = None,      
        ev_vecs: Optional[np.ndarray] = None,  
    ) -> EvaluationResult:
        # 1. Embed with STRICT shape handling
        if claim_vec is None:
            claim_vec_raw = self.embedder.embed([claim])
            if claim_vec_raw.shape[0] == 1:
                claim_vec = claim_vec_raw[0]
            else:
                raise ValueError(f"Unexpected claim embedding shape: {claim_vec_raw.shape}")
        
        if ev_vecs is None:
            ev_vecs = self.embedder.embed(evidence_texts)
        
        # 2. Compute base energy
        base = self.energy_computer.compute(claim_vec, ev_vecs)
        
        # 4. Robustness probe
        probe = self.energy_computer.compute_robustness_probe(claim_vec, ev_vecs)
        robust_var = float(np.var(probe))
        difficulty = DifficultyMetrics(
            evidence_count=ev_vecs.shape[0],
            effective_rank=base.effective_rank,
            sensitivity=base.sensitivity,
            robustness_variance=robust_var,
            hard_negative_gap=policy.hard_negative_gap  # injected from calibration
        )

        effectiveness = self.effectiveness_score(base.energy, policy.tau_accept)

        # 5. Policy decision
        verdict = policy.decide(base, difficulty, effectiveness)
        
        return EvaluationResult(
            claim=claim,
            evidence=evidence_texts,
            energy_result=base,
            verdict=verdict,
            policy_applied=policy.name,
            robustness_probe=probe
        ) 
    

    def effectiveness_score(self, energy: float, tau: float) -> float:
        """
        How much margin we have relative to policy threshold.
        """
        return max(0.0, (tau - energy) / max(tau, 1e-6))
