# src/dpgss/gate.py
from typing import List, Optional
from .types import EnergyResult, EvaluationResult
from .embedder import Embedder
from .energy import HallucinationEnergyComputer
from .policy import Policy

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
        evidence_texts: List[str]
    ) -> EnergyResult:
        """
        Compute hallucination energy WITHOUT policy decision.
        Used for calibration/sweeping — no dummy policy needed.
        """
        # Embed claim (flatten to 1D vector)
        claim_vec_raw = self.embedder.embed([claim])  # Returns (1, d)
        if claim_vec_raw.shape[0] == 1:
            claim_vec = claim_vec_raw[0]  # Flatten to (d,)
        else:
            raise ValueError(f"Unexpected claim embedding shape: {claim_vec_raw.shape}")
        
        # Embed evidence
        ev_vecs = self.embedder.embed(evidence_texts)  # (n, d)
        
        # Compute and return raw energy
        return self.energy_computer.compute(claim_vec, ev_vecs)
    
    def evaluate(
        self,
        claim: str,
        evidence_texts: List[str],
        policy: Policy,
        oracle_claim: Optional[str] = None
    ) -> EvaluationResult:
        # 1. Embed with STRICT shape handling
        claim_vec_raw = self.embedder.embed([claim])
        if claim_vec_raw.shape[0] == 1:
            claim_vec = claim_vec_raw[0]
        else:
            raise ValueError(f"Unexpected claim embedding shape: {claim_vec_raw.shape}")
        
        ev_vecs = self.embedder.embed(evidence_texts)
        
        # 2. Compute base energy
        base_energy = self.energy_computer.compute(claim_vec, ev_vecs)
        
        # 4. Robustness probe
        probe = self.energy_computer.compute_robustness_probe(claim_vec, ev_vecs)
        
        # 5. Policy decision
        verdict = policy.decide(base_energy)
        
        return EvaluationResult(
            claim=claim,
            evidence=evidence_texts,
            energy_result=base_energy,
            verdict=verdict,
            policy_applied=policy.name,
            robustness_probe=probe
        ) 