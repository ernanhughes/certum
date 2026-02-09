from typing import List, Optional
from .types import EvaluationResult
from .embedder import Embedder
from .energy import HallucinationEnergyComputer
from .oracle import OracleValidator
from .policy import Policy

class VerifiabilityGate:
    def __init__(
        self,
        embedder: Embedder,
        energy_computer: HallucinationEnergyComputer,
        oracle_validator: Optional[OracleValidator] = None
    ):
        self.embedder = embedder
        self.energy_computer = energy_computer
        self.oracle_validator = oracle_validator or OracleValidator()
    
    def evaluate(
        self,
        claim: str,
        evidence_texts: List[str],
        policy: Policy,
        oracle_claim: Optional[str] = None
    ) -> EvaluationResult:
        # 1. Embed with STRICT shape handling
        claim_vec_raw = self.embedder.embed([claim])  # Returns (1, d)
        if claim_vec_raw.shape[0] == 1:
            claim_vec = claim_vec_raw[0]  # CRITICAL: flatten to (d,)
        else:
            raise ValueError(f"Unexpected claim embedding shape: {claim_vec_raw.shape}")
        
        ev_vecs = self.embedder.embed(evidence_texts)  # (n, d)
        
        # 2. Compute base energy
        base_energy = self.energy_computer.compute(claim_vec, ev_vecs)
        
        # 3. Oracle calibration
        oracle_claim = oracle_claim or (evidence_texts[0] if evidence_texts else None)
        oracle_calibration = None
        if oracle_claim and evidence_texts:
            oracle_calibration = self.oracle_validator.validate(
                oracle_claim=oracle_claim,
                evidence_texts=evidence_texts,
                embedder=self.embedder,
                energy_computer=self.energy_computer
            )
        
        # 4. Robustness probe (NOW EXISTS)
        probe = self.energy_computer.compute_robustness_probe(claim_vec, ev_vecs)
        
        # 5. Policy decision
        verdict = policy.decide(base_energy, oracle_calibration)
        
        return EvaluationResult(
            claim=claim,
            evidence=evidence_texts,
            energy_result=base_energy,
            oracle_calibration=oracle_calibration,
            verdict=verdict,
            policy_applied=policy.name,
            robustness_probe=probe
        )