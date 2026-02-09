from typing import List, Tuple
from .types import OracleCalibration
from .embedder import Embedder
from .energy import HallucinationEnergyComputer

class OracleValidator:
    """
    Validates oracle construction quality.
    Prevents false confidence from trivial oracles.
    """
    
    def __init__(self, max_allowed_energy: float = 0.01):
        self.max_allowed_energy = max_allowed_energy
    
    def validate(
        self,
        oracle_claim: str,
        evidence_texts: List[str],
        embedder: Embedder,
        energy_computer: 'HallucinationEnergyComputer'
    ) -> OracleCalibration:
        """Compute oracle energy and validate it's non-trivial."""
        oracle_vec = embedder.embed([oracle_claim])[0]
        ev_vecs = embedder.embed(evidence_texts)
        
        oracle_res = energy_computer.compute(oracle_vec, ev_vecs)
        is_valid = oracle_res.energy < self.max_allowed_energy
        
        return OracleCalibration(
            oracle_energy=oracle_res.energy,
            energy_gap=0.0,  # Gap is 0 by definition for oracle itself
            is_valid=is_valid
        )

class NegativeControlGenerator:
    """
    Generates adversarial negative examples to stress-test the system.
    Critical for falsification testing (your "too good" concern).
    """
    
    def __init__(self, embedder: Embedder):
        self.embedder = embedder
    
    def create_domain_mismatch(
        self,
        claim: str,
        evidence_domain_a: List[str],
        evidence_domain_b: List[str]
    ) -> Tuple[str, List[str]]:
        """
        Create claim that blends domains → should show HIGH energy on mismatched evidence.
        Example: "Mitochondria exhibit quantum entanglement" + physics evidence
        """
        return claim, evidence_domain_b  # Deliberate mismatch
    
    def create_semantic_overreach(
        self,
        base_claim: str,
        evidence: List[str],
        overreach_level: int = 1  # 1=mild, 3=severe
    ) -> str:
        """
        Systematically inject unsupported assertions.
        Level 1: Add plausible but unverified qualifier
        Level 2: Introduce domain leap
        Level 3: Factual overreach
        """
        # Implementation would use LLM or templates - placeholder here
        modifiers = [
            lambda c: f"{c} with high confidence",
            lambda c: f"{c} as proven by recent studies",
            lambda c: f"{c} which definitively proves the underlying mechanism"
        ]
        return modifiers[min(overreach_level-1, 2)](base_claim)