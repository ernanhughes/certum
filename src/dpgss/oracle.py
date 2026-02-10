from typing import List
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
