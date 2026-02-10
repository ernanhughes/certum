from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Dict, Any
import numpy as np

class Verdict(Enum):
    ACCEPT = "accept"
    REVIEW = "review"
    REJECT = "reject"

@dataclass(frozen=True)
class EnergyResult:
    """Hallucination energy computation result."""
    energy: float          # [0.0, 1.0] - unsupported semantic mass
    explained: float       # ||U_r^T c||^2 - proportion explained by evidence subspace
    identity_error: float  # |1 - (explained + energy)| - numerical stability check
    topk: int              # evidence vectors used
    rank_r: int            # subspace rank used
    effective_rank: int    # actual rank from SVD
    used_count: int        # evidence vectors actually available
    sensitivity: float = 0.0

    def is_stable(self, threshold: float = 1e-4) -> bool:
        return self.identity_error < threshold

@dataclass(frozen=True)
class OracleCalibration:
    """Oracle-relative energy calibration."""
    oracle_energy: float   # Should be near 0.0 for valid oracle
    energy_gap: float      # claim_energy - oracle_energy
    is_valid: bool         # oracle_energy < threshold (e.g., 0.01)
    
    @property
    def gap_ratio(self) -> float:
        """Relative drift from oracle baseline."""
        if self.oracle_energy < 1e-7:
            return self.energy_gap  # Avoid division by near-zero
        return self.energy_gap / max(self.oracle_energy, 1e-7)

@dataclass(frozen=True)
class EvaluationResult:
    """Complete evaluation outcome."""
    claim: str
    evidence: List[str]
    energy_result: EnergyResult
    oracle_calibration: Optional[OracleCalibration]
    verdict: Verdict
    policy_applied: str
    robustness_probe: Optional[List[float]] = None  # Energy under param variations
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "claim": self.claim[:120] + "..." if len(self.claim) > 120 else self.claim,
            "energy": self.energy_result.energy,
            "oracle_energy": self.oracle_calibration.oracle_energy if self.oracle_calibration else None,
            "energy_gap": self.oracle_calibration.energy_gap if self.oracle_calibration else None,
            "verdict": self.verdict.value,
            "policy": self.policy_applied,
            "is_stable": self.energy_result.is_stable(),
            "probe_variance": float(np.var(self.robustness_probe)) if self.robustness_probe else None
        }