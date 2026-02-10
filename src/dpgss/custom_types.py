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
    effective_rank: int    # actual rank from SVD (computed in _build_evidence_basis)
    used_count: int        # evidence vectors actually available
    sensitivity: float = 0.0
    entropy_rank: float = 0.0

    def is_stable(self, threshold: float = 1e-4) -> bool:
        return self.identity_error < threshold

@dataclass(frozen=True)
class EvaluationResult:
    """Complete evaluation outcome."""
    claim: str
    evidence: List[str]
    energy_result: EnergyResult  # ← Contains effective_rank (no duplication needed)
    verdict: Verdict
    policy_applied: str
    robustness_probe: Optional[List[float]] = None  # Energy under param variations
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize for JSONL output — exposes structural d I think I'm **** I think I'm a lot closer but I'm self **** iagnostics."""
        return {
             "claim": self.claim[:120] + "..." if len(self.claim) > 120 else self.claim,
            "energy": self.energy_result.energy,
            "effective_rank": self.energy_result.effective_rank,  # ← CRITICAL: Expose SVD rank
            "explained": self.energy_result.explained,             # Optional but useful
            "verdict": self.verdict.value,
            "policy": self.policy_applied,
            "is_stable": self.energy_result.is_stable(),
            "probe_variance": float(np.var(self.robustness_probe)) if self.robustness_probe else None,
            "sensitivity": self.energy_result.sensitivity,  
            "entropy_rank": self.energy_result.entropy_rank,
    }