from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Dict, Any
import numpy as np

from dpgss.policy.decision_trace import DecisionTrace

class Verdict(Enum):
    ACCEPT = "accept"
    REVIEW = "review"
    REJECT = "reject"

@dataclass(frozen=True)
class EnergyResult:
    """Hallucination energy computation result."""
    energy: float       
    explained: float
    identity_error: float

    evidence_topk: int
    rank_cap: int
    effective_rank: int
    used_count: int

    sensitivity: float = 0.0
    entropy_rank: float = 0.0

    def is_stable(self, threshold: float = 1e-4) -> bool:
        return self.identity_error < threshold

    def to_dict(self) -> Dict[str, Any]:
        return {
            "value": self.energy,
            "explained": self.explained,
            "identity_error": self.identity_error,

            "config": {
                "evidence_topk": self.evidence_topk,
                "rank_cap": self.rank_cap,
            },

            "support": {
                "effective_rank": self.effective_rank,
                "used_count": self.used_count,
                "entropy_rank": self.entropy_rank,
            },

            "robustness": {
                "sensitivity": self.sensitivity,
            }
        }

@dataclass(frozen=True)
class EvaluationResult:
    """Complete evaluation outcome."""
    claim: str
    evidence: List[str]

    energy_result: EnergyResult  
    decision_trace: DecisionTrace
    verdict: Verdict
    policy_applied: str

    run_id: str
    split: str
    difficulty_value: float
    difficulty_bucket: str
    effectiveness: float

    embedding_info: Dict

    neg_mode: Optional[str]
    robustness_probe: Optional[List[float]] = None  # Energy under param variations


    def to_dict(self) -> Dict[str, Any]:
        return {
            "meta": {
                "run_id": self.run_id,
                "split": self.split,
                "neg_mode": self.neg_mode,
                "policy": self.policy_applied,
            },

            "claim": self.claim,
            "evidence": self.evidence,

            "energy": self.energy_result.to_dict(),

            "difficulty": {
                "value": self.difficulty_value,
                "bucket": self.difficulty_bucket,
            },

            "effectiveness": self.effectiveness,

            "embedding": self.embedding_info,

            "decision": {
                "verdict": self.verdict.value,
                "trace": self.decision_trace.to_dict(),
            },

            "stability": {
                "is_stable": self.energy_result.is_stable(),
                "probe_variance": float(np.var(self.robustness_probe)) if self.robustness_probe else None,
            }
        }
