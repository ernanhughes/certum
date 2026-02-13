from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Dict, Any
import numpy as np

from certum.policy.decision_trace import DecisionTrace


class Verdict(Enum):
    ACCEPT = "accept"
    REVIEW = "review"
    REJECT = "reject"


@dataclass(frozen=True)
class GeometryDiagnostics:
    """
    Intrinsic geometric properties of claim–evidence interaction.
    All values are computed at SVD time.
    """

    # Spectral structure
    sigma1_ratio: float
    sigma2_ratio: float
    spectral_sum: float
    participation_ratio: float

    effective_rank: int
    used_count: int

    # Alignment
    alignment_to_sigma1: float

    # Similarity geometry
    sim_top1: float
    sim_top2: float
    sim_margin: float

    # Concentration / brittleness
    sensitivity: float

    # Optional raw vector (for offline research only)
    v1: np.ndarray
    entropy_rank: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "spectral": {
                "sigma1_ratio": self.sigma1_ratio,
                "sigma2_ratio": self.sigma2_ratio,
                "spectral_sum": self.spectral_sum,
                "participation_ratio": self.participation_ratio,
                "effective_rank": self.effective_rank,
            },
            "alignment": {
                "alignment_to_sigma1": self.alignment_to_sigma1,
            },
            "similarity": {
                "sim_top1": self.sim_top1,
                "sim_top2": self.sim_top2,
                "sim_margin": self.sim_margin,
            },
            "robustness": {
                "sensitivity": self.sensitivity,
            },
            "support": {
                "effective_rank": self.effective_rank,
                "used_count": self.used_count,
                "entropy_rank": self.entropy_rank,
            },
        }


@dataclass(frozen=True)
class EnergyResult:
    energy: float
    explained: float
    identity_error: float

    evidence_topk: int
    rank_cap: int

    geometry: GeometryDiagnostics

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
            "geometry": self.geometry.to_dict(),
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
    effectiveness: float

    embedding_info: Dict

    robustness_probe: Optional[List[float]] = None  # Energy under param variations
    difficulty_value: Optional[float] = 0.0
    difficulty_bucket: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "meta": {
                "run_id": self.run_id,
                "split": self.split,
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
                "probe_variance": float(np.var(self.robustness_probe))
                if self.robustness_probe
                else None,
            },
        }
