from dataclasses import dataclass

@dataclass(frozen=True)
class DifficultyMetrics:
    evidence_count: int
    effective_rank: int
    sensitivity: float
    robustness_variance: float
    hard_negative_gap: float  # Δ energy between deranged vs hard_mined

    @property
    def difficulty_score(self) -> float:
        """
        Scalar difficulty proxy in [0,1].
        Conservative by design.
        """
        score = 0.0
        score += 0.3 * (1.0 if self.evidence_count <= 1 else 0.0)
        score += 0.3 * (1.0 - min(1.0, self.effective_rank / max(2, self.evidence_count)))
        score += 0.2 * min(1.0, self.sensitivity)
        score += 0.2 * min(1.0, self.robustness_variance)
        return min(1.0, score)
