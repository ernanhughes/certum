from __future__ import annotations
from dataclasses import dataclass
from typing import List
from dpgss.custom_types import EvaluationResult
from dpgss.difficulty.difficulty_metrics import DifficultyMetrics

@dataclass(frozen=True)
class DifficultyRanges:
    """Calibrated from POSITIVE examples ONLY (clear evidence regime)"""
    margin_p90: float      # 90th percentile sim_margin from deranged positives
    rank_ratio_p90: float  # 90th percentile (effective_rank / evidence_count)

class Difficulty:
    def __init__(self, ranges: DifficultyRanges):
        self.ranges = ranges

    def compute(self, m: "DifficultyMetrics") -> float:
        """
        Pure ambiguity index: policy-agnostic evidence uncertainty.
        
        High difficulty when:
        - Evidence support is diffuse (small sim_margin)
        - Many evidence items contribute weakly (high effective_rank ratio)
        - Sensitivity to perturbation is high
        """
        # ---- 1. Similarity Ambiguity (PRIMARY SIGNAL) ----
        # Normalize against POSITIVE regime's typical margin
        margin_norm = min(1.0, m.sim_margin / max(1e-4, self.ranges.margin_p90))
        margin_score = 1.0 - margin_norm  # 0=clear winner, 1=ambiguous tie

        # ---- 2. Support Diffuseness ----
        rank_ratio = min(1.0, m.effective_rank / max(1, m.evidence_count))
        # Normalize against POSITIVE regime's typical diffuseness
        rank_norm = min(1.0, rank_ratio / max(1e-4, self.ranges.rank_ratio_p90))
        rank_score = rank_norm  # 0=sharp focus, 1=diffuse support

        # ---- 3. Sensitivity (Optional robustness signal) ----
        sensitivity_score = min(1.0, max(0.0, m.sensitivity))

        # ---- Weighted Blend (tuned for ambiguity separation) ----
        difficulty = (
            0.60 * margin_score +    # Dominant: evidence ambiguity
            0.30 * rank_score +      # Secondary: support diffuseness
            0.10 * sensitivity_score # Tertiary: perturbation sensitivity
        )

        return float(max(0.0, min(1.0, difficulty)))
    
    @staticmethod
    def calibrate_from_positives(results: List[EvaluationResult]) -> DifficultyRanges:
        """
        Calibrate ranges using POSITIVE examples ONLY (clear evidence regime).
        Establishes the "easy" baseline for ambiguity measurement.
        
        Analogous to energy calibration against negatives — but for ambiguity,
        we calibrate against positives to define "low ambiguity" baseline.
        """
        # Filter positives only
        pos_results = [r for r in results if r.split == "pos"]
        if not pos_results:
            raise ValueError("No positive examples found for difficulty calibration")
        
        # Extract ambiguity signals
        margins = [r.energy_result.sim_margin for r in pos_results]
        rank_ratios = [
            r.energy_result.effective_rank / max(1, len(r.evidence))
            for r in pos_results
        ]
        
        # Use 90th percentile as "typical easy case" boundary
        # (90% of positives should be below this ambiguity level)
        margin_p90 = float(sorted(margins)[int(0.9 * len(margins))]) if margins else 0.25
        rank_ratio_p90 = float(sorted(rank_ratios)[int(0.9 * len(rank_ratios))]) if rank_ratios else 0.4
        
        # Safety floors to avoid division-by-zero in compute()
        return DifficultyRanges(
            margin_p90=max(0.05, margin_p90),
            rank_ratio_p90=max(0.1, rank_ratio_p90)
        )