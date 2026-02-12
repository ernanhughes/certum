import numpy as np
from dataclasses import dataclass

# src/dpgss/difficulty/difficulty_score.py

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class DifficultyConfig:
    """
    Calibration parameters for Difficulty score.
    Derived from POSITIVE regime only.
    """
    margin_p90: float
    rank_ratio_p90: float
    pr_p90: float

    # weights (must sum to 1.0)
    w_margin: float = 0.5
    w_rank: float = 0.25
    w_sensitivity: float = 0.15
    w_pr: float = 0.10

    def validate(self):
        total = self.w_margin + self.w_rank + self.w_sensitivity + self.w_pr
        if not np.isclose(total, 1.0):
            raise ValueError(f"Difficulty weights must sum to 1.0, got {total}")



def compute_difficulty(
    *,
    sim_margin: float,
    effective_rank: int,
    evidence_count: int,
    sensitivity: float,
    participation_ratio: float,
    config: DifficultyConfig
) -> float:
    """
    Formal Difficulty functional D(x) ∈ [0,1]

    D(x) =
        w_m * A_m
      + w_r * A_r
      + w_s * A_s
      + w_pr * A_pr
    """

    config.validate()

    eps = 1e-8

    # ----- 1. Margin Ambiguity -----
    margin_norm = min(1.0, sim_margin / max(config.margin_p90, eps))
    A_m = 1.0 - margin_norm

    # ----- 2. Rank Diffuseness -----
    rank_ratio = effective_rank / max(1, evidence_count)
    rank_norm = min(1.0, rank_ratio / max(config.rank_ratio_p90, eps))
    A_r = rank_norm

    # ----- 3. Sensitivity -----
    A_s = min(1.0, max(0.0, sensitivity))

    # ----- 4. Spectral Diffusion -----
    pr_norm = min(1.0, participation_ratio / max(config.pr_p90, eps))
    A_pr = pr_norm

    # ----- Weighted Blend -----
    D = (
        config.w_margin * A_m
        + config.w_rank * A_r
        + config.w_sensitivity * A_s
        + config.w_pr * A_pr
    )

    return float(np.clip(D, 0.0, 1.0))

def calibrate_difficulty_from_positives(results) -> DifficultyConfig:
    """
    results: iterable of JSON-decoded evaluation objects.
    Only positive regime should be passed.
    """

    margins = []
    rank_ratios = []
    prs = []

    for r in results:
        geo = r["energy"]["geometry"]

        sim_margin = geo["similarity"]["sim_margin"]
        effective_rank = geo["spectral"]["effective_rank"]
        evidence_count = r["embedding"]["evidence_count"]
        pr = geo["spectral"]["participation_ratio"]

        margins.append(sim_margin)
        rank_ratios.append(effective_rank / max(1, evidence_count))
        prs.append(pr)

    margin_p90 = float(np.percentile(margins, 90))
    rank_ratio_p90 = float(np.percentile(rank_ratios, 90))
    pr_p90 = float(np.percentile(prs, 90))

    return DifficultyConfig(
        margin_p90=max(0.05, margin_p90),
        rank_ratio_p90=max(0.1, rank_ratio_p90),
        pr_p90=max(1.0, pr_p90)
    )
