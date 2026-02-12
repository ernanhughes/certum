from __future__ import annotations
import numpy as np

from dpgss.difficulty.difficulty_metrics import DifficultyMetrics


class SpectralDifficulty:
    """
    Pure geometric difficulty measure based on
    spectral participation ratio of the evidence matrix.

    Difficulty reflects how diffusely evidence spans
    semantic space.

    0.0 → clear consensus (one dominant direction)
    1.0 → maximally diffuse support
    """

    def compute(self, m: DifficultyMetrics) -> float:
        """
        Compute normalized participation ratio from
        singular values of evidence embedding matrix.

        NOTE:
        This assumes effective_rank was computed from SVD
        inside the energy computer using singular values.
        If effective_rank is already participation-ratio-based,
        this becomes plug-compatible automatically.
        """

        k = max(1, m.evidence_count)

        # Edge case: single evidence sentence → no ambiguity
        if k <= 1:
            return 0.0

        # We assume effective_rank approximates PR
        # If effective_rank is true rank (integer),
        # this becomes a coarse approximation.
        pr = float(m.participation_ratio)

        # Normalize to [0,1]
        # PR range theoretically: 1 ≤ PR ≤ k
        difficulty = (pr - 1.0) / (k - 1.0)

        # Clamp for safety
        return float(np.clip(difficulty, 0.0, 1.0))
