# src/certum/evaluation/feature_builder.py

from typing import List

import pandas as pd


def extract_dataframe_from_results(results: List) -> pd.DataFrame:
    """
    Convert EvaluationResult objects into a flat DataFrame
    suitable for modeling and evaluation.

    Compatible with:
        - SummarizationRunner outputs
        - Any EvaluationResult containing support_diagnostics

    Includes:
        - Similarity metrics
        - Coverage metrics
        - Energy aggregates
        - Energy gap features
        - Entailment aggregates
        - Structural signals
    """

    rows = []

    for r in results:

        s = r.support_diagnostics

        # ----------------------------
        # Defensive compatibility
        # ----------------------------

        min_energy = getattr(s, "min_energy", None)
        high_energy_count = getattr(s, "high_energy_count", None)

        if min_energy is None:
            min_energy = s.max_energy  # neutral fallback

        if high_energy_count is None:
            high_energy_count = 0

        # ----------------------------
        # Build flat row
        # ----------------------------

        row = {
            "label": r.label,

            # --------------------
            # Similarity
            # --------------------
            "mean_sim_top1": s.mean_sim_top1,
            "min_sim_top1": s.min_sim_top1,
            "mean_sim_margin": s.mean_sim_margin,
            "min_sim_margin": s.min_sim_margin,

            # --------------------
            # Coverage
            # --------------------
            "mean_coverage": s.mean_coverage,
            "min_coverage": s.min_coverage,

            # --------------------
            # Energy
            # --------------------
            "max_energy": s.max_energy,
            "mean_energy": s.mean_energy,
            "p90_energy": s.p90_energy,
            "frac_above_threshold": s.frac_above_threshold,
            "min_energy": min_energy,
            "energy_gap": s.max_energy - min_energy,
            "high_energy_count": high_energy_count,

            # --------------------
            # Entailment
            # --------------------
            "max_entailment": s.max_entailment,
            "mean_entailment": s.mean_entailment,
            "min_entailment": s.min_entailment,
            "entailment_gap": s.max_entailment - s.min_entailment,

            # --------------------
            # Structural
            # --------------------
            "sentence_count": s.sentence_count,
            "paragraph_count": s.paragraph_count,
        }

        rows.append(row)

    df = pd.DataFrame(rows)

    # Ensure numeric dtype
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.fillna(0.0)

    return df
