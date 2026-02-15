"""
Residual axis independence test.

Goal:
Determine whether secondary axes provide independent predictive
signal beyond energy.

Method:
Compare AUC of:
    - Energy only
    - Energy + one additional axis

If AUC improves meaningfully, the axis contains independent signal.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


# -----------------------------------------------------
# Utilities
# -----------------------------------------------------

def _safe_auc(y_true, y_score):
    """
    Safe AUC computation.
    Returns None if only one class present.
    """
    if len(np.unique(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, y_score))


def _extract_axis(rows, axis_path):
    values = []
    for r in rows:
        val = r
        for key in axis_path:
            val = val.get(key, {})
        values.append(val if isinstance(val, (int, float)) else 0.0)
    return np.array(values, dtype=float)


# -----------------------------------------------------
# Main Residual Test
# -----------------------------------------------------

def residual_axis_test(pos_rows, neg_rows):
    """
    Evaluate whether axes add independent signal beyond energy.

    Returns:
        dict with AUC comparisons and deltas.
    """

    if len(pos_rows) == 0 or len(neg_rows) == 0:
        return {
            "valid": False,
            "reason": "Empty class."
        }

    all_rows = pos_rows + neg_rows
    y = np.array([1]*len(pos_rows) + [0]*len(neg_rows))

    # ---- Extract axes ----
    energy = _extract_axis(all_rows, ["energy", "value"])
    pr = _extract_axis(all_rows, ["energy", "geometry", "spectral", "participation_ratio"])
    sens = _extract_axis(all_rows, ["energy", "geometry", "robustness", "sensitivity"])
    align = _extract_axis(all_rows, ["energy", "geometry", "alignment", "alignment_to_sigma1"])

    # ---- Energy-only model ----
    model_energy = LogisticRegression(max_iter=1000)
    model_energy.fit(energy.reshape(-1, 1), y)
    p_energy = model_energy.predict_proba(energy.reshape(-1, 1))[:, 1]
    auc_energy = _safe_auc(y, p_energy)

    # ---- Energy + Participation ----
    X_pr = np.column_stack([energy, pr])
    model_pr = LogisticRegression(max_iter=1000)
    model_pr.fit(X_pr, y)
    p_pr = model_pr.predict_proba(X_pr)[:, 1]
    auc_pr = _safe_auc(y, p_pr)

    # ---- Energy + Sensitivity ----
    X_sens = np.column_stack([energy, sens])
    model_sens = LogisticRegression(max_iter=1000)
    model_sens.fit(X_sens, y)
    p_sens = model_sens.predict_proba(X_sens)[:, 1]
    auc_sens = _safe_auc(y, p_sens)

    # ---- Energy + Alignment ----
    X_align = np.column_stack([energy, align])
    model_align = LogisticRegression(max_iter=1000)
    model_align.fit(X_align, y)
    p_align = model_align.predict_proba(X_align)[:, 1]
    auc_align = _safe_auc(y, p_align)

    # ---- Deltas ----
    def delta(a):
        if auc_energy is None or a is None:
            return None
        return float(a - auc_energy)

    return {
        "valid": True,
        "auc_energy_only": auc_energy,
        "auc_energy_plus_participation": auc_pr,
        "auc_energy_plus_sensitivity": auc_sens,
        "auc_energy_plus_alignment": auc_align,
        "delta_participation": delta(auc_pr),
        "delta_sensitivity": delta(auc_sens),
        "delta_alignment": delta(auc_align),
    }
