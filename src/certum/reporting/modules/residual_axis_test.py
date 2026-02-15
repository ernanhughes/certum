import numpy as np
from sklearn.metrics import roc_auc_score


def extract_axis(rows, axis_path):
    values = []
    for r in rows:
        val = r
        for key in axis_path:
            val = val.get(key, {})
        values.append(val if isinstance(val, (int, float)) else 0.0)
    return np.array(values, dtype=float)


def residual_axis_auc(pos_rows, neg_rows, tau_energy, band_width=0.05):
    """
    Compute axis AUC inside energy uncertainty band.
    """

    all_rows = pos_rows + neg_rows
    labels = np.array([1] * len(pos_rows) + [0] * len(neg_rows))

    energies = np.array([r["energy"]["value"] for r in all_rows])

    # Uncertainty band
    mask = np.abs(energies - tau_energy) <= band_width

    if np.sum(mask) < 30:
        return {
            "n_in_band": int(np.sum(mask)),
            "auc_participation": None,
            "auc_sensitivity": None,
            "auc_alignment": None,
        }

    labels_band = labels[mask]

    def safe_auc(scores):
        if len(np.unique(labels_band)) < 2:
            return None
        return float(roc_auc_score(labels_band, scores))

    pr = extract_axis(
        all_rows,
        ["energy", "geometry", "spectral", "participation_ratio"]
    )[mask]

    sens = extract_axis(
        all_rows,
        ["energy", "geometry", "robustness", "sensitivity"]
    )[mask]

    align = extract_axis(
        all_rows,
        ["energy", "geometry", "alignment", "alignment_to_sigma1"]
    )[mask]

    # Higher PR and sensitivity → more negative
    auc_pr = safe_auc(-pr)
    auc_sens = safe_auc(-sens)

    # Higher alignment → more positive
    auc_align = safe_auc(align)

    return {
        "n_in_band": int(np.sum(mask)),
        "auc_participation": auc_pr,
        "auc_sensitivity": auc_sens,
        "auc_alignment": auc_align,
    }
