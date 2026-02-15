import numpy as np
from sklearn.metrics import roc_auc_score

from certum.axes.bundle import AxisBundle
from certum.policy.energy_only import EnergyOnlyPolicy
from certum.policy.policy import AdaptivePolicy
from certum.reporting.modules.auc import auc_from_curve
from certum.reporting.modules.policy_comparison import (evaluate_policy,
                                                        sweep_policy_curve)


def extract_energy(rows):
    return np.array([r["energy"]["value"] for r in rows], dtype=float)


def gap_region_rows(pos_rows, neg_rows):
    """
    Identify overlap band between positive and negative energy distributions.
    Returns gap_pos_rows, gap_neg_rows.
    """

    pos_e = extract_energy(pos_rows)
    neg_e = extract_energy(neg_rows)

    if len(pos_e) == 0 or len(neg_e) == 0:
        return [], []

    overlap_low = max(pos_e.min(), neg_e.min())
    overlap_high = min(pos_e.max(), neg_e.max())

    if overlap_low >= overlap_high:
        # No overlap region
        return [], []

    def in_gap(rows):
        return [
            r for r in rows
            if overlap_low <= r["energy"]["value"] <= overlap_high
        ]

    return in_gap(pos_rows), in_gap(neg_rows)


def gap_conditioned_analysis(pos_rows, neg_rows, tau_energy, tau_pr, tau_sens, gap_width):

    all_rows = pos_rows + neg_rows
    labels = np.array(
        [1] * len(pos_rows) + [0] * len(neg_rows)
    )

    energies = np.array([r["energy"]["value"] for r in all_rows])

    mask = np.abs(energies - tau_energy) <= gap_width

    gap_rows = [r for r, m in zip(all_rows, mask) if m]
    gap_labels = labels[mask]

    n_pos = int(np.sum(gap_labels == 1))
    n_neg = int(np.sum(gap_labels == 0))

    if n_pos == 0 or n_neg == 0:
        return {
            "has_gap": False,
            "n_in_gap": len(gap_rows),
            "n_pos_in_gap": n_pos,
            "n_neg_in_gap": n_neg,
            "message": "No class overlap inside tau-centered gap."
        }

    # Split back
    gap_pos = gap_rows[:n_pos]
    gap_neg = gap_rows[n_pos:]

    taus = np.linspace(0.0, 1.0, 100)

    energy_curve = sweep_policy_curve(
        gap_pos,
        gap_neg,
        lambda tau: EnergyOnlyPolicy(tau_energy=tau),
        taus
    )

    adaptive_curve = sweep_policy_curve(
        gap_pos,
        gap_neg,
        lambda tau: AdaptivePolicy(
            tau_energy=tau,
            tau_pr=tau_pr,
            tau_sensitivity=tau_sens,
            gap_width=gap_width
        ),
        taus
    )

    energy_auc = auc_from_curve(energy_curve)
    adaptive_auc = auc_from_curve(adaptive_curve)

    return {
        "has_gap": True,
        "n_pos_in_gap": n_pos,
        "n_neg_in_gap": n_neg,
        "auc_energy_gap": energy_auc,
        "auc_adaptive_gap": adaptive_auc,
        "auc_delta_gap": adaptive_auc - energy_auc,
        "adaptive_dominates_in_gap": adaptive_auc > energy_auc
    }

def gap_mask(rows, tau_energy, gap_width):
    energies = np.array([r["energy"]["value"] for r in rows])
    return np.abs(energies - tau_energy) <= gap_width


def extract_axis(rows, axis_path):
    values = []
    for r in rows:
        val = r
        for key in axis_path:
            val = val.get(key, {})
        values.append(val if isinstance(val, (int, float)) else 0.0)
    return np.array(values, dtype=float)


def conditional_axis_auc(pos_rows, neg_rows, tau_energy, gap_width):
    """
    Compute AUC of axes inside energy gap centered at tau_energy.
    """

    all_rows = pos_rows + neg_rows
    labels = np.array(
        [1] * len(pos_rows) + [0] * len(neg_rows)
    )

    energies = np.array([r["energy"]["value"] for r in all_rows])

    mask = np.abs(energies - tau_energy) <= gap_width
    n_in_gap = int(np.sum(mask))

    if n_in_gap < 20:
        return {
            "n_in_gap": n_in_gap,
            "n_pos_in_gap": None,
            "n_neg_in_gap": None,
            "auc_participation": None,
            "auc_sensitivity": None,
            "reason": "too_few_samples"
        }

    labels_gap = labels[mask]
    n_pos = int(np.sum(labels_gap == 1))
    n_neg = int(np.sum(labels_gap == 0))

    # --- CRITICAL FIX ---
    if n_pos == 0 or n_neg == 0:
        return {
            "n_in_gap": n_in_gap,
            "n_pos_in_gap": n_pos,
            "n_neg_in_gap": n_neg,
            "auc_participation": None,
            "auc_sensitivity": None,
            "reason": "single_class_in_gap"
        }

    pr = extract_axis(
        all_rows,
        ["energy", "geometry", "spectral", "participation_ratio"]
    )[mask]

    sens = extract_axis(
        all_rows,
        ["energy", "geometry", "robustness", "sensitivity"]
    )[mask]

    auc_pr = roc_auc_score(labels_gap, -pr)
    auc_sens = roc_auc_score(labels_gap, -sens)

    return {
        "n_in_gap": n_in_gap,
        "n_pos_in_gap": n_pos,
        "n_neg_in_gap": n_neg,
        "auc_participation": float(auc_pr),
        "auc_sensitivity": float(auc_sens),
    }
