import numpy as np

from certum.policy.energy_only import EnergyOnlyPolicy
from certum.policy.policy import AdaptivePolicy
from certum.reporting.modules.policy_comparison import evaluate_policy


def sweep_gap_width(
    pos_rows,
    neg_rows,
    tau_energy,
    tau_pr,
    tau_sensitivity,
    gap_values=np.linspace(0.0, 0.30, 16),
):
    """
    Sweep ambiguity band width and evaluate performance.

    Returns list of dicts with:
        - gap_width
        - equal_far_tpr_gain
        - auc_delta
        - adaptive_far
        - adaptive_tpr
    """

    results = []

    # Energy-only baseline
    energy_policy = EnergyOnlyPolicy(tau_energy=tau_energy)
    energy_metrics = evaluate_policy(pos_rows, neg_rows, energy_policy)

    for gap in gap_values:

        adaptive = AdaptivePolicy(
            tau_energy=tau_energy,
            tau_pr=tau_pr,
            tau_sensitivity=tau_sensitivity,
            gap_width=gap
        )

        adaptive_metrics = evaluate_policy(pos_rows, neg_rows, adaptive)

        # Equal-FAR comparison
        target_far = adaptive_metrics["far"]

        neg_energies = np.array([r["energy"]["value"] for r in neg_rows])
        tau_equal = np.quantile(neg_energies, target_far)

        energy_equal = EnergyOnlyPolicy(tau_energy=tau_equal)
        energy_equal_metrics = evaluate_policy(pos_rows, neg_rows, energy_equal)

        tpr_gain = adaptive_metrics["tpr"] - energy_equal_metrics["tpr"]

        results.append({
            "gap_width": float(gap),
            "adaptive_tpr": adaptive_metrics["tpr"],
            "adaptive_far": adaptive_metrics["far"],
            "energy_equal_far_tpr": energy_equal_metrics["tpr"],
            "equal_far_tpr_gain": float(tpr_gain),
            "energy_metrics": energy_metrics,
            "adaptive_metrics": adaptive_metrics,
        })

    return results
