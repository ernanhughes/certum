"""
High-level experiment orchestration.
"""

from typing import List

from .modeling import run_model
from .metrics import bootstrap_auc


def run_experiment(
    df,
    features: List[str],
    name: str,
):
    print(f"\n=== {name} ===")

    auc, coefs, y_test, probs = run_model(df, features)
    mean_boot, ci_low, ci_high = bootstrap_auc(y_test, probs)

    print(f"{name} AUC: {mean_boot:.4f}")
    print(f"95% CI: [{ci_low:.4f}, {ci_high:.4f}]")

    print("Coefficients:")
    for k, v in coefs.items():
        print(f"  {k}: {v:.4f}")

    return {
        "auc": auc,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "coefficients": coefs,
        "bootstrap_mean": mean_boot,
        "y_test": y_test,
        "probs": probs,
    }


def run_ablation(
    df,
    base_features: List[str],
    remove_sets: List[List[str]],
):
    print("\n=== Ablation Study ===")
    results = {}

    for remove in remove_sets:
        features = [f for f in base_features if f not in remove]

        auc, _, y_test, probs = run_model(df, features)
        mean_boot, ci_low, ci_high = bootstrap_auc(y_test, probs)

        print(f"\nRemoved: {remove}")
        print(f"AUC: {mean_boot:.4f}")
        print(f"CI: [{ci_low:.4f}, {ci_high:.4f}]")
        results[str(remove)] = {
            "auc": mean_boot,
            "ci": [ci_low, ci_high],
        }
    return results

