#!/usr/bin/env python
import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


# -------------------------------------------------------------
# Utilities
# -------------------------------------------------------------

def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def extract_axes(records):
    energy = []
    pr = []
    sens = []
    alignment = []
    margin = []
    verdict = []

    for r in records:
        e = r["energy"]["value"]
        g = r["energy"]["geometry"]

        energy.append(e)
        pr.append(g["spectral"]["participation_ratio"])
        sens.append(g["robustness"]["sensitivity"])
        alignment.append(g["alignment"]["alignment_to_sigma1"])
        margin.append(g["similarity"]["sim_margin"])
        verdict.append(r["decision"]["verdict"])

    return {
        "energy": np.array(energy),
        "pr": np.array(pr),
        "sens": np.array(sens),
        "alignment": np.array(alignment),
        "margin": np.array(margin),
        "verdict": verdict
    }


def summarize_axis(name, pos, neg):
    print(f"\n{name.upper()}")
    print("  POS:", {"mean": float(np.mean(pos)), "std": float(np.std(pos))})
    print("  NEG:", {"mean": float(np.mean(neg)), "std": float(np.std(neg))})
    print("  Δ:", float(np.mean(neg) - np.mean(pos)))


def correlations(label, axes):
    print(f"\n{label} CORRELATIONS")
    print({
        "energy_pr": float(np.corrcoef(axes["energy"], axes["pr"])[0,1]),
        "energy_sens": float(np.corrcoef(axes["energy"], axes["sens"])[0,1]),
        "pr_sens": float(np.corrcoef(axes["pr"], axes["sens"])[0,1]),
    })

def compute_roc(scores_pos, scores_neg, higher_is_positive=True):
    """
    Build ROC curve from POS/NEG score arrays.

    higher_is_positive:
        True  -> higher score = more likely POS
        False -> lower score = more likely POS (e.g., energy)
    """

    y_true = np.concatenate([
        np.ones(len(scores_pos)),
        np.zeros(len(scores_neg))
    ])

    scores = np.concatenate([scores_pos, scores_neg])

    if not higher_is_positive:
        scores = -scores  # invert if lower is better

    fpr, tpr, _ = roc_curve(y_true, scores)
    roc_auc = auc(fpr, tpr)

    return fpr, tpr, roc_auc


# -------------------------------------------------------------
# Main
# -------------------------------------------------------------

def main():

    run_dirs = sorted(Path("artifacts/runs").glob("*"))
    if not run_dirs:
        raise RuntimeError("No runs found.")

    run_dir = run_dirs[-1]
    print(f"\n📊 Using run: {run_dir}\n")

    pos_file = list(run_dir.glob("pos_hard_mined_v2*.jsonl"))[0]
    print(f"📄 POS file: {pos_file}")
    neg_file = list(run_dir.glob("neg_hard_mined_v2*.jsonl"))[0]
    print(f"📄 NEG file: {neg_file}")

    pos_records = load_jsonl(pos_file)
    neg_records = load_jsonl(neg_file)

    pos_axes = extract_axes(pos_records)
    neg_axes = extract_axes(neg_records)

    print("\n================ AXIS DISTRIBUTIONS ================")

    summarize_axis("energy", pos_axes["energy"], neg_axes["energy"])
    summarize_axis("participation_ratio", pos_axes["pr"], neg_axes["pr"])
    summarize_axis("sensitivity", pos_axes["sens"], neg_axes["sens"])

    print("\n================ CORRELATIONS ======================")
    correlations("POS", pos_axes)
    correlations("NEG", neg_axes)

    # -------------------------------------------------
    # Policy performance
    # -------------------------------------------------

    pos_verdict = np.array(pos_axes["verdict"])
    neg_verdict = np.array(neg_axes["verdict"])

    tpr = np.mean(pos_verdict == "accept")
    far = np.mean(neg_verdict == "accept")
    review_pos = np.mean(pos_verdict == "review")
    review_neg = np.mean(neg_verdict == "review")

    print("\n================ ROC ANALYSIS ======================")

    # Energy (lower = better POS)
    fpr_e, tpr_e, auc_e = compute_roc(
        pos_axes["energy"],
        neg_axes["energy"],
        higher_is_positive=False
    )

    print("Energy AUC:", float(auc_e))
    print("Energy TPR at FAR=1%:", float(np.interp(0.01, fpr_e, tpr_e)))

    # Participation Ratio (we assume higher = harder = more NEG)
    fpr_pr, tpr_pr, auc_pr = compute_roc(
        pos_axes["pr"],
        neg_axes["pr"],
        higher_is_positive=True
    )

    print("Participation Ratio AUC:", float(auc_pr))
    print("Participation Ratio TPR at FAR=1%:", float(np.interp(0.01, fpr_pr, tpr_pr)))
    # Energy + PR (simple linear fusion for now)
    # You can tune weights later
    fusion_pos = -pos_axes["energy"] + 0.1 * pos_axes["pr"]
    fusion_neg = -neg_axes["energy"] + 0.1 * neg_axes["pr"]

    fpr_f, tpr_f, auc_f = compute_roc(
        fusion_pos,
        fusion_neg,
        higher_is_positive=True
    )

    print("Energy+PR AUC:", float(auc_f))
    print("Energy+PR TPR at FAR=1%:", float(np.interp(0.01, fpr_f, tpr_f)))

    print("\n================ POLICY PERFORMANCE ================")
    print("TPR:", float(tpr))
    print("FAR:", float(far))
    print("review_pos:", float(review_pos))
    print("review_neg:", float(review_neg))

    # -------------------------------------------------
    # Plots
    # -------------------------------------------------

    plots_dir = run_dir / "analysis_plots"
    plots_dir.mkdir(exist_ok=True)

    # Energy distribution
    plt.figure()
    plt.hist(pos_axes["energy"], bins=50, alpha=0.6, label="POS")
    plt.hist(neg_axes["energy"], bins=50, alpha=0.6, label="NEG")
    plt.legend()
    plt.title("Energy Distribution")
    plt.savefig(plots_dir / "energy_distribution.png")
    plt.close()

    # Energy vs Sensitivity
    plt.figure()
    plt.scatter(pos_axes["energy"], pos_axes["sens"], s=5, alpha=0.5)
    plt.scatter(neg_axes["energy"], neg_axes["sens"], s=5, alpha=0.5)
    plt.title("Energy vs Sensitivity")
    plt.xlabel("Energy")
    plt.ylabel("Sensitivity")
    plt.savefig(plots_dir / "energy_vs_sensitivity.png")
    plt.close()

    # Energy vs Participation Ratio
    plt.figure()
    plt.scatter(pos_axes["energy"], pos_axes["pr"], s=5, alpha=0.5)
    plt.scatter(neg_axes["energy"], neg_axes["pr"], s=5, alpha=0.5)
    plt.title("Energy vs Participation Ratio")
    plt.xlabel("Energy")
    plt.ylabel("Participation Ratio")
    plt.savefig(plots_dir / "energy_vs_pr.png")
    plt.close()

    # ROC Plot
    plt.figure()
    plt.plot(fpr_e, tpr_e, label=f"Energy (AUC={auc_e:.3f})")
    plt.plot(fpr_pr, tpr_pr, label=f"PR (AUC={auc_pr:.3f})")
    plt.plot(fpr_f, tpr_f, label=f"Energy+PR (AUC={auc_f:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend()
    plt.savefig(plots_dir / "roc_curves.png")
    plt.close()

    print(f"\nPlots saved to: {plots_dir}\n")

    summary = {
        "axis_stats": {
            "energy": {
                "pos_mean": float(np.mean(pos_axes["energy"])),
                "neg_mean": float(np.mean(neg_axes["energy"])),
            },
            "participation_ratio": {
                "pos_mean": float(np.mean(pos_axes["pr"])),
                "neg_mean": float(np.mean(neg_axes["pr"])),
            },
            "sensitivity": {
                "pos_mean": float(np.mean(pos_axes["sens"])),
                "neg_mean": float(np.mean(neg_axes["sens"])),
            },
        },
        "separation": {
            "energy_delta": float(np.mean(neg_axes["energy"]) - np.mean(pos_axes["energy"])),
            "pr_delta": float(np.mean(neg_axes["pr"]) - np.mean(pos_axes["pr"])),
            "sens_delta": float(np.mean(neg_axes["sens"]) - np.mean(pos_axes["sens"])),
        },
        "policy_performance": {
            "TPR": float(tpr),
            "FAR": float(far),
            "review_pos": float(review_pos),
            "review_neg": float(review_neg),
        },
        "roc": {
            "energy_auc": float(auc_e),
            "pr_auc": float(auc_pr),
            "fusion_auc": float(auc_f),
        },
    }

    with open(run_dir / "analysis_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
