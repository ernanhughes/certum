"""
Plotting utilities for evaluation.
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (auc, average_precision_score,
                             precision_recall_curve, roc_curve)


def plot_roc(y_true, probs, out_path):
    # Defensive conversion
    y_true = np.asarray(y_true).reshape(-1)
    probs = np.asarray(probs).reshape(-1)

    fpr, tpr, _ = roc_curve(y_true, probs)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_precision_recall(y_true, probs, out_path: str):
    precision, recall, _ = precision_recall_curve(y_true, probs)
    ap = average_precision_score(y_true, probs)

    plt.figure()
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision–Recall (AP={ap:.3f})")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

    return ap


def plot_calibration(y_true, probs, out_path: str):
    prob_true, prob_pred = calibration_curve(
        y_true,
        probs,
        n_bins=10,
    )

    plt.figure()
    plt.plot(prob_pred, prob_true, marker="o")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("Predicted Probability")
    plt.ylabel("True Probability")
    plt.title("Calibration Curve")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
