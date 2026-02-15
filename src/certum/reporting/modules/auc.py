import numpy as np
from sklearn.metrics import auc


def auc_from_curve(curve):
    """
    Compute AUC from ROC curve list.
    curve: list of {"tau", "tpr", "far"}
    """

    fars = np.array([p["far"] for p in curve])
    tprs = np.array([p["tpr"] for p in curve])

    # Sort by FAR ascending
    idx = np.argsort(fars)
    fars = fars[idx]
    tprs = tprs[idx]

    return float(auc(fars, tprs))
