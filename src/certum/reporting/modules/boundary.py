import numpy as np


def dominates(curve_a, curve_b):
    """
    Check if curve A dominates curve B.
    A dominates if for all FAR, TPR_A >= TPR_B.
    """

    fars_a = np.array([p["far"] for p in curve_a])
    tprs_a = np.array([p["tpr"] for p in curve_a])

    fars_b = np.array([p["far"] for p in curve_b])
    tprs_b = np.array([p["tpr"] for p in curve_b])

    # interpolate B at A FAR values
    tprs_b_interp = np.interp(fars_a, fars_b, tprs_b)

    return bool(np.all(tprs_a >= tprs_b_interp))
