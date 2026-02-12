
import numpy as np

def compute_effectiveness_curve(
    *,
    difficulties,
    verdicts,
    labels,
    n_bins=10
):
    """
    difficulties: array-like D(x)
    verdicts: array-like ['accept','reject','review']
    labels: array-like ground truth (1=positive, 0=negative)

    Returns:
        dict with bin centers and effectiveness values
    """

    difficulties = np.asarray(difficulties)
    labels = np.asarray(labels)

    # correctness logic
    correct = []
    for v, y in zip(verdicts, labels):
        if v == "review":
            correct.append(1)  # abstention not penalized
        elif v == "accept" and y == 1:
            correct.append(1)
        elif v == "reject" and y == 0:
            correct.append(1)
        else:
            correct.append(0)

    correct = np.asarray(correct)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = []
    effectiveness = []
    counts = []

    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        mask = (difficulties >= lo) & (difficulties < hi)

        if np.sum(mask) == 0:
            effectiveness.append(np.nan)
            counts.append(0)
        else:
            eff = np.mean(correct[mask])
            effectiveness.append(float(eff))
            counts.append(int(np.sum(mask)))

        bin_centers.append((lo + hi) / 2.0)

    return {
        "bin_centers": np.array(bin_centers),
        "effectiveness": np.array(effectiveness),
        "counts": np.array(counts),
    }
