import numpy as np


def extract_axis(rows, axis_name):
    values = []
    labels = []

    for r in rows:
        val = None
        if axis_name == "energy":
            val = r["energy"]["value"]
        else:
            val = r["energy"]["geometry"]["spectral"].get(axis_name)

        if val is not None:
            values.append(val)
            labels.append(1 if r["decision"]["verdict"] == "accept" else 0)

    return np.array(values), np.array(labels)


def correlation_with_correctness(rows, axis_name):
    vals, labels = extract_axis(rows, axis_name)
    if len(vals) != len(labels):
        return None
    if np.std(vals) < 1e-12 or np.std(labels) < 1e-12:
        return None
    if len(vals) < 5 or len(labels) == 0 or labels.mean() == 0:
        return None
    return float(np.corrcoef(vals, labels)[0, 1])


def region_split_by_energy_overlap(pos_rows, neg_rows):
    pos_e = np.array([r["energy"]["value"] for r in pos_rows])
    neg_e = np.array([r["energy"]["value"] for r in neg_rows])

    overlap_low = max(pos_e.min(), neg_e.min())
    overlap_high = min(pos_e.max(), neg_e.max())

    def in_gap(r):
        return overlap_low <= r["energy"]["value"] <= overlap_high

    gap_rows = [r for r in pos_rows + neg_rows if in_gap(r)]
    non_gap_rows = [r for r in pos_rows + neg_rows if not in_gap(r)]

    return gap_rows, non_gap_rows


def axis_shift_analysis(pos_rows, neg_rows):

    gap_rows, non_gap_rows = region_split_by_energy_overlap(pos_rows, neg_rows)
    all_rows = pos_rows + neg_rows

    axes = ["energy", "participation_ratio", "sensitivity"]

    results = {}

    for axis in axes:
        results[axis] = {
            "corr_global": correlation_with_correctness(all_rows, axis),
            "corr_gap": correlation_with_correctness(gap_rows, axis),
            "corr_non_gap": correlation_with_correctness(non_gap_rows, axis),
        }

    return results
