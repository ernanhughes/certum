#!/usr/bin/env python
import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score



# -------------------------------------------------------------
# Utilities
# -------------------------------------------------------------

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def build_feature_matrix(pos, neg):
    def extract(record):
        g = record["energy"]["geometry"]
        return [
            record["energy"]["value"],
            g["spectral"]["participation_ratio"],
            g["robustness"]["sensitivity"],
            g["similarity"]["sim_margin"],
            g["support"]["effective_rank"],
            g["support"]["entropy_rank"],
            g["alignment"]["alignment_to_sigma1"],
        ]

    X_pos = np.array([extract(r) for r in pos])
    X_neg = np.array([extract(r) for r in neg])

    X = np.vstack([X_pos, X_neg])
    y = np.array([1]*len(X_pos) + [0]*len(X_neg))

    feature_names = [
        "energy",
        "participation_ratio",
        "sensitivity",
        "sim_margin",
        "effective_rank",
        "entropy_rank",
        "alignment"
    ]

    return X, y, feature_names


def extract_axes(records):
    out = {
        "energy": [],
        "pr": [],
        "sens": [],
        "alignment": [],
        "sim_margin": [],
        "effective_rank": [],
        "entropy_rank": [],
        "verdict": [],
    }

    for r in records:
        g = r["energy"]["geometry"]

        out["energy"].append(r["energy"]["value"])
        out["pr"].append(g["spectral"]["participation_ratio"])
        out["sens"].append(g["robustness"]["sensitivity"])
        out["alignment"].append(g["alignment"]["alignment_to_sigma1"])
        out["sim_margin"].append(g["similarity"]["sim_margin"])
        out["effective_rank"].append(g["support"]["effective_rank"])
        out["entropy_rank"].append(g["support"]["entropy_rank"])
        out["verdict"].append(r["decision"]["verdict"])

    for k in out:
        if k != "verdict":
            out[k] = np.array(out[k])

    return out


def compute_roc(pos_scores, neg_scores, higher_is_pos=True):
    y_true = np.concatenate([
        np.ones(len(pos_scores)),
        np.zeros(len(neg_scores))
    ])

    scores = np.concatenate([pos_scores, neg_scores])

    if not higher_is_pos:
        scores = -scores

    fpr, tpr, _ = roc_curve(y_true, scores)
    roc_auc = auc(fpr, tpr)

    return fpr, tpr, roc_auc


def tpr_at_far(fpr, tpr, far=0.01):
    return float(np.interp(far, fpr, tpr))

def boundary_composite_test(pos_axes, neg_axes, tau, margin):

    lower = tau - margin
    upper = tau + margin

    pos_mask = (pos_axes["energy"] >= lower) & (pos_axes["energy"] <= upper)
    neg_mask = (neg_axes["energy"] >= lower) & (neg_axes["energy"] <= upper)

    if pos_mask.sum() < 20 or neg_mask.sum() < 20:
        print("⚠️ Not enough boundary samples.")
        return

    pos_b = {k: v[pos_mask] for k, v in pos_axes.items() if isinstance(v, np.ndarray)}
    neg_b = {k: v[neg_mask] for k, v in neg_axes.items() if isinstance(v, np.ndarray)}

    diff_pos_b, diff_neg_b = build_composite_difficulty(pos_b, neg_b)

    fpr_b, tpr_b, auc_b = compute_roc(
        diff_pos_b,
        diff_neg_b,
        higher_is_positive=True
    )

    print("\nBoundary Composite Difficulty AUC:", float(auc_b))
    print("Boundary Composite TPR@1% FAR:", float(np.interp(0.01, fpr_b, tpr_b)))

def build_composite_difficulty(pos_axes, neg_axes):
    """
    Train logistic regression on geometric ambiguity signals
    to build composite difficulty score.
    """

    # Features: ambiguity-style metrics
    X_pos = np.stack([
        pos_axes["pr"],
        1 - pos_axes["sim_margin"],
        pos_axes["sens"],
        pos_axes["entropy_rank"],
        pos_axes["effective_rank"],
    ], axis=1)

    X_neg = np.stack([
        neg_axes["pr"],
        1 - neg_axes["sim_margin"],
        neg_axes["sens"],
        neg_axes["entropy_rank"],
        neg_axes["effective_rank"],
    ], axis=1)

    X = np.vstack([X_pos, X_neg])
    y = np.concatenate([
        np.ones(len(X_pos)),
        np.zeros(len(X_neg))
    ])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_scaled, y)

    difficulty_scores = clf.predict_proba(X_scaled)[:, 1]

    diff_pos = difficulty_scores[:len(X_pos)]
    diff_neg = difficulty_scores[len(X_pos):]

    print("\nLogistic Coefficients:")
    print(clf.coef_)

    return diff_pos, diff_neg


def boundary_analysis(pos_axes, neg_axes, tau, margin):

    print("\n================ BOUNDARY ANALYSIS ======================")

    lower = tau - margin
    upper = tau + margin

    pos_mask = (pos_axes["energy"] >= lower) & (pos_axes["energy"] <= upper)
    neg_mask = (neg_axes["energy"] >= lower) & (neg_axes["energy"] <= upper)

    pos_b = {k: v[pos_mask] if isinstance(v, np.ndarray) else v
             for k, v in pos_axes.items() if isinstance(v, np.ndarray)}

    neg_b = {k: v[neg_mask] if isinstance(v, np.ndarray) else v
             for k, v in neg_axes.items() if isinstance(v, np.ndarray)}

    print(f"Energy band: [{lower:.4f}, {upper:.4f}]")
    print(f"POS in band: {len(pos_b['energy'])}")
    print(f"NEG in band: {len(neg_b['energy'])}")

    if len(pos_b["energy"]) < 20 or len(neg_b["energy"]) < 20:
        print("⚠️ Not enough samples in boundary.")
        return

    # --- Test axes inside boundary ---
    axes_to_test = {
        "participation_ratio": ("pr", True),
        "sensitivity": ("sens", True),
        "alignment": ("alignment", True),
        "sim_margin": ("sim_margin", True),
        "effective_rank": ("effective_rank", True),
        "entropy_rank": ("entropy_rank", True),
    }

    for name, (key, higher_is_pos) in axes_to_test.items():
        fpr, tpr, roc_auc = compute_roc(
            pos_b[key],
            neg_b[key],
            higher_is_pos=higher_is_pos
        )

        tpr_at_1 = float(np.interp(0.01, fpr, tpr))

        print(f"\n{name}")
        print(f"  AUC (boundary): {roc_auc:.4f}")
        print(f"  TPR@1% FAR (boundary): {tpr_at_1:.4f}")

    print("\nBoundary correlations (POS)")
    print({
        "energy_pr": float(np.corrcoef(pos_b["energy"], pos_b["pr"])[0,1]),
        "energy_sens": float(np.corrcoef(pos_b["energy"], pos_b["sens"])[0,1]),
    })

    print("\nBoundary correlations (NEG)")
    print({
        "energy_pr": float(np.corrcoef(neg_b["energy"], neg_b["pr"])[0,1]),
        "energy_sens": float(np.corrcoef(neg_b["energy"], neg_b["sens"])[0,1]),
    })


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
    neg_file = list(run_dir.glob("neg_hard_mined_v2*.jsonl"))[0]

    print(f"📄 POS file: {pos_file}")
    print(f"📄 NEG file: {neg_file}")

    pos_records = load_jsonl(pos_file)
    pos = extract_axes(pos_records)
    neg_records = load_jsonl(neg_file)
    neg = extract_axes(neg_records)

    # -------------------------------------------------------------
    # Baseline: Energy
    # -------------------------------------------------------------

    fpr_e, tpr_e, auc_e = compute_roc(
        pos["energy"],
        neg["energy"],
        higher_is_pos=False  # lower energy = more positive
    )

    tpr_e_1 = tpr_at_far(fpr_e, tpr_e)

    print("\n================ BASELINE ======================")
    print(f"Energy AUC: {auc_e:.4f}")
    print(f"Energy TPR@1% FAR: {tpr_e_1:.4f}")

    # -------------------------------------------------------------
    # Candidate axes
    # -------------------------------------------------------------

    candidates = {
        "pr": (pos["pr"], neg["pr"], False),               # higher PR = more NEG
        "sens": (pos["sens"], neg["sens"], False),         # higher sens often NEG
        "alignment": (pos["alignment"], neg["alignment"], True), # higher alignment = POS
        "sim_margin": (pos["sim_margin"], neg["sim_margin"], True),
        "effective_rank": (pos["effective_rank"], neg["effective_rank"], False),
        "entropy_rank": (pos["entropy_rank"], neg["entropy_rank"], False),
    }

    results = []


    print("\n================ BOUNDARY TEST ======================")

    # Extract tau and margin from first record
    tau = pos_records[0]["decision"]["trace"]["tau_accept"]
    print(f"\nExtracted tau_accept: {tau:.4f}")
    margin = pos_records[0]["decision"]["trace"]["margin_band"]
    print(f"Extracted margin_band: {margin:.4f}")

    boundary_analysis(pos, neg, tau, margin)


    print("\n================ SECONDARY AXIS TEST ======================")

    for name, (pos_scores, neg_scores, higher_is_pos) in candidates.items():

        # Correlation with energy
        corr = np.corrcoef(
            np.concatenate([pos["energy"], neg["energy"]]),
            np.concatenate([pos_scores, neg_scores])
        )[0, 1]

        fpr, tpr, roc_auc = compute_roc(
            pos_scores,
            neg_scores,
            higher_is_pos=higher_is_pos
        )

        tpr1 = tpr_at_far(fpr, tpr)

        # Fusion with Energy (simple linear)
        if higher_is_pos:
            fusion_pos = -pos["energy"] + pos_scores
            fusion_neg = -neg["energy"] + neg_scores
        else:
            fusion_pos = -pos["energy"] - pos_scores
            fusion_neg = -neg["energy"] - neg_scores

        fpr_f, tpr_f, auc_f = compute_roc(
            fusion_pos,
            fusion_neg,
            higher_is_pos=True
        )

        tpr_f1 = tpr_at_far(fpr_f, tpr_f)

        delta_auc = auc_f - auc_e
        delta_tpr = tpr_f1 - tpr_e_1

        print(f"\n{name}")
        print(f"  AUC: {roc_auc:.4f}")
        print(f"  Fusion AUC: {auc_f:.4f}")
        print(f"  ΔAUC vs Energy: {delta_auc:.4f}")
        print(f"  Fusion TPR@1%: {tpr_f1:.4f}")
        print(f"  ΔTPR@1% vs Energy: {delta_tpr:.4f}")
        print(f"  Corr(Energy, {name}): {corr:.4f}")

        results.append({
            "name": name,
            "delta_auc": delta_auc,
            "delta_tpr": delta_tpr,
            "corr": abs(corr)
        })

    # -------------------------------------------------------------
    # Rank secondary axes
    # -------------------------------------------------------------

    ranked = sorted(
        results,
        key=lambda x: (x["delta_tpr"], x["delta_auc"], -x["corr"]),
        reverse=True
    )

    print("\n================ RANKED SECONDARY AXES ======================")
    for r in ranked:
        print(r)

    best = ranked[0]["name"]
    print(f"\n🏆 Best Secondary Axis: {best}")


    print("\n================ COMPOSITE DIFFICULTY ======================")

    diff_pos, diff_neg = build_composite_difficulty(pos, neg)

    fpr_d, tpr_d, auc_d = compute_roc(
        diff_pos,
        diff_neg,
        higher_is_pos=True
    )

    print("Composite Difficulty AUC:", float(auc_d))
    print("Composite Difficulty TPR@1% FAR:", float(np.interp(0.01, fpr_d, tpr_d)))


    fusion_pos = -pos["energy"] + diff_pos
    fusion_neg = -neg["energy"] + diff_neg

    fpr_f, tpr_f, auc_f = compute_roc(
        fusion_pos,
        fusion_neg,
        higher_is_pos=True
    )

    print("\nEnergy + Difficulty Fusion AUC:", float(auc_f))
    print("Fusion TPR@1% FAR:", float(np.interp(0.01, fpr_f, tpr_f)))


    print("\n================ NONLINEAR MODELS (GLOBAL) ======================")

    X, y, feature_names = build_feature_matrix(pos_records, neg_records)

    # ---------------- Random Forest ----------------
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        random_state=42
    )

    rf.fit(X, y)
    rf_scores = rf.predict_proba(X)[:, 1]
    auc_rf = roc_auc_score(y, rf_scores)

    print(f"\nRandom Forest AUC: {auc_rf:.4f}")

    print("\nFeature Importances (RF):")
    for name, imp in sorted(zip(feature_names, rf.feature_importances_), key=lambda x: -x[1]):
        print(f"{name:20s} {imp:.4f}")

    # ---------------- Gradient Boosting ----------------
    gb = GradientBoostingClassifier(
        n_estimators=300,
        max_depth=3,
        random_state=42
    )

    gb.fit(X, y)
    gb_scores = gb.predict_proba(X)[:, 1]
    auc_gb = roc_auc_score(y, gb_scores)

    print(f"\nGradient Boosting AUC: {auc_gb:.4f}")


    print("\n================ NONLINEAR MODELS (BOUNDARY ONLY) ======================")

    # Build full feature matrix again
    X_full, y_full, feature_names = build_feature_matrix(pos_records, neg_records)

    # Extract energy column index
    energy_idx = feature_names.index("energy")
    energy_vals = X_full[:, energy_idx]

    # Create boundary mask
    lower = tau - margin
    upper = tau + margin

    boundary_mask = (energy_vals >= lower) & (energy_vals <= upper)

    X_boundary = X_full[boundary_mask]
    y_boundary = y_full[boundary_mask]

    print(f"Boundary samples: {len(X_boundary)}")

    if len(X_boundary) > 50:  # avoid tiny sample instability

        rf_b = RandomForestClassifier(
            n_estimators=300,
            max_depth=6,
            random_state=42
        )

        rf_b.fit(X_boundary, y_boundary)
        rf_b_scores = rf_b.predict_proba(X_boundary)[:, 1]
        auc_rf_b = roc_auc_score(y_boundary, rf_b_scores)

        print(f"\nBoundary RF AUC: {auc_rf_b:.4f}")

        print("\nBoundary Feature Importances (RF):")
        for name, imp in sorted(zip(feature_names, rf_b.feature_importances_), key=lambda x: -x[1]):
            print(f"{name:20s} {imp:.4f}")

        gb_b = GradientBoostingClassifier(
            n_estimators=300,
            max_depth=3,
            random_state=42
        )

        gb_b.fit(X_boundary, y_boundary)
        gb_b_scores = gb_b.predict_proba(X_boundary)[:, 1]
        auc_gb_b = roc_auc_score(y_boundary, gb_b_scores)

        print(f"\nBoundary Gradient Boost AUC: {auc_gb_b:.4f}")

    else:
        print("⚠️ Not enough boundary samples for nonlinear modeling.")

    # -------------------------------------------------------------
    # Low-FAR zoom plot
    # -------------------------------------------------------------

    plots_dir = run_dir / "analysis_plots"
    plots_dir.mkdir(exist_ok=True)

    plt.figure()
    plt.plot(fpr_e, tpr_e, label=f"Energy ({auc_e:.3f})")
    plt.xlim(0, 0.02)
    plt.ylim(0, 0.5)
    plt.xlabel("FAR (0–2%)")
    plt.ylabel("TPR")
    plt.title("Low-FAR Zoom")
    plt.legend()
    plt.savefig(plots_dir / "roc_low_far_zoom.png")
    plt.close()

    # -------------------------------------------------------------
    # Energy vs Best Axis scatter
    # -------------------------------------------------------------

    plt.figure()
    plt.scatter(pos["energy"], pos[best], s=5, alpha=0.5)
    plt.scatter(neg["energy"], neg[best], s=5, alpha=0.5)
    plt.xlabel("Energy")
    plt.ylabel(best)
    plt.title(f"Energy vs {best}")
    plt.savefig(plots_dir / f"energy_vs_{best}.png")
    plt.close()

    print(f"\nPlots saved to: {plots_dir}")


if __name__ == "__main__":
    main()