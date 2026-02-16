import json
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

from certum.embedding.hf_embedder import HFEmbedder
from certum.embedding.sqlite_embedding_backend import SQLiteEmbeddingBackend
from certum.geometry.claim_evidence import ClaimEvidenceGeometry
from certum.geometry.nli_wrapper import EntailmentModel
from certum.geometry.sentence_support import SentenceSupportAnalyzer
from certum.orchestration.summarization_runner import SummarizationRunner
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve, average_precision_score

warnings.simplefilter("ignore", FutureWarning)


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def run_model(df, features):

    X = df[features].values
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = LogisticRegression(max_iter=3000)
    model.fit(X_train, y_train)

    probs = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, probs)

    return auc, dict(zip(features, model.coef_[0])), y_test, probs


def ablation_study(df, base_features, remove_sets):
    print("\n=== Ablation Study ===")

    for remove in remove_sets:
        features = [f for f in base_features if f not in remove]

        auc, _, y_test, probs = run_model(df, features)
        mean_boot, low, high = bootstrap_auc(y_test, probs)

        print(f"\nRemoved: {remove}")
        print(f"AUC: {auc:.4f}")
        print(f"Bootstrap CI: [{low:.4f}, {high:.4f}]")


def bootstrap_auc(y_true, y_probs, n_bootstrap=1000, seed=42):
    rng = np.random.RandomState(seed)
    aucs = []

    for _ in range(n_bootstrap):
        indices = rng.randint(0, len(y_true), len(y_true))
        if len(np.unique(y_true[indices])) < 2:
            continue  # skip degenerate resample

        auc = roc_auc_score(y_true[indices], y_probs[indices])
        aucs.append(auc)

    lower = np.percentile(aucs, 2.5)
    upper = np.percentile(aucs, 97.5)

    return np.mean(aucs), lower, upper


def run_model_cv(df, features, folds=5):

    X = df[features].values
    y = df["label"].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)

    aucs = []

    for train_idx, test_idx in skf.split(X, y):
        model = LogisticRegression(max_iter=3000)
        model.fit(X[train_idx], y[train_idx])

        probs = model.predict_proba(X[test_idx])[:, 1]
        auc = roc_auc_score(y[test_idx], probs)
        aucs.append(auc)

    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)

    return mean_auc, std_auc


def plot_roc(df, features, filename):

    X = df[features].values
    y = df["label"].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    model = LogisticRegression(max_iter=3000)
    model.fit(X, y)

    probs = model.predict_proba(X)[:, 1]
    fpr, tpr, _ = roc_curve(y, probs)

    plt.figure()
    plt.plot(fpr, tpr)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.savefig(filename)
    plt.close()



def plot_precision_recall(y_true, probs, title="PR Curve"):
    precision, recall, _ = precision_recall_curve(y_true, probs)
    ap = average_precision_score(y_true, probs)

    plt.figure()
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"{title} (AP={ap:.3f})")
    plt.tight_layout()
    plt.savefig("precision_recall_curve.png")
    plt.close()

    return ap


def plot_calibration(y_true, probs, title="Calibration Curve"):
    prob_true, prob_pred = calibration_curve(y_true, probs, n_bins=10)

    plt.figure()
    plt.plot(prob_pred, prob_true, marker='o')
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel("Predicted Probability")
    plt.ylabel("True Probability")
    plt.title(title)
    plt.tight_layout()
    plt.savefig("calibration_curve.png")
    plt.close()


def extract_dataframe_from_results(results):
    rows = []

    for r in results:
        s = r.support_diagnostics

        # Defensive handling in case older results don't include new fields
        min_energy = getattr(s, "min_energy", None)
        high_energy_count = getattr(s, "high_energy_count", None)

        if min_energy is None:
            min_energy = s.max_energy  # fallback (neutral gap)

        if high_energy_count is None:
            high_energy_count = 0

        row = {
            "label": r.label,

            # --------------------
            # Similarity
            # --------------------
            "mean_sim_top1": s.mean_sim_top1,
            "min_sim_top1": s.min_sim_top1,
            "mean_sim_margin": s.mean_sim_margin,
            "min_sim_margin": s.min_sim_margin,

            # --------------------
            # Coverage
            # --------------------
            "mean_coverage": s.mean_coverage,
            "min_coverage": s.min_coverage,

            # --------------------
            # Energy
            # --------------------
            "max_energy": s.max_energy,
            "mean_energy": s.mean_energy,
            "p90_energy": s.p90_energy,
            "frac_above_threshold": s.frac_above_threshold,
            "min_energy": min_energy,
            "energy_gap": s.max_energy - min_energy,
            "high_energy_count": high_energy_count,

            # --------------------
            # Entailment
            # --------------------
            "max_entailment": s.max_entailment,
            "mean_entailment": s.mean_entailment,
            "min_entailment": s.min_entailment,
            "entailment_gap": s.max_entailment - s.min_entailment,

            # --------------------
            # Counts
            # --------------------
            "sentence_count": s.sentence_count,
            "paragraph_count": s.paragraph_count,
        }

        rows.append(row)

    return pd.DataFrame(rows)


def main():

    print("\n=== Loading Samples ===")
    samples = load_jsonl("E:\\data\\halueval_test_v1.jsonl")

    # limit = 3000
    # samples = samples[:limit]
    print(f"Loaded {len(samples)} samples for analysis.")

    backend = SQLiteEmbeddingBackend("E:\\data\\global_embeddings.db")
    embedder = HFEmbedder(
        "sentence-transformers/all-MiniLM-L6-v2",
        backend=backend
    )

    energy_computer = ClaimEvidenceGeometry(
        top_k=1000,
        rank_r=32
    )

    entailment_model = EntailmentModel(
        model_name="MoritzLaurer/deberta-v3-base-mnli-fever-anli",
        batch_size=32
    )

    # ----------------------------------------
    # NEW SUPPORT ANALYZER
    # ----------------------------------------

    support_analyzer = SentenceSupportAnalyzer(
        embedder=embedder,
        energy_computer=energy_computer,
        entailment_model=entailment_model,
        top_k=3
    )

    runner = SummarizationRunner(
        support_analyzer=support_analyzer
    )

    print("\n=== Running Summarization Runner ===")
    results = runner.run(
        samples,
        out_path="summary_results_20k.jsonl"
    )

    print("\nSaved structured results to summary_results_20k.jsonl")

    # ----------------------------------------
    # Convert to DataFrame
    # ----------------------------------------

    df = extract_dataframe_from_results(results)

    print("\n=== Correlation Matrix ===")

    corr = df.corr(numeric_only=True)
    print(corr.round(3))

    corr.to_csv("feature_correlation_matrix.csv")

    print("\nEnergy vs Similarity:")
    print(df[["mean_energy", "mean_sim_top1"]].corr())

    print("\nEnergy vs Entailment:")
    print(df[["mean_energy", "mean_entailment"]].corr())




    # ----------------------------------------
    # MODEL 1: Geometry-only
    # ----------------------------------------

    print("\n=== Modeling: Geometry Only ===")

    features_geometry = [
        # Similarity
        "mean_sim_top1",
        "min_sim_top1",
        "mean_sim_margin",
        "min_sim_margin",

        # Coverage
        "mean_coverage",
        "min_coverage",

        # Energy
        "max_energy",
        "mean_energy",
        "p90_energy",
        "frac_above_threshold",
        "min_energy",
        "energy_gap",
        "high_energy_count",
    ]

    auc_geo, coefs_geo, y_test, probs = run_model(df, features_geometry)

    mean_boot, ci_low, ci_high = bootstrap_auc(y_test, probs)

    print(f"Geometry only AUC: {mean_boot:.4f}")
    print(f"95% CI: [{ci_low:.4f}, {ci_high:.4f}]")

    print("AUC (geometry only):", auc_geo)
    for k, v in coefs_geo.items():
        print(f"  {k}: {v:.4f}")

    pd.DataFrame([coefs_geo]).to_csv("coefficients_geometry.csv", index=False)



    # ----------------------------------------
    # MODEL 2: Entailment-only
    # ----------------------------------------

    print("\n=== Modeling: Entailment Only ===")

    features_entailment = [
        "max_entailment",
        "mean_entailment",
        "min_entailment",
        "entailment_gap",
    ]

    auc_ent, coefs_ent, y_test, probs = run_model(df, features_entailment)
    mean_boot, ci_low, ci_high = bootstrap_auc(y_test, probs)

    print(f"Entailment only AUC: {mean_boot:.4f}")
    print(f"95% CI: [{ci_low:.4f}, {ci_high:.4f}]")

    print("AUC (entailment only):", auc_ent)
    for k, v in coefs_ent.items():
        print(f"  {k}: {v:.4f}")

    pd.DataFrame([coefs_ent]).to_csv("coefficients_entailment.csv", index=False)


    # ----------------------------------------
    # MODEL 3A: Full Fusion (WITH length)
    # ----------------------------------------

    print("\n=== Modeling: Full Fusion (WITH length) ===")

    features_full_with_length = features_geometry + features_entailment + [
        "sentence_count",
        "paragraph_count",
    ]

    auc_full_len, coefs_full_len, y_test, probs = run_model(df, features_full_with_length)
    mean_boot, ci_low, ci_high = bootstrap_auc(y_test, probs)

    print(f"Full Fusion (WITH length) AUC: {mean_boot:.4f}")
    print(f"95% CI: [{ci_low:.4f}, {ci_high:.4f}]")

    print("AUC (full with length):", auc_full_len)
    for k, v in coefs_full_len.items():
        print(f"  {k}: {v:.4f}")

    pd.DataFrame([coefs_full_len]).to_csv("coefficients_full_with_length.csv", index=False)


    # ----------------------------------------
    # MODEL 3B: Full Fusion (NO length)
    # ----------------------------------------

    print("\n=== Modeling: Full Fusion (NO length) ===")

    features_full_no_length = features_geometry + features_entailment

    auc_full_nolen, coefs_full_nolen, y_test, probs = run_model(df, features_full_no_length)
    mean_boot, ci_low, ci_high = bootstrap_auc(y_test, probs)

    print(f"Full Fusion (NO length) AUC: {mean_boot:.4f}")
    print(f"95% CI: [{ci_low:.4f}, {ci_high:.4f}]")

    print("AUC (full no length):", auc_full_nolen)
    for k, v in coefs_full_nolen.items():
        print(f"  {k}: {v:.4f}")

    pd.DataFrame([coefs_full_nolen]).to_csv("coefficients_full_no_length.csv", index=False)

    print("\n=== Modeling: Full Fusion (NO length) ===")

    mean_auc, std_auc = run_model_cv(df, features_full_with_length)

    print(f"AUC (5-fold CV): {mean_auc:.4f} ± {std_auc:.4f}")


    print(f"AUC 95% CI: [{ci_low:.4f}, {ci_high:.4f}]")

    plot_roc(df, features_geometry, "roc_geometry.png")
    plot_roc(df, features_entailment, "roc_entailment.png")
    plot_roc(df, features_full_with_length, "roc_full_with_length.png")
    plot_roc(df, features_full_no_length, "roc_full_no_length.png")

    plot_calibration(y_test, probs)
    ap = plot_precision_recall(y_test, probs)
    print(f"Average Precision: {ap:.4f}")

    ablation_study(
        df,
        features_full_with_length,
        remove_sets=[
            ["energy_gap"],
            ["high_energy_count"],
            ["energy_gap", "high_energy_count"],
        ]
    )

    print("Feature Stability")

    for seed in range(5):
        mean_auc, std_auc = run_model_cv(df.sample(frac=1, random_state=seed), features_full_with_length)
        print(seed, mean_auc)

    print("\n=== Experiment Complete ===")


if __name__ == "__main__":
    main()
