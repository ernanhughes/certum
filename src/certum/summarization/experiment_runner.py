import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, precision_recall_curve,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler


class SummarizationExperimentRunner:

    def __init__(self, support_runner):
        self.support_runner = support_runner

    # ---------------------------------------------------------
    # Public Entry
    # ---------------------------------------------------------

    def run(self, samples, limit: int = None):

        run_id = f"summarization_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        run_dir = Path("runs") / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        if limit:
            samples = samples[:limit]

        print(f"\nRun ID: {run_id}")
        print(f"Using {len(samples)} samples")

        # -------------------------------------------------
        # Write config.json (Reproducibility)
        # -------------------------------------------------

        config = {
            "run_id": run_id,
            "sample_count": len(samples),
            "limit": limit,
            "model_type": "logistic_regression",
            "scaling": "StandardScaler",
            "cv_folds": 5,
            "bootstrap_samples": 1000,
        }

        with open(run_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)

        # -------------------------------------------------
        # 1️⃣ Run support analysis
        # -------------------------------------------------

        results_path = run_dir / "support_results.jsonl"

        results = self.support_runner.run(
            samples=samples,
            out_path=results_path,
        )

        # -------------------------------------------------
        # 2️⃣ Build DataFrame
        # -------------------------------------------------

        df = self._extract_dataframe(results)
        df.to_csv(run_dir / "features.csv", index=False)

        corr = df.corr(numeric_only=True)
        corr.to_csv(run_dir / "feature_correlation.csv")

        # -------------------------------------------------
        # Feature groups
        # -------------------------------------------------

        features_geometry = [
            "mean_sim_top1",
            "min_sim_top1",
            "mean_sim_margin",
            "min_sim_margin",
            "mean_coverage",
            "min_coverage",
            "max_energy",
            "mean_energy",
            "p90_energy",
            "frac_above_threshold",
            "min_energy",
            "energy_gap",
            "high_energy_count",
        ]

        features_entailment = [
            "max_entailment",
            "mean_entailment",
            "min_entailment",
            "entailment_gap",
        ]

        features_full_with_length = (
            features_geometry
            + features_entailment
            + ["sentence_count", "paragraph_count"]
        )

        features_full_no_length = (
            features_geometry
            + features_entailment
        )

        # -------------------------------------------------
        # Modeling Blocks
        # -------------------------------------------------

        report = {}

        report["geometry"] = self._run_model_block(
            df, features_geometry, run_dir, "geometry"
        )

        report["entailment"] = self._run_model_block(
            df, features_entailment, run_dir, "entailment"
        )

        report["full_with_length"] = self._run_model_block(
            df, features_full_with_length, run_dir, "full_with_length"
        )

        report["full_no_length"] = self._run_model_block(
            df, features_full_no_length, run_dir, "full_no_length"
        )

        # -------------------------------------------------
        # Ablation Study (Energy components)
        # -------------------------------------------------

        report["ablation"] = self._run_ablation(
            df,
            base_features=features_full_with_length,
            remove_sets=[
                ["energy_gap"],
                ["high_energy_count"],
                ["energy_gap", "high_energy_count"],
            ],
        )

        # -------------------------------------------------
        # Write final report
        # -------------------------------------------------

        with open(run_dir / "report.json", "w") as f:
            json.dump(report, f, indent=2)

        print("\nRun complete.")
        print(f"Artifacts saved to: {run_dir}")

    # ---------------------------------------------------------
    # Core Modeling Block
    # ---------------------------------------------------------

    def _run_model_block(self, df, features, run_dir, name):

        X = df[features].values
        y = df["label"].values

        # ---------------- Train/Test ----------------

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, stratify=y, random_state=42
        )

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        model = LogisticRegression(max_iter=3000)
        model.fit(X_train, y_train)

        probs = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, probs)

        # ---------------- Bootstrap CI ----------------

        mean_boot, low, high = self._bootstrap_auc(y_test, probs)

        # ---------------- Cross Validation ----------------

        cv_mean, cv_std = self._cross_validate(df, features)

        # ---------------- Save Coefficients ----------------

        coefs = dict(zip(features, model.coef_[0]))
        pd.DataFrame([coefs]).to_csv(
            run_dir / f"coefficients_{name}.csv", index=False
        )

        # ---------------- ROC ----------------

        fpr, tpr, _ = roc_curve(y_test, probs)
        plt.figure()
        plt.plot(fpr, tpr)
        plt.title(f"ROC - {name}")
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.tight_layout()
        plt.savefig(run_dir / f"roc_{name}.png")
        plt.close()

        # ---------------- PR Curve ----------------

        precision, recall, _ = precision_recall_curve(y_test, probs)
        ap = average_precision_score(y_test, probs)

        plt.figure()
        plt.plot(recall, precision)
        plt.title(f"PR - {name} (AP={ap:.3f})")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.tight_layout()
        plt.savefig(run_dir / f"pr_{name}.png")
        plt.close()

        # ---------------- Calibration ----------------

        prob_true, prob_pred = calibration_curve(y_test, probs, n_bins=10)

        plt.figure()
        plt.plot(prob_pred, prob_true, marker="o")
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.title(f"Calibration - {name}")
        plt.tight_layout()
        plt.savefig(run_dir / f"calibration_{name}.png")
        plt.close()

        return {
            "auc": float(auc),
            "bootstrap_mean": mean_boot,
            "ci_low": low,
            "ci_high": high,
            "cv_mean_auc": cv_mean,
            "cv_std_auc": cv_std,
            "average_precision": float(ap),
        }

    # ---------------------------------------------------------
    # Cross Validation
    # ---------------------------------------------------------

    def _cross_validate(self, df, features, folds=5):

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
            aucs.append(roc_auc_score(y[test_idx], probs))

        return float(np.mean(aucs)), float(np.std(aucs))

    # ---------------------------------------------------------
    # Ablation
    # ---------------------------------------------------------

    def _run_ablation(self, df, base_features, remove_sets):

        results = {}

        for remove in remove_sets:
            features = [f for f in base_features if f not in remove]

            X = df[features].values
            y = df["label"].values

            scaler = StandardScaler()
            X = scaler.fit_transform(X)

            model = LogisticRegression(max_iter=3000)
            model.fit(X, y)

            probs = model.predict_proba(X)[:, 1]
            auc = roc_auc_score(y, probs)

            results[str(remove)] = float(auc)

        return results

    # ---------------------------------------------------------
    # Bootstrap
    # ---------------------------------------------------------

    def _bootstrap_auc(self, y_true, y_probs, n_bootstrap=1000):

        rng = np.random.RandomState(42)
        aucs = []

        for _ in range(n_bootstrap):
            idx = rng.randint(0, len(y_true), len(y_true))

            if len(np.unique(y_true[idx])) < 2:
                continue

            auc = roc_auc_score(y_true[idx], y_probs[idx])
            aucs.append(auc)

        return (
            float(np.mean(aucs)),
            float(np.percentile(aucs, 2.5)),
            float(np.percentile(aucs, 97.5)),
        )

    # ---------------------------------------------------------
    # Feature Extraction
    # ---------------------------------------------------------

    def _extract_dataframe(self, results):

        rows = []

        for r in results:
            s = r.support_diagnostics

            min_energy = getattr(s, "min_energy", s.max_energy)
            high_energy_count = getattr(s, "high_energy_count", 0)

            rows.append({
                "label": r.label,

                "mean_sim_top1": s.mean_sim_top1,
                "min_sim_top1": s.min_sim_top1,
                "mean_sim_margin": s.mean_sim_margin,
                "min_sim_margin": s.min_sim_margin,

                "mean_coverage": s.mean_coverage,
                "min_coverage": s.min_coverage,

                "max_energy": s.max_energy,
                "mean_energy": s.mean_energy,
                "p90_energy": s.p90_energy,
                "frac_above_threshold": s.frac_above_threshold,
                "min_energy": min_energy,
                "energy_gap": s.max_energy - min_energy,
                "high_energy_count": high_energy_count,

                "max_entailment": s.max_entailment,
                "mean_entailment": s.mean_entailment,
                "min_entailment": s.min_entailment,
                "entailment_gap": s.max_entailment - s.min_entailment,

                "sentence_count": s.sentence_count,
                "paragraph_count": s.paragraph_count,
            })

        return pd.DataFrame(rows)
