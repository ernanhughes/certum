# src/certum/evaluation/runner.py

import argparse
import json
import logging
import uuid
import warnings
from pathlib import Path

import numpy as np

from certum.embedding.hf_embedder import HFEmbedder
from certum.embedding.sqlite_embedding_backend import SQLiteEmbeddingBackend
from certum.geometry.claim_evidence import ClaimEvidenceGeometry
from certum.geometry.nli_wrapper import EntailmentModel
from certum.geometry.sentence_support import SentenceSupportAnalyzer
from certum.orchestration.summarization_runner import SummarizationRunner


from .experiments import run_ablation, run_experiment
from .feature_builder import extract_dataframe_from_results
from .metrics import bootstrap_auc
from .modeling import run_model_cv, run_xgb_model, run_xgb_model_cv
from .plots import plot_calibration, plot_precision_recall, plot_roc
from .pipeline import run_summarization_pipeline

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# =========================================================
# Evaluation Runner (Certum-style aligned)
# =========================================================

class EvaluationRunner:

    def run(
        self,
        *,
        input_jsonl: Path,
        embedding_model: str,
        embedding_db: Path,
        nli_model: str,
        top_k: int,
        limit: int,
        seed: int,
        dataset_name: str,
        out_dir: Path,
        entailment_db: Path,
        geometry_top_k: int = 1000,
        rank_r: int = 32,
    ) -> None:

        # =====================================================
        # 0️⃣ Setup
        # =====================================================

        run_id = str(uuid.uuid4())
        out_dir.mkdir(parents=True, exist_ok=True)

        np.random.seed(seed)

        logger.info(f"Starting evaluation run {run_id}")
        logger.info(f"Run folder: {out_dir}")

        # =====================================================
        # 1️⃣ Load dataset
        # =====================================================

        samples = self._load_jsonl(input_jsonl)

        if limit:
            samples = samples[:limit]

        logger.info(f"Loaded {len(samples)} samples.")

        # =====================================================
        # 2️⃣ Build Core Pipeline (NO hardcoding)
        # =====================================================

        results_path = out_dir / "summary_results.jsonl"

        results = run_summarization_pipeline(
            samples=samples,
            embedding_model=embedding_model,
            embedding_db=embedding_db,
            nli_model=nli_model,
            entailment_db=entailment_db,
            top_k=top_k,
            geometry_top_k=geometry_top_k,
            rank_r=rank_r,
            out_path=results_path,
        )

        # =====================================================
        # 3️⃣ Feature Extraction
        # =====================================================

        df = extract_dataframe_from_results(results)
        logger.info(f"Extracted feature dataframe shape: {df.shape}")

        # =====================================================
        # 4️⃣ Write Config
        # =====================================================

        config = {
            "run_id": run_id,
            "dataset": dataset_name,
            "input_jsonl": str(input_jsonl),
            "n_samples": len(df),
            "embedding_model": embedding_model,
            "embedding_db": str(embedding_db),
            "nli_model": nli_model,
            "entailment_db": str(entailment_db),
            "top_k_sentence_support": top_k,
            "geometry_top_k": geometry_top_k,
            "geometry_rank_r": rank_r,
            "limit": limit,
            "seed": seed,
        }

        self._write_json(out_dir / "config.json", config)

        # =====================================================
        # 5️⃣ Correlation Matrix
        # =====================================================

        corr = df.corr(numeric_only=True)
        corr.to_csv(out_dir / "feature_correlation.csv")

        # =====================================================
        # 6️⃣ Feature Sets
        # =====================================================

        feature_sets = self._build_feature_sets()

        results_summary = {}

        # =====================================================
        # 7️⃣ Logistic Experiments
        # =====================================================

        for name, features in feature_sets.items():

            logger.info(f"Running experiment: {name}")

            result = run_experiment(df, features, name)

            mean_boot, ci_low, ci_high = bootstrap_auc(
                result["y_test"],
                result["probs"],
            )

            mean_cv, std_cv = run_model_cv(df, features)

            plot_roc(
                result["y_test"],
                result["probs"],
                out_dir / f"roc_{name}.png",
            )

            if name == "full":
                ap = plot_precision_recall(
                    result["y_test"],
                    result["probs"],
                    out_dir / "precision_recall.png",
                )

                plot_calibration(
                    result["y_test"],
                    result["probs"],
                    out_dir / "calibration.png",
                )
            else:
                ap = None

            results_summary[name] = {
                "auc": float(result["auc"]),
                "bootstrap_mean_auc": float(mean_boot),
                "bootstrap_ci": [float(ci_low), float(ci_high)],
                "cv_mean_auc": float(mean_cv),
                "cv_std_auc": float(std_cv),
                "average_precision": float(ap) if ap else None,
                "coefficients": result["coefficients"],
            }

        # =====================================================
        # 8️⃣ Ablation (Full Model)
        # =====================================================

        ablation = run_ablation(
            df,
            feature_sets["full"],
            remove_sets=[
                ["energy_gap"],
                ["high_energy_count"],
                ["energy_gap", "high_energy_count"],
            ],
        )

        results_summary["ablation"] = ablation

        # =====================================================
        # 9️⃣ XGBoost (Full Features Only)
        # =====================================================

        full_features = feature_sets["full"]

        xgb_auc, xgb_importance, y_test, probs = run_xgb_model(
            df,
            full_features,
        )

        xgb_cv_mean, xgb_cv_std = run_xgb_model_cv(
            df,
            full_features,
        )

        logger.info(f"XGBoost AUC: {xgb_auc:.4f}")
        logger.info(f"XGBoost CV AUC: {xgb_cv_mean:.4f} ± {xgb_cv_std:.4f}")

        plot_roc(
            y_test,
            probs,
            out_dir / "roc_xgboost.png",
        )

        # =====================================================
        # 🔟 Final Report
        # =====================================================

        report = {
            "run_id": run_id,
            "dataset": dataset_name,
            "n_samples": len(df),
            "results": results_summary,
            "xgboost": {
                "auc": float(xgb_auc),
                "cv_mean_auc": float(xgb_cv_mean),
                "cv_std_auc": float(xgb_cv_std),
                "feature_importance": xgb_importance,
            },
        }

        self._write_json(out_dir / "report.json", report)

        logger.info("Evaluation complete.")

    # =====================================================
    # Utilities
    # =====================================================

    def _write_json(self, path: Path, obj: dict):

        def convert(o):
            if isinstance(o, (np.float32, np.float64)):
                return float(o)
            if isinstance(o, (np.int32, np.int64)):
                return int(o)
            if isinstance(o, np.ndarray):
                return o.tolist()
            raise TypeError

        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, default=convert)

    def _load_jsonl(self, path: Path):
        with open(path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f]

    def _build_feature_sets(self):

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

        features_full = (
            features_geometry
            + features_entailment
            + ["sentence_count", "paragraph_count"]
        )

        return {
            "geometry": features_geometry,
            "entailment": features_entailment,
            "full": features_full,
        }


# =========================================================
# CLI ENTRYPOINT
# =========================================================

def main():

    ap = argparse.ArgumentParser(description="Certum Summarization Evaluation")

    ap.add_argument("--input_jsonl", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--dataset_name", type=str, default="unknown")
    ap.add_argument("--embedding_model", type=str, required=True)
    ap.add_argument("--embedding_db", type=Path, required=True)
    ap.add_argument("--nli_model", type=str, required=True)
    ap.add_argument("--top_k", type=int, default=3)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--entailment_db", type=Path, default=Path("entailment_cache.db"))
    ap.add_argument("--geometry_top_k", type=int, default=1000)
    ap.add_argument("--rank_r", type=int, default=32)

    args = ap.parse_args()

    runner = EvaluationRunner()

    runner.run(
        input_jsonl=args.input_jsonl,
        embedding_model=args.embedding_model,
        embedding_db=args.embedding_db,
        nli_model=args.nli_model,
        top_k=args.top_k,
        limit=args.limit,
        seed=args.seed,
        dataset_name=args.dataset_name,
        out_dir=args.out_dir,
        entailment_db=args.entailment_db,
        geometry_top_k=args.geometry_top_k,
        rank_r=args.rank_r,
    )


if __name__ == "__main__":
    main()
