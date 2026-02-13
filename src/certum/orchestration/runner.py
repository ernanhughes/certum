# src/certum/runner.py

from pathlib import Path
from typing import Optional
from time import time
import uuid
import json
import random
import numpy as np
import logging

from tqdm import tqdm

from certum.evidence.sqlite_evidence_store import SQLiteEvidenceStore
from certum.policy.policy import AdaptivePolicy
from certum.calibration import AdaptiveCalibrator
from certum.orchestration.audit import AuditLogger
from certum.adversarial import get_adversarial_generator
from certum.dataset.loader import load_examples
from certum.plot import plot_distributions
from certum.orchestration.factory import CertumFactory
import argparse

logger = logging.getLogger(__name__)


class CertumRunner:
    """
    Certum Orchestration Layer.

    Responsible for:
        - Dataset loading
        - Calibration
        - Policy construction
        - Evaluation
        - Reporting

    No model logic here.
    No energy logic here.
    No policy logic here.
    """

    def __init__(self):
        self.factory = CertumFactory()

    def run(
        self,
        *,
        kind: str,
        in_path: Path,
        model: str,
        cache_db: Path,
        embedding_db: Path,
        regime: str,
        far: float,
        cal_frac: float,
        n: int,
        seed: int,
        neg_mode: str,
        out_report: Path,
        out_pos_scored: Path,
        out_neg_scored: Path,
        neg_offset: Optional[int] = None,
        plot_png: Optional[Path] = None,
    ):

        start_time = time()
        run_id = str(uuid.uuid4())

        logger.info(f"Starting Certum run {run_id}")

        random.seed(seed)
        np.random.seed(seed)

        # -------------------------------------------------
        # 1. Load dataset
        # -------------------------------------------------

        evidence_store = SQLiteEvidenceStore(cache_db)

        samples, load_stats = load_examples(
            kind,
            in_path,
            n,
            seed,
            evidence_store=evidence_store,
            model=model,
        )

        if len(samples) < 50:
            raise RuntimeError("Too few usable examples.")

        cal_n = int(len(samples) * cal_frac)
        cal_samples = samples[:cal_n]
        eval_samples = samples[cal_n:]

        # -------------------------------------------------
        # 2. Build core objects
        # -------------------------------------------------

        embedder = self.factory.build_embedder(model=model, embedding_db=embedding_db)

        self._ensure_vectors(samples, embedder)

        energy_computer = self.factory.build_energy_computer()
        gate = self.factory.build_gate(embedder, energy_computer)

        # -------------------------------------------------
        # 3. Calibration
        # -------------------------------------------------

        calibrator = AdaptiveCalibrator(gate, embedder=embedder)

        claim_vec_cache = {}

        sweep_results = calibrator.run_sweep(
            claims=[s["claim"] for s in cal_samples],
            evidence_sets=[s["evidence"] for s in cal_samples],
            evidence_vecs=[s["evidence_vecs"] for s in cal_samples],
            percentiles=[int(far * 100)],
            neg_mode=neg_mode,
            neg_offset=neg_offset or 37,
            seed=seed,
            claim_vec_cache=claim_vec_cache,
        )

        policy = AdaptivePolicy(
            tau_energy=sweep_results["tau_energy"],
            tau_pr=sweep_results["tau_pr"],
            tau_sensitivity=sweep_results["tau_sensitivity"],
            hard_negative_gap=sweep_results["hard_negative_gap"],
        )

        # -------------------------------------------------
        # 4. Evaluate positives
        # -------------------------------------------------

        pos_results = []

        for sample in tqdm(eval_samples, desc="POS"):
            try:
                result = gate.evaluate(
                    sample["claim"],
                    sample["evidence"],
                    policy,
                    run_id=run_id,
                )
                pos_results.append(result)
            except Exception as e:
                logger.warning(f"Skipping POS: {e}")

        # -------------------------------------------------
        # 5. Generate negatives
        # -------------------------------------------------

        adv_gen = get_adversarial_generator(
            neg_mode,
            neg_offset=neg_offset,
        )

        neg_pairs, neg_meta = adv_gen.generate(
            pairs=eval_samples,
            seed=seed,
            embedder=embedder,
            energy_computer=energy_computer,
        )

        neg_results = []

        for pair in tqdm(neg_pairs, desc="NEG"):
            try:
                result = gate.evaluate(
                    pair["claim"],
                    pair["evidence"],
                    policy,
                    run_id=run_id,
                )
                neg_results.append(result)
            except Exception as e:
                logger.warning(f"Skipping NEG: {e}")

        # -------------------------------------------------
        # 6. Reporting
        # -------------------------------------------------

        self._write_outputs(
            run_id,
            model,
            far,
            neg_mode,
            seed,
            cal_frac,
            n,
            load_stats,
            sweep_results,
            pos_results,
            neg_results,
            neg_meta,
            out_report,
            out_pos_scored,
            out_neg_scored,
        )

        if plot_png:
            plot_distributions(
                pos_energies=[r.energy_result.energy for r in pos_results],
                neg_energies=[r.energy_result.energy for r in neg_results],
                title=f"Certum | {neg_mode} | FAR={far}",
                out_path=plot_png,
                tau=sweep_results["tau_energy"],
            )

        logger.info(f"Run completed in {time() - start_time:.2f}s")

    # -----------------------------------------------------
    # Utilities
    # -----------------------------------------------------

    def _ensure_vectors(self, samples, embedder):
        for s in samples:
            if "evidence_vecs" not in s or s["evidence_vecs"] is None:
                s["evidence_vecs"] = embedder.embed(s["evidence"])
            if "claim_vec" not in s:
                s["claim_vec"] = embedder.embed([s["claim"]])[0]

    def _write_outputs(
        self,
        run_id,
        model,
        far,
        neg_mode,
        seed,
        cal_frac,
        n,
        load_stats,
        sweep_results,
        pos_results,
        neg_results,
        neg_meta,
        out_report,
        out_pos_scored,
        out_neg_scored,
    ):

        AuditLogger.write_evaluation_dump(pos_results, out_pos_scored)
        AuditLogger.write_evaluation_dump(neg_results, out_neg_scored)

        pos_summary = AuditLogger.generate_summary_report(pos_results)
        neg_summary = AuditLogger.generate_summary_report(neg_results)

        report = {
            "run_id": run_id,
            "model": model,
            "far": far,
            "neg_mode": neg_mode,
            "seed": seed,
            "cal_frac": cal_frac,
            "n_total": n,
            "load_stats": load_stats,
            "calibration": sweep_results,
            "positive": pos_summary,
            "negative": neg_summary,
            "neg_meta": neg_meta,
        }

        out_report.parent.mkdir(parents=True, exist_ok=True)
        with open(out_report, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--kind", required=True)
    ap.add_argument("--in_path", type=Path, required=True)
    ap.add_argument("--cache_db", type=Path, required=True)
    ap.add_argument("--embedding_db", type=Path, required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--regime", required=True)
    ap.add_argument("--far", type=float, required=True)
    ap.add_argument("--cal_frac", type=float, required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--neg_mode", required=True)
    ap.add_argument("--neg_offset", type=int, default=37)
    ap.add_argument("--out_report", type=Path, required=True)
    ap.add_argument("--out_pos_scored", type=Path, required=True)
    ap.add_argument("--out_neg_scored", type=Path, required=True)
    ap.add_argument("--plot_png", type=Path, default=None)

    args = ap.parse_args()

    runner = CertumRunner()
    runner.run(**vars(args))

if __name__ == "__main__":
    main()
