# src/certum/runner.py

import argparse
import hashlib
import json
import logging
import random
import uuid
from pathlib import Path
from time import time
from typing import Optional

import numpy as np
from tqdm import tqdm

from certum.adversarial import get_adversarial_generator
from certum.calibration import AdaptiveCalibrator
from certum.dataset.loader import load_examples
from certum.evidence.sqlite_evidence_store import SQLiteEvidenceStore
from certum.orchestration.audit import AuditLogger
from certum.orchestration.factory import CertumFactory
from certum.policy.policies import build_policies
from certum.policy.policy import AdaptivePolicy
from certum.reporting.modules.plot import plot_distributions
from certum.utils.id_utils import compute_ids
from certum.utils.math_utils import accept_margin_ratio

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
        policies: str,
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
        out_pos_policies: Optional[Path] = None,
        out_neg_policies: Optional[Path] = None,        
        neg_offset: Optional[int] = None,
        out_duckdb: Optional[Path] = None,
        plot_png: Optional[Path] = None,
        gap_width: float = 0.1,  
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
        logger.info(f"Calibration complete. Sweep results: {sweep_results}")

        policy_names = [p.strip() for p in policies.split(",") if p.strip()]
        logger.info(f"Building policies: {policy_names}")

        policies = build_policies(
            policy_names,
            tau_energy=sweep_results["tau_energy"],
            tau_pr=sweep_results["tau_pr"],
            tau_sensitivity=sweep_results["tau_sensitivity"],
            gap_width=gap_width,
        )

        sweep_pos_rows: list[dict] = []
        sweep_neg_rows: list[dict] = []
        
        policy = AdaptivePolicy(
            tau_energy=sweep_results["tau_energy"],
            tau_pr=sweep_results["tau_pr"],
            tau_sensitivity=sweep_results["tau_sensitivity"],
            hard_negative_gap=sweep_results["hard_negative_gap"],
            gap_width=gap_width,
        )

        # -------------------------------------------------
        # 4. Evaluate positives
        # -------------------------------------------------

        pos_results = []

        for sample in tqdm(eval_samples, desc="POS"):
            try:
                # Policy sweep (optional)
                if out_pos_policies is not None:
                    sweep_pos_rows.extend(self._evaluate_policy_suite(
                        gate=gate,
                        sample=sample,
                        policies_list=policies,
                        run_id=run_id,
                        split="pos",
                    ))

                result = gate.evaluate(
                    sample["claim"],
                    sample["evidence"],
                    policy,
                    run_id=run_id,
                )
                pos_results.append(result)
            except Exception as e:
                logger.exception("Policy sweep failed (POS). sample=%r, error=%r", type(sample), e)

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
                # Policy sweep (optional)
                if out_neg_policies is not None:
                    sweep_neg_rows.extend(self._evaluate_policy_suite(
                        gate=gate,
                        sample=pair,
                        policies_list=policies,
                        run_id=run_id,
                        split="neg",
                    ))

                result = gate.evaluate(
                    pair["claim"],
                    pair["evidence"],
                    policy,
                    run_id=run_id,
                )
                neg_results.append(result)
            except Exception as e:
                logger.exception("Policy sweep failed (POS). sample=%r, error=%r", type(sample), e)

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

        if out_pos_policies is not None:
            self._write_policy_rows(out_pos_policies, sweep_pos_rows)
        if out_neg_policies is not None:
            self._write_policy_rows(out_neg_policies, sweep_neg_rows)

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

    def _write_policy_rows(self, path: Path, rows: list[dict]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")


    def _stable_sample_id(self, sample: dict) -> str:
        """
        Stable ID for joining across tables/files.
        Prefer dataset-provided 'id'; otherwise hash claim+evidence.
        """
        sid = sample.get("id", None)
        if sid is not None:
            return str(sid)

        claim = sample.get("claim", "") or ""
        evidence = sample.get("evidence", []) or []
        blob = claim + "\n" + "\n".join(map(str, evidence))
        return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:16]


    def _row_id(self, sample_id: str, policy_name: str, split: str) -> str:
        blob = f"{split}|{sample_id}|{policy_name}"
        return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:16]

    def _evaluate_policy_suite(
        self,
        *,
        gate,
        sample: dict,
        policies_list: list,
        run_id: str,
        split: str,
    ) -> list[dict]:
        """
        Returns JSON-serializable rows (one per policy) for policy sweep outputs.
        Does NOT change your existing main outputs.
        """
        # Compute axes once (vector-aware) — requires gate.evaluate to accept claim_vec/evidence_vecs,
        # OR you can call gate.compute_axes if you added it.
        base, axes, embedding_info = gate.compute_axes(
            sample["claim"],
            sample["evidence"],
            claim_vec=sample.get("claim_vec"),
            evidence_vecs=sample.get("evidence_vecs"),
        ) 

        pair_id, claim_id, evidence_id = compute_ids(sample["claim"], sample["evidence"])

        rows: list[dict] = []
        for policy in policies_list:
            tau = getattr(policy, "tau_accept", None)
            if tau is None:
                eff = 0.0  # or None, but your Policy.decide expects a float
            else:
                eff = accept_margin_ratio(energy=float(axes.get("energy")), tau=float(tau))

            verdict = policy.decide(axes, float(eff))
            g = base.geometry
            rows.append({
                "run_id": run_id,
                "split": split,

                "sample_id": sample.get("id"),  
                "id": pair_id,
                "pair_id": pair_id,
                "claim_id": claim_id,
                "evidence_id": evidence_id,
                "row_id": self._row_id(sample.get("id"), policy.name, split),


                "policy_name": policy.name,
                "policy_key": getattr(policy, "key", None),  # harmless if absent
                "verdict": verdict.value,
                "effectiveness": float(eff),

                # a few extra geometry fields (high-value)
                "effective_rank": int(getattr(g, "effective_rank", 0)),
                "used_count": int(getattr(g, "used_count", 0)),
                "sigma1_ratio": float(getattr(g, "sigma1_ratio", 0.0)),
                "sigma2_ratio": float(getattr(g, "sigma2_ratio", 0.0)),
                "entropy_rank": float(getattr(g, "entropy_rank", 0.0)),
                "sim_top1": float(getattr(g, "sim_top1", 0.0)),
                "sim_top2": float(getattr(g, "sim_top2", 0.0)),

                "embedding_backend": embedding_info.get("embedding_backend"),
                "claim_dim": embedding_info.get("claim_dim"),
                "evidence_count": embedding_info.get("evidence_count"),

                "tau_accept": policy.tau_accept,

                # keep full nested structure for later deep dives
                "energy": base.energy,
                "energy_result": base.to_dict(),
            })
        return rows

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
    ap.add_argument("--policies", default="adaptive", help="Comma-separated policy names to evaluate (e.g. energy_only,axis_only,monotone_adaptive).")
    ap.add_argument("--regime", required=True)
    ap.add_argument("--far", type=float, required=True)
    ap.add_argument("--cal_frac", type=float, required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--neg_mode", required=True)
    ap.add_argument("--out_pos_policies", type=Path, default=None, help="Optional: write policy-sweep POS rows (one row per sample per policy).")
    ap.add_argument("--out_neg_policies", type=Path, default=None, help="Optional: write policy-sweep NEG rows (one row per sample per policy).")
    ap.add_argument("--neg_offset", type=int, default=37)
    ap.add_argument("--out_report", type=Path, required=True)
    ap.add_argument("--out_pos_scored", type=Path, required=True)
    ap.add_argument("--out_neg_scored", type=Path, required=True)
    ap.add_argument("--out_duckdb", type=Path, default=None, help="Optional: build a DuckDB DB in the run folder.")
    ap.add_argument("--plot_png", type=Path, default=None)
    ap.add_argument("--gap_width", type=float, default=0.1, help="Width of the ambiguity band as a fraction of tau_energy (default: 0.1 for 10%)")  

    args = ap.parse_args()

    runner = CertumRunner()
    runner.run(**vars(args))

if __name__ == "__main__":
    main()
