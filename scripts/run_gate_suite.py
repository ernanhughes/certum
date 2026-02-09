#!/usr/bin/env python
"""
Gate suite runner built on dpgss class architecture.
Replaces procedural gate_suite.py with explicit class boundaries.
"""
import argparse
import json
import random
from pathlib import Path
from typing import Optional

import numpy as np

# Your new class architecture
from dpgss.embedder import HFEmbedder
from dpgss.energy import HallucinationEnergyComputer
from dpgss.oracle import OracleValidator
from dpgss.policy import AdaptivePercentilePolicy
from dpgss.gate import VerifiabilityGate
from dpgss.calibration import AdaptiveCalibrator
from dpgss.audit import AuditLogger
from dpgss.adversarial import (
    DerangedGenerator, OffsetGenerator, CyclicGenerator,
    PermuteGenerator, HardMinedGenerator, AdversarialClaimGenerator
)
from dpgss.dataset import load_feverous_samples
from dpgss.plot import plot_distributions

def get_adversarial_generator(mode: str, **kwargs) -> AdversarialClaimGenerator:
    """Factory for negative calibration modes."""
    if mode == "deranged":
        return DerangedGenerator()
    elif mode == "offset":
        return OffsetGenerator(offset=int(kwargs.get("neg_offset", 37)))
    elif mode == "cyclic":
        return CyclicGenerator()
    elif mode == "permute":
        return PermuteGenerator()
    elif mode == "hard_mined":
        return HardMinedGenerator()
    else:
        raise ValueError(f"Unknown neg_mode: {mode}")


def run_gate_suite(
    in_path: Path,
    model_name: str,
    cache_db: Path,
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
    # Determinism
    random.seed(seed)
    np.random.seed(seed)
    
    # 1. Load data
    samples = load_feverous_samples(in_path, n=n, seed=seed, model=model_name, cache_db=cache_db)  # Returns [(claim, evidence_list), ...]
    cal_n = int(len(samples) * cal_frac)
    eval_n = len(samples) - cal_n
    
    cal_samples = samples[:cal_n]
    eval_samples = samples[cal_n:cal_n + eval_n]
    
    # 2. Build gate
    embedder = HFEmbedder(model_name=model_name)
    energy_computer = HallucinationEnergyComputer(top_k=12, rank_r=8)
    oracle_validator = OracleValidator(max_allowed_energy=0.01)  # CRITICAL: rejects trivial oracles
    gate = VerifiabilityGate(embedder, energy_computer, oracle_validator)
    
    # 3. Calibrate adaptive policy
    calibrator = AdaptiveCalibrator(gate)
    cal_claims = [c for c, _ in cal_samples]
    cal_evidence = [e for _, e in cal_samples]
    
    sweep_results = calibrator.run_sweep(
        claims=cal_claims,
        evidence_sets=cal_evidence,
        percentiles=[int(far * 100)],  # e.g., FAR=0.01 → P1
        oracle_claims=None  # Uses evidence[0] as oracle
    )
    
    # Validate oracle quality BEFORE proceeding (your "too good" safeguard)
    oracle_valid_rate = sweep_results["oracle_validity_rate"]
    if oracle_valid_rate < 0.95:
        print(
            f"⚠️  Oracle validity rate too low ({oracle_valid_rate:.1%})! "
            "Your evidence may be incoherent or oracle construction broken. "
            "This is GOOD — it means your gate isn't trivially accepting everything."
        )
    
    # 4. Get policy
    tau = sweep_results["tau_by_percentile"][int(far * 100)]
    policy = AdaptivePercentilePolicy(percentile=int(far * 100), calibration_energies=sweep_results["energy_gaps"])
    
    # 5. Evaluate POSITIVE samples
    pos_results = []
    for claim, evidence in eval_samples:
        try:
            result = gate.evaluate(claim, evidence, policy)
            pos_results.append(result)
        except Exception as e:
            print(f"⚠️  Skipping POS sample due to error: {e}")
            continue
    
    # 6. Generate NEGATIVE samples via adversarial transformation
    adv_gen = get_adversarial_generator(neg_mode, neg_offset=neg_offset)
    neg_results = []
    for claim, evidence in eval_samples:
        adv_claim = adv_gen.transform(claim, evidence, seed=seed)
        try:
            result = gate.evaluate(adv_claim, evidence, policy)
            neg_results.append(result)
        except Exception as e:
            print(f"⚠️  Skipping NEG sample due to error: {e}")
            continue
    
    # 7. Write outputs
    AuditLogger.write_evaluation_dump(pos_results, out_pos_scored)
    AuditLogger.write_evaluation_dump(neg_results, out_neg_scored)
    
    # 8. Generate report
    pos_summary = AuditLogger.generate_summary_report(pos_results)
    neg_summary = AuditLogger.generate_summary_report(neg_results)
    
    report = {
        "config": {
            "model": model_name,
            "regime": regime,
            "far": far,
            "cal_frac": cal_frac,
            "n_total": n,
            "neg_mode": neg_mode,
            "seed": seed
        },
        "calibration": sweep_results,
        "oracle_validity_rate": oracle_valid_rate,  # CRITICAL METRIC FOR YOUR CONCERN
        "positive_samples": pos_summary,
        "negative_samples": neg_summary,
        "separation": {
            "pos_energy_mean": pos_summary["energy_stats"]["mean"],
            "neg_energy_mean": neg_summary["energy_stats"]["mean"],
            "delta": pos_summary["energy_stats"]["mean"] - neg_summary["energy_stats"]["mean"]
        }
    }
    
    out_report.parent.mkdir(parents=True, exist_ok=True)
    with open(out_report, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    
    # 9. Plot (optional - reuse your existing viz code)
    if plot_png:
        plot_distributions(
            pos_energies=[r.energy_result.energy for r in pos_results],
            neg_energies=[r.energy_result.energy for r in neg_results],
            title=f"FEVEROUS | {neg_mode} | FAR={far}",
            out_path=plot_png
        )
    
    # FINAL SAFETY CHECK: Print oracle validity to console
    print(f"\n✅ Oracle validity rate: {oracle_valid_rate:.1%}")
    if oracle_valid_rate > 0.99:
        print("⚠️  WARNING: Oracle validity near 100% — your gate may be trivially accepting. Check energy distributions!")
    else:
        print("✅ Gate is non-trivial (oracle energy > 0.01 in some samples)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", required=True, choices=["feverous"])
    ap.add_argument("--in_path", type=Path, required=True)
    ap.add_argument("--cache_db", type=Path, default=None)  # Optional for future caching
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--regime", default="standard")
    ap.add_argument("--far", type=float, default=0.01)
    ap.add_argument("--cal_frac", type=float, default=0.5)
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--neg_mode", required=True, 
                    choices=["deranged", "offset", "cyclic", "permute", "hard_mined"])
    ap.add_argument("--neg_offset", type=int, default=37)
    ap.add_argument("--out_report", type=Path, required=True)
    ap.add_argument("--out_pos_scored", type=Path, required=True)
    ap.add_argument("--out_neg_scored", type=Path, required=True)
    ap.add_argument("--plot_png", type=Path, default=None)
    args = ap.parse_args()
    
    run_gate_suite(
        in_path=args.in_path,
        model_name=args.model,
        cache_db=args.cache_db,
        regime=args.regime,
        far=args.far,
        cal_frac=args.cal_frac,
        n=args.n,
        seed=args.seed,
        neg_mode=args.neg_mode,
        neg_offset=args.neg_offset,
        out_report=args.out_report,
        out_pos_scored=args.out_pos_scored,
        out_neg_scored=args.out_neg_scored,
        plot_png=args.plot_png,
    )


if __name__ == "__main__":
    main()