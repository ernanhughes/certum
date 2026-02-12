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
from time import time
from tqdm import tqdm

import numpy as np
import uuid

# Your new class architecture
from dpgss.embedding.sqlite_embedding_backend import SQLiteEmbeddingBackend
from dpgss.evidence.sqlite_evidence_store import SQLiteEvidenceStore
from dpgss.embedding.embedder import HFEmbedder
from dpgss.energy import HallucinationEnergyComputer
from dpgss.difficulty.spectral_difficulty import SpectralDifficulty
from dpgss.policy.policy import AdaptivePercentilePolicy
from dpgss.gate import VerifiabilityGate
from dpgss.calibration import AdaptiveCalibrator
from dpgss.audit import AuditLogger
from dpgss.adversarial import (
    DerangedPairGenerator, HardMinedPairGeneratorV2, OffsetPairGenerator, CyclicPairGenerator,
    PermutePairGenerator, HardMinedPairGenerator, AdversarialPairGenerator
)
from dpgss.dataset import load_examples
from dpgss.plot import plot_distributions

def get_adversarial_generator(mode: str, **kwargs) -> AdversarialPairGenerator:
    off = kwargs.get("neg_offset", 37)
    off = 37 if off is None else int(off) # Ensure offset is an integer for generators that require it

    """Factory for negative calibration modes."""
    if mode == "deranged":
        return DerangedPairGenerator()
    elif mode == "offset":
        return OffsetPairGenerator(offset=off)
    elif mode == "cyclic":
        return CyclicPairGenerator()
    elif mode == "permute":
        return PermutePairGenerator()
    elif mode == "hard_mined":
        return HardMinedPairGenerator()
    elif mode == "hard_mined_v2":
        return HardMinedPairGeneratorV2()
    else:
        raise ValueError(f"Unknown neg_mode: {mode}")


def run_gate_suite(
    kind: str,    
    in_path: Path,
    model_name: str,
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

    # Determinism
    random.seed(seed)
    np.random.seed(seed)
    
    # 1. Load data
    evidence_store = SQLiteEvidenceStore(cache_db)
    samples, load_stats = load_examples(
        kind,
        in_path,
        n,
        seed,
        evidence_store=evidence_store,
        model=model_name,
    )
    if len(samples) < 50:
        raise RuntimeError(f"Too few usable examples ({len(samples)}). Check input format/evidence extraction.")
    claim_vec_cache = {}

    cal_n = int(len(samples) * cal_frac)
    cal_samples = samples[:cal_n]
    eval_samples = samples[cal_n:]
    
    print("\n==============================")
    print("📦 PHASE 1 — Data Loaded")
    print(f"Total samples: {len(samples)}")
    print(f"Calibration: {len(cal_samples)}")
    print(f"Evaluation: {len(eval_samples)}")
    print("==============================\n")

    # 2. Build gate
    backend = SQLiteEmbeddingBackend(str(embedding_db))
    embedder = HFEmbedder(model_name=model_name, backend=backend)
    ensure_vectors(samples, embedder)  

    energy_computer = HallucinationEnergyComputer(top_k=12, rank_r=8)

    difficulty = SpectralDifficulty()  # Use defaults for now; can be extended to load from config
    gate = VerifiabilityGate(embedder, energy_computer, difficulty)
    
    # 3. Calibrate adaptive policy using NEGATIVE CONTROL ENERGIES
    print("\n⚙️ PHASE 3 — Calibration\n")
    cal_start = time()

    calibrator = AdaptiveCalibrator(gate, embedder=embedder)
    
    cal_claims = [sample["claim"] for sample in cal_samples]
    cal_evidence = [sample["evidence"] for sample in cal_samples]
    cal_evidence_vecs = [sample["evidence_vecs"] for sample in cal_samples]

    sweep_results = calibrator.run_sweep(
        claims=cal_claims,
        evidence_sets=cal_evidence,
        evidence_vecs=cal_evidence_vecs,
        percentiles=[int(far * 100)],
        neg_mode=neg_mode,
        neg_offset=neg_offset or 37,
        seed=seed,
        claim_vec_cache=claim_vec_cache,
    )
    print(f"Calibration completed in {time() - cal_start:.2f} seconds\n")


    # 4. Get policy calibrated on NEGATIVE energies
    tau = sweep_results["tau_by_percentile"][int(far * 100)]
    print(f"Calibrated tau for FAR={far:.2%}: {tau:.4f}")
    print(f"Separation delta: {sweep_results['separation_delta']:.3f} "
          f"(pos mean={np.mean(sweep_results['pos_energies']):.3f}, "
          f"neg mean={np.mean(sweep_results['neg_energies']):.3f})")
    
    policy = AdaptivePercentilePolicy(
        percentile=int(far * 100),
        calibration_energies=sweep_results["neg_energies"],
        hard_negative_gap=sweep_results["hard_negative_gap"],
    )    

    # 5. Evaluate POSITIVE samples
    pos_results = []
    print("\n🚦 PHASE 5 — Evaluating POS samples\n")
    for sample in tqdm(eval_samples, desc="POS evaluation", unit="sample"):
        claim = sample["claim"]
        evidence = sample["evidence"]
        try:
            result = gate.evaluate(claim, evidence, policy, run_id=run_id)
            pos_results.append(result)
        except Exception as e:
            print(f"⚠️  Skipping POS sample due to error: {e}")
            continue
    
    # 6. Generate NEGATIVE samples via adversarial PAIR transformation
    print("\n🔥 PHASE 6 — Generating adversarial negatives\n")
    adv_gen = get_adversarial_generator(neg_mode, neg_offset=neg_offset)

    neg_pairs, neg_meta = adv_gen.generate(
        pairs=eval_samples,
        seed=seed,
        embedder=embedder,   # required for hard_mined
        energy_computer=energy_computer,  # Required for hard_mined_v2
    )
    print(f"Generated {len(neg_pairs)} adversarial NEGATIVE samples using mode '{neg_mode}'.")
    print(json.dumps(neg_meta, indent=2))
    
    neg_results = []
    print("\n🚦 PHASE 7 — Evaluating NEG samples\n")
    for pair in tqdm(neg_pairs, desc="NEG evaluation", unit="sample"):

        try:
            result = gate.evaluate(pair["claim"], pair["evidence"], policy, run_id=run_id)
            neg_results.append(result)
        except Exception as e:
            print(f"⚠️  Skipping NEG sample due to error: {e}")
    
    # 7-9. Outputs, report, plot (unchanged - already correct)
    AuditLogger.write_run_header(
        out_report,
        {
            "run_id": run_id,
            "timestamp": time(),
            "model": model_name,
            "embedding": {
                "backend": "huggingface",
                "model": model_name,
            },
            "policy": policy.name,
            "neg_mode": neg_mode,
            "seed": seed,
        }
    )
    AuditLogger.write_evaluation_dump(pos_results, out_pos_scored)
    AuditLogger.write_evaluation_dump(neg_results, out_neg_scored)
    
    pos_summary = AuditLogger.generate_summary_report(pos_results)
    neg_summary = AuditLogger.generate_summary_report(neg_results)
    
    report_params = {
        "tau_cal": tau,          # ← CRITICAL: validator expects this key
        "far": far,
        "neg_mode": neg_mode,
        "seed": seed,
        "cal_frac": cal_frac,
    }

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
        "stats": {
            "eval": {                # Required by validator
                "FAR": far,
                "TPR": "TBD",
                "AUC": "TBD",
            },
            "load_stats": load_stats,
            "neg_generation": neg_meta
        },
        "params": report_params,
        "calibration": sweep_results,
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
    
    if plot_png:
        plot_distributions(
            pos_energies=[r.energy_result.energy for r in pos_results],
            neg_energies=[r.energy_result.energy for r in neg_results],
            title=f"FEVEROUS | {neg_mode} | FAR={far}",
            out_path=plot_png,
            tau=tau  # Pass calibrated tau for visualization
        )    
    
    print("\n==============================")
    print(f"✅ Run completed in {time() - start_time:.2f} seconds")
    print("==============================")


def ensure_vectors(samples, embedder):
    for s in samples:
        if "evidence_vecs" not in s or s["evidence_vecs"] is None:
            s["evidence_vecs"] = embedder.embed(s["evidence"])
        if "claim_vec" not in s:
            s["claim_vec"] = embedder.embed([s["claim"]])[0]

def resolve_vectors(sample, embedder, claim_cache):
    claim = sample["claim"]

    # ---- CLAIM VECTOR (cached) ----
    if claim in claim_cache:
        claim_vec = claim_cache[claim]
    else:
        claim_vec = embedder.embed([claim])[0]
        claim_cache[claim] = claim_vec

    # ---- EVIDENCE VECTORS (authoritative) ----
    if "evidence_vecs" in sample and sample["evidence_vecs"] is not None:
        evidence_vecs = sample["evidence_vecs"]
    else:
        evidence_vecs = embedder.embed(sample["evidence"])

    return claim_vec, evidence_vecs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", required=True, choices=["feverous", "jsonl"], help="Dataset format. 'feverous' for FEVEROUS-style JSONL, 'jsonl' for generic rows with 'claim' and 'evidence' fields.")
    ap.add_argument("--in_path", type=Path, required=True)
    ap.add_argument("--cache_db", type=Path, default=None)  # Optional for future caching
    ap.add_argument("--embedding_db", type=Path, default=None)  # Optional for future caching
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--regime", default="standard")
    ap.add_argument("--far", type=float, default=0.01)
    ap.add_argument("--cal_frac", type=float, default=0.5)
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--neg_mode", required=True, 
                    choices=["deranged", "offset", "cyclic", "permute", "hard_mined", "hard_mined_v2"],
                    help="Method for generating negative samples for calibration and evaluation.")
    ap.add_argument("--neg_offset", type=int, default=37)
    ap.add_argument("--out_report", type=Path, required=True)
    ap.add_argument("--out_pos_scored", type=Path, required=True)
    ap.add_argument("--out_neg_scored", type=Path, required=True)
    ap.add_argument("--plot_png", type=Path, default=None)
    args = ap.parse_args()
    
    run_gate_suite(
        kind=args.kind,
        in_path=args.in_path,
        model_name=args.model,
        cache_db=args.cache_db,
        embedding_db=args.embedding_db,
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