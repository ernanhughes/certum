#!/usr/bin/env python
"""
Clean policy instrumentation harness.

This script:
- Runs the real gate
- Uses the real policy
- Collects decision traces
- Measures axis separation
- Measures correlations
- Reports policy performance

No decision logic lives here.
"""

import json
from pathlib import Path
from time import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from certum.gate import VerifiabilityGate
from certum.policy.policy import AdaptivePolicy
from certum.geometry.claim_evidence import ClaimEvidenceGeometry
from certum.embedding.hf_embedder import HFEmbedder
from certum.embedding.sqlite_embedding_backend import SQLiteEmbeddingBackend
from certum.evidence.sqlite_evidence_store import SQLiteEvidenceStore
from certum.dataset.loader import load_examples
from certum.adversarial import get_adversarial_generator


# =====================================================
# Utilities
# =====================================================

def collect_axes(results):
    return {
        "energy": [r.decision_trace.energy for r in results],
        "participation_ratio": [
            r.decision_trace.participation_ratio for r in results
        ],
        "sensitivity": [
            r.decision_trace.sensitivity for r in results
        ],
    }


def axis_summary(values):
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
    }


def axis_delta(pos_vals, neg_vals):
    return float(np.mean(neg_vals) - np.mean(pos_vals))


def correlation_block(axes):
    e = axes["energy"]
    pr = axes["participation_ratio"]
    s = axes["sensitivity"]

    def safe_corr(a, b):
        if np.std(a) == 0 or np.std(b) == 0:
            return 0.0
        return float(np.corrcoef(a, b)[0, 1])

    return {
        "energy_pr": safe_corr(e, pr),
        "energy_sens": safe_corr(e, s),
        "pr_sens": safe_corr(pr, s),
    }


def compute_policy_metrics(pos_results, neg_results):
    pos_accept = sum(r.verdict.value == "accept" for r in pos_results)
    neg_accept = sum(r.verdict.value == "accept" for r in neg_results)

    pos_review = sum(r.verdict.value == "review" for r in pos_results)
    neg_review = sum(r.verdict.value == "review" for r in neg_results)

    return {
        "TPR": pos_accept / len(pos_results),
        "FAR": neg_accept / len(neg_results),
        "review_pos": pos_review / len(pos_results),
        "review_neg": neg_review / len(neg_results),
    }


# =====================================================
# Plotting
# =====================================================

def plot_energy_distribution(pos_axes, neg_axes, tau, out_path):
    plt.figure()
    plt.hist(pos_axes["energy"], bins=40, alpha=0.5, label="POS")
    plt.hist(neg_axes["energy"], bins=40, alpha=0.5, label="NEG")
    plt.axvline(tau)
    plt.legend()
    plt.title("Energy Distribution")
    plt.xlabel("Energy")
    plt.ylabel("Count")
    plt.savefig(out_path)
    plt.close()


def plot_energy_vs_sensitivity(pos_axes, neg_axes, out_path):
    plt.figure()
    plt.scatter(pos_axes["energy"], pos_axes["sensitivity"], s=5)
    plt.scatter(neg_axes["energy"], neg_axes["sensitivity"], s=5)
    plt.title("Energy vs Sensitivity")
    plt.xlabel("Energy")
    plt.ylabel("Sensitivity")
    plt.savefig(out_path)
    plt.close()


# =====================================================
# Main
# =====================================================

def main():
    run_dirs = sorted(Path("artifacts/runs").glob("*"))
    if not run_dirs:
        raise RuntimeError("No runs found in artifacts/runs/")
    run_dir = run_dirs[-1]
    print(f"📊 Using run: {run_dir}")

    ap = argparse.ArgumentParser()
    ap.add_argument("--in_path", type=Path, default=run_dir)
    ap.add_argument("--embedding_db", type=Path, default="E:/data/global_embeddings.db")
    ap.add_argument("--cache_db", type=Path, default="E:/data/global_embeddings.db")
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--far", type=float, default=0.01)
    ap.add_argument("--neg_mode", required=True)
    ap.add_argument("--out_report", type=Path, default=run_dir)
    ap.add_argument("--plot_dir", type=Path, default=run_dir)
    args = ap.parse_args()

    start = time()

    # -------------------------------------------------
    # Load Data
    # -------------------------------------------------

    evidence_store = SQLiteEvidenceStore(args.cache_db)
    samples, _ = load_examples(
        "jsonl",
        args.in_path,
        args.n,
        1337,
        evidence_store=evidence_store,
        model=args.model,
    )

    cal_frac = 0.5
    cal_n = int(len(samples) * cal_frac)
    cal_samples = samples[:cal_n]
    eval_samples = samples[cal_n:]

    # -------------------------------------------------
    # Build Gate
    # -------------------------------------------------

    backend = SQLiteEmbeddingBackend(str(args.embedding_db))
    embedder = HFEmbedder(model_name=args.model, backend=backend)
    energy_computer = ClaimEvidenceGeometry(top_k=12, rank_r=8)

    gate = VerifiabilityGate(embedder, energy_computer)

    # -------------------------------------------------
    # Calibration (Energy only)
    # -------------------------------------------------

    print("Calibrating policy energy threshold...")
    neg_gen = get_adversarial_generator(args.neg_mode)
    print(f"Using negative generator: {neg_gen.__class__.__name__}")
    neg_pairs, _ = neg_gen.generate(
        pairs=cal_samples,
        seed=1337,
        embedder=embedder,
        energy_computer=energy_computer,
    )

    neg_energies = []
    for pair in neg_pairs:
        res = gate.compute_energy(pair["claim"], pair["evidence"])
        neg_energies.append(res.energy)

    print(f"Calibrated on {len(cal_samples)} samples, generated {len(neg_pairs)} neg pairs")

    tau = np.percentile(neg_energies, args.far * 100)

    policy = AdaptivePolicy(
        tau_energy=tau,
        tau_pr=0.5,
        tau_sensitivity=0.5,
    )

    # -------------------------------------------------
    # Evaluate POS
    # -------------------------------------------------

    pos_results = []
    for s in tqdm(eval_samples, desc="POS"):
        r = gate.evaluate(s["claim"], s["evidence"], policy, run_id="run")
        pos_results.append(r)

    # -------------------------------------------------
    # Evaluate NEG
    # -------------------------------------------------

    neg_pairs, _ = neg_gen.generate(
        pairs=eval_samples,
        seed=1337,
        embedder=embedder,
        energy_computer=energy_computer,
    )

    neg_results = []
    for p in tqdm(neg_pairs, desc="NEG"):
        r = gate.evaluate(p["claim"], p["evidence"], policy, run_id="run")
        neg_results.append(r)

    # -------------------------------------------------
    # Measurements
    # -------------------------------------------------

    pos_axes = collect_axes(pos_results)
    neg_axes = collect_axes(neg_results)

    policy_metrics = compute_policy_metrics(pos_results, neg_results)

    print("\n================ AXIS DISTRIBUTIONS ================")

    for axis in pos_axes:
        print(f"\n{axis.upper()}")
        print("  POS:", axis_summary(pos_axes[axis]))
        print("  NEG:", axis_summary(neg_axes[axis]))

    print("\n================ SEPARATION ========================")
    for axis in pos_axes:
        print(
            f"{axis}: Δ =",
            axis_delta(pos_axes[axis], neg_axes[axis])
        )

    print("\n================ CORRELATIONS ======================")
    print("POS:", correlation_block(pos_axes))
    print("NEG:", correlation_block(neg_axes))

    print("\n================ POLICY PERFORMANCE ================")
    for k, v in policy_metrics.items():
        print(f"{k}: {v:.4f}")

    # -------------------------------------------------
    # Save Report
    # -------------------------------------------------

    report = {
        "tau_energy": float(tau),
        "policy_metrics": policy_metrics,
        "axis_stats": {
            "pos": {a: axis_summary(pos_axes[a]) for a in pos_axes},
            "neg": {a: axis_summary(neg_axes[a]) for a in neg_axes},
        },
        "axis_separation": {
            a: axis_delta(pos_axes[a], neg_axes[a]) for a in pos_axes
        },
        "correlations": {
            "pos": correlation_block(pos_axes),
            "neg": correlation_block(neg_axes),
        },
    }

    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_report, "w") as f:
        json.dump(report, f, indent=2)

    # -------------------------------------------------
    # Plots
    # -------------------------------------------------

    args.plot_dir.mkdir(parents=True, exist_ok=True)

    plot_energy_distribution(
        pos_axes,
        neg_axes,
        tau,
        args.plot_dir / "energy_distribution.png"
    )

    plot_energy_vs_sensitivity(
        pos_axes,
        neg_axes,
        args.plot_dir / "energy_vs_sensitivity.png"
    )

    print(f"\nCompleted in {time() - start:.2f} seconds")


if __name__ == "__main__":
    main()
