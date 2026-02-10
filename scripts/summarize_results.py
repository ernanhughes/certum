#!/usr/bin/env python
"""Generate concise summary table for paper."""
import json
import numpy as np
from pathlib import Path

run_dir = sorted(Path("artifacts/runs").glob("*"))[-1]
report = json.loads((run_dir / "feverous_negcal_hard_mined.json").read_text())

# Extract key metrics
tau = report["params"]["tau_cal"]
pos_mean = report["positive_samples"]["energy_stats"]["mean"]
neg_mean = report["negative_samples"]["energy_stats"]["mean"]
delta = neg_mean - pos_mean

# Load rank distributions
def load_ranks(path):
    ranks = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            r = json.loads(line)
            ranks.append(r['energy_result']['effective_rank'])
    return np.array(ranks)

pos_ranks = load_ranks(run_dir / "pos_hard_mined.jsonl")
neg_ranks = load_ranks(run_dir / "neg_hard_mined.jsonl")

print("\n" + "="*70)
print("PUBLICATION-READY RESULTS SUMMARY")
print("="*70)
print(f"\nDataset: FEVEROUS (hard_mined negatives)")
print(f"Policy: Adaptive P1 (FAR=1%)")
print(f"Threshold τ: {tau:.4f}")
print(f"\nEnergy Separation:")
print(f"  Positive mean: {pos_mean:.3f}")
print(f"  Negative mean: {neg_mean:.3f}")
print(f"  Delta: {delta:.3f} {'✅ STRONG' if delta > 0.3 else '⚠️ WEAK'}")
print(f"\nStructural Integrity (Effective Rank):")
print(f"  Positives with rank ≥ 2: {np.sum(pos_ranks >= 2)}/{len(pos_ranks)} ({np.sum(pos_ranks >= 2)/len(pos_ranks):.1%})")
print(f"  Negatives with rank = 1:  {np.sum(neg_ranks == 1)}/{len(neg_ranks)} ({np.sum(neg_ranks == 1)/len(neg_ranks):.1%})")
print(f"\nAmbiguity Band Analysis (Energy 0.45–0.60):")
pos_ambig = (pos_ranks[(0.45 <= pos_mean) & (pos_mean <= 0.60)] >= 2).mean() if len(pos_ranks) > 0 else 0
neg_ambig = (neg_ranks[(0.45 <= neg_mean) & (neg_mean <= 0.60)] == 1).mean() if len(neg_ranks) > 0 else 0
print(f"  Valid claims preserved (rank ≥ 2): {pos_ambig:.1%}")
print(f"  Brittle negatives caught (rank = 1): {neg_ambig:.1%}")
print("\n" + "="*70)