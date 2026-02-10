#!/usr/bin/env python
"""Scatter plot: Energy (Y) vs Effective Rank (X) — works with FLAT JSONL format."""
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_energies(jsonl_path: Path):
    """Load ONLY energy values (flat format)."""
    energies = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            r = json.loads(line)
            energies.append(r['energy'])
    return np.array(energies)

# Load latest run
run_dirs = sorted(Path("artifacts/runs").glob("*"))
if not run_dirs:
    raise RuntimeError("No runs found in artifacts/runs/")
run_dir = run_dirs[-1]
print(f"📊 Loading data from: {run_dir}")

# Load energies (flat format)
pos_e = load_energies(run_dir / "pos_hard_mined.jsonl")
neg_e = load_energies(run_dir / "neg_hard_mined.jsonl")

# Load report to get tau
report = json.loads((run_dir / "feverous_negcal_hard_mined.json").read_text())
tau = report["params"]["tau_cal"]

# Create plot: Energy histogram with tau line
fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

# Histograms
ax.hist(pos_e, bins=30, alpha=0.7, color='#2E7D32', label='Positive (Valid)', edgecolor='white', linewidth=0.5)
ax.hist(neg_e, bins=30, alpha=0.7, color='#C62828', label='Negative (Brittle)', edgecolor='white', linewidth=0.5)

# Tau threshold
ax.axvline(x=tau, color='#1565C0', linestyle='--', linewidth=2.5, label=f'τ = {tau:.3f} (FAR=1%)')

# Ambiguity band
ax.axvspan(0.45, 0.60, alpha=0.15, color='gray', label='Ambiguity Band (0.45–0.60)')

# Styling
ax.set_xlabel('Hallucination Energy', fontsize=13, fontweight='bold')
ax.set_ylabel('Count', fontsize=13, fontweight='bold')
ax.set_title('Hard-Mined Separation: Energy Distribution', fontsize=14, fontweight='bold', pad=15)
ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
ax.set_axisbelow(True)
ax.legend(loc='upper right', framealpha=0.95, fontsize=11)
ax.set_xlim(-0.05, 1.05)

plt.tight_layout()
out_path = run_dir / "energy_histogram.png"
plt.savefig(out_path, dpi=180, bbox_inches='tight')
plt.close()

print(f"✅ Saved energy histogram: {out_path}")
print("\n📊 Key metrics:")
print(f"   τ (threshold): {tau:.4f}")
print(f"   Positive mean energy: {np.mean(pos_e):.3f} ± {np.std(pos_e):.3f}")
print(f"   Negative mean energy: {np.mean(neg_e):.3f} ± {np.std(neg_e):.3f}")
print(f"   Separation delta: {np.mean(neg_e) - np.mean(pos_e):.3f}")
