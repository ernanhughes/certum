import matplotlib

matplotlib.use('Agg')  # Non-interactive backend for script use
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np


def plot_distributions(
    pos_energies: List[float],
    neg_energies: List[float],
    title: str,
    out_path: Path,
    tau: Optional[float] = None,
    figsize: tuple = (8, 5),
    dpi: int = 180,
):
    """
    Plot histogram comparing positive (supported) vs negative (adversarial) energy distributions.
    
    Args:
        pos_energies: Energy values for claims with real evidence
        neg_energies: Energy values for claims with mismatched/adversarial evidence
        title: Plot title (e.g., "FEVEROUS | hard_mined | FAR=0.01")
        out_path: Output PNG path
        tau: Optional calibrated threshold (vertical line)
        figsize: Figure dimensions in inches
        dpi: Output resolution
    """
    # Convert to numpy arrays and filter invalid values
    pos = np.asarray(pos_energies, dtype=np.float32)
    neg = np.asarray(neg_energies, dtype=np.float32)
    
    pos = pos[np.isfinite(pos)]
    neg = neg[np.isfinite(neg)]
    
    if len(pos) == 0 and len(neg) == 0:
        raise ValueError("No valid energy values to plot")
    
    # Determine binning strategy
    bins = np.linspace(0.0, 1.0, 41)  # 40 bins across [0, 1] range
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # Plot distributions
    if len(pos) > 0:
        ax.hist(
            pos,
            bins=bins,
            alpha=0.75,
            color='#2E7D32',  # Dark green for positives (supported claims)
            edgecolor='white',
            linewidth=0.5,
            label=f'Positive (n={len(pos)})',
            density=False
        )
    
    if len(neg) > 0:
        ax.hist(
            neg,
            bins=bins,
            alpha=0.75,
            color='#C62828',  # Dark red for negatives (adversarial claims)
            edgecolor='white',
            linewidth=0.5,
            label=f'Negative (n={len(neg)})',
            density=False
        )
    
    # Add tau threshold line if provided
    if tau is not None and 0.0 <= tau <= 1.0:
        ax.axvline(
            tau,
            color='#1565C0',  # Dark blue
            linestyle='--',
            linewidth=2.5,
            label=f'τ = {tau:.3f}',
            zorder=10
        )
    
    # Styling
    ax.set_xlabel('Hallucination Energy', fontsize=12, fontweight='bold')
    ax.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.set_xlim(0.0, 1.0)
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Legend positioning (adaptive based on data overlap)
    if len(pos) > 0 and len(neg) > 0:
        # Check separation quality
        pos_mean = np.mean(pos) if len(pos) > 0 else 0.0
        neg_mean = np.mean(neg) if len(neg) > 0 else 1.0
        if abs(pos_mean - neg_mean) < 0.15:
            legend_loc = 'upper center'
        else:
            legend_loc = 'upper right'
        ax.legend(loc=legend_loc, framealpha=0.95, fontsize=10)
    
    # Spine styling
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_linewidth(1.2)
    
    # Tight layout and save
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    
    print(f"✅ Saved distribution plot: {out_path}")


# Optional: Standalone test/example
if __name__ == "__main__":
    # Example usage
    rng = np.random.default_rng(42)
    pos_sample = rng.beta(2, 5, size=500) * 0.6  # Low energies (supported claims)
    neg_sample = 0.4 + rng.beta(3, 2, size=500) * 0.6  # Higher energies (adversarial)
    
    plot_distributions(
        pos_energies=pos_sample.tolist(),
        neg_energies=neg_sample.tolist(),
        title="FEVEROUS | hard_mined | FAR=0.01",
        tau=0.45,
        out_path=Path("artifacts/test_distribution.png")
    )