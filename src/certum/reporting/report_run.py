#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
from certum.reporting.modules.auc import auc_from_curve
from certum.reporting.modules.boundary import dominates
from certum.reporting.modules.correlation_matrix import (correlation_eigenvalues,
                                                  correlation_matrix,
                                                  detect_axis_collapse)
from certum.reporting.modules.formatter import build_markdown
from certum.reporting.modules.geometry_stats import (alignment_vs_correctness,
                                              effectiveness_mean,
                                              energy_vs_participation,
                                              hard_negative_gap_distribution, hard_negative_gap_per_row,
                                              stability_fraction)
from certum.reporting.modules.loader import discover_modes, load_json, load_jsonl
from certum.reporting.modules.metrics import (compute_auc, compute_tpr_at_far,
                                       extract_energies,
                                       summarize_distribution,
                                       summarize_geometry, summarize_verdicts)
from certum.reporting.modules.policy_comparison import sweep_policy_curve
from certum.reporting.modules.scatter_plots import scatter_plot
from certum.reporting.validate_gate_artifacts import validate_run_directory
from certum.reporting.modules.gap_analysis import gap_conditioned_analysis
from certum.reporting.modules.gap_analysis import conditional_axis_auc

from certum.policy.energy_only import EnergyOnlyPolicy
from certum.policy.policy import AdaptivePolicy
from certum.reporting.modules.gap_axis_shift import axis_shift_analysis
from certum.reporting.modules.gap_sweep import sweep_gap_width


def main():
    run_dirs = sorted(Path("artifacts/runs").glob("*"))
    if not run_dirs:
        raise RuntimeError("No runs found.")

    default_run = run_dirs[-1]

    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=default_run)
    args = parser.parse_args()

    run_dir = args.run

    # ---------------------------------------------------
    # 1. Validate artifacts
    # ---------------------------------------------------
    validate_run_directory(run_dir)

    # ---------------------------------------------------
    # 2. Discover modes
    # ---------------------------------------------------
    modes = discover_modes(run_dir)
    if not modes:
        raise RuntimeError("No valid modes discovered.")

    summary = {
        "run_dir": str(run_dir),
        "modes": {}
    }

    plot_dir = run_dir / "report_plots"
    plot_dir.mkdir(exist_ok=True)

    # ---------------------------------------------------
    # 3. Compute metrics per mode
    # ---------------------------------------------------
    for mode, paths in modes.items():

        print(f"\nProcessing mode: {mode}")

        pos_rows = load_jsonl(paths["pos"])
        neg_rows = load_jsonl(paths["neg"])
        cal = load_json(paths["calibration"])

        tau = cal["calibration"]["tau_energy"]

        from certum.reporting.modules.policy_comparison import compare_energy_vs_policy

        # --- Basic metrics ---
        auc = compute_auc(pos_rows, neg_rows)
        tpr_far = compute_tpr_at_far(pos_rows, neg_rows, tau)

        pos_energy = extract_energies(pos_rows)
        neg_energy = extract_energies(neg_rows)

        mode_summary = {
            "auc": auc,
            "tau_energy": tau,
            "tpr_at_tau": tpr_far["tpr"],
            "far_at_tau": tpr_far["far"],
            "n_pos": len(pos_rows),
            "n_neg": len(neg_rows),

            "pos_energy_distribution": summarize_distribution(pos_energy),
            "neg_energy_distribution": summarize_distribution(neg_energy),

            "pos_verdicts": summarize_verdicts(pos_rows),
            "neg_verdicts": summarize_verdicts(neg_rows),

            "pos_geometry_means": summarize_geometry(pos_rows),
            "neg_geometry_means": summarize_geometry(neg_rows),
        }

        # --- Advanced diagnostics ---
        mode_summary.update({
            "energy_vs_participation_corr_pos": energy_vs_participation(pos_rows),
            "energy_vs_participation_corr_neg": energy_vs_participation(neg_rows),

            "alignment_vs_correctness_corr_pos": alignment_vs_correctness(pos_rows),

            "hard_negative_gap_pos": hard_negative_gap_distribution(pos_rows),
            "hard_negative_gap_neg": hard_negative_gap_distribution(neg_rows),

            "stability_fraction_pos": stability_fraction(pos_rows),
            "stability_fraction_neg": stability_fraction(neg_rows),

            "effectiveness_mean_pos": effectiveness_mean(pos_rows),
            "effectiveness_mean_neg": effectiveness_mean(neg_rows),
        })

        # --- Correlation matrix (POS + NEG) ---
        corr_pos, dropped_pos = correlation_matrix(pos_rows)
        eigvals_pos = correlation_eigenvalues(corr_pos)

        corr_neg, dropped_neg = correlation_matrix(neg_rows)
        eigvals_neg = correlation_eigenvalues(corr_neg)


        mode_summary["correlation_matrix_pos"] = corr_pos
        mode_summary["correlation_matrix_neg"] = corr_neg

        mode_summary["correlation_eigenvalues_pos"] = eigvals_pos
        mode_summary["correlation_eigenvalues_neg"] = eigvals_neg
    
        mode_summary["dropped_axes_pos"] = dropped_pos
        mode_summary["dropped_axes_neg"] = dropped_neg

        mode_summary["axis_collapse_pos"] = detect_axis_collapse(corr_pos)
        mode_summary["axis_collapse_neg"] = detect_axis_collapse(corr_neg)

        summary["modes"][mode] = mode_summary


        comparison = compare_energy_vs_policy(pos_rows, neg_rows, tau)
        mode_summary["policy_comparison"] = comparison


        taus = np.linspace(0.0, 1.0, 100)

        energy_curve = sweep_policy_curve(
            pos_rows,
            neg_rows,
            lambda tau: EnergyOnlyPolicy(tau_energy=tau),
            taus
        )
        current_pr_threshold = cal["calibration"].get("tau_pr")
        current_sens_threshold = cal["calibration"].get("tau_sensitivity")

        adaptive_curve = sweep_policy_curve(
            pos_rows,
            neg_rows,
            lambda tau: AdaptivePolicy(
                tau_energy=tau,
                tau_pr=current_pr_threshold,
                tau_sensitivity=current_sens_threshold
            ),
            taus
        )

        mode_summary["roc_energy"] = energy_curve
        mode_summary["roc_adaptive"] = adaptive_curve
    
        energy_auc = auc_from_curve(energy_curve)
        adaptive_auc = auc_from_curve(adaptive_curve)

        mode_summary["auc_energy_only"] = energy_auc
        mode_summary["auc_adaptive"] = adaptive_auc
        mode_summary["auc_delta"] = adaptive_auc - energy_auc


        mode_summary["adaptive_dominates_energy"] = dominates(
            adaptive_curve,
            energy_curve
        )



        gap_width = 0.1  # or sweep this
        gap_results = gap_conditioned_analysis(
            pos_rows,
            neg_rows,
            tau,
            current_pr_threshold,
            current_sens_threshold,
            gap_width
        )   

        mode_summary["gap_conditioned_analysis"] = gap_results


        mode_summary["axis_shift"] = axis_shift_analysis(pos_rows, neg_rows)



        gap_sweep_results = sweep_gap_width(
            pos_rows,
            neg_rows,
            tau,
            current_pr_threshold,
            current_sens_threshold
        )

        mode_summary["gap_sweep"] = gap_sweep_results


        mode_summary["gap_axis_signal"] = conditional_axis_auc(
            pos_rows,
            neg_rows,
            tau,
            gap_width
        )

        # ---------------------------------------------------
        # 4. Scatter plots (per mode)
        # ---------------------------------------------------

        scatter_plot(
            pos_rows,
            ["energy", "value"],
            ["energy", "geometry", "spectral", "participation_ratio"],
            plot_dir / f"{mode}_energy_vs_pr.png",
            f"{mode} POS Energy vs Participation"
        )

        scatter_plot(
            pos_rows,
            ["energy", "value"],
            ["energy", "geometry", "alignment", "alignment_to_sigma1"],
            plot_dir / f"{mode}_energy_vs_alignment.png",
            f"{mode} POS Energy vs Alignment"
        )

        scatter_plot(
            pos_rows,
            ["energy", "geometry", "alignment", "alignment_to_sigma1"],
            ["effectiveness"],
            plot_dir / f"{mode}_alignment_vs_effectiveness.png",
            f"{mode} POS Alignment vs Effectiveness"
        )

        scatter_plot(
            pos_rows,
            ["energy", "geometry", "robustness", "sensitivity"],
            ["energy", "geometry", "spectral", "participation_ratio"],
            plot_dir / f"{mode}_sensitivity_vs_pr.png",
            f"{mode} POS Sensitivity vs Participation"
        )

    neg_deragend = []
    neg_hard = []
    for mode, paths in modes.items():
        if mode == "deranged":
            neg_deragend = load_jsonl(paths["neg"])
        elif mode == "hard_mined_v2":
            neg_hard = load_jsonl(paths["neg"])

    gap_stats = hard_negative_gap_per_row(
        neg_deragend,
        neg_hard
    )

    summary["hard_negative_gap_distribution"] = gap_stats

    # ---------------------------------------------------
    # 5. Write outputs
    # ---------------------------------------------------

    json_path = run_dir / "report_summary.json"
    md_path = run_dir / "report_summary.md"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    md = build_markdown(summary)

    with md_path.open("w", encoding="utf-8") as f:
        f.write(md)

    print(f"\nReport written to: {json_path}")
    print(f"Markdown report: {md_path}")


if __name__ == "__main__":
    main()
