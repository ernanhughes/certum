import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict
import numpy as np
import matplotlib.pyplot as plt


# -----------------------------
# Pure Ambiguity Calibration (Policy-Agnostic Difficulty)
# -----------------------------

@dataclass(frozen=True)
class DifficultyV2Ranges:
    """Calibrated from POSITIVE examples ONLY (clear evidence regime)"""
    margin_p90: float      # 90th percentile sim_margin from deranged positives
    rank_ratio_p90: float  # 90th percentile (effective_rank / evidence_count)

def calibrate_difficulty_ranges_from_positives(pos_results: List[Dict]) -> DifficultyV2Ranges:
    """
    Calibrate ambiguity baseline using POSITIVE examples ONLY.
    Establishes "easy" regime boundary (90th percentile).
    """
    margins = []
    rank_ratios = []
    
    for r in pos_results:
        if r["meta"]["split"] != "pos":
            continue
        e = r["energy"]
        margins.append(e["similarity"]["sim_margin"])
        rank_ratios.append(
            e["support"]["effective_rank"] / max(1, len(r["evidence"]))
        )
    
    if not margins:
        raise ValueError("No positive examples found for difficulty calibration")
    
    # 90th percentile defines "typical easy case" boundary
    margin_p90 = float(sorted(margins)[int(0.9 * len(margins))])
    rank_ratio_p90 = float(sorted(rank_ratios)[int(0.9 * len(rank_ratios))])
    
    return DifficultyV2Ranges(
        margin_p90=max(0.05, margin_p90),   # Safety floor
        rank_ratio_p90=max(0.1, rank_ratio_p90)
    )

def recompute_difficulty_pure(
    sim_margin: float,
    effective_rank: int,
    evidence_count: int,
    sensitivity: float,
    ranges: DifficultyV2Ranges,
) -> float:
    """
    Pure ambiguity index: policy-agnostic evidence uncertainty.
    Matches DifficultyV2.compute() logic but standalone.
    """
    # 1. Similarity Ambiguity (PRIMARY)
    margin_norm = min(1.0, sim_margin / max(1e-4, ranges.margin_p90))
    margin_score = 1.0 - margin_norm  # 0=clear winner, 1=ambiguous tie

    # 2. Support Diffuseness
    rank_ratio = min(1.0, effective_rank / max(1, evidence_count))
    rank_norm = min(1.0, rank_ratio / max(1e-4, ranges.rank_ratio_p90))
    rank_score = rank_norm  # 0=sharp focus, 1=diffuse support

    # 3. Sensitivity (robustness)
    sensitivity_score = min(1.0, max(0.0, sensitivity))

    # Weighted blend (tuned for ambiguity separation)
    difficulty = (
        0.60 * margin_score +    # Dominant: evidence ambiguity
        0.30 * rank_score +      # Secondary: support diffuseness
        0.10 * sensitivity_score # Tertiary: perturbation sensitivity
    )
    return float(max(0.0, min(1.0, difficulty)))

def patch_results_with_pure_difficulty(
    results: List[Dict],
    ranges: DifficultyV2Ranges,
) -> List[Dict]:
    """
    Recompute difficulty using pure ambiguity index.
    Operates on existing JSONL results — NO gate re-evaluation needed.
    """
    patched = []
    for r in results:
        # Extract raw metrics already stored in your schema
        e = r["energy"]
        sim_margin = e["similarity"]["sim_margin"]
        effective_rank = e["support"]["effective_rank"]
        evidence_count = len(r["evidence"])
        sensitivity = e["robustness"]["sensitivity"]
        
        # Recompute policy-agnostic difficulty
        new_diff = recompute_difficulty_pure(
            sim_margin, effective_rank, evidence_count, sensitivity, ranges
        )
        
        # Patch result dict (shallow copy)
        patched_r = r.copy()
        patched_r["difficulty"] = {
            "value": new_diff,
            "bucket": "easy" if new_diff <= 0.40 else ("medium" if new_diff <= 0.75 else "hard")
        }
        patched.append(patched_r)
    return patched


# -----------------------------
# Data loading (unchanged)
# -----------------------------

@dataclass(frozen=True)
class Point:
    energy: float
    difficulty: float

def load_results(jsonl_path: Path) -> List[Dict]:
    rows = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows

def load_points(jsonl_path: Path) -> List[Point]:
    pts: List[Point] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)

            # Your stable schema already has these:
            e = float(r["energy"]["value"])
            d = float(r["difficulty"]["value"])

            # defensively clamp
            e = min(1.0, max(0.0, e))
            d = min(1.0, max(0.0, d))
            pts.append(Point(e, d))
    return pts


# -----------------------------
# Policy model (unchanged)
# -----------------------------

@dataclass(frozen=True)
class PolicyKnobs:
    # difficulty bands
    difficulty_low: float = 0.40
    difficulty_high: float = 0.75

    # effectiveness thresholds
    eff_min_review: float = 0.05
    eff_min_accept: float = 0.25

    # margin band around tau_accept (fraction of tau)
    margin_frac: float = 0.10

    # tau_review multiplier (matches your code)
    tau_review_mult: float = 1.25


def effectiveness(energy: float, tau: float) -> float:
    # matches your gate.effectiveness_score()
    return max(0.0, (tau - energy) / max(tau, 1e-6))


def decide_energy_only(energy: float, tau: float, knobs: PolicyKnobs) -> str:
    """
    Simple baseline: no difficulty axis.
    Uses tau_accept + tau_review (review band).
    """
    tau_review = knobs.tau_review_mult * tau
    margin = knobs.margin_frac * tau

    eff = effectiveness(energy, tau)

    # Region C
    if energy > tau_review or eff < knobs.eff_min_review:
        return "reject"

    # Region B
    if abs(energy - tau) <= margin or eff < knobs.eff_min_accept:
        return "review"

    # Region A
    return "accept"

def decide_threshold_only(energy: float, tau: float) -> str:
    return "accept" if energy <= tau else "reject"

def decide_energy_plus_difficulty(energy: float, difficulty: float, tau: float, knobs: PolicyKnobs) -> str:
    """
    Your 2D surface: energy + difficulty.
    """
    tau_review = knobs.tau_review_mult * tau
    margin = knobs.margin_frac * tau
    eff = effectiveness(energy, tau)

    # Region C (unsafe)
    if difficulty > knobs.difficulty_high or energy > tau_review or eff < knobs.eff_min_review:
        return "reject"

    # Region B (ambiguous)
    if difficulty > knobs.difficulty_low or abs(energy - tau) <= margin or eff < knobs.eff_min_accept:
        return "review"

    # Region A (safe)
    return "accept"


def auc_score(pos_vals: List[float], neg_vals: List[float]) -> float:
    """
    Compute AUC via rank comparison (no sklearn dependency).
    AUC = P(neg > pos)
    """
    pos = np.array(pos_vals)
    neg = np.array(neg_vals)

    if len(pos) == 0 or len(neg) == 0:
        return float("nan")

    # Pairwise comparison
    total = 0
    correct = 0

    for n in neg:
        for p in pos:
            total += 1
            if n > p:
                correct += 1
            elif n == p:
                correct += 0.5

    return correct / total

def extract_metric(results: List[Dict], key: str) -> List[float]:
    vals = []
    for r in results:
        e = r["energy"]

        if key == "PR":
            vals.append(e["participation_ratio"])
        elif key == "sigma1":
            vals.append(e["spectral"]["sigma1_ratio"])
        elif key == "sim_margin":
            vals.append(e["similarity"]["sim_margin"])
        elif key == "sensitivity":
            vals.append(e["robustness"]["sensitivity"])
        else:
            raise ValueError(f"Unknown metric: {key}")

    return vals

# -----------------------------
# Calibration at fixed FAR (unchanged)
# -----------------------------

def find_tau_for_far_budget(neg: List[Point], target_far: float) -> float:
    n = len(neg)
    if n == 0:
        return 1e-4

    k = int(np.floor(target_far * n))  # allowed accepted negatives
    energies = np.sort([p.energy for p in neg])

    if k <= 0:
        # Must accept 0 negatives -> set tau just below the minimum negative energy
        return float(max(1e-6, energies[0] - 1e-6))

    # Accept exactly k negatives: threshold is kth smallest energy (1-indexed)
    return float(energies[k - 1])


def far_for_tau(neg: List[Point], tau: float, *, use_difficulty: bool, knobs: PolicyKnobs) -> float:
    acc = 0
    for p in neg:
        v = (
            decide_energy_plus_difficulty(p.energy, p.difficulty, tau, knobs)
            if use_difficulty
            else decide_energy_only(p.energy, tau, knobs)
        )
        if v == "accept":
            acc += 1
    return acc / max(1, len(neg))

def metrics_at_tau(
    pos: List[Point],
    neg: List[Point],
    tau: float,
    *,
    use_difficulty: bool,
    knobs: PolicyKnobs,
) -> Dict[str, float]:
    def verdict(p: Point) -> str:
        return (
            decide_energy_plus_difficulty(p.energy, p.difficulty, tau, knobs)
            if use_difficulty
            else decide_energy_only(p.energy, tau, knobs)
        )

    pos_v = [verdict(p) for p in pos]
    neg_v = [verdict(p) for p in neg]

    tpr = sum(v == "accept" for v in pos_v) / max(1, len(pos))
    far = sum(v == "accept" for v in neg_v) / max(1, len(neg))
    review_pos = sum(v == "review" for v in pos_v) / max(1, len(pos))
    review_neg = sum(v == "review" for v in neg_v) / max(1, len(neg))

    return {
        "tau": float(tau),
        "TPR": float(tpr),
        "FAR": float(far),
        "REVIEW_POS": float(review_pos),
        "REVIEW_NEG": float(review_neg),
    }


# -----------------------------
# Ablation: "difficulty contribution" (unchanged)
# -----------------------------

def ablation_report(pos: List[Point], neg: List[Point], target_far: float, knobs: PolicyKnobs) -> Dict[str, Dict[str, float]]:
    # Calibrate tau separately for each policy to hit the SAME FAR
    tau = find_tau_for_far_budget(neg, target_far)  # one tau only

    m_E = metrics_at_tau(pos, neg, tau, use_difficulty=False, knobs=knobs)
    m_ED = metrics_at_tau(pos, neg, tau, use_difficulty=True, knobs=knobs)

    # Difficulty contribution: delta TPR at fixed FAR
    m_ED["ΔTPR_vs_energy_only"] = m_ED["TPR"] - m_E["TPR"]
    m_ED["ΔREVIEW_POS_vs_energy_only"] = m_ED["REVIEW_POS"] - m_E["REVIEW_POS"]

    return {
        "tau": float(tau), 
        "energy_only": m_E,
        "energy_plus_difficulty": m_ED,
    }



# -----------------------------
# Region visualization + area coverage (unchanged)
# -----------------------------

def region_id(energy: float, difficulty: float, tau: float, knobs: PolicyKnobs, use_difficulty: bool) -> int:
    v = (
        decide_energy_plus_difficulty(energy, difficulty, tau, knobs)
        if use_difficulty
        else decide_energy_only(energy, tau, knobs)
    )
    return {"accept": 0, "review": 1, "reject": 2}[v]

def compute_tpr(results):
    pos = [r for r in results if r["meta"]["split"] == "pos"]
    accepted = [r for r in pos if r["decision"]["verdict"] == "accept"]
    return len(accepted) / max(1, len(pos))

def extract_rank_stats(results):
    ranks = [
        r["energy"]["support"]["effective_rank"]
        for r in results
    ]
    return {
        "mean": float(np.mean(ranks)),
        "std": float(np.std(ranks)),
        "p50": float(np.percentile(ranks, 50)),
        "p90": float(np.percentile(ranks, 90)),
    }

def difficulty_stats(results):
    vals = [
        r["difficulty"]["value"]
        for r in results
    ]
    return {
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals)),
        "p50": float(np.percentile(vals, 50)),
        "p90": float(np.percentile(vals, 90)),
    }


def region_area_coverage(
    tau: float,
    knobs: PolicyKnobs,
    *,
    use_difficulty: bool,
    samples: int = 200_000,
    seed: int = 1337,
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    ds = rng.random(samples)
    es = rng.random(samples)

    counts = np.zeros(3, dtype=np.int64)
    for e, d in zip(es, ds):
        counts[region_id(float(e), float(d), tau, knobs, use_difficulty)] += 1

    total = counts.sum()
    return {
        "accept_area": float(counts[0] / total),
        "review_area": float(counts[1] / total),
        "reject_area": float(counts[2] / total),
    }

def sweep_tau_vs_far(pos: List[Point], neg: List[Point], fars: List[float]):
    taus = []
    for f in fars:
        tau = find_tau_for_far_budget(neg, f)
        taus.append(tau)
    return taus

def sweep_tpr_vs_far(pos, neg, fars, knobs):
    tprs = []
    for f in fars:
        tau = find_tau_for_far_budget(neg, f)
        m = metrics_at_tau(pos, neg, tau, use_difficulty=False, knobs=knobs)
        tprs.append(m["TPR"])
    return tprs

def corr_energy_difficulty(rows, name: str):
    e = np.array([float(r["energy"]["value"]) for r in rows], dtype=np.float32)
    d = np.array([float(r["difficulty"]["value"]) for r in rows], dtype=np.float32)
    c = float(np.corrcoef(e, d)[0, 1]) if len(rows) > 2 else float("nan")
    print(f"Corr(energy,difficulty) {name}: {c:.4f}")

def spectral_stats(results):
    sigma1 = [r["energy"]["spectral"]["sigma1_ratio"] for r in results]
    sigma2 = [r["energy"]["spectral"]["sigma2_ratio"] for r in results]
    pr = [r["energy"]["participation_ratio"] for r in results]

    return {
        "mean_sigma1_ratio": float(np.mean(sigma1)),
        "mean_sigma2_ratio": float(np.mean(sigma2)),
        "mean_PR": float(np.mean(pr)),
    }


def plot_rank_vs_energy(results, out_path: Path, title: str):
    ranks = []
    energies = []

    for r in results:
        if "energy" not in r:
            continue
        ranks.append(r["energy"]["support"]["effective_rank"])
        energies.append(r["energy"]["value"])

    if not ranks:
        print(f"No valid rows for {title}")
        return

    plt.figure()
    plt.scatter(ranks, energies, alpha=0.4, s=10)
    plt.xlabel("effective_rank")
    plt.ylabel("energy")
    plt.title(title)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved: {out_path}")

def plot_similarity_diagnostics(results, out_prefix: Path, title_prefix: str):
    sim_margin = []
    sim_top1 = []
    sim_top2 = []
    energy = []
    rank = []

    for r in results:
        e = r["energy"]
        energy.append(e["value"])
        rank.append(e["support"]["effective_rank"])
        s = r["energy"]["similarity"]
        sim_margin.append(s.get("sim_margin", 0.0))
        sim_top1.append(s.get("sim_top1", 0.0))
        sim_top2.append(s.get("sim_top2", 0.0))

    sim_margin = np.array(sim_margin)
    energy = np.array(energy)
    rank = np.array(rank)

    # ---- Plot 1: sim_margin vs energy ----
    plt.figure()
    plt.scatter(sim_margin, energy, s=10, alpha=0.4)
    plt.xlabel("sim_margin (top1 - top2)")
    plt.ylabel("energy")
    plt.title(f"{title_prefix}: sim_margin vs energy")
    plt.savefig(out_prefix.with_name(out_prefix.name + "_margin_vs_energy.png"))
    plt.close()

    # ---- Plot 2: sim_margin vs effective_rank ----
    plt.figure()
    plt.scatter(sim_margin, rank, s=10, alpha=0.4)
    plt.xlabel("sim_margin")
    plt.ylabel("effective_rank")
    plt.title(f"{title_prefix}: sim_margin vs rank")
    plt.savefig(out_prefix.with_name(out_prefix.name + "_margin_vs_rank.png"))
    plt.close()

    # ---- Plot 3: sim_margin histogram ----
    plt.figure()
    plt.hist(sim_margin, bins=40)
    plt.xlabel("sim_margin")
    plt.ylabel("count")
    plt.title(f"{title_prefix}: sim_margin distribution")
    plt.savefig(out_prefix.with_name(out_prefix.name + "_margin_hist.png"))
    plt.close()

    # ---- Correlations ----
    if len(sim_margin) > 1:
        corr_energy = float(np.corrcoef(sim_margin, energy)[0, 1])
        corr_rank = float(np.corrcoef(sim_margin, rank)[0, 1])
        print(f"{title_prefix} correlation margin↔energy:", corr_energy)
        print(f"{title_prefix} correlation margin↔rank:", corr_rank)


def plot_shaded_surface(
    points: List[Point],
    tau: float,
    knobs: PolicyKnobs,
    *,
    use_difficulty: bool,
    title: str,
    out_path: Path,
    grid: int = 250,
):
    # create region grid
    xs = np.linspace(0, 1, grid)
    ys = np.linspace(0, 1, grid)
    Z = np.zeros((grid, grid), dtype=np.int32)

    for i, d in enumerate(xs):
        for j, e in enumerate(ys):
            Z[j, i] = region_id(float(e), float(d), tau, knobs, use_difficulty)

    plt.figure()
    # 0=accept,1=review,2=reject
    plt.imshow(Z, origin="lower", extent=(0, 1, 0, 1), aspect="auto", alpha=0.25)

    # scatter points
    for p in points:
        v = (
            decide_energy_plus_difficulty(p.energy, p.difficulty, tau, knobs)
            if use_difficulty
            else decide_energy_only(p.energy, tau, knobs)
        )
        plt.scatter([p.difficulty], [p.energy], s=10, label=v)

    # de-duplicate legend entries
    handles, labels = plt.gca().get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    plt.legend(uniq.values(), uniq.keys())

    # draw policy lines
    plt.axvline(knobs.difficulty_low)
    plt.axvline(knobs.difficulty_high)

    tau_review = knobs.tau_review_mult * tau
    margin = knobs.margin_frac * tau
    plt.axhline(tau)
    plt.axhline(tau - margin, linestyle="--")
    plt.axhline(tau + margin, linestyle="--")
    plt.axhline(tau_review, color="red")

    plt.title(title)
    plt.xlabel("difficulty")
    plt.ylabel("energy")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


# -----------------------------
# Main (CALIBRATION-AWARE)
# -----------------------------

if __name__ == "__main__":
    run_dirs = sorted(Path("artifacts/runs").glob("*"))
    if not run_dirs:
        raise RuntimeError("No runs found in artifacts/runs/")
    run_dir = run_dirs[-1]
    print(f"📊 Using run: {run_dir}")

    # ---- STEP 1: Load POSITIVE examples for calibration ----
    pos_der_raw = load_results(run_dir / "pos_deranged.jsonl")
    if not pos_der_raw:
        raise RuntimeError("No positive examples found for difficulty calibration")
    
    # Calibrate ambiguity baseline from POSITIVES ONLY
    ranges = calibrate_difficulty_ranges_from_positives(pos_der_raw)
    print(f"✅ Calibrated DifficultyV2Ranges:")
    print(f"   margin_p90={ranges.margin_p90:.3f} | rank_ratio_p90={ranges.rank_ratio_p90:.3f}")

    # ---- STEP 2: Patch ALL result sets with pure ambiguity index ----
    pos_der = patch_results_with_pure_difficulty(pos_der_raw, ranges)
    neg_der = patch_results_with_pure_difficulty(
        load_results(run_dir / "neg_deranged.jsonl"), ranges
    )
    pos_hard = patch_results_with_pure_difficulty(
        load_results(run_dir / "pos_hard_mined_v2.jsonl"), ranges
    )
    neg_hard = patch_results_with_pure_difficulty(
        load_results(run_dir / "neg_hard_mined_v2.jsonl"), ranges
    )

    print(f"POS deranged count: {len(pos_der)}")
    print(f"NEG deranged count: {len(neg_der)}")
    print(f"POS hard-mined count: {len(pos_hard)}")
    print(f"NEG hard-mined count: {len(neg_hard)}")

    # ---- STEP 3: Convert to Point objects (using PATCHED difficulty) ----
    def to_points(results: List[Dict]) -> List[Point]:
        pts = []
        for r in results:
            e = float(r["energy"]["value"])
            d = float(r["difficulty"]["value"])
            e = min(1.0, max(0.0, e))
            d = min(1.0, max(0.0, d))
            pts.append(Point(e, d))
        return pts

    pos_der_pts = to_points(pos_der)
    neg_der_pts = to_points(neg_der)
    pos_hard_pts = to_points(pos_hard)
    neg_hard_pts = to_points(neg_hard)

    # ---- STEP 4: Proceed with EXISTING analysis (now on calibrated difficulty) ----
    # Pick regime for initial ablation
    pos = pos_hard_pts
    neg = neg_hard_pts
    nneg = len(neg)
    step = 1 / max(1, nneg)
    target_far = 0.01
    budget = int(np.floor(target_far * nneg))
    print(f"NEG n={nneg} -> FAR step={step:.4f} -> target_far={target_far:.3f} -> budget accepts={budget}")

    energies = np.sort([p.energy for p in neg])
    print("min neg energy:", energies[0], "p1:", np.percentile(energies, 1), "p2:", np.percentile(energies, 2))

    knobs = PolicyKnobs(
        difficulty_low=0.40,
        difficulty_high=0.75,
        eff_min_review=0.05,
        eff_min_accept=0.25,
        margin_frac=0.10,
        tau_review_mult=1.25,
    )

    rep = ablation_report(pos, neg, target_far, knobs)

    print("\n=== Fixed-FAR Ablation (target FAR = %.3f) ===" % target_far)
    print("Energy-only:", rep["energy_only"])
    print("Energy+Difficulty:", rep["energy_plus_difficulty"])
    print(f"NEG n={len(neg)} -> FAR step = {1/len(neg):.4f}")

    # Use the 2D policy's tau for surfaces and area
    tau = rep["tau"]

    print("\n--- NEG ACCEPTED ENERGIES (patched difficulty) ---")
    for p in neg:
        if decide_energy_plus_difficulty(p.energy, p.difficulty, tau, knobs) == "accept":
            print(f"  energy={p.energy:.4f} | difficulty={p.difficulty:.4f}")
    print("--- END NEG ACCEPTED ENERGIES ---")

    areas_E  = region_area_coverage(tau, knobs, use_difficulty=False)
    areas_ED = region_area_coverage(tau, knobs, use_difficulty=True)

    print("\n=== Region Area Coverage (uniform over [0,1]^2) ===")
    print("Energy-only areas:", areas_E)
    print("Energy+Difficulty areas:", areas_ED)

    plot_shaded_surface(
        pos, tau, knobs,
        use_difficulty=True,
        title="POS: shaded policy regions (E+D, patched)",
        out_path=run_dir / "policy_surface_pos_patched.png"
    )

    plot_shaded_surface(
        neg, tau, knobs,
        use_difficulty=True,
        title="NEG: shaded policy regions (E+D, patched)",
        out_path=run_dir / "policy_surface_neg_patched.png"
    )

    # Correlation diagnostics (now on patched difficulty)
    corr_energy_difficulty(pos_der,  "POS deranged (patched)")
    corr_energy_difficulty(neg_der,  "NEG deranged (patched)")
    corr_energy_difficulty(pos_hard, "POS hard_mined_v2 (patched)")
    corr_energy_difficulty(neg_hard, "NEG hard_mined_v2 (patched)")

    # ---- Compute per-distribution taus ----
    tau_der = find_tau_for_far_budget(neg_der_pts, target_far)
    tau_hard = find_tau_for_far_budget(neg_hard_pts, target_far)

    print("\n=== Calibrated Taus (FAR=%.3f) ===" % target_far)
    print("tau_deranged:", tau_der)
    print("tau_hard_mined_v2:", tau_hard)

    # ---- Compute TPR under shared tau (hard tau) ----
    shared_tau = tau_hard

    tpr_der_shared = metrics_at_tau(
        pos_der_pts, neg_der_pts, shared_tau,
        use_difficulty=True, knobs=knobs
    )["TPR"]

    tpr_hard_shared = metrics_at_tau(
        pos_hard_pts, neg_hard_pts, shared_tau,
        use_difficulty=True, knobs=knobs
    )["TPR"]

    print("\n=== TPR using SHARED tau (hard_mined_v2 calibrated) ===")
    print("TPR (deranged, shared tau):", tpr_der_shared)
    print("TPR (hard_mined_v2, shared tau):", tpr_hard_shared)
    print("ΔTPR (shared tau):", tpr_der_shared - tpr_hard_shared)

    # ---- Compute TPR under own tau ----
    tpr_der_own = metrics_at_tau(
        pos_der_pts, neg_der_pts, tau_der,
        use_difficulty=True, knobs=knobs
    )["TPR"]

    tpr_hard_own = metrics_at_tau(
        pos_hard_pts, neg_hard_pts, tau_hard,
        use_difficulty=True, knobs=knobs
    )["TPR"]

    print("\n=== TPR using OWN tau per distribution ===")
    print("TPR (deranged, own tau):", tpr_der_own)
    print("TPR (hard_mined_v2, own tau):", tpr_hard_own)
    print("ΔTPR (own tau):", tpr_der_own - tpr_hard_own)

    print("\n=== Rank Statistics (patched) ===")
    print("Deranged POS rank:", extract_rank_stats(pos_der))
    print("Hard POS rank:", extract_rank_stats(pos_hard))

    print("\n=== Difficulty Statistics (patched) ===")
    print("POS difficulty:", difficulty_stats(pos_der))
    print("NEG difficulty:", difficulty_stats(neg_der))

    # Diagnostic: Print patched difficulty separation
    pos_diff_mean = difficulty_stats(pos_der)["mean"]
    neg_diff_mean = difficulty_stats(neg_der)["mean"]
    print(f"\n✅ Difficulty separation (patched): POS={pos_diff_mean:.3f} vs NEG={neg_diff_mean:.3f} | Δ={abs(pos_diff_mean - neg_diff_mean):.3f}")

    fars = np.linspace(0.005, 0.05, 20)

    tau_der_curve = sweep_tau_vs_far(pos_der_pts, neg_der_pts, fars)
    tau_hard_curve = sweep_tau_vs_far(pos_hard_pts, neg_hard_pts, fars)

    plt.figure()
    plt.plot(fars, tau_der_curve, label="deranged")
    plt.plot(fars, tau_hard_curve, label="hard_mined_v2")
    plt.xlabel("FAR")
    plt.ylabel("Calibrated tau")
    plt.legend()
    plt.title("Tau vs FAR (patched difficulty)")
    plt.savefig(run_dir / "tau_vs_far_patched.png")
    plt.close()

    tpr_der_curve = sweep_tpr_vs_far(pos_der_pts, neg_der_pts, fars, knobs)
    tpr_hard_curve = sweep_tpr_vs_far(pos_hard_pts, neg_hard_pts, fars, knobs)

    plt.figure()
    plt.plot(fars, tpr_der_curve, label="deranged")
    plt.plot(fars, tpr_hard_curve, label="hard_mined_v2")
    plt.xlabel("FAR")
    plt.ylabel("TPR")
    plt.legend()
    plt.title("TPR vs FAR (patched difficulty)")
    plt.savefig(run_dir / "tpr_vs_far_patched.png")
    plt.close()

    # Rank vs Energy plots
    plot_rank_vs_energy(
        pos_der,
        run_dir / "rank_vs_energy_pos_deranged_patched.png",
        "POS Deranged: rank vs energy (patched)"
    )

    plot_rank_vs_energy(
        pos_hard,
        run_dir / "rank_vs_energy_pos_hard_mined_v2_patched.png",
        "POS Hard-Mined: rank vs energy (patched)"
    )

    plot_rank_vs_energy(
        neg_der,
        run_dir / "rank_vs_energy_neg_deranged_patched.png",
        "NEG Deranged: rank vs energy (patched)"
    )

    plot_rank_vs_energy(
        neg_hard,
        run_dir / "rank_vs_energy_neg_hard_mined_v2_patched.png",
        "NEG Hard-Mined: rank vs energy (patched)"
    )

    plot_similarity_diagnostics(
        pos_der,
        run_dir / "pos_deranged_patched",
        "POS Deranged (patched)"
    )

    plot_similarity_diagnostics(
        pos_hard,
        run_dir / "pos_hard_mined_v2_patched",
        "POS Hard-Mined (patched)"
    )

    plot_similarity_diagnostics(
        neg_der,
        run_dir / "neg_deranged_patched",
        "NEG Deranged (patched)"
    )

    plot_similarity_diagnostics(
        neg_hard,
        run_dir / "neg_hard_mined_v2_patched",
        "NEG Hard-Mined (patched)"
    )

    print("\n=== Spectral Diagnostics ===")
    print("POS deranged:", spectral_stats(pos_der))
    print("NEG deranged:", spectral_stats(neg_der))
    print("POS hard-mined:", spectral_stats(pos_hard))
    print("NEG hard-mined:", spectral_stats(neg_hard))


    print("\n=== AUC Diagnostics (POS vs NEG) ===")
    metrics = ["PR", "sigma1", "sim_margin", "sensitivity"]
    for m in metrics:
        pos_vals = extract_metric(pos_hard, m)
        neg_vals = extract_metric(neg_hard, m)

        auc = auc_score(pos_vals, neg_vals)
        print(f"AUC({m}) = {auc:.4f}")
    
    print("\n=== AUC Diagnostics (DERANGED) ===")
    for m in metrics:
        pos_vals = extract_metric(pos_der, m)
        neg_vals = extract_metric(neg_der, m)

        auc = auc_score(pos_vals, neg_vals)
        print(f"AUC({m}) = {auc:.4f}")


    print("\nCorrelation rank↔energy (deranged POS, patched):",
        np.corrcoef(
            [r["energy"]["support"]["effective_rank"] for r in pos_der],
            [r["energy"]["value"] for r in pos_der]
        )[0,1])

    print("\n✅ Pure ambiguity calibration complete.")
    print("   Next: Check correlation diagnostics for orthogonality (target: |Corr(E,D)| > 0.20 on hard-mined negatives)")