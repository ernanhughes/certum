import json
from dataclasses import dataclass
from operator import neg
from pathlib import Path
from typing import List, Dict
import numpy as np
import matplotlib.pyplot as plt


# -----------------------------
# Data loading
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
# Policy model (recomputed)
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


# -----------------------------
# Calibration at fixed FAR
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
# Ablation: "difficulty contribution"
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
# Region visualization + area coverage
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
# Main
# -----------------------------

if __name__ == "__main__":
    run_dirs = sorted(Path("artifacts/runs").glob("*"))
    if not run_dirs:
        raise RuntimeError("No runs found in artifacts/runs/")
    run_dir = run_dirs[-1]
    print(f"📊 Using run: {run_dir}")

    # Pick regime
    pos_path = run_dir / "pos_hard_mined_v2.jsonl"
    neg_path = run_dir / "neg_hard_mined_v2.jsonl"
    if not pos_path.exists() or not neg_path.exists():
        raise RuntimeError(f"Missing expected files: {pos_path.name}, {neg_path.name}")

    pos = load_points(pos_path)
    neg = load_points(neg_path)
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


    print("\n--- NEG ACCEPTED ENERGIES ---")
    for p in neg:
        if decide_energy_plus_difficulty(p.energy, p.difficulty, tau, knobs) == "accept":
            print(p.energy, p.difficulty)
    print("\n--- END NEG ACCEPTED ENERGIES ---")

    areas_E  = region_area_coverage(tau, knobs, use_difficulty=False)
    areas_ED = region_area_coverage(tau, knobs, use_difficulty=True)

    print("\n=== Region Area Coverage (uniform over [0,1]^2) ===")
    print("Energy-only areas:", areas_E)
    print("Energy+Difficulty areas:", areas_ED)

    plot_shaded_surface(
        pos, tau, knobs,
        use_difficulty=True,
        title="POS: shaded policy regions (E+D)",
        out_path=run_dir / "policy_surface_pos.png"
    )

    plot_shaded_surface(
        neg, tau, knobs,
        use_difficulty=True,
        title="NEG: shaded policy regions (E+D)",
        out_path=run_dir / "policy_surface_neg.png"
    )

    # Load scored JSONLs
    pos_der = load_results(run_dir /"pos_deranged.jsonl")
    neg_der = load_results(run_dir /"neg_deranged.jsonl")

    pos_hard = load_results(run_dir /"pos_hard_mined_v2.jsonl")
    neg_hard = load_results(run_dir /"neg_hard_mined_v2.jsonl")

    # Convert to Point objects for tau calibration
    neg_der_pts = [
        Point(
            energy=r["energy"]["value"],
            difficulty=r["difficulty"]["value"]
        )
        for r in neg_der
    ]

    neg_hard_pts = [
        Point(
            energy=r["energy"]["value"],
            difficulty=r["difficulty"]["value"]
        )
        for r in neg_hard
    ]

    pos_der_pts = [
        Point(
            energy=r["energy"]["value"],
            difficulty=r["difficulty"]["value"]
        )
        for r in pos_der
    ]

    pos_hard_pts = [
        Point(
            energy=r["energy"]["value"],
            difficulty=r["difficulty"]["value"]
        )
        for r in pos_hard
    ]


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

    print("\n=== Calibrated Taus (FAR=%.3f) ===" % target_far)
    print("tau_deranged:", tau_der)
    print("tau_hard_mined_v2:", tau_hard)

    print("\n=== TPR using SHARED tau (hard_mined_v2 calibrated) ===")
    print("TPR (deranged, shared tau):", tpr_der_shared)
    print("TPR (hard_mined_v2, shared tau):", tpr_hard_shared)
    print("ΔTPR (shared tau):", tpr_der_shared - tpr_hard_shared)

    print("\n=== TPR using OWN tau per distribution ===")
    print("TPR (deranged, own tau):", tpr_der_own)
    print("TPR (hard_mined_v2, own tau):", tpr_hard_own)
    print("ΔTPR (own tau):", tpr_der_own - tpr_hard_own)

    print("Deranged POS rank:", extract_rank_stats(pos_der))
    print("Hard POS rank:", extract_rank_stats(pos_hard))

    print("POS difficulty:", difficulty_stats(pos_der))
    print("NEG difficulty:", difficulty_stats(neg_der))

    fars = np.linspace(0.005, 0.05, 20)

    tau_der_curve = sweep_tau_vs_far(pos_der_pts, neg_der_pts, fars)
    tau_hard_curve = sweep_tau_vs_far(pos_hard_pts, neg_hard_pts, fars)

    plt.figure()
    plt.plot(fars, tau_der_curve, label="deranged")
    plt.plot(fars, tau_hard_curve, label="hard_mined_v2")
    plt.xlabel("FAR")
    plt.ylabel("Calibrated tau")
    plt.legend()
    plt.savefig(run_dir / "tau_vs_far.png")
    plt.close()

    tpr_der_curve = sweep_tpr_vs_far(pos_der_pts, neg_der_pts, fars, knobs)
    tpr_hard_curve = sweep_tpr_vs_far(pos_hard_pts, neg_hard_pts, fars, knobs)

    plt.figure()
    plt.plot(fars, tpr_der_curve, label="deranged")
    plt.plot(fars, tpr_hard_curve, label="hard_mined_v2")
    plt.xlabel("FAR")
    plt.ylabel("TPR")
    plt.legend()
    plt.savefig(run_dir / "tpr_vs_far.png")
    plt.close()

    # Rank vs Energy plots
    plot_rank_vs_energy(
        pos_der,
        run_dir / "rank_vs_energy_pos_deranged.png",
        "POS Deranged: rank vs energy"
    )

    plot_rank_vs_energy(
        pos_hard,
        run_dir / "rank_vs_energy_pos_hard_mined_v2.png",
        "POS Hard-Mined: rank vs energy"
    )

    plot_rank_vs_energy(
        neg_der,
        run_dir / "rank_vs_energy_neg_deranged.png",
        "NEG Deranged: rank vs energy"
    )

    plot_rank_vs_energy(
        neg_hard,
        run_dir / "rank_vs_energy_neg_hard_mined_v2.png",
        "NEG Hard-Mined: rank vs energy"
    )

    print("Correlation rank↔energy (deranged POS):",
        np.corrcoef(
            [r["energy"]["support"]["effective_rank"] for r in pos_der],
            [r["energy"]["value"] for r in pos_der]
        )[0,1])

