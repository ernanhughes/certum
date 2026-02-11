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

def load_points(jsonl_path: Path) -> List[Point]:
    pts: List[Point] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)

            # Your stable schema already has these:
            e = float(r["energy"]["energy"])
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


def plot_shaded_surface(
    points: List[Point],
    tau: float,
    knobs: PolicyKnobs,
    *,
    use_difficulty: bool,
    title: str,
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
    plt.show()


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

    plot_shaded_surface(pos, tau, knobs, use_difficulty=True, title="POS: shaded policy regions (E+D)")
    plot_shaded_surface(neg, tau, knobs, use_difficulty=True, title="NEG: shaded policy regions (E+D)")
