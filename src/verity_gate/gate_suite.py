from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from tqdm import tqdm


# -----------------------------------------------------------------------------
# Optional imports: keep your existing embedder + feverous loader if present.
# -----------------------------------------------------------------------------
from .embedder import HFEmbedder  # expects .embed(list[str]) -> (n,d)

from .dataset import load_feverous, extract_evidence


# -----------------------------------------------------------------------------
# Policy (fixed threshold + margin)
# -----------------------------------------------------------------------------
POLICIES = {
    "editorial": 0.55,
    "standard": 0.45,
    "strict": 0.30,
}


def apply_policy(energy: float, regime: str, delta: float = 0.10) -> str:
    tau = float(POLICIES[regime])
    if energy <= tau:
        return "accept"
    if energy <= tau + float(delta):
        return "review"
    return "reject"


# -----------------------------------------------------------------------------
# Hallucination Energy (your current SVD residual)
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class EnergyResult:
    energy: float           # H_E in [0,1]
    explained: float        # ||U_r^T c||^2
    identity_error: float   # |1 - (explained + energy)|
    topk: int
    rank_r: int
    effective_rank: int
    sv: Optional[List[float]]
    topk_scores: Optional[List[float]]
    used_count: int


def _unit_norm(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = float(np.linalg.norm(x))
    if n < eps:
        return x * 0.0
    return x / n


def _unit_norm_rows(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms < eps, 1.0, norms)
    return X / norms


def cosine_scores(c: np.ndarray, E: np.ndarray) -> np.ndarray:
    # c: (d,) unit, E: (n,d) unit rows
    return (E @ c).astype(np.float32)


def build_evidence_basis(
    c: np.ndarray,
    E: np.ndarray,
    *,
    top_k: int,
    rank_r: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if E.size == 0:
        return (
            np.zeros((c.shape[0], 0), dtype=np.float32),
            np.array([], dtype=np.float32),
            np.array([], dtype=np.int64),
        )

    top_k = max(1, int(top_k))
    rank_r = max(1, int(rank_r))

    scores = cosine_scores(c, E)
    k = min(top_k, E.shape[0])
    idx = np.argsort(-scores)[:k]
    E_k = E[idx]  # (k,d)

    # Thin SVD
    U_e, S, Vt = np.linalg.svd(E_k, full_matrices=False)

    r_full = Vt.shape[0]
    r = min(rank_r, r_full)
    U_r = Vt[:r].T  # (d,r)

    return U_r.astype(np.float32), S.astype(np.float32), idx.astype(np.int64)


def hallucination_energy_svd(
    claim_vec: np.ndarray,
    evidence_vecs: np.ndarray,
    *,
    top_k: int = 12,
    rank_r: int = 8,
    return_debug: bool = True,
) -> EnergyResult:
    if claim_vec is None or evidence_vecs is None:
        return EnergyResult(
            energy=1.0,
            explained=0.0,
            identity_error=1.0,
            topk=int(top_k),
            rank_r=int(rank_r),
            effective_rank=0,
            sv=None,
            topk_scores=None,
            used_count=0,
        )

    c = _unit_norm(np.asarray(claim_vec, dtype=np.float32))

    E = np.asarray(evidence_vecs, dtype=np.float32)
    if E.ndim != 2 or E.shape[0] == 0:
        return EnergyResult(
            energy=1.0,
            explained=0.0,
            identity_error=1.0,
            topk=int(top_k),
            rank_r=int(rank_r),
            effective_rank=0,
            sv=None,
            topk_scores=None,
            used_count=0,
        )

    E = _unit_norm_rows(E)

    U_r, S, idx = build_evidence_basis(c, E, top_k=top_k, rank_r=rank_r)

    if U_r.shape[1] == 0:
        return EnergyResult(
            energy=1.0,
            explained=0.0,
            identity_error=1.0,
            topk=int(top_k),
            rank_r=int(rank_r),
            effective_rank=0,
            sv=None,
            topk_scores=None,
            used_count=0,
        )

    proj_coords = U_r.T @ c
    explained = float(np.dot(proj_coords, proj_coords))
    energy = 1.0 - explained

    explained = max(0.0, min(1.0, explained))
    energy = max(0.0, min(1.0, energy))

    identity_error = abs(1.0 - (explained + energy))
    effective_rank = int(np.sum(S > 1e-6))

    if not return_debug:
        return EnergyResult(
            energy=energy,
            explained=explained,
            identity_error=identity_error,
            topk=int(top_k),
            rank_r=int(rank_r),
            effective_rank=effective_rank,
            sv=None,
            topk_scores=None,
            used_count=int(len(idx)),
        )

    scores = cosine_scores(c, E)
    topk_scores = scores[idx].tolist() if idx.size else []
    sv_list = S.tolist() if S.size else []

    return EnergyResult(
        energy=energy,
        explained=explained,
        identity_error=identity_error,
        topk=int(top_k),
        rank_r=int(rank_r),
        effective_rank=effective_rank,
        sv=sv_list,
        topk_scores=topk_scores,
        used_count=int(len(idx)),
    )


# -----------------------------------------------------------------------------
# Evaluate claim (no oracle in v1 main path)
# -----------------------------------------------------------------------------
def evaluate_claim(
    claim_vec: np.ndarray,
    evidence_vecs: np.ndarray,
    regime: str,
    *,
    top_k: int = 12,
    rank_r: int = 8,
) -> Tuple[EnergyResult, str, List[float]]:
    base = hallucination_energy_svd(claim_vec, evidence_vecs, top_k=top_k, rank_r=rank_r)

    # robustness probe (same idea you already had)
    probe = []
    for k in (8, 12, 20):
        kk = min(int(k), evidence_vecs.shape[0]) if hasattr(evidence_vecs, "shape") else int(k)
        r = hallucination_energy_svd(
            claim_vec,
            evidence_vecs,
            top_k=max(1, kk),
            rank_r=rank_r,
            return_debug=False,
        )
        probe.append(float(r.energy))

    decision_fixed = apply_policy(base.energy, regime)
    return base, decision_fixed, probe


# -----------------------------------------------------------------------------
# IO helpers
# -----------------------------------------------------------------------------
def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def stable_unique(xs: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in xs:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


def load_examples(kind: str, path: Path, n: int, seed: int) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    out: List[Dict[str, Any]] = []

    if kind == "feverous":
        if load_feverous is None or extract_evidence is None:
            raise RuntimeError("feverous kind requires verity_gate.dataset.load_feverous + extract_evidence")

        rows = list(load_feverous(path))
        rng.shuffle(rows)
        for r in rows:
            claim = r.get("claim")
            if not isinstance(claim, str) or not claim.strip():
                continue
            ev = extract_evidence(r)
            if not ev:
                continue
            ev = stable_unique([str(x).strip() for x in ev if str(x).strip()])
            if not ev:
                continue
            out.append({"claim": claim.strip(), "evidence": ev, "label": r.get("label")})
            if len(out) >= n:
                break
        return out

    if kind == "gate_jsonl":
        rows = list(iter_jsonl(path))
        rng.shuffle(rows)
        for r in rows:
            claim = r.get("claim")
            ev = r.get("evidence_texts")
            if not isinstance(claim, str) or not claim.strip():
                continue
            if not isinstance(ev, list) or not ev:
                continue
            ev2 = stable_unique([str(x).strip() for x in ev if str(x).strip()])
            if not ev2:
                continue
            out.append({"claim": claim.strip(), "evidence": ev2, "label": r.get("label")})
            if len(out) >= n:
                break
        return out

    raise ValueError("kind must be: feverous | gate_jsonl")


def make_shuffled_negatives(pairs: List[Dict[str, Any]], offset: int = 37) -> List[Dict[str, Any]]:
    n = len(pairs)
    neg = []
    for i in range(n):
        neg.append({"claim": pairs[i]["claim"], "evidence": pairs[(i + offset) % n]["evidence"]})
    return neg


def summarize(x: np.ndarray) -> Dict[str, float]:
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return {"count": 0.0}
    return {
        "count": float(x.size),
        "mean": float(x.mean()),
        "std": float(x.std()),
        "p50": float(np.percentile(x, 50)),
        "p90": float(np.percentile(x, 90)),
        "p95": float(np.percentile(x, 95)),
        "p99": float(np.percentile(x, 99)),
        "min": float(x.min()),
        "max": float(x.max()),
    }


def auc_lower_is_positive(pos: np.ndarray, neg: np.ndarray) -> float:
    # AUC where lower scores indicate "positive"
    scores = np.concatenate([pos, neg]).astype(np.float64)
    labels = np.concatenate([np.ones_like(pos, dtype=np.int32), np.zeros_like(neg, dtype=np.int32)])

    order = np.argsort(scores)  # ascending
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(scores), dtype=np.float64)

    pos_ranks = ranks[labels == 1]
    return float((pos_ranks.sum() - (len(pos) * (len(pos) - 1) / 2.0)) / (len(pos) * len(neg)))


# -----------------------------------------------------------------------------
# Main: calibrate tau on negatives (FAR-bounded), then report performance.
# -----------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", required=True, choices=["feverous", "gate_jsonl"])
    ap.add_argument("--in_path", required=True, type=Path)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--regime", type=str, default="standard", choices=["standard", "strict", "editorial"])
    ap.add_argument("--delta", type=float, default=0.10)

    ap.add_argument("--top_k", type=int, default=12)
    ap.add_argument("--rank_r", type=int, default=8)

    ap.add_argument("--far", type=float, default=0.01, help="Target false-accept rate on shuffled negatives (accept if energy<=tau)")
    ap.add_argument("--neg_offset", type=int, default=37)

    ap.add_argument("--out_report", type=Path, default=Path("artifacts/negcal_report.json"))
    ap.add_argument("--out_scored", type=Path, default=None, help="Optional JSONL of scored positives")
    ap.add_argument("--plot_png", type=Path, default=None, help="Optional PNG histogram (pos vs neg)")
    args = ap.parse_args()

    if HFEmbedder is None:
        raise RuntimeError("Could not import verity_gate.embedder.HFEmbedder")

    embedder = HFEmbedder()

    # --- Load positives ---
    pos_pairs = load_examples(args.kind, args.in_path, args.n, args.seed)
    if len(pos_pairs) < 50:
        raise RuntimeError(f"Too few usable examples ({len(pos_pairs)}). Check input format/extraction.")

    # --- Build shuffled negatives ---
    neg_pairs = make_shuffled_negatives(pos_pairs, offset=int(args.neg_offset))

    # --- Compute energies ---
    pos_e = []
    neg_e = []

    for ex in tqdm(pos_pairs, desc="Compute POS energies"):
        claim_vec = embedder.embed([ex["claim"]])[0]
        ev_vecs = embedder.embed(ex["evidence"])
        base, decision_fixed, probe = evaluate_claim(
            claim_vec, ev_vecs, args.regime, top_k=min(args.top_k, len(ex["evidence"])), rank_r=args.rank_r
        )
        pos_e.append(float(base.energy))

    for ex in tqdm(neg_pairs, desc="Compute NEG energies (shuffled)"):
        claim_vec = embedder.embed([ex["claim"]])[0]
        ev_vecs = embedder.embed(ex["evidence"])
        base, _, _ = evaluate_claim(
            claim_vec, ev_vecs, args.regime, top_k=min(args.top_k, len(ex["evidence"])), rank_r=args.rank_r
        )
        neg_e.append(float(base.energy))

    pos_e = np.asarray(pos_e, dtype=np.float32)
    neg_e = np.asarray(neg_e, dtype=np.float32)

    # --- Calibrate tau for FAR ---
    # accept if energy <= tau, so tau is FAR percentile of negatives.
    far = float(args.far)
    far = max(0.0, min(1.0, far))
    tau_cal = float(np.percentile(neg_e, far * 100.0))

    # --- Evaluate with calibrated tau ---
    FAR = float((neg_e <= tau_cal).mean())
    TPR = float((pos_e <= tau_cal).mean())
    AUC = auc_lower_is_positive(pos_e, neg_e)

    report = {
        "kind": args.kind,
        "in_path": str(args.in_path),
        "n": int(len(pos_pairs)),
        "params": {
            "regime": args.regime,
            "delta": float(args.delta),
            "top_k": int(args.top_k),
            "rank_r": int(args.rank_r),
            "neg_offset": int(args.neg_offset),
            "far_target": float(args.far),
            "tau_cal": float(tau_cal),
        },
        "stats": {
            "pos": summarize(pos_e),
            "neg_shuffled": summarize(neg_e),
            "FAR": FAR,
            "TPR": TPR,
            "AUC": AUC,
        },
        "interpretation": {
            "accept_rule": "accept if energy <= tau",
            "note": "tau calibrated on shuffled negatives to bound false-accept rate",
        },
    }

    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    args.out_report.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n=== Negative-control calibration (v1, no oracle) ===")
    print(f"tau_cal (accept if energy<=tau): {tau_cal:.4f}")
    print(f"FAR on shuffled negatives:       {FAR:.3%} (target {args.far:.3%})")
    print(f"TPR on real pairs:              {TPR:.3%}")
    print(f"AUC (pos vs neg):               {AUC:.3f}")
    print(f"Wrote report -> {args.out_report}")

    # --- Optional: write scored positives JSONL ---
    if args.out_scored is not None:
        args.out_scored.parent.mkdir(parents=True, exist_ok=True)
        with args.out_scored.open("w", encoding="utf-8") as f:
            for ex in tqdm(pos_pairs, desc="Write scored positives"):
                claim_vec = embedder.embed([ex["claim"]])[0]
                ev_vecs = embedder.embed(ex["evidence"])
                base, decision_fixed, probe = evaluate_claim(
                    claim_vec, ev_vecs, args.regime, top_k=min(args.top_k, len(ex["evidence"])), rank_r=args.rank_r
                )
                decision_cal = "accept" if float(base.energy) <= tau_cal else ("review" if float(base.energy) <= tau_cal + float(args.delta) else "reject")
                f.write(json.dumps({
                    "claim": ex["claim"],
                    "energy": float(base.energy),
                    "coverage": float(base.explained),
                    "effective_rank": int(base.effective_rank),
                    "used_count": int(base.used_count),
                    "probe": [float(x) for x in probe],
                    "decision_fixed": decision_fixed,
                    "decision_calibrated": decision_cal,
                    "tau_cal": float(tau_cal),
                    "label": ex.get("label"),
                }, ensure_ascii=False) + "\n")
        print(f"Wrote scored positives -> {args.out_scored}")

    # --- Optional: plot histograms ---
    if args.plot_png is not None:
        try:
            import matplotlib.pyplot as plt
            args.plot_png.parent.mkdir(parents=True, exist_ok=True)
            plt.figure()
            plt.hist(pos_e, bins=40, alpha=0.7, label="pos (real evidence)")
            plt.hist(neg_e, bins=40, alpha=0.7, label="neg (shuffled)")
            plt.axvline(tau_cal, linestyle="--", label=f"tau_cal={tau_cal:.3f}")
            plt.xlabel("Hallucination Energy")
            plt.ylabel("Count")
            plt.legend()
            plt.title("Energy separation: real vs shuffled evidence")
            plt.savefig(args.plot_png, dpi=160)
            plt.close()
            print(f"Wrote plot -> {args.plot_png}")
        except Exception as e:
            print(f"Plot skipped (matplotlib issue): {e}")


if __name__ == "__main__":
    main()
