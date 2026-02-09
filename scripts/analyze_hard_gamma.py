#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from tqdm import tqdm

# ----------------------------
# Cache DB helpers
# ----------------------------
def connect_db(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    return conn

def fetch_vecs(conn: sqlite3.Connection, element_ids: List[str]) -> np.ndarray:
    """
    Fetch embedding vectors for the provided element_ids.

    Assumes a table 'embeddings' with:
      - element_id TEXT
      - vec BLOB (np.float32 bytes)  OR a JSON string, depending on your schema

    You may need to adjust decoding to match your DB.
    """
    if not element_ids:
        return np.zeros((0, 384), dtype=np.float32)

    q = ",".join(["?"] * len(element_ids))
    rows = conn.execute(
        f"SELECT element_id, vec FROM embeddings WHERE element_id IN ({q})",
        element_ids
    ).fetchall()

    # preserve input order
    by_id = {r["element_id"]: r["vec"] for r in rows}
    vecs = []
    for eid in element_ids:
        blob = by_id.get(eid)
        if blob is None:
            continue
        # --- decode ---
        # If stored as float32 bytes:
        v = np.frombuffer(blob, dtype=np.float32)
        vecs.append(v)

    if not vecs:
        return np.zeros((0, 384), dtype=np.float32)
    return np.vstack(vecs)

def unit(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x)
    if n == 0:
        return x
    return x / n

# ----------------------------
# Energy + contributions
# ----------------------------
def projection_energy(claim: np.ndarray, E: np.ndarray, rank_r: int) -> Tuple[float, np.ndarray]:
    """
    Compute residual energy of claim after projection into rank-r subspace
    of evidence matrix E (m x d). Returns:
      energy in [0,1] (if vectors are unit),
      per-evidence contributions proxy (m,) based on squared dot with basis.
    """
    if E.shape[0] == 0:
        return 1.0, np.zeros((0,), dtype=np.float32)

    # Normalize rows
    E = np.stack([unit(v) for v in E], axis=0)
    c = unit(claim)

    # Build low-rank basis via SVD of E
    # E = U S Vt ; rows span is in Vt
    # We want top r right-singular vectors
    try:
        _, _, Vt = np.linalg.svd(E, full_matrices=False)
    except np.linalg.LinAlgError:
        return 1.0, np.zeros((E.shape[0],), dtype=np.float32)

    r = max(1, min(rank_r, Vt.shape[0]))
    B = Vt[:r].T  # d x r, orthonormal-ish

    # Project claim into span(B)
    proj = B @ (B.T @ c)
    resid = c - proj
    energy = float(np.linalg.norm(resid) ** 2)  # since unit vectors => in [0,1-ish]

    # Contribution proxy: how much each evidence row aligns with claim
    # (cheap proxy; you can replace with your exact basis coefficient logic later)
    dots = (E @ c)
    contrib = (dots ** 2).astype(np.float32)

    return energy, contrib

def gamma_dropout(claim: np.ndarray, E: np.ndarray, rank_r: int) -> float:
    """
    Γ_drop = max_i (E_without_i - E_full)
    """
    full_e, _ = projection_energy(claim, E, rank_r)
    if E.shape[0] <= 1:
        return 0.0
    deltas = []
    for i in range(E.shape[0]):
        Em = np.delete(E, i, axis=0)
        e_i, _ = projection_energy(claim, Em, rank_r)
        deltas.append(e_i - full_e)
    return float(np.max(deltas))

def gamma_concentration(contrib: np.ndarray) -> float:
    """
    Γ_conc: concentration of support across evidence items.
    Uses max-share (in [0,1]) where 1 = all support from one sentence.
    """
    s = float(np.sum(contrib))
    if s <= 0:
        return 1.0
    p = contrib / s
    return float(np.max(p))

# ----------------------------
# Dump loading
# ----------------------------
def load_dump(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows

def quantiles(xs: List[float]) -> Dict[str, float]:
    a = np.array(xs, dtype=np.float32)
    return {
        "n": int(a.size),
        "min": float(np.min(a)),
        "p01": float(np.percentile(a, 1)),
        "p10": float(np.percentile(a, 10)),
        "p50": float(np.percentile(a, 50)),
        "p90": float(np.percentile(a, 90)),
        "p99": float(np.percentile(a, 99)),
        "max": float(np.max(a)),
        "mean": float(np.mean(a)),
    }

# ----------------------------
# Main analysis
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", type=Path, required=True)
    ap.add_argument("--cache_db", type=Path, required=True)
    ap.add_argument("--rank_r", type=int, default=8)
    ap.add_argument("--tau_delta", type=float, required=True)
    ap.add_argument("--target_far", type=float, default=0.01)
    ap.add_argument("--out", type=Path, default=Path("artifacts/hard_gamma_report.json"))
    ap.add_argument("--out_csv", type=Path, default=Path("artifacts/delta_vs_gamma.csv"))
    args = ap.parse_args()

    conn = connect_db(args.cache_db)
    rows = load_dump(args.dump)

    # Focus on EVAL split for analysis (you can include CAL too)
    ev = [r for r in rows if str(r.get("split","")).startswith("eval")]
    pos = [r for r in ev if r.get("label") == "POS"]
    neg = [r for r in ev if r.get("label") == "NEG" and r.get("neg_mode") == "hard_mined"]

    if not neg:
        raise SystemExit("No eval hard_mined negatives found in dump. Check dump labels/modes.")

    # You need claim embeddings; if not in dump, fetch from cache via claim_id.
    # For now assume claim embedding is stored as an element_id as well (claim_id in dump).
    # If not, adapt: embed claim text on the fly using your embedder (preferred).
    if "claim_id" not in neg[0]:
        raise SystemExit("Dump must include claim_id (element_id for claim embedding) or claim text + embedder hook.")

    # Compute gamma features
    records = []
    for r in tqdm(pos + neg, desc="Compute gamma features"):
        claim_id = r["claim_id"]
        ev_ids = r.get("evidence_ids", [])

        claim_vecs = fetch_vecs(conn, [claim_id])
        if claim_vecs.shape[0] != 1:
            continue
        c = claim_vecs[0]

        E = fetch_vecs(conn, ev_ids)
        if E.shape[0] == 0:
            continue

        # Delta recompute (optional cross-check)
        delta, contrib = projection_energy(c, E, args.rank_r)
        g_drop = gamma_dropout(c, E, args.rank_r)
        g_conc = gamma_concentration(contrib)

        records.append({
            "label": r["label"],
            "neg_mode": r.get("neg_mode", "real"),
            "delta": float(delta),
            "gamma_drop": float(g_drop),
            "gamma_conc": float(g_conc),
        })

    # Separate arrays
    pos_rec = [x for x in records if x["label"] == "POS"]
    neg_rec = [x for x in records if x["label"] == "NEG" and x["neg_mode"] == "hard_mined"]

    # Choose gamma threshold from hard negatives to meet FAR on gamma *conditional on delta accept*
    # You can use AND/OR; here we show a 2D rule:
    # Accept if delta <= tau_delta_easy AND gamma_drop <= tau_gamma
    neg_cond = [x for x in neg_rec if x["delta"] <= args.tau_delta]
    if not neg_cond:
        # If none are under delta threshold, gamma isn't needed for that tau; but hard case usually has many.
        neg_cond = neg_rec

    gammas = [x["gamma_drop"] for x in neg_cond]
    tau_gamma = float(np.percentile(np.array(gammas, dtype=np.float32), args.target_far * 100.0))

    # Compute FAR/TPR for the 2D rule on eval set
    def accept(x):
        return (x["delta"] <= args.tau_delta) and (x["gamma_drop"] <= tau_gamma)

    far = np.mean([accept(x) for x in neg_rec]) if neg_rec else 0.0
    tpr = np.mean([accept(x) for x in pos_rec]) if pos_rec else 0.0

    out = {
        "params": {
            "rank_r": args.rank_r,
            "tau_delta": args.tau_delta,
            "target_far": args.target_far,
            "tau_gamma_drop": tau_gamma,
        },
        "eval": {
            "pos": {
                "delta": quantiles([x["delta"] for x in pos_rec]),
                "gamma_drop": quantiles([x["gamma_drop"] for x in pos_rec]),
                "gamma_conc": quantiles([x["gamma_conc"] for x in pos_rec]),
            },
            "neg_hard": {
                "delta": quantiles([x["delta"] for x in neg_rec]),
                "gamma_drop": quantiles([x["gamma_drop"] for x in neg_rec]),
                "gamma_conc": quantiles([x["gamma_conc"] for x in neg_rec]),
            },
            "2d_gate": {
                "FAR": float(far),
                "TPR": float(tpr),
                "rule": "accept iff (delta<=tau_delta) AND (gamma_drop<=tau_gamma_drop)",
            }
        }
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")

    # Write CSV for plotting
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", encoding="utf-8") as f:
        f.write("label,neg_mode,delta,gamma_drop,gamma_conc\n")
        for x in records:
            f.write(f"{x['label']},{x['neg_mode']},{x['delta']:.6f},{x['gamma_drop']:.6f},{x['gamma_conc']:.6f}\n")

    print(f"✅ wrote {args.out}")
    print(f"✅ wrote {args.out_csv}")
    print(f"2D gate: FAR={far:.3%}  TPR={tpr:.3%}  tau_gamma_drop={tau_gamma:.6f}")

if __name__ == "__main__":
    main()
