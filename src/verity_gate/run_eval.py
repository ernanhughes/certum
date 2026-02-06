# src/verity_gate/run_eval.py
from pathlib import Path
import json
import numpy as np

from verity_gate.dataset import load_feverous, extract_evidence
from verity_gate.embedder import HFEmbedder
from verity_gate.gate import evaluate_claim


DATASET = Path("datasets/feverous/feverous_dev_challenges.jsonl")
OUT_PATH = Path("artifacts/results_full.jsonl")


def main():
    embedder = HFEmbedder()
    results = []

    for ex in load_feverous(DATASET):
        evidence = extract_evidence(ex)
        if not evidence:
            continue

        claim = ex["claim"]

        # --------------------------------------------------
        # Embeddings
        # --------------------------------------------------
        ev_vecs = embedder.embed(evidence)
        claim_vec = embedder.embed([claim])[0]

        # --------------------------------------------------
        # Oracle vector (grounded synthesis)
        # NOTE:
        # For FEVEROUS, we treat the *mean evidence embedding*
        # as the oracle-supported semantic span.
        # --------------------------------------------------
        oracle_vec = np.mean(ev_vecs, axis=0)

        # --------------------------------------------------
        # Evaluation
        # --------------------------------------------------
        base, decision, probe, oracle_energy, energy_gap = evaluate_claim(
            claim_vec,
            ev_vecs,
            oracle_vec=oracle_vec,
            regime="standard",
            top_k=12,
            rank_r=8,
        )

        # Check 2: Energy ordering consistency
        assert oracle_energy <= base.energy + 1e-6

        # --------------------------------------------------
        # Record result row
        # --------------------------------------------------
        row = {
            "claim": claim,

            # Core energy metrics
            "energy": base.energy,
            "coverage": base.explained,
            "oracle_energy": oracle_energy,
            "energy_gap": energy_gap,

            # Structural diagnostics
            "effective_rank": base.rank_r,
            "topk": base.topk,
            "rank_r": base.rank_r,

            # Robustness
            "energy_probe": probe,
            "energy_probe_var": float(np.var(probe)),

            # Policy outcome
            "decision": decision,
        }

        results.append(row)

    # --------------------------------------------------
    # Write results
    # --------------------------------------------------
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"✅ Wrote {len(results)} rows → {OUT_PATH}")


if __name__ == "__main__":
    main()
