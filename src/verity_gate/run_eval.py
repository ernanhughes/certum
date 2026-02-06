# src/verity_gate/run_eval.py
from pathlib import Path
from verity_gate.dataset import load_feverous, extract_evidence
from verity_gate.embedder import HFEmbedder
from verity_gate.gate import evaluate_claim
import json
import numpy as np

DATASET = Path("datasets/feverous/feverous_dev_challenges.jsonl")

def main():
    embedder = HFEmbedder()
    results = []

    for ex in load_feverous(DATASET):
        evidence = extract_evidence(ex)
        if not evidence:
            continue

        claim = ex["claim"]

        ev_vecs = embedder.embed(evidence)
        claim_vec = embedder.embed([claim])[0]

        res, decision, probe = evaluate_claim(
            claim_vec,
            ev_vecs,
            regime="standard",
            top_k=12,
            rank_r=8,
        )

        results.append({
            "claim": claim,
            "energy": res.energy,
            "coverage": res.explained,
            "identity_error": res.identity_error,
            "effective_rank": res.effective_rank,
            "topk": res.topk,
            "rank_r": res.rank_r,
            "energy_probe": probe,
            "energy_probe_var": float(np.var(probe)),
            "decision": decision,
        })

    with open("artifacts/results_full.jsonl", "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

if __name__ == "__main__":
    main()
