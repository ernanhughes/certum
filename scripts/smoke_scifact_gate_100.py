from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

# Matches your repo’s canonical embedder + gate surface
from verity_gate.embedder import HFEmbedder
from verity_gate.gate import evaluate_claim


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_scifact_corpus(corpus_path: Path) -> Dict[str, List[str]]:
    """
    SciFact corpus.jsonl lines typically contain:
      { "doc_id": int, "title": str, "abstract": [sentence, ...], ... }
    """
    corpus: Dict[str, List[str]] = {}
    for row in iter_jsonl(corpus_path):
        doc_id = row.get("doc_id", row.get("id"))
        if doc_id is None:
            continue
        doc_id = str(doc_id)
        abstract = row.get("abstract") or []
        if isinstance(abstract, str):
            # fallback if someone stored it as one blob
            abstract = [s.strip() for s in abstract.split(".") if s.strip()]
        if not isinstance(abstract, list):
            abstract = []
        corpus[doc_id] = [str(s) for s in abstract if str(s).strip()]
    return corpus


def pick_claim_text(row: dict) -> Optional[str]:
    for k in ("claim", "claim_text", "text"):
        v = row.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def pick_evidence_texts_from_row(row: dict) -> Optional[List[str]]:
    """
    If your scifact_dev_rationale.jsonl already contains rationale/evidence text,
    use it directly (works even without corpus.jsonl).
    """
    for k in ("evidence_texts", "rationale", "rationale_texts", "evidence_sentence_texts"):
        v = row.get(k)
        if isinstance(v, list) and v and all(isinstance(x, str) for x in v):
            out = [x.strip() for x in v if x.strip()]
            if out:
                return out
    v = row.get("evidence_text")
    if isinstance(v, str) and v.strip():
        return [v.strip()]
    return None


def evidence_from_corpus(
    row: dict, corpus: Dict[str, List[str]]
) -> Optional[Tuple[str, List[str]]]:
    """
    Matches the schema in the SciFact dataset script: evidence_doc_id + evidence_sentences. :contentReference[oaicite:4]{index=4}
    """
    doc_id = row.get("evidence_doc_id", row.get("doc_id"))
    if doc_id is None:
        return None
    doc_id = str(doc_id)

    sent_ids = row.get("evidence_sentences") or row.get("sent_ids") or row.get("sentences") or []
    if not isinstance(sent_ids, list):
        sent_ids = []

    abstract = corpus.get(doc_id)
    if not abstract:
        # try integer-normalized lookup
        try:
            abstract = corpus.get(str(int(doc_id)))
        except Exception:
            abstract = None
    if not abstract:
        return None

    picked: List[str] = []
    for s in sent_ids:
        try:
            idx = int(s)
        except Exception:
            continue
        if 0 <= idx < len(abstract):
            picked.append(abstract[idx])

    if picked:
        return (doc_id, picked)

    # fallback: if sentence ids missing, take first few sentences
    fallback = abstract[:10]
    return (doc_id, fallback) if fallback else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--claims", type=Path, required=True, help="e.g. datasets/scifact/scifact_dev_rationale.jsonl")
    ap.add_argument("--corpus", type=Path, default=None, help="Optional: SciFact corpus.jsonl")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--regime", type=str, default="standard", choices=["standard", "strict", "lenient"])
    ap.add_argument("--top_k", type=int, default=12)
    ap.add_argument("--rank_r", type=int, default=8)
    ap.add_argument("--out", type=Path, default=Path("artifacts/scifact_smoke_100.jsonl"))
    args = ap.parse_args()

    corpus = load_scifact_corpus(args.corpus) if args.corpus else None
    embedder = HFEmbedder()

    results: List[dict] = []
    fixed_counts = Counter()
    adaptive_counts = Counter()
    energies: List[float] = []
    gaps: List[float] = []

    for row in iter_jsonl(args.claims):
        if len(results) >= args.n:
            break

        claim = pick_claim_text(row)
        if not claim:
            continue

        evidence_texts = pick_evidence_texts_from_row(row)

        doc_id = None
        if evidence_texts is None and corpus is not None:
            got = evidence_from_corpus(row, corpus)
            if got is not None:
                doc_id, evidence_texts = got

        if not evidence_texts:
            # skip if we still have nothing usable
            continue

        claim_vec = embedder.embed([claim])[0]
        ev_vecs = embedder.embed(evidence_texts)

        top_k = min(args.top_k, len(evidence_texts))
        base, decision_fixed, decision_adaptive, probe, oracle_energy, energy_gap = evaluate_claim(
            claim_vec=claim_vec,
            evidence_vecs=ev_vecs,
            regime=args.regime,
            top_k=top_k,
            rank_r=args.rank_r,
            embedder=embedder,
            evidence_texts=evidence_texts,
        )

        fixed_counts[decision_fixed] += 1
        adaptive_counts[decision_adaptive] += 1
        energies.append(float(base.energy))
        if oracle_energy is not None and energy_gap is not None:
            gaps.append(float(energy_gap))

        results.append(
            {
                "claim": claim,
                "doc_id": doc_id,
                "evidence_texts": evidence_texts,
                "energy": float(base.energy),
                "oracle_energy": None if oracle_energy is None else float(oracle_energy),
                "energy_gap": None if energy_gap is None else float(energy_gap),
                "decision_fixed": decision_fixed,
                "decision_adaptive": decision_adaptive,
                "probe": probe,
                "meta": {
                    "regime": args.regime,
                    "top_k": top_k,
                    "rank_r": args.rank_r,
                },
            }
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    if not results:
        print("No usable examples found. If your JSONL only has ids, pass --corpus corpus.jsonl.")
        return

    e = np.array(energies, dtype=np.float64)
    print(f"\nLoaded {len(results)} examples from {args.claims}")
    print(f"Regime={args.regime} top_k={args.top_k} rank_r={args.rank_r}")
    print("\nDecision counts (fixed):", dict(fixed_counts))
    print("Decision counts (adaptive):", dict(adaptive_counts))
    print("\nEnergy percentiles: p50=%.3f p90=%.3f p95=%.3f p99=%.3f" % tuple(np.percentile(e, [50, 90, 95, 99])))
    print("Energy mean/std: %.3f / %.3f" % (e.mean(), e.std()))
    if gaps:
        g = np.array(gaps, dtype=np.float64)
        print("\nEnergy-gap percentiles: p50=%.3f p90=%.3f p95=%.3f p99=%.3f" % tuple(np.percentile(g, [50, 90, 95, 99])))
        print("Energy-gap mean/std: %.3f / %.3f" % (g.mean(), g.std()))

    # Show top 5 highest-energy cases so you can *see* it differs from FEVEROUS
    top = sorted(results, key=lambda r: r["energy"], reverse=True)[:5]
    print("\nTop-5 by energy:")
    for i, r in enumerate(top, 1):
        c = (r["claim"][:120] + "…") if len(r["claim"]) > 120 else r["claim"]
        print(f"{i:02d}) E={r['energy']:.3f} gap={r['energy_gap']} fixed={r['decision_fixed']} adapt={r['decision_adaptive']} :: {c}")

    print(f"\nWrote: {args.out}")


if __name__ == "__main__":
    main()
