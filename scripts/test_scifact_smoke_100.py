# tests/test_scifact_smoke_100.py
import json
from pathlib import Path
import numpy as np

from verity_gate.embedder import HFEmbedder
from verity_gate.gate import evaluate_claim

ROOT = Path(__file__).resolve().parents[1]
SCIFACT = ROOT / "datasets" / "scifact" / "scifact_dev_rationale.jsonl"

def iter_jsonl(p: Path):
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def pick_claim(row):
    for k in ("claim", "claim_text", "text"):
        v = row.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None

def pick_evidence_texts(row):
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

def test_scifact_smoke_100():
    assert SCIFACT.exists(), f"Missing {SCIFACT}"

    embedder = HFEmbedder()
    energies = []
    decisions = []

    n = 0
    for row in iter_jsonl(SCIFACT):
        if n >= 100:
            break
        claim = pick_claim(row)
        ev = pick_evidence_texts(row)
        if not claim or not ev:
            continue

        claim_vec = embedder.embed([claim])[0]
        ev_vecs = embedder.embed(ev)

        base, decision_fixed, decision_adaptive, *_ = evaluate_claim(
            claim_vec=claim_vec,
            evidence_vecs=ev_vecs,
            regime="standard",
            top_k=min(12, len(ev)),
            rank_r=8,
            embedder=embedder,
            evidence_texts=ev,
        )

        energies.append(float(base.energy))
        decisions.append(decision_fixed)
        n += 1

    assert n >= 50, "Too few usable SciFact rows (need claim + evidence_texts)"
    assert len(set(decisions)) >= 2, "Degenerate: all decisions identical"
    e = np.array(energies)
    assert e.min() >= 0.0 and e.max() <= 1.0, "Energy out of expected [0,1] range"
    assert np.std(e) > 0.02, "Degenerate: energy distribution too narrow"
