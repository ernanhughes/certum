# Project Context: scripts
# Path: C:\Users\ernan\Project\Deterministic-Policy-Gates-for-Stochastic-Systems\scripts
# Generated for AI Review


==================================================
FILE: download_feverous.py
==================================================

from pathlib import Path
import requests


def download_dataset(url: str, dest: Path, *, overwrite: bool = False) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and not overwrite:
        return dest

    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

    return dest

path = download_dataset(
    url="https://fever.ai/download/feverous/feverous_dev_challenges.jsonl",
    dest=Path("datasets/feverous/feverous_dev_challenges.jsonl"),
)
print(f"Downloaded to {path}")


==================================================
FILE: download_hover.py
==================================================

from pathlib import Path
import requests


def download_dataset(url: str, dest: Path, *, overwrite: bool = False) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and not overwrite:
        return dest

    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

    return dest

path = download_dataset(
    url="https://raw.githubusercontent.com/hover-nlp/hover/refs/heads/main/data/hover/hover_dev_release_v1.1.json",
    dest=Path("datasets/hover/hover_dev_release_v1.1.json"),
)
print(f"Downloaded to {path}")





==================================================
FILE: download_scifact.py
==================================================

import tarfile
import io
import requests
from pathlib import Path

URL = "https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz"

def main(out_dir="datasets/scifact_raw"):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("Downloading:", URL)
    r = requests.get(URL, timeout=300)
    r.raise_for_status()

    with tarfile.open(fileobj=io.BytesIO(r.content), mode="r:gz") as tf:
        tf.extractall(out)

    print("Extracted to:", out.resolve())

if __name__ == "__main__":
    main()


==================================================
FILE: prepare_hover_gate.py
==================================================

import json
import re
import requests
from pathlib import Path

SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")

import time
import requests
from urllib.parse import quote

SESSION = requests.Session()
SESSION.headers.update({
    # IMPORTANT: put something real here (repo + contact)
    "User-Agent": "DeterministicPolicyGates/1.0 (contact: ernan@example.com)",
    "Accept": "application/json",
})

def wiki_extract(title: str, *, max_retries: int = 5, base_sleep: float = 0.5) -> str:
    """
    Fetch plaintext extract from Wikipedia via Action API.
    Requires a non-generic User-Agent or Wikimedia may return 403.
    """
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "formatversion": 2,
        "prop": "extracts",
        "explaintext": 1,
        "redirects": 1,
        "titles": title,
    }

    last_status = None
    for attempt in range(max_retries):
        r = SESSION.get(url, params=params, timeout=60)
        last_status = r.status_code

        # Rate limiting / transient errors
        if r.status_code in (429, 503, 502):
            time.sleep(base_sleep * (2 ** attempt))
            continue

        # Forbidden: usually User-Agent policy or network policy
        if r.status_code == 403:
            raise RuntimeError(
                f"Wikipedia API returned 403 for title={title!r}. "
                f"Ensure User-Agent is informative per Wikimedia policy."
            )

        r.raise_for_status()
        data = r.json()

        pages = data.get("query", {}).get("pages", [])
        if not pages:
            return ""

        page = pages[0]
        return (page.get("extract") or "").strip()

    raise RuntimeError(f"Failed to fetch title={title!r} after retries; last_status={last_status}")

def get_sentence(title: str, sent_id: int, cache_dir: Path) -> str | None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    safe = quote(title, safe="")  # safe filename
    cache_file = cache_dir / f"{safe}.txt"


    if cache_file.exists():
        text = cache_file.read_text(encoding="utf-8", errors="ignore")
    else:
        text = wiki_extract(title)
        cache_file.write_text(text, encoding="utf-8")

    sents = SENT_SPLIT.split(text)
    if 0 <= sent_id < len(sents):
        return sents[sent_id].strip()
    return None

def main(in_path="datasets/hover/hover_dev.json", out_path="datasets/hover/hover_dev_gate.jsonl"):
    cache_dir = Path("datasets/hover/wiki_cache")
    src = Path(in_path)

    examples = json.loads(src.read_text(encoding="utf-8"))
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    with out.open("w", encoding="utf-8") as f:
        for ex in examples:
            claim = ex["claim"]
            label = ex.get("label", "")
            # supporting_facts: [[title, sent_id], ...]
            sfs = ex.get("supporting_facts", []) or []

            evidence_texts = []
            for title, sent_id in sfs:
                sent = get_sentence(title, int(sent_id), cache_dir)
                if sent:
                    evidence_texts.append(sent)

            evidence_texts = list(dict.fromkeys(evidence_texts))  # de-dupe, keep order
            if not evidence_texts:
                continue

            row = {
                "dataset": "hover",
                "split": "dev",
                "id": str(ex.get("uid", "")),
                "claim": claim,
                "label": label,
                "evidence_texts": evidence_texts,
                "meta": {"supporting_facts": sfs},
            }
            f.write(json.dumps(row) + "\n")
            n += 1

    print(f"Wrote {n} rows -> {out}")

if __name__ == "__main__":
    main()


==================================================
FILE: prepare_scifact_gate.py
==================================================

import json
from pathlib import Path

def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def main(raw_dir="datasets/scifact_raw/data", split="dev", mode="rationale", out_path=None):
    raw = Path(raw_dir)
    corpus_path = raw / "corpus.jsonl"
    claims_path = raw / f"claims_{split}.jsonl"

    # doc_id -> abstract sentences
    corpus = {}
    for d in read_jsonl(corpus_path):
        corpus[int(d["doc_id"])] = d["abstract"]

    rows = []
    for ex in read_jsonl(claims_path):
        claim_id = ex.get("id", None)
        claim = ex["claim"]

        # evidence: {doc_id: [ {label: "...", sentences: [...]}, ... ] }
        evidence = ex.get("evidence", {}) or {}

        for doc_id_str, ev_list in evidence.items():
            doc_id = int(doc_id_str)
            abstract = corpus.get(doc_id, [])
            for ev in ev_list:
                ev_label = ev.get("label", "")
                sent_ids = ev.get("sentences", []) or []

                if mode == "rationale":
                    evidence_texts = [abstract[i] for i in sent_ids if 0 <= i < len(abstract)]
                elif mode == "abstract":
                    evidence_texts = list(abstract)
                else:
                    raise ValueError("mode must be 'rationale' or 'abstract'")

                if not evidence_texts:
                    continue

                rows.append({
                    "dataset": "scifact",
                    "split": split,
                    "id": str(claim_id) if claim_id is not None else None,
                    "claim": claim,
                    "label": ev_label,
                    "evidence_texts": evidence_texts,
                    "meta": {"doc_id": doc_id, "sent_ids": sent_ids},
                })

    if out_path is None:
        out_path = f"datasets/scifact/scifact_{split}_{mode}.jsonl"

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    print(f"Wrote {len(rows)} rows -> {out}")

if __name__ == "__main__":
    main()


==================================================
FILE: prepare_scifact_gate_hf.py
==================================================

import json
from pathlib import Path
from datasets import load_dataset

def main(split="validation", mode="rationale", out_path=None):
    # Load SciFact
    claims = load_dataset("allenai/scifact", "claims", split=split)
    corpus = load_dataset("allenai/scifact", "corpus", split="train")

    # doc_id -> abstract sentences
    doc_abs = {int(d["doc_id"]): d["abstract"] for d in corpus}

    rows = []
    for ex in claims:
        claim = ex["claim"]
        doc_id = int(ex["evidence_doc_id"]) if ex.get("evidence_doc_id") else None
        sent_ids = ex.get("evidence_sentences") or []
        label = ex.get("evidence_label") or ""

        if doc_id is None or doc_id not in doc_abs:
            continue

        abstract = doc_abs[doc_id]
        if mode == "rationale":
            evidence_texts = [abstract[i] for i in sent_ids if 0 <= i < len(abstract)]
        elif mode == "abstract":
            evidence_texts = list(abstract)
        else:
            raise ValueError("mode must be 'rationale' or 'abstract'")

        if not evidence_texts:
            continue

        rows.append({
            "dataset": "scifact",
            "split": split,
            "id": str(ex["id"]),
            "claim": claim,
            "label": label,
            "evidence_texts": evidence_texts,
            "meta": {"doc_id": doc_id, "sent_ids": sent_ids},
        })

    if out_path is None:
        out_path = f"datasets/scifact/scifact_{split}_{mode}.jsonl"

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    print(f"Wrote {len(rows)} rows -> {out}")

if __name__ == "__main__":
    main()


==================================================
FILE: prepare_scifact_gate_release.py
==================================================

import json
from pathlib import Path

def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def main(split="dev", mode="rationale",
         raw_dir="datasets/scifact_raw/data",
         out_path=None):

    raw = Path(raw_dir)
    corpus_path = raw / "corpus.jsonl"
    claims_path = raw / f"claims_{split}.jsonl"

    # doc_id -> abstract sentences (list[str])
    corpus = {}
    for d in read_jsonl(corpus_path):
        corpus[int(d["doc_id"])] = d["abstract"]

    rows = []
    for ex in read_jsonl(claims_path):
        claim_id = ex.get("id")
        claim = ex["claim"]

        # evidence is a mapping: doc_id -> list[ {label, sentences:[idx]} ]
        evidence = ex.get("evidence") or {}
        for doc_id_str, ev_list in evidence.items():
            doc_id = int(doc_id_str)
            abstract = corpus.get(doc_id, [])
            if not abstract:
                continue

            for ev in ev_list:
                label = ev.get("label", "")
                sent_ids = ev.get("sentences") or []

                if mode == "rationale":
                    evidence_texts = [abstract[i] for i in sent_ids if 0 <= i < len(abstract)]
                elif mode == "abstract":
                    evidence_texts = list(abstract)
                else:
                    raise ValueError("mode must be 'rationale' or 'abstract'")

                if not evidence_texts:
                    continue

                rows.append({
                    "dataset": "scifact",
                    "split": split,
                    "id": str(claim_id) if claim_id is not None else None,
                    "claim": claim,
                    "label": label,
                    "evidence_texts": evidence_texts,
                    "meta": {"doc_id": doc_id, "sent_ids": sent_ids},
                })

    if out_path is None:
        out_path = f"datasets/scifact/scifact_{split}_{mode}.jsonl"

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Wrote {len(rows)} rows -> {out}")

if __name__ == "__main__":
    main()


==================================================
FILE: smoke_scifact_gate_100.py
==================================================

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


==================================================
FILE: test_scifact_smoke_100.py
==================================================

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


==================================================
FILE: tree.py
==================================================

#!/usr/bin/env python3
"""
dir_tree.py — portable repo tree dumper (text or JSON)

Features
- Cross-platform (Windows/macOS/Linux), no third-party deps
- Exclude patterns (glob): ".git,node_modules,__pycache__,*.egg-info,.venv,venv,dist,build"
- Depth limit (default 4)
- Show aggregate sizes and file/dir counts
- Output formats: text (tree) or JSON
- Optionally only list directories
- Optional anonymization by hashing names for public sharing

Examples
  python dir_tree.py --root . --out repo-tree.txt
  python dir_tree.py --root . --format json --out repo-tree.json
  python dir_tree.py --root . --max-depth 5 --only-dirs --show-sizes
  python dir_tree.py --root . --exclude ".git,node_modules,dist,build,__pycache__,.venv,*.egg-info" --out repo-tree.txt
  python dir_tree.py --root . --anonymize --format json --out repo-anon.json
"""
import argparse
import fnmatch
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

def _hash(data: bytes, algo: str = "sha256") -> str:
    """Internal helper that does the actual hashing."""
    hasher = hashlib.new(algo)
    hasher.update(data)
    return hasher.hexdigest()


def hash_text(text: str, algorithm: str = "sha256") -> str:
    """
    Generate a hash for the given text using the specified algorithm.

    Args:
        text (str): The input text to hash.
        algorithm (str): Hash algorithm, e.g., 'sha256', 'sha1', or 'md5'.

    Returns:
        str: The hexadecimal digest of the hash.

    Usage:
        digest = hash_text("hello world")
    """
    if text is None:
        text = ""
    if not isinstance(text, str):
        raise TypeError(f"hash_text expected str, got {type(text)}")
    return _hash(text.encode("utf-8"), algo=algorithm)

def hash_list(names: List[str]) -> str:
    h = hashlib.sha256()
    for n in names:
        h.update((n + "\n").encode("utf-8"))
    return h.hexdigest()[:16]

def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


DEFAULT_EXCLUDES = [
    ".git", ".hg", ".svn",
    ".venv", "venv", ".tox",
    "node_modules",
    "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    "dist", "build", ".eggs", "*.egg-info",
    ".ipynb_checkpoints", ".idea", ".DS_Store",
]

def human_size(n: int) -> str:
    units = ["B","KB","MB","GB","TB"]
    i = 0
    f = float(n)
    while f >= 1024 and i < len(units)-1:
        f /= 1024.0
        i += 1
    return f"{f:.1f} {units[i]}"

def sha8(s: str) -> str:
    return hash_text(s)[:8]

@dataclass
class Node:
    name: str
    path: str
    type: str  # "dir" | "file"
    size: int = 0
    files: int = 0
    dirs: int = 0
    children: Optional[List["Node"]] = None
    pruned: int = 0  # children not shown due to depth limit

    def to_dict(self, anonymize: bool = False) -> Dict[str, Any]:
        nm = self.name
        pth = self.path
        if anonymize:
            nm = f"{'D' if self.type=='dir' else 'F'}-{sha8(self.name)}"
            pth = "/".join("H-"+sha8(seg) for seg in self.path.replace("\\","/").split("/"))
        d = {
            "name": nm,
            "path": pth,
            "type": self.type,
            "size": self.size,
            "files": self.files,
            "dirs": self.dirs,
            "pruned": self.pruned,
        }
        if self.children is not None:
            d["children"] = [c.to_dict(anonymize=anonymize) for c in self.children]
        return d

def compile_patterns(patterns_csv: str) -> List[str]:
    pats: List[str] = []
    for raw in patterns_csv.split(","):
        s = raw.strip()
        if s:
            pats.append(s)
    return pats

def is_excluded(rel_path: str, name: str, patterns: List[str], include_hidden: bool) -> bool:
    # Hidden file/dir (starts with .) handling
    if not include_hidden and name.startswith("."):
        return True
    # Glob match by name and by relative path
    for pat in patterns:
        if fnmatch.fnmatch(name, pat) or fnmatch.fnmatch(rel_path, pat):
            return True
        # Also check any path segment match (common for folder globs)
        parts = rel_path.replace("\\", "/").split("/")
        if any(fnmatch.fnmatch(seg, pat) for seg in parts):
            return True
    return False

def safe_getsize(path: str) -> int:
    try:
        return os.path.getsize(path)
    except OSError:
        return 0

def scan(root: str,
         rel_root: str,
         max_depth: int,
         depth: int,
         excludes: List[str],
         include_hidden: bool,
         only_dirs: bool,
         follow_symlinks: bool) -> Node:
    name = os.path.basename(root) or root
    node = Node(name=name, path=rel_root or ".", type="dir", size=0, files=0, dirs=1, children=[])
    try:
        entries = list(os.scandir(root))
    except OSError:
        # unreadable directory
        node.pruned = 0
        return node

    # Sort: directories first, then files, alphabetically
    entries.sort(key=lambda e: (not e.is_dir(follow_symlinks=follow_symlinks), e.name.lower()))

    if depth >= max_depth:
        # We won't descend further; just count how many we’re not showing
        node.pruned = len(entries)
        # Still accumulate summary sizes/counts shallowly
        for e in entries:
            rel = os.path.join(rel_root, e.name) if rel_root else e.name
            if is_excluded(rel, e.name, excludes, include_hidden):
                continue
            if e.is_file(follow_symlinks=False):
                node.files += 1
                node.size += safe_getsize(e.path)
            elif e.is_dir(follow_symlinks=follow_symlinks):
                node.dirs += 1
        return node

    for e in entries:
        rel = os.path.join(rel_root, e.name) if rel_root else e.name
        if is_excluded(rel, e.name, excludes, include_hidden):
            continue
        try:
            if e.is_dir(follow_symlinks=follow_symlinks):
                child = scan(e.path, rel, max_depth, depth+1, excludes, include_hidden, only_dirs, follow_symlinks)
                node.size += child.size
                node.files += child.files
                node.dirs += child.dirs
                if node.children is not None:
                    node.children.append(child)
            elif not only_dirs and e.is_file(follow_symlinks=False):
                fsize = safe_getsize(e.path)
                node.size += fsize
                node.files += 1
                if node.children is not None:
                    node.children.append(Node(name=e.name, path=rel, type="file", size=fsize, files=1, dirs=0))
        except OSError:
            # Skip problematic entries
            continue
    return node

def print_tree(node: Node, show_sizes: bool, out, prefix: str = "", is_last: bool = True):
    connector = "└── " if is_last else "├── "
    label = node.name
    if show_sizes:
        if node.type == "dir":
            label += f"  [dirs:{node.dirs-1 if node.dirs>0 else 0} files:{node.files} size:{human_size(node.size)}]"
        else:
            label += f"  [{human_size(node.size)}]"
    if node.pruned:
        label += f"  …(+{node.pruned} more)"
    out.write(prefix + connector + label + "\n")

    if node.children:
        new_prefix = prefix + ("    " if is_last else "│   ")
        for i, ch in enumerate(node.children):
            print_tree(ch, show_sizes, out, new_prefix, i == len(node.children)-1)

def main():
    ap = argparse.ArgumentParser(description="Dump a directory tree (portable, no deps).")
    ap.add_argument("--root", default="./packages", help="Root directory to scan")
    ap.add_argument("--max-depth", type=int, default=6, help="Max depth to display (default: 4)")
    ap.add_argument("--exclude", default=",".join(DEFAULT_EXCLUDES), help="Comma-separated glob patterns to exclude")
    ap.add_argument("--include-hidden", action="store_true", help="Include dotfiles/directories")
    ap.add_argument("--only-dirs", action="store_true", help="List directories only (no files)")
    ap.add_argument("--follow-symlinks", action="store_true", help="Follow symlinks for directories")
    ap.add_argument("--format", choices=["text", "json"], default="text", help="Output format")
    ap.add_argument("--show-sizes", action="store_true", help="Show human-readable sizes in text mode")
    ap.add_argument("--anonymize", action="store_true", help="Hash names/paths for sharing publicly")
    ap.add_argument("--out", default="-", help="Output file path or '-' for stdout")
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    if not os.path.isdir(root):
        print(f"error: not a directory: {root}", file=sys.stderr)
        sys.exit(1)

    rel_root = os.path.basename(root.rstrip(os.sep))
    excludes = compile_patterns(args.exclude) if args.exclude else []

    node = scan(root, rel_root, args.max_depth, 0, excludes, args.include_hidden, args.only_dirs, args.follow_symlinks)

    if args.anonymize:
        # Convert to dict with anonymized names/paths
        data = node.to_dict(anonymize=True)
    else:
        data = node.to_dict(anonymize=False)

    # Output
    if args.out == "-":
        out = sys.stdout
        must_close = False
    else:
        out = open(args.out, "w", encoding="utf-8")
        must_close = True

    try:
        if args.format == "json":
            json.dump(data, out, indent=2)
            out.write("\n")
        else:
            # Print header line with totals
            header = f"{data['name']}  [dirs:{data['dirs']-1 if data['dirs']>0 else 0} files:{data['files']} size:{human_size(data['size'])}]"
            out.write(header + ("\n" if not header.endswith("\n") else ""))
            # Render tree
            # Rebuild Node from dict (for simplicity reuse print logic on the original node)
            print_tree(node if not args.anonymize else Node(**{
                "name": data["name"], "path": data["path"], "type": data["type"], "size": data["size"],
                "files": data["files"], "dirs": data["dirs"], "children": [],
                "pruned": data.get("pruned", 0)
            }), args.show_sizes, out, prefix="", is_last=True)
            # The header is shown; tree root printed as last line too for consistency.
    finally:
        if must_close:
            out.close()

if __name__ == "__main__":
    main()


==================================================
FILE: zip_project.py
==================================================

import os
import zipfile

EXCLUDE_DIRS = {"venv", "__pycache__", ".ruff_cache", ".git", ".idea", ".vscode"}
EXCLUDE_SUFFIXES = (".pyc", ".egg-info")
EXCLUDE_FILES = (".DS_Store",)

def should_exclude(path):
    parts = path.split(os.sep)
    return any(part in EXCLUDE_DIRS or part.endswith(EXCLUDE_SUFFIXES) for part in parts)

def zip_project_directory(source_dir: str, output_filename: str = "verity.zip"):
    with zipfile.ZipFile(output_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(source_dir):
            # Remove excluded directories from the walk
            dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS and not d.endswith(EXCLUDE_SUFFIXES)]
            for file in files:
                if file in EXCLUDE_FILES or file.endswith(EXCLUDE_SUFFIXES):
                    continue
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, source_dir)
                if should_exclude(rel_path):
                    continue
                zipf.write(full_path, rel_path)
    print(f"✅ Project zipped to {output_filename}")

# Example usage
if __name__ == "__main__":
    zip_project_directory("packages/")

