# src/verity_gate/dataset.py
import json
from pathlib import Path
from typing import Iterator, Dict

def load_feverous(path: Path) -> Iterator[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def extract_evidence(example: Dict) -> list[str]:
    texts = []
    for ev in example.get("evidence", []):
        for ctx in ev.get("context", {}).values():
            texts.extend(ctx)
    return list(set(texts))
