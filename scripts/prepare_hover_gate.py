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
