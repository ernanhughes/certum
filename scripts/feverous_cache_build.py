from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
from tqdm import tqdm


# ============================================================
# Evidence-id parsing
# ============================================================

# FEVEROUS element IDs look like:
#   "Algebraic logic_sentence_0"
#   "Algebraic logic_cell_0_1_1"
#   "Algebraic logic_header_cell_0_0_1"
#   "Algebraic logic_section_4"
#   "Algebraic logic_title"
#   "The Discoverie of Witchcraft_table_caption_0"
#
# We parse by matching the LAST evidence-type token.
EID_RE = re.compile(
    r"^(?P<page>.+)_(?P<kind>sentence|cell|header_cell|item|section|table_caption|title)(?:_(?P<rest>.*))?$"
)

def _norm(s: str) -> str:
    return unicodedata.normalize("NFC", s).strip()

def split_evidence_id(eid: str) -> Tuple[Optional[str], Optional[str]]:
    """
    "Manchester Cancer Research Centre_sentence_0" -> ("Manchester Cancer Research Centre", "sentence_0")
    "Some Page_title" -> ("Some Page", "title")
    """
    eid = _norm(eid)
    m = EID_RE.match(eid)
    if not m:
        return None, None

    page = _norm(m.group("page"))
    kind = m.group("kind")
    rest = m.group("rest")

    if kind == "title":
        return page, "title"

    if not rest:
        return None, None

    local = f"{kind}_{rest}"
    return page, _norm(local)


# ============================================================
# Simple wiki markup cleanup
# ============================================================

WIKI_LINK_RE = re.compile(r"\[\[([^\]|]+)\|([^\]]+)\]\]|\[\[([^\]]+)\]\]")

def strip_wiki_markup(text: str) -> str:
    """
    [[A|B]] -> B
    [[A]] -> A
    """
    def repl(m: re.Match) -> str:
        if m.group(2):
            return m.group(2)
        return m.group(3) or m.group(1) or ""
    return WIKI_LINK_RE.sub(repl, text)


# ============================================================
# Validation
# ============================================================

def is_valid_text(text: str, *, min_chars: int = 5, max_chars: int = 20000) -> bool:
    if not text:
        return False
    t = text.strip()
    if len(t) < min_chars:
        return False
    if len(t) > max_chars:
        return False
    if not re.search(r"[A-Za-z0-9]", t):
        return False
    return True

def sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()


# ============================================================
# Wiki resolver (reads your feverous_wikiv1.db)
# ============================================================

class WikiSqliteResolver:
    """
    Expects SQLite:
      CREATE TABLE wiki (id PRIMARY KEY, data json)
    where wiki.id == Wikipedia page title (text),
    wiki.data == JSON string for that page.
    """

    def __init__(self, wiki_db_path: str | Path):
        self.wiki_db_path = str(wiki_db_path)
        self.conn = sqlite3.connect(self.wiki_db_path)
        self.conn.row_factory = sqlite3.Row

    def close(self) -> None:
        self.conn.close()

    def get_page_json(self, page_id: str) -> Optional[dict]:
        page_id = _norm(page_id)
        cur = self.conn.execute("SELECT data FROM wiki WHERE id = ?", (page_id,))
        row = cur.fetchone()
        if not row:
            return None
        raw = row["data"]
        try:
            return json.loads(raw)
        except Exception:
            return None

    def resolve_local(self, page_json: dict, local_id: str) -> Optional[str]:
        # title
        if local_id == "title":
            v = page_json.get("title")
            return str(v) if isinstance(v, str) else None

        # sentence_#
        if local_id.startswith("sentence_"):
            v = page_json.get(local_id)
            return str(v) if isinstance(v, str) else None

        # section_# -> {"value": "...", "level": n}
        if local_id.startswith("section_"):
            sec = page_json.get(local_id)
            if isinstance(sec, dict):
                v = sec.get("value")
                return str(v) if isinstance(v, str) else None
            return None

        # item_*_* lives inside list_*
        if local_id.startswith("item_"):
            for k, v in page_json.items():
                if not k.startswith("list_") or not isinstance(v, dict):
                    continue
                items = v.get("list")
                if not isinstance(items, list):
                    continue
                for it in items:
                    if isinstance(it, dict) and it.get("id") == local_id:
                        val = it.get("value")
                        return str(val) if isinstance(val, str) else None
            return None

        # cell/header_cell live inside table_*["table"]
        if local_id.startswith(("cell_", "header_cell_")):
            for k, v in page_json.items():
                if not k.startswith("table_") or not isinstance(v, dict):
                    continue
                table = v.get("table")
                if not isinstance(table, list):
                    continue
                for row in table:
                    if not isinstance(row, list):
                        continue
                    for cell in row:
                        if isinstance(cell, dict) and cell.get("id") == local_id:
                            val = cell.get("value")
                            return str(val) if isinstance(val, str) else None
            return None

        # table_caption_# stored as table_#["caption"]
        if local_id.startswith("table_caption_"):
            try:
                idx = int(local_id.split("_")[-1])
            except Exception:
                return None
            tkey = f"table_{idx}"
            t = page_json.get(tkey)
            if isinstance(t, dict):
                cap = t.get("caption")
                return str(cap) if isinstance(cap, str) else None
            return None

        return None

    def resolve_evidence_id(self, evidence_id: str) -> Tuple[str, str, Optional[str], Optional[str]]:
        """
        Returns: (page_id, local_id, text, error)
        """
        page_id, local_id = split_evidence_id(evidence_id)
        if not page_id or not local_id:
            return "", "", None, "bad_evidence_id_format"

        page_json = self.get_page_json(page_id)
        if page_json is None:
            return page_id, local_id, None, "page_not_found_or_bad_json"

        text = self.resolve_local(page_json, local_id)
        if text is None:
            return page_id, local_id, None, "local_id_not_found"

        return page_id, local_id, text, None


# ============================================================
# Cache DB (resolved + embeddings)
# ============================================================

class EvidenceCacheDB:
    def __init__(self, cache_db_path: str | Path):
        self.cache_db_path = str(cache_db_path)
        self.conn = sqlite3.connect(self.cache_db_path)
        self.conn.row_factory = sqlite3.Row
        self._init()

    def close(self) -> None:
        self.conn.close()

    def _init(self) -> None:
        cur = self.conn.cursor()
        cur.execute("PRAGMA journal_mode=WAL;")
        cur.execute("PRAGMA synchronous=NORMAL;")
        cur.execute("PRAGMA temp_store=MEMORY;")

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS resolved_items (
              evidence_id TEXT PRIMARY KEY,
              page_id TEXT NOT NULL,
              local_id TEXT NOT NULL,
              text TEXT,
              text_hash TEXT,
              status TEXT NOT NULL,          -- ok | failed
              error TEXT,
              cleaned INTEGER NOT NULL DEFAULT 0,
              created_at REAL NOT NULL,
              updated_at REAL NOT NULL
            );
            """
        )

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS embeddings (
              evidence_id TEXT NOT NULL,
              model_name TEXT NOT NULL,
              dim INTEGER NOT NULL,
              dtype TEXT NOT NULL,           -- float32
              vector BLOB NOT NULL,
              created_at REAL NOT NULL,
              PRIMARY KEY (evidence_id, model_name),
              FOREIGN KEY (evidence_id) REFERENCES resolved_items(evidence_id)
            );
            """
        )

        cur.execute("CREATE INDEX IF NOT EXISTS idx_resolved_status ON resolved_items(status);")
        self.conn.commit()

    def get_resolved_row(self, evidence_id: str) -> Optional[sqlite3.Row]:
        cur = self.conn.execute(
            "SELECT * FROM resolved_items WHERE evidence_id = ?",
            (evidence_id,),
        )
        return cur.fetchone()

    def upsert_resolved_ok(self, evidence_id: str, page_id: str, local_id: str, text: str, *, cleaned: bool) -> None:
        now = time.time()
        self.conn.execute(
            """
            INSERT INTO resolved_items(evidence_id, page_id, local_id, text, text_hash, status, error, cleaned, created_at, updated_at)
            VALUES(?,?,?,?,?,'ok',NULL,?,?,?)
            ON CONFLICT(evidence_id) DO UPDATE SET
              page_id=excluded.page_id,
              local_id=excluded.local_id,
              text=excluded.text,
              text_hash=excluded.text_hash,
              status='ok',
              error=NULL,
              cleaned=excluded.cleaned,
              updated_at=excluded.updated_at
            """,
            (evidence_id, page_id, local_id, text, sha1_text(text), int(cleaned), now, now),
        )

    def upsert_resolved_failed(self, evidence_id: str, page_id: str, local_id: str, error: str) -> None:
        now = time.time()
        self.conn.execute(
            """
            INSERT INTO resolved_items(evidence_id, page_id, local_id, text, text_hash, status, error, cleaned, created_at, updated_at)
            VALUES(?,?,?,NULL,NULL,'failed',?,0,?,?)
            ON CONFLICT(evidence_id) DO UPDATE SET
              page_id=excluded.page_id,
              local_id=excluded.local_id,
              status='failed',
              error=excluded.error,
              updated_at=excluded.updated_at
            """,
            (evidence_id, page_id, local_id, error, now, now),
        )

    def has_embedding(self, evidence_id: str, model_name: str) -> bool:
        cur = self.conn.execute(
            "SELECT 1 FROM embeddings WHERE evidence_id=? AND model_name=?",
            (evidence_id, model_name),
        )
        return cur.fetchone() is not None

    def upsert_embedding(self, evidence_id: str, model_name: str, vec: np.ndarray) -> None:
        vec = np.asarray(vec, dtype=np.float32)
        now = time.time()
        self.conn.execute(
            """
            INSERT INTO embeddings(evidence_id, model_name, dim, dtype, vector, created_at)
            VALUES(?,?,?,?,?,?)
            ON CONFLICT(evidence_id, model_name) DO UPDATE SET
              dim=excluded.dim,
              dtype=excluded.dtype,
              vector=excluded.vector,
              created_at=excluded.created_at
            """,
            (evidence_id, model_name, int(vec.size), "float32", vec.tobytes(), now),
        )


# ============================================================
# Embedder (SentenceTransformers)
# ============================================================

class STEmbedder:
    def __init__(self, model_name: str, batch_size: int = 64):
        self.model_name = model_name
        self.batch_size = batch_size
        from sentence_transformers import SentenceTransformer  # type: ignore
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        arr = self.model.encode(
            list(texts),
            batch_size=self.batch_size,
            show_progress_bar=False,
            normalize_embeddings=False,
        )
        return np.asarray(arr, dtype=np.float32)


# ============================================================
# FEVEROUS parsing helpers
# ============================================================

def collect_evidence_ids(example: dict, include_context: bool) -> Set[str]:
    ids: Set[str] = set()

    ev = example.get("evidence", [])
    # some files contain a schema line with evidence = "" (string)
    if isinstance(ev, str):
        return ids
    if isinstance(ev, dict):
        ev = [ev]

    if not isinstance(ev, list):
        return ids

    for evset in ev:
        if not isinstance(evset, dict):
            continue

        content = evset.get("content", [])
        if isinstance(content, list):
            for x in content:
                if isinstance(x, str) and x:
                    ids.add(_norm(x))

        if include_context:
            ctx = evset.get("context", {})
            if isinstance(ctx, dict):
                for k, v in ctx.items():
                    if isinstance(k, str) and k:
                        ids.add(_norm(k))
                    if isinstance(v, list):
                        for y in v:
                            if isinstance(y, str) and y:
                                ids.add(_norm(y))

    return ids


# ============================================================
# Cache builder
# ============================================================

@dataclass
class CacheStats:
    requested: int = 0
    embedding_hits: int = 0
    embedding_new: int = 0
    resolved_hit_ok: int = 0
    resolved_hit_failed: int = 0
    resolved_new_ok: int = 0
    resolved_new_failed: int = 0

    def to_dict(self) -> Dict[str, float | int]:
        hit_rate = (self.embedding_hits / self.requested) if self.requested else 0.0
        resolve_ok = self.resolved_hit_ok + self.resolved_new_ok
        resolve_rate = (resolve_ok / self.requested) if self.requested else 0.0
        return {
            "requested": self.requested,
            "embedding_hits": self.embedding_hits,
            "embedding_new": self.embedding_new,
            "embedding_hit_rate": hit_rate,
            "resolved_hit_ok": self.resolved_hit_ok,
            "resolved_hit_failed": self.resolved_hit_failed,
            "resolved_new_ok": self.resolved_new_ok,
            "resolved_new_failed": self.resolved_new_failed,
            "resolver_success_rate": resolve_rate,
        }


class FeverousCacheBuilder:
    def __init__(
        self,
        *,
        wiki_db_path: str | Path,
        cache_db_path: str | Path,
        model_name: str,
        embed_batch_size: int,
        include_context: bool,
        clean_wiki: bool,
        commit_every: int = 500,
    ):
        self.include_context = include_context
        self.clean_wiki = clean_wiki
        self.commit_every = commit_every

        self.resolver = WikiSqliteResolver(wiki_db_path)
        self.cache = EvidenceCacheDB(cache_db_path)
        self.embedder = STEmbedder(model_name, batch_size=embed_batch_size)

    def close(self) -> None:
        self.resolver.close()
        self.cache.conn.commit()
        self.cache.close()

    def ensure_cached(self, evidence_ids: Iterable[str]) -> CacheStats:
        stats = CacheStats()
        model = self.embedder.model_name

        ids = [_norm(x) for x in evidence_ids if x]
        ids = list(dict.fromkeys(ids))
        stats.requested = len(ids)

        # 1) embedding hits
        to_process: List[str] = []
        p = tqdm(ids, desc="🔎 Checking embedding cache", unit="id")
        for eid in p:
            if self.cache.has_embedding(eid, model):
                stats.embedding_hits += 1
            else:
                to_process.append(eid)

            # live counters
            if (stats.embedding_hits + len(to_process)) % 500 == 0:
                p.set_postfix_str(
                    f"hit={stats.embedding_hits} miss={len(to_process)} hit_rate={(stats.embedding_hits / max(1, (stats.embedding_hits+len(to_process)))):.2%}"
                )
        p.close()

        # 2) resolve missing embeddings
        to_embed_ids: List[str] = []
        to_embed_texts: List[str] = []
        pending_ops = 0

        p = tqdm(to_process, desc="🧩 Resolving IDs", unit="id")
        for eid in p:
            row = self.cache.get_resolved_row(eid)

            if row and row["status"] == "ok":
                stats.resolved_hit_ok += 1
                text = row["text"] or ""
                if is_valid_text(text):
                    to_embed_ids.append(eid)
                    to_embed_texts.append(text)
                else:
                    page_id, local_id = split_evidence_id(eid)
                    self.cache.upsert_resolved_failed(eid, page_id or "", local_id or "", "cached_text_invalid")
                    stats.resolved_new_failed += 1
                continue

            if row and row["status"] == "failed":
                stats.resolved_hit_failed += 1
                continue

            page_id, local_id, text, err = self.resolver.resolve_evidence_id(eid)
            if err:
                self.cache.upsert_resolved_failed(eid, page_id or "", local_id or "", err)
                stats.resolved_new_failed += 1
                pending_ops += 1
            else:
                assert text is not None
                cleaned = False
                if self.clean_wiki:
                    text = strip_wiki_markup(text)
                    cleaned = True

                if not is_valid_text(text):
                    self.cache.upsert_resolved_failed(eid, page_id, local_id, "validation_failed")
                    stats.resolved_new_failed += 1
                    pending_ops += 1
                else:
                    self.cache.upsert_resolved_ok(eid, page_id, local_id, text, cleaned=cleaned)
                    stats.resolved_new_ok += 1
                    pending_ops += 1
                    to_embed_ids.append(eid)
                    to_embed_texts.append(text)

            if pending_ops >= self.commit_every:
                self.cache.conn.commit()
                pending_ops = 0

            if (stats.resolved_hit_ok + stats.resolved_new_ok + stats.resolved_new_failed) % 500 == 0:
                p.set_postfix_str(
                    f"ok(hit/new)={stats.resolved_hit_ok}/{stats.resolved_new_ok} fail(new)={stats.resolved_new_failed} queued_embed={len(to_embed_texts)}"
                )

        p.close()

        if pending_ops:
            self.cache.conn.commit()

        # 3) embed + store (show progress by chunking)
        if to_embed_texts:
            bs = self.embedder.batch_size
            p = tqdm(range(0, len(to_embed_texts), bs), desc="🧠 Embedding", unit="batch")
            for i in p:
                batch_texts = to_embed_texts[i : i + bs]
                batch_ids = to_embed_ids[i : i + bs]
                vecs = self.embedder.encode(batch_texts)

                for eid, v in zip(batch_ids, vecs):
                    self.cache.upsert_embedding(eid, model, v)
                    stats.embedding_new += 1

                if (i // bs) % 10 == 0:
                    p.set_postfix_str(f"new_embeds={stats.embedding_new} dim={int(vecs.shape[1])}")

            p.close()
            self.cache.conn.commit()

        return stats


# ============================================================
# Main
# ============================================================

def count_lines(path: str | Path) -> int:
    n = 0
    with open(path, "rb") as f:
        for _ in f:
            n += 1
    return n

def iter_jsonl(path: str | Path, limit: Optional[int] = None):
    n = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)

            # Skip the schema/header line sometimes present in these files
            if isinstance(obj, dict) and obj.get("id", None) == "" and obj.get("claim", "") == "" and obj.get("label", "") == "":
                continue

            yield obj
            n += 1
            if limit is not None and n >= limit:
                return


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="Path to FEVEROUS jsonl (dev challenges, dev, train, etc.)")
    ap.add_argument("--wiki-db", required=True, help="Path to feverous_wikiv1.db (wiki(id,data))")
    ap.add_argument("--cache-db", required=True, help="Path to cache db to create/use (resolved + embeddings)")
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--embed-batch-size", type=int, default=64)
    ap.add_argument("--limit", type=int, default=None, help="Optional limit on number of examples to process")
    ap.add_argument("--include-context", action="store_true", help="Also cache+embed context element IDs")
    ap.add_argument("--no-clean", action="store_true", help="Do not strip simple wiki markup [[A|B]]")
    ap.add_argument("--report-json", default=None, help="Optional path to write a JSON report")
    args = ap.parse_args()

    dataset = Path(args.dataset)
    wiki_db = Path(args.wiki_db)
    cache_db = Path(args.cache_db)

    if not dataset.exists():
        raise FileNotFoundError(dataset)
    if not wiki_db.exists():
        raise FileNotFoundError(wiki_db)

    t0 = time.time()

    # collect ids
    total_lines = count_lines(dataset)
    print(f"Dataset lines (approx): {total_lines}")

    all_ids: Set[str] = set()
    examples = 0

    p = tqdm(iter_jsonl(dataset, limit=args.limit), total=(args.limit or total_lines), desc="📄 Scanning dataset", unit="ex")
    for ex in p:
        examples += 1
        all_ids |= collect_evidence_ids(ex, include_context=args.include_context)
        if examples % 200 == 0:
            p.set_postfix_str(f"examples={examples} unique_ids={len(all_ids)}")
    p.close()

    t1 = time.time()

    builder = FeverousCacheBuilder(
        wiki_db_path=wiki_db,
        cache_db_path=cache_db,
        model_name=args.model,
        embed_batch_size=args.embed_batch_size,
        include_context=args.include_context,
        clean_wiki=(not args.no_clean),
    )

    stats = builder.ensure_cached(all_ids)
    builder.close()

    t2 = time.time()

    report = {
        "dataset": str(dataset),
        "wiki_db": str(wiki_db),
        "cache_db": str(cache_db),
        "model": args.model,
        "include_context": bool(args.include_context),
        "examples_seen": examples,
        "unique_element_ids": len(all_ids),
        "timing_seconds": {
            "collect_ids": round(t1 - t0, 3),
            "resolve_embed_cache": round(t2 - t1, 3),
            "total": round(t2 - t0, 3),
        },
        "stats": stats.to_dict(),
    }

    print("\n==== FEVEROUS CACHE BUILD REPORT ====")
    print(json.dumps(report, indent=2, ensure_ascii=False))

    if args.report_json:
        out = Path(args.report_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nWrote report -> {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
