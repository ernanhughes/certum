from __future__ import annotations

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


# -----------------------------
# Evidence-id parsing
# -----------------------------

EID_RE = re.compile(
    r"^(?P<page>.+)_(?P<kind>sentence|cell|header_cell|item|section|table_caption|title)(?:_(?P<rest>.*))?$"
)

def _norm(s: str) -> str:
    return unicodedata.normalize("NFC", s).strip()

def split_evidence_id(eid: str) -> Tuple[Optional[str], Optional[str]]:
    """
    "Manchester Cancer Research Centre_sentence_0" ->
        ("Manchester Cancer Research Centre", "sentence_0")
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


# -----------------------------
# Basic wiki markup cleanup
# -----------------------------

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
    text = WIKI_LINK_RE.sub(repl, text)
    return text


# -----------------------------
# Validation
# -----------------------------

def is_valid_text(text: str, *, min_chars: int = 5, max_chars: int = 20000) -> bool:
    if not text:
        return False
    t = text.strip()
    if len(t) < min_chars:
        return False
    if len(t) > max_chars:
        return False
    # reject "pure punctuation / whitespace"
    if not re.search(r"[A-Za-z0-9]", t):
        return False
    return True

def sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()


# -----------------------------
# Wiki resolver (reads your wiki sqlite db)
# -----------------------------

class WikiSqliteResolver:
    """
    Expects a sqlite DB with:
      CREATE TABLE wiki (id PRIMARY KEY, data json)
    where id == page title (with spaces), and data is JSON string.
    """

    def __init__(self, wiki_db_path: str | Path):
        self.wiki_db_path = str(wiki_db_path)
        self._conn = sqlite3.connect(self.wiki_db_path)
        self._conn.row_factory = sqlite3.Row

    def close(self) -> None:
        self._conn.close()

    def get_page_json(self, page_id: str) -> Optional[dict]:
        cur = self._conn.execute("SELECT data FROM wiki WHERE id = ?", (page_id,))
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

        # item_*_* lives inside some list_*
        if local_id.startswith("item_"):
            target = local_id
            for k, v in page_json.items():
                if not k.startswith("list_"):
                    continue
                if not isinstance(v, dict):
                    continue
                items = v.get("list")
                if not isinstance(items, list):
                    continue
                for it in items:
                    if isinstance(it, dict) and it.get("id") == target:
                        val = it.get("value")
                        return str(val) if isinstance(val, str) else None
            return None

        # cell / header_cell live in some table_*
        if local_id.startswith(("cell_", "header_cell_")):
            target = local_id
            for k, v in page_json.items():
                if not k.startswith("table_"):
                    continue
                if not isinstance(v, dict):
                    continue
                table = v.get("table")
                if not isinstance(table, list):
                    continue
                for row in table:
                    if not isinstance(row, list):
                        continue
                    for cell in row:
                        if isinstance(cell, dict) and cell.get("id") == target:
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


# -----------------------------
# Cache DB
# -----------------------------

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
              status TEXT NOT NULL,         -- ok | failed
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
              dtype TEXT NOT NULL,          -- float32
              vector BLOB NOT NULL,
              created_at REAL NOT NULL,
              PRIMARY KEY (evidence_id, model_name),
              FOREIGN KEY (evidence_id) REFERENCES resolved_items(evidence_id)
            );
            """
        )

        cur.execute("CREATE INDEX IF NOT EXISTS idx_resolved_status ON resolved_items(status);")
        self.conn.commit()

    def get_resolved_status(self, evidence_id: str) -> Optional[str]:
        cur = self.conn.execute(
            "SELECT status FROM resolved_items WHERE evidence_id = ?",
            (evidence_id,),
        )
        row = cur.fetchone()
        return row["status"] if row else None

    def get_resolved_text(self, evidence_id: str) -> Optional[str]:
        cur = self.conn.execute(
            "SELECT text, status FROM resolved_items WHERE evidence_id = ?",
            (evidence_id,),
        )
        row = cur.fetchone()
        if not row:
            return None
        if row["status"] != "ok":
            return None
        return row["text"]

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
        self.conn.commit()

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
        self.conn.commit()

    def has_embedding(self, evidence_id: str, model_name: str) -> bool:
        cur = self.conn.execute(
            "SELECT 1 FROM embeddings WHERE evidence_id=? AND model_name=?",
            (evidence_id, model_name),
        )
        return cur.fetchone() is not None

    def get_embedding(self, evidence_id: str, model_name: str) -> Optional[np.ndarray]:
        cur = self.conn.execute(
            "SELECT dim, dtype, vector FROM embeddings WHERE evidence_id=? AND model_name=?",
            (evidence_id, model_name),
        )
        row = cur.fetchone()
        if not row:
            return None
        dim = int(row["dim"])
        dtype = row["dtype"]
        if dtype != "float32":
            raise ValueError(f"Unsupported dtype in cache: {dtype}")
        vec = np.frombuffer(row["vector"], dtype=np.float32)
        if vec.size != dim:
            raise ValueError(f"Corrupt embedding: expected dim={dim}, got {vec.size}")
        return vec

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
        self.conn.commit()


# -----------------------------
# Embedder interface + helper
# -----------------------------

class Embedder:
    model_name: str
    def encode(self, texts: Sequence[str]) -> np.ndarray:
        raise NotImplementedError

class SentenceTransformersEmbedder(Embedder):
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", batch_size: int = 64):
        self.model_name = model_name
        self.batch_size = batch_size
        from sentence_transformers import SentenceTransformer  # type: ignore
        self._m = SentenceTransformer(model_name)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        # returns (n, d)
        return np.asarray(
            self._m.encode(list(texts), batch_size=self.batch_size, show_progress_bar=False, normalize_embeddings=False),
            dtype=np.float32,
        )


# -----------------------------
# Main cache builder
# -----------------------------

@dataclass
class CacheStats:
    requested: int = 0
    resolved_hit_ok: int = 0
    resolved_hit_failed: int = 0
    resolved_new_ok: int = 0
    resolved_new_failed: int = 0
    embedded_hit: int = 0
    embedded_new: int = 0

class FeverousEvidenceCacheBuilder:
    def __init__(
        self,
        *,
        wiki_db_path: str | Path,
        cache_db_path: str | Path,
        embedder: Embedder,
        clean_wiki: bool = True,
    ):
        self.resolver = WikiSqliteResolver(wiki_db_path)
        self.cache = EvidenceCacheDB(cache_db_path)
        self.embedder = embedder
        self.clean_wiki = clean_wiki

    def close(self) -> None:
        self.resolver.close()
        self.cache.close()

    def ensure_cached(self, evidence_ids: Iterable[str]) -> CacheStats:
        stats = CacheStats()
        model = self.embedder.model_name

        # 1) dedupe
        ids = [_norm(x) for x in evidence_ids if x]
        ids = list(dict.fromkeys(ids))
        stats.requested = len(ids)

        # 2) embedding hits first
        to_resolve: List[str] = []
        for eid in ids:
            if self.cache.has_embedding(eid, model):
                stats.embedded_hit += 1
                continue
            to_resolve.append(eid)

        # 3) resolve missing embeddings (with resolved-items cache helping)
        to_embed_ids: List[str] = []
        to_embed_texts: List[str] = []

        for eid in to_resolve:
            status = self.cache.get_resolved_status(eid)
            if status == "ok":
                stats.resolved_hit_ok += 1
                text = self.cache.get_resolved_text(eid)
                if text and is_valid_text(text):
                    to_embed_ids.append(eid)
                    to_embed_texts.append(text)
                else:
                    # if somehow invalid, mark failed so we stop retrying
                    page_id, local_id = split_evidence_id(eid)
                    self.cache.upsert_resolved_failed(eid, page_id or "", local_id or "", "cached_text_invalid")
                    stats.resolved_new_failed += 1
                continue

            if status == "failed":
                stats.resolved_hit_failed += 1
                continue

            # not cached: resolve from wiki
            page_id, local_id, text, err = self.resolver.resolve_evidence_id(eid)
            if err:
                self.cache.upsert_resolved_failed(eid, page_id or "", local_id or "", err)
                stats.resolved_new_failed += 1
                continue

            assert text is not None
            cleaned = False
            if self.clean_wiki:
                text = strip_wiki_markup(text)
                cleaned = True

            if not is_valid_text(text):
                self.cache.upsert_resolved_failed(eid, page_id, local_id, "validation_failed")
                stats.resolved_new_failed += 1
                continue

            self.cache.upsert_resolved_ok(eid, page_id, local_id, text, cleaned=cleaned)
            stats.resolved_new_ok += 1
            to_embed_ids.append(eid)
            to_embed_texts.append(text)

        # 4) embed in batches and store
        if to_embed_texts:
            vecs = self.embedder.encode(to_embed_texts)  # (n, d)
            for eid, v in zip(to_embed_ids, vecs):
                self.cache.upsert_embedding(eid, model, v)
                stats.embedded_new += 1

        return stats


# -----------------------------
# FEVEROUS ID collection helper
# -----------------------------

def collect_feverous_evidence_ids(example: dict, *, include_context: bool = True) -> Set[str]:
    ids: Set[str] = set()

    ev = example.get("evidence", [])
    # FEVEROUS uses list of evidence-sets; each set is a dict {content, context}
    if isinstance(ev, dict):
        ev = [ev]

    if isinstance(ev, list):
        for evset in ev:
            if not isinstance(evset, dict):
                continue
            content = evset.get("content", [])
            if isinstance(content, list):
                ids.update([_norm(x) for x in content if isinstance(x, str)])

            if include_context:
                ctx = evset.get("context", {})
                if isinstance(ctx, dict):
                    ids.update([_norm(k) for k in ctx.keys() if isinstance(k, str)])
                    for v in ctx.values():
                        if isinstance(v, list):
                            ids.update([_norm(x) for x in v if isinstance(x, str)])

    return ids
