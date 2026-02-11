from __future__ import annotations

import argparse
import json
import re
import sqlite3
import time
import unicodedata
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from feverous_wiki_resolver import FeverousWikiDB

import numpy as np
from tqdm import tqdm


# ----------------------------
# Element ID parsing
# ----------------------------

ELEM_KINDS = [
    "header_cell",
    "table_caption",
    "sentence",
    "cell",
    "item",
    "title",
    "section",
    "list",
]

def parse_element_id(element_id: str) -> Tuple[str, str, str]:
    """
    Returns (page_id, kind, rest).
    Examples:
      "Algebraic logic_sentence_0" -> ("Algebraic logic","sentence","0")
      "John Laurie_cell_2_13_1"    -> ("John Laurie","cell","2_13_1")
      "Foo_title"                  -> ("Foo","title","")
      "Foo_section_4"              -> ("Foo","section","4")
    """
    # Handle suffix-only types first
    if element_id.endswith("_title"):
        return element_id[:-6], "title", ""
    m = re.match(r"^(.*)_section_(\d+)$", element_id)
    if m:
        return m.group(1), "section", m.group(2)
    m = re.match(r"^(.*)_list_(\d+)$", element_id)
    if m:
        return m.group(1), "list", m.group(2)

    # Handle _<kind>_... forms; page_id may contain underscores/spaces
    for kind in ["header_cell", "table_caption", "sentence", "cell", "item"]:
        token = f"_{kind}_"
        idx = element_id.rfind(token)
        if idx != -1:
            page_id = element_id[:idx]
            rest = element_id[idx + len(token):]
            return page_id, kind, rest

    raise ValueError(f"Unrecognized element_id: {element_id}")


# ----------------------------
# Wiki DB access + resolver
# ----------------------------

def _normalize_title_variants(title: str) -> List[str]:
    """
    Generate reasonable variants for SQLite id lookups.
    This is important for unicode normalization (– vs -, NFC/NFKC, etc.).
    """
    title = title.strip()
    variants = []

    def add(t: str):
        t = t.strip()
        if t and t not in variants:
            variants.append(t)

    add(title)
    add(unicodedata.normalize("NFC", title))
    add(unicodedata.normalize("NFKC", title))
    # dash normalization
    add(title.replace("–", "-").replace("—", "-"))
    add(unicodedata.normalize("NFC", title.replace("–", "-").replace("—", "-")))
    # underscore/space swap (some dumps differ)
    add(title.replace("_", " "))
    add(title.replace(" ", "_"))

    return variants


class WikiDB:
    def __init__(self, path: str):
        self.path = path
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self._page_id_cache: Dict[str, Optional[str]] = {}

    def close(self) -> None:
        self.conn.close()

    def resolve_page_id(self, page_id: str) -> Optional[str]:
        if page_id in self._page_id_cache:
            return self._page_id_cache[page_id]

        for cand in _normalize_title_variants(page_id):
            row = self.conn.execute("SELECT id FROM wiki WHERE id = ? LIMIT 1", (cand,)).fetchone()
            if row:
                self._page_id_cache[page_id] = row["id"]
                return row["id"]

        self._page_id_cache[page_id] = None
        return None

    def load_page_json(self, resolved_page_id: str) -> Dict[str, Any]:
        row = self.conn.execute("SELECT data FROM wiki WHERE id = ? LIMIT 1", (resolved_page_id,)).fetchone()
        if not row:
            raise KeyError(f"page not found: {resolved_page_id}")
        data = row["data"]
        if isinstance(data, (bytes, bytearray)):
            data = data.decode("utf-8")
        return json.loads(data)


@dataclass
class PageIndex:
    title: str
    sentences: Dict[str, str]
    sections: Dict[str, str]          # "section_4" -> "Physics"
    lists: Dict[str, str]             # "list_0" -> joined bullet string
    items: Dict[str, str]             # "item_0_0" -> item value
    cells: Dict[str, str]             # "cell_0_1_1" -> cell value
    header_cells: Dict[str, str]      # "header_cell_0_0_1" -> header cell value
    table_captions: Dict[str, str]    # "table_caption_0" -> caption string

    @staticmethod
    def from_page_json(page: Dict[str, Any]) -> "PageIndex":
        title = page.get("title", "")

        # Sentences and sections are direct keys in the JSON
        sentences = {}
        sections = {}
        lists = {}
        items = {}
        cells = {}
        header_cells = {}
        table_captions = {}

        for k, v in page.items():
            if k.startswith("sentence_") and isinstance(v, str):
                sentences[k] = v
            elif k.startswith("section_") and isinstance(v, dict):
                # store the visible section title / value
                sections[k] = str(v.get("value", "")).strip()
            elif k.startswith("list_") and isinstance(v, dict):
                # list: {type:..., list:[{id,value,level,type},...]}
                li = v.get("list", [])
                lines = []
                for it in li:
                    if not isinstance(it, dict):
                        continue
                    iid = it.get("id")
                    ival = str(it.get("value", "")).strip()
                    if iid:
                        items[iid] = ival
                    if ival:
                        lines.append(ival)
                lists[k] = "\n".join(lines).strip()

            elif k.startswith("table_") and isinstance(v, dict):
                # captions
                cap = v.get("caption")
                if cap:
                    # FEVEROUS refers to "table_caption_0" etc
                    table_idx = k.split("_", 1)[1]
                    table_captions[f"table_caption_{table_idx}"] = str(cap).strip()

                # cells inside v["table"] (list of rows)
                t = v.get("table", [])
                for row in t:
                    if not isinstance(row, list):
                        continue
                    for cell in row:
                        if not isinstance(cell, dict):
                            continue
                        cid = cell.get("id")
                        val = str(cell.get("value", "")).strip()
                        if not cid:
                            continue
                        if cid.startswith("header_cell_"):
                            header_cells[cid] = val
                        elif cid.startswith("cell_"):
                            cells[cid] = val

        return PageIndex(
            title=title,
            sentences=sentences,
            sections=sections,
            lists=lists,
            items=items,
            cells=cells,
            header_cells=header_cells,
            table_captions=table_captions,
        )


class WikiResolver:
    def __init__(self, wiki_db: WikiDB, page_cache_size: int = 256):
        self.wiki_db = wiki_db
        self._page_cache: Dict[str, PageIndex] = {}
        self._page_cache_order: List[str] = []
        self.page_cache_size = page_cache_size

    def _get_page_index(self, page_id_raw: str) -> Optional[PageIndex]:
        resolved = self.wiki_db.resolve_page_id(page_id_raw)
        if not resolved:
            return None

        if resolved in self._page_cache:
            return self._page_cache[resolved]

        page = self.wiki_db.load_page_json(resolved)
        idx = PageIndex.from_page_json(page)

        # Simple LRU
        self._page_cache[resolved] = idx
        self._page_cache_order.append(resolved)
        if len(self._page_cache_order) > self.page_cache_size:
            old = self._page_cache_order.pop(0)
            self._page_cache.pop(old, None)

        return idx

    def resolve_text(self, element_id: str) -> Optional[str]:
        page_id, kind, rest = parse_element_id(element_id)
        idx = self._get_page_index(page_id)
        if idx is None:
            return None

        if kind == "title":
            return idx.title.strip() or page_id.strip()

        if kind == "sentence":
            key = f"sentence_{rest}"
            return (idx.sentences.get(key) or "").strip() or None

        if kind == "section":
            key = f"section_{rest}"
            return (idx.sections.get(key) or "").strip() or None

        if kind == "list":
            key = f"list_{rest}"
            return (idx.lists.get(key) or "").strip() or None

        if kind == "item":
            key = f"item_{rest}"
            return (idx.items.get(key) or "").strip() or None

        if kind == "cell":
            key = f"cell_{rest}"
            return (idx.cells.get(key) or "").strip() or None

        if kind == "header_cell":
            key = f"header_cell_{rest}"
            return (idx.header_cells.get(key) or "").strip() or None

        if kind == "table_caption":
            key = f"table_caption_{rest}"
            return (idx.table_captions.get(key) or "").strip() or None

        return None


# ----------------------------
# Cache DB (resolved + embeddings)
# ----------------------------

class CacheDB:
    def __init__(self, path: str):
        self.path = path
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self._setup()

    def close(self) -> None:
        self.conn.close()

    def _setup(self) -> None:
        cur = self.conn.cursor()
        cur.execute("PRAGMA journal_mode=WAL;")
        cur.execute("PRAGMA synchronous=NORMAL;")
        cur.execute("PRAGMA temp_store=MEMORY;")

        cur.execute("""
        CREATE TABLE IF NOT EXISTS resolved (
            element_id TEXT PRIMARY KEY,
            ok INTEGER NOT NULL,
            text TEXT,
            error TEXT,
            kind TEXT,
            page_id TEXT,
            updated_at REAL NOT NULL
        );
        """)

        cur.execute("""
        CREATE TABLE IF NOT EXISTS embeddings (
            element_id TEXT PRIMARY KEY,
            model TEXT NOT NULL,
            dim INTEGER NOT NULL,
            vec BLOB NOT NULL,
            updated_at REAL NOT NULL
        );
        """)

        cur.execute("CREATE INDEX IF NOT EXISTS idx_resolved_ok ON resolved(ok);")
        self.conn.commit()

    def has_embedding(self, element_id: str, model: str) -> bool:
        row = self.conn.execute(
            "SELECT 1 FROM embeddings WHERE element_id = ? AND model = ? LIMIT 1",
            (element_id, model),
        ).fetchone()
        return row is not None

    def has_ok_embedding(self, element_id: str, model: str) -> bool:
        row = self.conn.execute(
            """
            SELECT 1
            FROM resolved r
            JOIN embeddings e ON e.element_id = r.element_id
            WHERE r.element_id = ? AND r.ok = 1 AND e.model = ?
            LIMIT 1
            """,
            (element_id, model),
        ).fetchone()
        return row is not None

    def put_resolved_ok(self, element_id: str, text: str) -> None:
        page_id, kind, _ = parse_element_id(element_id)
        self.conn.execute(
            """
            INSERT INTO resolved(element_id, ok, text, error, kind, page_id, updated_at)
            VALUES(?, 1, ?, NULL, ?, ?, ?)
            ON CONFLICT(element_id) DO UPDATE SET
                ok=1, text=excluded.text, error=NULL, kind=excluded.kind, page_id=excluded.page_id, updated_at=excluded.updated_at
            """,
            (element_id, text, kind, page_id, time.time()),
        )

    def put_resolved_fail(self, element_id: str, error: str) -> None:
        page_id, kind, _ = parse_element_id(element_id)
        self.conn.execute(
            """
            INSERT INTO resolved(element_id, ok, text, error, kind, page_id, updated_at)
            VALUES(?, 0, NULL, ?, ?, ?, ?)
            ON CONFLICT(element_id) DO UPDATE SET
                ok=0, text=NULL, error=excluded.error, kind=excluded.kind, page_id=excluded.page_id, updated_at=excluded.updated_at
            """,
            (element_id, error[:500], kind, page_id, time.time()),
        )

    def put_embeddings(self, element_ids: Sequence[str], vecs: np.ndarray, model: str) -> None:
        assert vecs.ndim == 2 and len(element_ids) == vecs.shape[0]
        dim = int(vecs.shape[1])
        now = time.time()

        rows = []
        for eid, v in zip(element_ids, vecs):
            v = np.asarray(v, dtype=np.float32)
            rows.append((eid, model, dim, v.tobytes(), now))

        self.conn.executemany(
            """
            INSERT INTO embeddings(element_id, model, dim, vec, updated_at)
            VALUES(?, ?, ?, ?, ?)
            ON CONFLICT(element_id) DO UPDATE SET
                model=excluded.model, dim=excluded.dim, vec=excluded.vec, updated_at=excluded.updated_at
            """,
            rows,
        )


# ----------------------------
# Dataset reading + completeness
# ----------------------------

def iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def required_ids_for_evidence_set(eset: Dict[str, Any], include_context: bool) -> Set[str]:
    req = set(eset.get("content", []))
    if include_context:
        ctx = eset.get("context", {}) or {}
        for cid in list(req):
            for x in ctx.get(cid, []) or []:
                req.add(x)
    return req


def evidence_set_is_complete(eset: Dict[str, Any], cache: CacheDB, model: str, include_context: bool) -> bool:
    req = required_ids_for_evidence_set(eset, include_context=include_context)
    return all(cache.has_ok_embedding(eid, model) for eid in req)


# ----------------------------
# Embedder hook (plug in yours)
# ----------------------------

class Embedder:
    """
    Adapter around whatever you already use.
    This version tries to import your HFEmbedder; if not available, falls back to sentence-transformers.
    """
    def __init__(self, model_name: str):
        self.model_name = model_name

        self._impl = None
        try:
            # If your project has this
            from dpgss.embedding.embedder import HFEmbedder  # type: ignore
            self._impl = HFEmbedder(model_name=model_name, embedding_db="E:\\data\\global_embeddings.db")
        except Exception:
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore
                self._impl = SentenceTransformer(model_name)
            except Exception as e:
                raise RuntimeError(
                    "No embedder found. Install sentence-transformers or ensure verity_gate.embedder.HFEmbedder is importable."
                ) from e

    def embed(self, texts: List[str]) -> np.ndarray:
        if hasattr(self._impl, "embed"):
            vecs = self._impl.embed(texts)
            return np.asarray(vecs, dtype=np.float32)
        # sentence-transformers
        vecs = self._impl.encode(texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vecs, dtype=np.float32)


# ----------------------------
# Cache build + filtering
# ----------------------------

def collect_unique_ids(dataset_path: str, include_context: bool) -> Tuple[int, Set[str]]:
    n = 0
    ids: Set[str] = set()
    for ex in iter_jsonl(dataset_path):
        n += 1
        for eset in ex.get("evidence", []) or []:
            ids.update(eset.get("content", []) or [])
            if include_context:
                ctx = eset.get("context", {}) or {}
                for cid in eset.get("content", []) or []:
                    ids.update(ctx.get(cid, []) or [])
    return n, ids


def build_cache(
    dataset_path: str,
    wiki_db_path: str,
    cache_db_path: str,
    model_name: str,
    include_context: bool,
    batch_size: int = 64,
) -> Dict[str, Any]:
    t0 = time.time()

    examples_seen, unique_ids = collect_unique_ids(dataset_path, include_context=include_context)
    ids_list = sorted(unique_ids)

    wiki = FeverousWikiDB(wiki_db_path)
    resolver = wiki 
    cache = CacheDB(cache_db_path)
    embedder = Embedder(model_name)

    hits = 0
    new_ok = 0
    new_fail = 0

    pending_ids: List[str] = []
    pending_texts: List[str] = []

    t_resolve_embed0 = time.time()
    pbar = tqdm(ids_list, desc="Resolve+Embed (cached)", total=len(ids_list))
    for eid in pbar:
        if cache.has_embedding(eid, model_name):
            hits += 1
            if (hits + new_ok + new_fail) % 500 == 0:
                resolved = new_ok + new_fail
                pbar.set_postfix({
                    "hit": hits,
                    "ok": new_ok,
                    "fail": new_fail,
                    "succ%": round(100.0 * new_ok / max(1, resolved), 2),
                })
            continue

        res = resolver.resolve_text(eid)
        if isinstance(res, tuple):
            ok, txt, err = res
        else:
            ok, txt, err = (res is not None and str(res).strip() != ""), res, None

        if not ok or not txt:
            cache.put_resolved_fail(eid, err or "resolve_failed")
            new_fail += 1
            continue

        cache.put_resolved_ok(eid, txt)
        pending_ids.append(eid)
        pending_texts.append(txt)
        new_ok += 1

        # Periodically update progress bar with live rates
        if (hits + new_ok + new_fail) % 250 == 0:
            resolved = new_ok + new_fail
            pbar.set_postfix({
                "hit": hits,
                "ok": new_ok,
                "fail": new_fail,
                "succ%": round(100.0 * new_ok / max(1, resolved), 2),
            })

        if len(pending_ids) >= batch_size:
            vecs = embedder.embed(pending_texts)
            cache.put_embeddings(pending_ids, vecs, model_name)
            cache.conn.commit()
            pending_ids.clear()
            pending_texts.clear()

    if pending_ids:
        vecs = embedder.embed(pending_texts)
        cache.put_embeddings(pending_ids, vecs, model_name)
        cache.conn.commit()

    t_resolve_embed = time.time() - t_resolve_embed0

    # wiki.close()
    cache.close()

    return {
        "dataset": dataset_path,
        "wiki_db": wiki_db_path,
        "cache_db": cache_db_path,
        "model": model_name,
        "include_context": include_context,
        "examples_seen": examples_seen,
        "unique_element_ids": len(ids_list),
        "timing_seconds": {
            "resolve_embed_cache": round(t_resolve_embed, 3),
            "total": round(time.time() - t0, 3),
        },
        "stats": {
            "requested": len(ids_list),
            "embedding_hits": hits,
            "embedding_new": new_ok,
            "embedding_hit_rate": float(hits / max(1, len(ids_list))),
            "resolved_new_ok": new_ok,
            "resolved_new_failed": new_fail,
            "resolver_success_rate": float(new_ok / max(1, new_ok + new_fail)),
        },
    }


def filter_complete_dataset(
    dataset_path: str,
    out_path: str,
    cache_db_path: str,
    model_name: str,
    include_context: bool,
    keep_mode: str = "any_set",
) -> Dict[str, Any]:
    """
    keep_mode:
      - "any_set": keep claim if >=1 evidence set complete; drop incomplete sets
      - "all_sets": keep claim only if ALL sets complete (very strict)
    """
    cache = CacheDB(cache_db_path)

    kept_claims = 0
    dropped_claims = 0
    kept_sets = 0
    dropped_sets = 0

    with open(out_path, "w", encoding="utf-8") as w:
        for ex in tqdm(list(iter_jsonl(dataset_path)), desc="Filter complete", total=None):
            ev = ex.get("evidence", []) or []
            complete_sets = []
            for eset in ev:
                if evidence_set_is_complete(eset, cache, model_name, include_context=include_context):
                    complete_sets.append(eset)
                else:
                    dropped_sets += 1

            if keep_mode == "all_sets":
                ok = (len(complete_sets) == len(ev)) and len(ev) > 0
            else:  # any_set
                ok = len(complete_sets) > 0

            if ok:
                kept_claims += 1
                kept_sets += len(complete_sets)
                ex2 = dict(ex)
                ex2["evidence"] = complete_sets
                w.write(json.dumps(ex2, ensure_ascii=False) + "\n")
            else:
                dropped_claims += 1

    cache.close()

    return {
        "dataset_in": dataset_path,
        "dataset_out": out_path,
        "cache_db": cache_db_path,
        "model": model_name,
        "include_context": include_context,
        "keep_mode": keep_mode,
        "kept_claims": kept_claims,
        "dropped_claims": dropped_claims,
        "kept_evidence_sets": kept_sets,
        "dropped_evidence_sets": dropped_sets,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--wiki_db", required=True)
    ap.add_argument("--cache_db", required=True)
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--include_context", action="store_true")
    ap.add_argument("--batch_size", type=int, default=64)

    ap.add_argument("--filter_out", default=None, help="If set, write filtered JSONL containing only complete evidence sets.")
    ap.add_argument("--keep_mode", default="any_set", choices=["any_set", "all_sets"])

    args = ap.parse_args()

    report = build_cache(
        dataset_path=args.dataset,
        wiki_db_path=args.wiki_db,
        cache_db_path=args.cache_db,
        model_name=args.model,
        include_context=args.include_context,
        batch_size=args.batch_size,
    )
    print("==== FEVEROUS CACHE BUILD REPORT ====")
    print(json.dumps(report, indent=2))

    if args.filter_out:
        frep = filter_complete_dataset(
            dataset_path=args.dataset,
            out_path=args.filter_out,
            cache_db_path=args.cache_db,
            model_name=args.model,
            include_context=args.include_context,
            keep_mode=args.keep_mode,
        )
        print("==== FEVEROUS COMPLETE-FILTER REPORT ====")
        print(json.dumps(frep, indent=2))


if __name__ == "__main__":
    main()
