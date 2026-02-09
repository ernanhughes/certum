from pathlib import Path
from typing import List, Tuple, Dict, Optional
import numpy as np
import sqlite3


class FeverousCache:
    """Read-only helper for feverous_cache.db.

    We use this to:
      1) validate an evidence set is *complete* (every required element_id resolved+embedded)
      2) retrieve the resolved text + cached embedding vectors for those element_ids
    """

    def __init__(self, cache_db: Path):
        self.cache_db = Path(cache_db)
        self.conn = sqlite3.connect(str(self.cache_db))
        self.conn.row_factory = sqlite3.Row

        # Small speedups; safe for read-only usage
        cur = self.conn.cursor()
        cur.execute("PRAGMA journal_mode=WAL")
        cur.execute("PRAGMA synchronous=NORMAL")
        cur.execute("PRAGMA temp_store=MEMORY")
        cur.close()

    def close(self) -> None:
        try:
            self.conn.close()
        except Exception:
            pass

    def has_ok_embedding(self, element_id: str, model: str) -> bool:
        cur = self.conn.cursor()
        cur.execute(
            """
            SELECT 1
            FROM resolved r
            JOIN embeddings e
              ON e.element_id = r.element_id
            WHERE r.element_id = ?
              AND r.ok = 1
              AND e.model = ?
            LIMIT 1
            """,
            (element_id, model),
        )
        row = cur.fetchone()
        cur.close()
        return row is not None

    def get_texts_and_vecs(self, element_ids: List[str], model: str) -> Tuple[List[str], np.ndarray, List[str]]:
        """Return (texts, vecs, missing_ids) for requested element_ids.

        Cache DB schema expectation:
          - resolved.ok = 1 indicates resolved text is valid
          - embeddings has (element_id, model, dim, vec) where vec is raw float32 bytes (v.tobytes()).
        """
        if not element_ids:
            return [], np.zeros((0, 0), dtype=np.float32), []

        # SQLite has a variable limit; chunk to be safe
        rows: Dict[str, sqlite3.Row] = {}
        chunk = 900
        for i in range(0, len(element_ids), chunk):
            sub = element_ids[i : i + chunk]
            q_marks = ",".join(["?"] * len(sub))
            sql = f"""
                SELECT r.element_id, r.text, e.vec, e.dim
                FROM resolved r
                JOIN embeddings e
                  ON e.element_id = r.element_id
                WHERE r.ok = 1
                  AND e.model = ?
                  AND r.element_id IN ({q_marks})
            """
            cur = self.conn.cursor()
            cur.execute(sql, [model, *sub])
            for r in cur.fetchall():
                rows[str(r["element_id"])] = r
            cur.close()

        texts: List[str] = []
        vecs_list: List[np.ndarray] = []
        missing: List[str] = []
        dim_expected: Optional[int] = None

        for eid in element_ids:
            r = rows.get(eid)
            if r is None:
                missing.append(eid)
                continue

            txt = r["text"]
            vec_blob = r["vec"]
            dim = int(r["dim"])

            if dim_expected is None:
                dim_expected = dim
            if dim_expected != dim:
                missing.append(eid)
                continue

            v = np.frombuffer(vec_blob, dtype=np.float32)
            if v.ndim != 1 or v.shape[0] != dim:
                missing.append(eid)
                continue

            texts.append(str(txt))
            vecs_list.append(v)

        if not vecs_list:
            return [], np.zeros((0, 0), dtype=np.float32), missing

        vecs = np.stack(vecs_list, axis=0).astype(np.float32, copy=False)
        return texts, vecs, missing

