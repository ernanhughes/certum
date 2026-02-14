import sqlite3
from pathlib import Path
from typing import Dict, List, Tuple

from certum.protocols.evidence_store import EvidenceStore
from certum.utils.text_utils import clean_wiki_markup


class SQLiteEvidenceStore(EvidenceStore):
    """
    SQLite-backed store for supporting evidence.

    Neutral to dataset (not feverous-specific).
    Assumes schema:

        resolved(element_id TEXT PRIMARY KEY, text TEXT, ok INTEGER)
        embeddings(element_id TEXT, model TEXT, dim INTEGER, vec BLOB)
    """

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row

        self._init_pragmas()

    def _init_pragmas(self):
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

    # -------------------------------------------------
    # API
    # -------------------------------------------------

    def get_texts(
        self,
        element_ids: List[str],
    ) -> Tuple[List[str], List[str]]:

        if not element_ids:
            return [], []

        rows: Dict[str, sqlite3.Row] = {}
        chunk = 900

        for i in range(0, len(element_ids), chunk):
            sub = element_ids[i:i+chunk]
            q_marks = ",".join(["?"] * len(sub))

            sql = f"""
                SELECT element_id, text
                FROM resolved
                WHERE ok = 1
                AND element_id IN ({q_marks})
            """

            cur = self.conn.cursor()
            cur.execute(sql, sub)
            for r in cur.fetchall():
                rows[str(r["element_id"])] = r
            cur.close()

        texts: List[str] = []
        missing: List[str] = []

        for eid in element_ids:
            r = rows.get(eid)
            if r is None:
                missing.append(eid)
            else:
                texts.append(clean_wiki_markup(str(r["text"])))

        return texts, missing
