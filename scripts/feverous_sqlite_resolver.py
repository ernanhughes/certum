from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


# -----------------------------
# Helpers
# -----------------------------
def _is_texty(col: str) -> bool:
    c = col.lower()
    return any(k in c for k in ("text", "content", "sentence", "cell", "value", "string"))

def _is_idy(col: str) -> bool:
    c = col.lower()
    return c in ("id", "_id", "key", "eid", "evidence_id") or c.endswith("_id")

def _is_pagey(col: str) -> bool:
    c = col.lower()
    return any(k in c for k in ("page", "title", "doc", "document", "wiki"))

def _titles_to_try(page: str) -> List[str]:
    # Evidence IDs sometimes use spaces; some DBs store underscores, etc.
    cands = [page]
    if "_" in page:
        cands.append(page.replace("_", " "))
    if " " in page:
        cands.append(page.replace(" ", "_"))
    # De-dupe preserving order
    out: List[str] = []
    seen = set()
    for t in cands:
        if t not in seen:
            out.append(t)
            seen.add(t)
    return out


_SENT_RE = re.compile(r"^(?P<page>.+)_sentence_(?P<idx>\d+)$")
_CELL_RE = re.compile(r"^(?P<page>.+)_cell_(?P<t>\d+)_(?P<r>\d+)_(?P<c>\d+)$")


@dataclass(frozen=True)
class TableInfo:
    name: str
    cols: List[str]

    @property
    def lower_cols(self) -> List[str]:
        return [c.lower() for c in self.cols]


class FeverousSQLiteResolver:
    """
    Resolves FEVEROUS evidence IDs like:
      - '{page}_sentence_{i}'
      - '{page}_cell_{table}_{row}_{col}'
    to actual text using feverous_wikiv1.db (SQLite).

    It does NOT assume a fixed schema:
      - It introspects tables/columns and tries a small set of robust query strategies.
    """

    def __init__(self, db_path: str | Path, *, read_only: bool = True, debug: bool = False):
        self.db_path = str(db_path)
        self.read_only = bool(read_only)
        self.debug = bool(debug)

        self._con: Optional[sqlite3.Connection] = None
        self._tables: List[TableInfo] = []

        # Cache: evidence_id -> text
        self._cache: Dict[str, Optional[str]] = {}

        # Once we find “the right” table for a strategy, store it for speed.
        self._direct_id_strategy: Optional[Tuple[str, str, str]] = None  # (table, id_col, text_col)
        self._sentence_strategy: Optional[Tuple[str, str, str, str]] = None  # (table, page_col, idx_col, text_col)
        self._cell_strategy: Optional[Tuple[str, str, str, str, str, str]] = None  # (table, page_col, tcol, rcol, ccol,_
