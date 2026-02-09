# scripts/feverous_wiki_resolver.py
from __future__ import annotations

import json
import sqlite3
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Optional, Tuple, Iterable


# ---- parsing ----

# Known FEVEROUS local-id suffix patterns (most common)
LOCAL_PREFIXES = (
    "sentence_", "section_", "list_", "item_",
    "cell_", "header_cell_",
    "table_caption_",
    "title",
)

def nfc(s: str) -> str:
    # Normalize unicode (important for titles like Shōnen / Jérôme etc.)
    return unicodedata.normalize("NFC", s)

def parse_feverous_element_id(full_id: str) -> Tuple[str, str]:
    """
    Split a FEVEROUS element id into (page_id, local_id).

    Example:
      'Algebraic logic_cell_0_1_1' -> ('Algebraic logic', 'cell_0_1_1')
      '1976 United States presidential election_header_cell_3_0_0'
         -> ('1976 United States presidential election', 'header_cell_3_0_0')
    """
    s = nfc(full_id)

    # Fast path: detect the last '_sentence_', '_cell_', '_header_cell_' etc.
    # We scan for the earliest match from the RIGHT side.
    for prefix in ("_header_cell_", "_table_caption_", "_sentence_", "_cell_", "_item_", "_section_", "_list_", "_title"):
        idx = s.rfind(prefix)
        if idx != -1:
            page = s[:idx]
            local = s[idx + 1:]  # drop leading underscore
            return nfc(page), nfc(local)

    # Fallback: attempt last underscore split (better than crashing)
    # (If this triggers often, your IDs are nonstandard and we should inspect.)
    a, b = s.rsplit("_", 1)
    return nfc(a), nfc(b)


# ---- page indexing ----

def index_wiki_page(page_json: dict) -> Dict[str, str]:
    """
    Build {local_id -> text/value} for every resolvable element in a wiki page record.
    """
    out: Dict[str, str] = {}

    # Title
    title = page_json.get("title")
    if title:
        out["title"] = str(title)

    # Ordered elements
    order = page_json.get("order", [])
    for key in order:
        val = page_json.get(key)

        # sentence_N is a string
        if key.startswith("sentence_") and isinstance(val, str):
            out[key] = val
            continue

        # section_N is dict {value, level}
        if key.startswith("section_") and isinstance(val, dict):
            # 'value' is the section heading text
            if "value" in val:
                out[key] = str(val["value"])
            continue

        # list_N is dict {type, list:[{id,value,...},...]}
        if key.startswith("list_") and isinstance(val, dict):
            items = val.get("list", [])
            for item in items:
                iid = item.get("id")
                ival = item.get("value")
                if iid is not None and ival is not None:
                    out[str(iid)] = str(ival)
            continue

        # table_N is dict {type, table:[[cell_dict...]], caption?}
        if key.startswith("table_") and isinstance(val, dict):
            # caption becomes table_caption_N
            cap = val.get("caption")
            if cap is not None:
                # table_3 -> table_caption_3
                n = key.split("_", 1)[1]
                out[f"table_caption_{n}"] = str(cap)

            table = val.get("table", [])
            # Scan every cell dict and index by its explicit "id"
            for row in table:
                if not isinstance(row, list):
                    continue
                for cell in row:
                    if not isinstance(cell, dict):
                        continue
                    cid = cell.get("id")
                    cval = cell.get("value")
                    if cid is not None and cval is not None:
                        out[str(cid)] = str(cval)
            continue

    return out


# ---- resolver ----

@dataclass(frozen=True)
class FeverousWikiDB:
    wiki_db_path: str

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.wiki_db_path)
        con.row_factory = sqlite3.Row
        return con

    @lru_cache(maxsize=4096)
    def _load_page(self, page_id: str) -> Optional[dict]:
        page_id = nfc(page_id)
        with self._connect() as con:
            row = con.execute("SELECT data FROM wiki WHERE id = ?", (page_id,)).fetchone()
            if not row:
                return None
            data = row["data"]
            # 'data' may already be text JSON
            if isinstance(data, (bytes, bytearray)):
                data = data.decode("utf-8", errors="replace")
            return json.loads(data)

    @lru_cache(maxsize=4096)
    def _page_index(self, page_id: str) -> Optional[Dict[str, str]]:
        page = self._load_page(page_id)
        if page is None:
            return None
        return index_wiki_page(page)

    def resolve_text(self, full_element_id: str) -> Tuple[bool, Optional[str], str]:
        """
        Returns (ok, text, error_msg).
        """
        page_id, local_id = parse_feverous_element_id(full_element_id)
        idx = self._page_index(page_id)
        if idx is None:
            return False, None, f"missing_page:{page_id}"

        # local ids in page data are like 'cell_0_1_1' etc.
        text = idx.get(local_id)
        if text is None:
            return False, None, f"missing_element:{page_id}:{local_id}"

        return True, text, ""


def main():
    WIKI_DB = r"E:\\data\\feverous_wikiv1.db"

    FAILED = [
        # paste a few from your list:
        "1909–10 Welsh Amateur Cup_cell_3_1_3",
        "1976 United States presidential election_header_cell_3_0_0",
        "2011 Pan Arab Games_header_cell_4_7_19",
    ]

    db = FeverousWikiDB(WIKI_DB)

    for fid in FAILED:
        ok, text, err = db.resolve_text(fid)
        print(fid, "OK" if ok else "FAIL", err)
        if ok:
            print("  ->", text[:160].replace("\n", " "))

if __name__ == "__main__":
    main()    