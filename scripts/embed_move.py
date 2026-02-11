import sqlite3
import hashlib
import time
from pathlib import Path
import numpy as np


# --------------------------------------------------
# CONFIG
# --------------------------------------------------

OLD_DBS = [
    Path(r"E:\\data\\feverous_cache.db"),
    Path(r"E:\\data\\feverous_cache_bge.db"),
]

NEW_DB = Path(r"E:\\data\\global_embeddings.db")


# --------------------------------------------------
# Helpers
# --------------------------------------------------

def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def ensure_new_schema(conn: sqlite3.Connection):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS embeddings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT NOT NULL,
            text_hash TEXT NOT NULL,
            model TEXT NOT NULL,
            dim INTEGER NOT NULL,
            vec BLOB NOT NULL,
            updated_at REAL NOT NULL,
            UNIQUE(text_hash, model)
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_hash_model
        ON embeddings(text_hash, model)
    """)
    conn.commit()


# --------------------------------------------------
# Migration
# --------------------------------------------------

def migrate_one_db(old_db: Path, new_conn: sqlite3.Connection):
    print(f"\n📦 Migrating from: {old_db}")

    old_conn = sqlite3.connect(str(old_db))
    old_conn.row_factory = sqlite3.Row

    cur = old_conn.cursor()

    # Assumes:
    # embeddings(element_id, model, dim, vec)
    # resolved(element_id, text, ok)

    cur.execute("""
        SELECT e.element_id, e.model, e.dim, e.vec, r.text
        FROM embeddings e
        JOIN resolved r ON r.element_id = e.element_id
        WHERE r.ok = 1
    """)

    rows = cur.fetchall()
    print(f"   Found {len(rows)} rows")

    inserted = 0
    skipped = 0
    now = time.time()

    new_cur = new_conn.cursor()

    for r in rows:
        text = r["text"]
        model = r["model"]
        dim = int(r["dim"])
        vec_blob = r["vec"]

        if not text or vec_blob is None:
            skipped += 1
            continue

        vec = np.frombuffer(vec_blob, dtype=np.float32)

        if vec.shape[0] != dim:
            skipped += 1
            continue

        text_hash = hash_text(text)

        try:
            new_cur.execute("""
                INSERT OR IGNORE INTO embeddings
                (text, text_hash, model, dim, vec, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                text,
                text_hash,
                model,
                dim,
                vec_blob,
                now
            ))
            if new_cur.rowcount > 0:
                inserted += 1
            else:
                skipped += 1

        except Exception as e:
            print(f"   ⚠ Error inserting row: {e}")
            skipped += 1

    new_conn.commit()
    new_cur.close()
    old_conn.close()

    print(f"   ✅ Inserted: {inserted}")
    print(f"   ⏭ Skipped:  {skipped}")


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    print(f"\n🔄 Creating / opening new DB: {NEW_DB}")
    new_conn = sqlite3.connect(str(NEW_DB))
    new_conn.execute("PRAGMA journal_mode=WAL")
    new_conn.execute("PRAGMA synchronous=NORMAL")
    new_conn.execute("PRAGMA temp_store=MEMORY")
    ensure_new_schema(new_conn)

    for db in OLD_DBS:
        if not db.exists():
            print(f"⚠ Missing DB: {db}")
            continue
        migrate_one_db(db, new_conn)

    new_conn.close()

    print("\n🎉 Migration complete.")


if __name__ == "__main__":
    main()
