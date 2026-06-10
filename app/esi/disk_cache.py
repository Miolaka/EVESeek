"""
Persistent disk cache for ESI market orders.
Stores raw JSON in a SQLite DB at data/esi_cache.db with TTL.
Survives server restarts; acts as L2 behind the in-memory L1 cache.
"""
import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Optional

_DB_PATH = Path(__file__).parent.parent.parent / "data" / "esi_cache.db"
_ready = False


def _init() -> None:
    global _ready
    if _ready:
        return
    with sqlite3.connect(_DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS cache (
                key      TEXT PRIMARY KEY,
                value    TEXT NOT NULL,
                expires  REAL NOT NULL
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS ix_expires ON cache(expires)")
    _ready = True


def get(key: str) -> Optional[Any]:
    _init()
    try:
        with sqlite3.connect(_DB_PATH) as conn:
            row = conn.execute(
                "SELECT value FROM cache WHERE key = ? AND expires > ?",
                (key, time.time()),
            ).fetchone()
            if row:
                return json.loads(row[0])
    except Exception:
        pass
    return None


def put(key: str, value: Any, ttl: int) -> None:
    _init()
    try:
        with sqlite3.connect(_DB_PATH) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO cache (key, value, expires) VALUES (?, ?, ?)",
                (key, json.dumps(value), time.time() + ttl),
            )
    except Exception:
        pass


def evict_expired() -> None:
    """Remove stale rows — call periodically to keep the DB small."""
    _init()
    try:
        with sqlite3.connect(_DB_PATH) as conn:
            conn.execute("DELETE FROM cache WHERE expires <= ?", (time.time(),))
    except Exception:
        pass
