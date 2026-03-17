from __future__ import annotations

import os
import sqlite3


class UserPreferenceStore:
    """SQLite-backed key-value store for user preferences."""

    def __init__(self, db_path: str) -> None:
        self._db_path = os.path.expanduser(db_path)
        if dirname := os.path.dirname(self._db_path):
            os.makedirs(dirname, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS preferences "
                "(key TEXT PRIMARY KEY, value TEXT)"
            )

    def set(self, key: str, value: str) -> None:
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                "INSERT INTO preferences (key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, value),
            )

    def get(self, key: str, default: str = "") -> str:
        with sqlite3.connect(self._db_path) as conn:
            row = conn.execute(
                "SELECT value FROM preferences WHERE key = ?", (key,)
            ).fetchone()
        return row[0] if row else default

    def all(self) -> dict[str, str]:
        with sqlite3.connect(self._db_path) as conn:
            rows = conn.execute("SELECT key, value FROM preferences").fetchall()
        return {k: v for k, v in rows}
