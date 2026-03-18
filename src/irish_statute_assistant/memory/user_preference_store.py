from __future__ import annotations

import os
import sqlite3


class UserPreferenceStore:
    """SQLite-backed key-value store for user preferences.

    Preferences are detected from query text (e.g. "I'm a solicitor") and
    from repeated evaluator signals. They persist across sessions and are
    injected into analyst and writer prompts.

    Args:
        db_path: Path to the SQLite database file. Supports ~ expansion.

    Example:
        store = UserPreferenceStore("~/.irish_statute_assistant/preferences.db")
        store.set("language_level", "technical")
        print(store.all())  # {"language_level": "technical"}
    """

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
        """Set a preference key to a value (upsert).

        Args:
            key: Preference key (e.g. "language_level", "verbosity", "user_type").
            value: Preference value (e.g. "plain", "brief", "solicitor").
        """
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                "INSERT INTO preferences (key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, value),
            )

    def get(self, key: str, default: str = "") -> str:
        """Get a preference value by key.

        Args:
            key: Preference key to look up.
            default: Value to return if the key is not set.

        Returns:
            The stored value, or default if not found.
        """
        with sqlite3.connect(self._db_path) as conn:
            row = conn.execute(
                "SELECT value FROM preferences WHERE key = ?", (key,)
            ).fetchone()
        return row[0] if row else default

    def all(self) -> dict[str, str]:
        """Return all stored preferences.

        Returns:
            Dict of all key-value pairs currently stored.
        """
        with sqlite3.connect(self._db_path) as conn:
            rows = conn.execute("SELECT key, value FROM preferences").fetchall()
        return {k: v for k, v in rows}
