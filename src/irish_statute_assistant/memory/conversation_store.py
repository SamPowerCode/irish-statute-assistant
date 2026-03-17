from __future__ import annotations

import os
import sqlite3


class ConversationStore:
    """SQLite-backed conversation history. Drop-in replacement for SessionMemory."""

    def __init__(self, db_path: str, history_limit: int = 20) -> None:
        self._db_path = os.path.expanduser(db_path)
        self._history_limit = history_limit
        if dirname := os.path.dirname(self._db_path):
            os.makedirs(dirname, exist_ok=True)
        self._init_db()
        self._history = self._load_history()

    def _init_db(self) -> None:
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS exchanges "
                "(id INTEGER PRIMARY KEY AUTOINCREMENT, user TEXT, assistant TEXT)"
            )

    def _load_history(self) -> list[dict[str, str]]:
        with sqlite3.connect(self._db_path) as conn:
            rows = conn.execute(
                "SELECT user, assistant FROM exchanges "
                "ORDER BY id DESC LIMIT ?",
                (self._history_limit,),
            ).fetchall()
        return [{"user": r[0], "assistant": r[1]} for r in reversed(rows)]

    def add_exchange(self, user: str, assistant: str) -> None:
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                "INSERT INTO exchanges (user, assistant) VALUES (?, ?)",
                (user, assistant),
            )
        self._history.append({"user": user, "assistant": assistant})
        if len(self._history) > self._history_limit:
            self._history = self._history[-self._history_limit:]

    def get_history(self) -> list[dict[str, str]]:
        return list(self._history)

    def format_for_prompt(self) -> str:
        if not self._history:
            return ""
        lines = []
        for exchange in self._history:
            lines.append(f"User: {exchange['user']}")
            lines.append(f"Assistant: {exchange['assistant']}")
        return "\n".join(lines)
