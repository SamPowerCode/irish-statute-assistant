from __future__ import annotations

import os
import sqlite3


class ConversationStore:
    """SQLite-backed conversation history that persists across process restarts.

    Stores user/assistant exchange pairs. Loads the most recent history_limit
    exchanges on construction. Each add_exchange call writes to SQLite immediately.
    The DB directory is created automatically on first use.

    Args:
        db_path: Path to the SQLite database file. Supports ~ expansion.
        history_limit: Maximum number of exchanges to load and maintain in memory.

    Example:
        store = ConversationStore("~/.irish_statute_assistant/conversations.db")
        store.add_exchange(user="What are my rights?", assistant="That depends...")
        print(store.format_for_prompt())
    """

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
        """Persist a user/assistant exchange and update the in-memory history.

        Args:
            user: The user's message.
            assistant: The assistant's response.
        """
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                "INSERT INTO exchanges (user, assistant) VALUES (?, ?)",
                (user, assistant),
            )
        self._history.append({"user": user, "assistant": assistant})
        if len(self._history) > self._history_limit:
            self._history = self._history[-self._history_limit:]

    def get_history(self) -> list[dict[str, str]]:
        """Return a copy of the in-memory exchange history.

        Returns:
            List of dicts with 'user' and 'assistant' keys, oldest first.
        """
        return list(self._history)

    def format_for_prompt(self) -> str:
        """Format conversation history for injection into an LLM prompt.

        Returns:
            A string of alternating "User: ..." and "Assistant: ..." lines,
            or an empty string if there is no history.
        """
        if not self._history:
            return ""
        lines = []
        for exchange in self._history:
            lines.append(f"User: {exchange['user']}")
            lines.append(f"Assistant: {exchange['assistant']}")
        return "\n".join(lines)
