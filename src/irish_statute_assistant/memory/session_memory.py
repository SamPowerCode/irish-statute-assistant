class SessionMemory:
    """Stores conversation exchanges for a single session."""

    def __init__(self) -> None:
        self._history: list[dict[str, str]] = []

    def add_exchange(self, user: str, assistant: str) -> None:
        self._history.append({"user": user, "assistant": assistant})

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
