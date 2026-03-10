from typing import Optional


class SessionCache:
    """In-session dict cache keyed by URL. Not shared across instances."""

    def __init__(self) -> None:
        self._store: dict[str, list[str]] = {}

    def get(self, url: str) -> Optional[list[str]]:
        return self._store.get(url)

    def set(self, url: str, sections: list[str]) -> None:
        self._store[url] = sections
