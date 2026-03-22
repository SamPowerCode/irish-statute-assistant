from __future__ import annotations

import logging

from irish_statute_assistant.agents.base_agent import BaseAgent
from irish_statute_assistant.config import Config
from irish_statute_assistant.exceptions import StatuteNotFoundError
from irish_statute_assistant.models.schemas import ActSection, ResearcherOutput
from irish_statute_assistant.tools.session_cache import SessionCache
from irish_statute_assistant.tools.statute_fetcher import StatuteFetcher
from irish_statute_assistant.tools.vector_store import get_vector_store

logger = logging.getLogger(__name__)


class ResearcherAgent(BaseAgent):
    """Retrieves relevant Irish statute sections for a query.

    Searches the vector store (ChromaDB or Qdrant) first. Falls back to a
    live HTTP fetch from irishstatutebook.ie if the store is empty or returns
    no results. Rate-limited and retried automatically.

    Args:
        config: Application configuration.
        cache: In-memory session cache to avoid duplicate fetches.
        fetcher: HTTP statute fetcher with rate limiting.
    """

    def __init__(self, config: Config, cache: SessionCache, fetcher: StatuteFetcher) -> None:
        self._config = config
        self._cache = cache
        self._fetcher = fetcher
        self._vector_store = get_vector_store(config)
        self._last_source: str = ""

    @property
    def last_source(self) -> str:
        """The data source used in the most recent run(): 'vector store' or 'live fetch'."""
        return self._last_source

    def run(self, query: str) -> ResearcherOutput:
        """Retrieve statute sections relevant to the query.

        Args:
            query: The user's legal question.

        Returns:
            ResearcherOutput containing a list of Acts with their sections.

        Raises:
            StatuteNotFoundError: If no relevant statutes are found.
        """
        if self._vector_store.is_populated():
            result = self._run_vector(query)
            self._last_source = "vector store"
            return result
        logger.warning(
            "Vector store is not populated — falling back to live HTTP fetch. "
            "Run `python -m irish_statute_assistant.indexer` to build the index."
        )
        result = self._run_live(query)
        self._last_source = "live fetch"
        return result

    def _run_vector(self, query: str) -> ResearcherOutput:
        results = self._vector_store.search(query, top_k=10)
        if not results:
            raise StatuteNotFoundError(f"No Acts found for query: {query!r}")

        grouped: dict[str, list[dict]] = {}
        for r in results:
            grouped.setdefault(r["url"], []).append(r)

        acts = []
        for url, group in grouped.items():
            group.sort(key=lambda r: r["section_index"])
            acts.append(ActSection(
                title=group[0]["title"],
                url=url,
                sections=[r["page_content"] for r in group],
            ))

        return ResearcherOutput(acts=acts)

    def _run_live(self, query: str) -> ResearcherOutput:
        results = self._fetcher.search(query)
        if not results:
            raise StatuteNotFoundError(f"No Acts found for query: {query!r}")

        acts = []
        for result in results:
            sections = self._fetcher.fetch(result["url"], self._cache)
            acts.append(ActSection(
                title=result["title"],
                url=result["url"],
                sections=sections,
            ))

        return ResearcherOutput(acts=acts)
