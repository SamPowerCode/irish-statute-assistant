from __future__ import annotations

import logging

from irish_statute_assistant.config import Config
from irish_statute_assistant.models.schemas import ActSection, ResearcherOutput
from irish_statute_assistant.tools.session_cache import SessionCache
from irish_statute_assistant.tools.statute_fetcher import fetch_act_sections, search_statutes
from irish_statute_assistant.tools.vector_store import VectorStore

logger = logging.getLogger(__name__)


class ResearcherAgent:
    def __init__(self, config: Config, cache: SessionCache) -> None:
        self._config = config
        self._cache = cache
        self._search = search_statutes
        self._fetch = fetch_act_sections
        self._vector_store = VectorStore(config)

    def run(self, query: str) -> ResearcherOutput:
        if self._vector_store.is_populated():
            return self._run_vector(query)
        logger.warning(
            "Vector store is not populated — falling back to live HTTP fetch. "
            "Run `python -m irish_statute_assistant.indexer` to build the index."
        )
        return self._run_live(query)

    def _run_vector(self, query: str) -> ResearcherOutput:
        results = self._vector_store.search(query, top_k=10)
        if not results:
            raise ValueError(f"No Acts found for query: {query!r}")

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
        results = self._search(query)
        if not results:
            raise ValueError(f"No Acts found for query: {query!r}")

        acts = []
        for result in results:
            sections = self._fetch(result["url"], self._cache)
            acts.append(ActSection(
                title=result["title"],
                url=result["url"],
                sections=sections,
            ))

        return ResearcherOutput(acts=acts)
