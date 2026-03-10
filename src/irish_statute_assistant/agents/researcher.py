from irish_statute_assistant.config import Config
from irish_statute_assistant.models.schemas import ActSection, ResearcherOutput
from irish_statute_assistant.tools.session_cache import SessionCache
from irish_statute_assistant.tools.statute_fetcher import fetch_act_sections, search_statutes


class ResearcherAgent:
    def __init__(self, config: Config, cache: SessionCache) -> None:
        self._cache = cache
        self._search = search_statutes
        self._fetch = fetch_act_sections

    def run(self, query: str) -> ResearcherOutput:
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
