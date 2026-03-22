"""Indexer — builds the ChromaDB vector store from irishstatutebook.ie.

Run:
    python -m irish_statute_assistant.indexer
"""

from __future__ import annotations

import logging

from irish_statute_assistant.config import Config
from irish_statute_assistant.tools.session_cache import SessionCache
from irish_statute_assistant.tools.statute_fetcher import StatuteFetcher
from irish_statute_assistant.tools.vector_store import get_vector_store

logger = logging.getLogger(__name__)


def main() -> None:
    config = Config()
    logging.basicConfig(
        level=getattr(logging, config.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s — %(message)s",
    )
    logger.info("Starting indexer — %d categories, up to %d acts each",
                len(config.index_categories), config.acts_per_category)

    store = get_vector_store(config)
    cache = SessionCache()
    fetcher = StatuteFetcher(
        rate_limit_delay=config.rate_limit_delay,
        max_retries=config.max_retries,
    )

    seen_urls: set[str] = set()
    all_sections: list[dict] = []
    total_acts = 0

    for category in config.index_categories:
        logger.info("Category: %s", category)
        results = fetcher.search(category, limit=config.acts_per_category)
        collected = 0
        for result in results:
            if collected >= config.acts_per_category:
                break
            url = result["url"]
            if url in seen_urls:
                logger.debug("Skipping duplicate: %s", url)
                continue
            seen_urls.add(url)
            collected += 1
            total_acts += 1

            logger.info("  Fetching [%d] %s", total_acts, result["title"])
            sections = fetcher.fetch(url, cache)
            logger.info("    → %d sections", len(sections))
            for i, text in enumerate(sections):
                all_sections.append({
                    "page_content": text,
                    "title": result["title"],
                    "url": url,
                    "section_index": i,
                })

    logger.info("Adding %d sections to vector store…", len(all_sections))
    store.add_sections(all_sections)
    k = len(config.index_categories)
    logger.info("Done — indexed %d sections from %d Acts across %d categories.",
                len(all_sections), total_acts, k)


if __name__ == "__main__":
    main()
