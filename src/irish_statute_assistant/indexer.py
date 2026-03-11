"""Indexer — builds the ChromaDB vector store from irishstatutebook.ie.

Run:
    python -m irish_statute_assistant.indexer
"""

from __future__ import annotations

from irish_statute_assistant.config import Config
from irish_statute_assistant.tools.session_cache import SessionCache
from irish_statute_assistant.tools.statute_fetcher import fetch_act_sections, search_statutes
from irish_statute_assistant.tools.vector_store import VectorStore


def main() -> None:
    config = Config()
    store = VectorStore(config)
    cache = SessionCache()

    seen_urls: set[str] = set()
    all_sections: list[dict] = []
    total_acts = 0

    for category in config.index_categories:
        results = search_statutes(category)
        collected = 0
        for result in results:
            if collected >= config.acts_per_category:
                break
            url = result["url"]
            if url in seen_urls:
                continue
            seen_urls.add(url)
            collected += 1
            total_acts += 1

            sections = fetch_act_sections(url, cache)
            for i, text in enumerate(sections):
                all_sections.append({
                    "page_content": text,
                    "title": result["title"],
                    "url": url,
                    "section_index": i,
                })

    store.add_sections(all_sections)
    k = len(config.index_categories)
    print(f"Indexed {len(all_sections)} sections from {total_acts} Acts across {k} categories.")


if __name__ == "__main__":
    main()
