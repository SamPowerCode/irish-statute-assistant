# Vector Store Indexer Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace live HTTP fetching in ResearcherAgent with a local ChromaDB vector store backed by a one-time indexer script.

**Architecture:** A new `VectorStore` wrapper handles ChromaDB reads and writes; a new `indexer.py` script populates it once by category; `ResearcherAgent` is updated to query the store first and fall back to live HTTP when the store is empty. The `ResearcherAgent`'s public interface is unchanged.

**Tech Stack:** `langchain-chroma`, `langchain-huggingface`, `sentence-transformers` (model `all-MiniLM-L6-v2`), ChromaDB persisted to `./data/chroma/`.

---

## Spec reference

`docs/superpowers/specs/2026-03-10-vector-store-indexer-design.md`

---

## File map

| File | Action |
|------|--------|
| `requirements.txt` | Add 3 deps |
| `.gitignore` | Add `data/chroma/` |
| `src/irish_statute_assistant/config.py` | Add 4 fields |
| `src/irish_statute_assistant/tools/vector_store.py` | Create |
| `src/irish_statute_assistant/indexer.py` | Create |
| `src/irish_statute_assistant/agents/researcher.py` | Update |
| `tests/test_vector_store.py` | Create |
| `tests/test_indexer.py` | Create |
| `tests/test_researcher.py` | Add test cases |

---

## Chunk 1: Config, dependencies, and VectorStore

### Task 1: Config fields, dependencies, .gitignore

**Files:**
- Modify: `requirements.txt`
- Modify: `.gitignore`
- Modify: `src/irish_statute_assistant/config.py`
- Test: `tests/test_config.py` (existing file — add test cases)

- [ ] **Step 1: Write failing config tests**

Open `tests/test_config.py` (existing). Add at the end:

```python
def test_config_chroma_defaults(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    config = Config()
    assert config.chroma_db_path == "./data/chroma"
    assert config.embedding_model == "all-MiniLM-L6-v2"
    assert config.acts_per_category == 5
    assert "employment" in config.index_categories
    assert "housing" in config.index_categories
```

- [ ] **Step 2: Run to verify failure**

```bash
cd /home/sam/projects/AI/langflow_learning_club
pytest tests/test_config.py::test_config_chroma_defaults -v
```
Expected: FAIL — `Config` has no `chroma_db_path` attribute.

- [ ] **Step 3: Add the four new fields to Config**

In `src/irish_statute_assistant/config.py`, add these four lines **before** the `model_config = ...` line (do not modify any existing fields):

```python
    chroma_db_path: str = "./data/chroma"
    embedding_model: str = "all-MiniLM-L6-v2"
    index_categories: list[str] = [
        "employment", "housing", "family", "criminal", "contract",
        "personal injury", "planning", "company", "tax", "consumer",
    ]
    acts_per_category: int = 5
```

After editing, the full file should look like:

```python
from pydantic_settings import BaseSettings


class Config(BaseSettings):
    anthropic_api_key: str
    model_name: str = "claude-sonnet-4-6"
    evaluator_pass_threshold: float = 0.7
    max_refinement_rounds: int = 2
    max_retries: int = 3
    token_budget_per_query: int = 4000
    rate_limit_delay: float = 1.0
    chroma_db_path: str = "./data/chroma"
    embedding_model: str = "all-MiniLM-L6-v2"
    index_categories: list[str] = [
        "employment", "housing", "family", "criminal", "contract",
        "personal injury", "planning", "company", "tax", "consumer",
    ]
    acts_per_category: int = 5

    model_config = {"env_file": ".env"}
```

- [ ] **Step 4: Verify test passes**

```bash
pytest tests/test_config.py::test_config_chroma_defaults -v
```
Expected: PASS.

- [ ] **Step 5: Add dependencies to requirements.txt**

```
langchain-chroma>=0.1
langchain-huggingface>=0.1
sentence-transformers>=3.0
```

- [ ] **Step 6: Add data/chroma/ to .gitignore**

Add this line to `.gitignore`:
```
data/chroma/
```

- [ ] **Step 7: Install new dependencies**

```bash
pip install langchain-chroma langchain-huggingface sentence-transformers
```

- [ ] **Step 8: Run full test suite to verify nothing broke**

```bash
pytest tests/ -v
```
Expected: all existing tests pass.

- [ ] **Step 9: Commit**

```bash
git add requirements.txt .gitignore src/irish_statute_assistant/config.py tests/test_config.py
git commit -m "feat: add vector store config fields and dependencies"
```

---

### Task 2: VectorStore

**Files:**
- Create: `src/irish_statute_assistant/tools/vector_store.py`
- Create: `tests/test_vector_store.py`

> **Note on `delete_collection`:** The wipe in `add_sections` uses `self._chroma._client.delete_collection(_COLLECTION_NAME)` — a direct call on the ChromaDB client — rather than `self._chroma.delete_collection()`. The public wrapper method's behaviour varies across `langchain-chroma` minor versions and may leave `self._chroma` in an inconsistent state before the reassignment. The `_client` call reliably removes the underlying collection, and the following `Chroma(...)` constructor always creates a fresh one.

---

- [ ] **Step 1: Write test — is_populated() on fresh store returns False**

Create `tests/test_vector_store.py`:

```python
import pytest
from irish_statute_assistant.config import Config
from irish_statute_assistant.tools.vector_store import VectorStore


class _StubEmbeddings:
    """4-dim fixed-vector stub. Avoids downloading sentence-transformer models in tests."""
    dim = 4

    def embed_documents(self, texts):
        return [[1.0] * self.dim for _ in texts]

    def embed_query(self, text):
        return [1.0] * self.dim


def make_store(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    config = Config(chroma_db_path=str(tmp_path / "chroma"))
    return VectorStore(config, embeddings=_StubEmbeddings())


def test_is_populated_false_on_fresh_store(tmp_path, monkeypatch):
    store = make_store(tmp_path, monkeypatch)
    assert store.is_populated() is False
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/test_vector_store.py::test_is_populated_false_on_fresh_store -v
```
Expected: FAIL — `VectorStore` not importable.

- [ ] **Step 3: Implement VectorStore (constructor + is_populated only)**

Create `src/irish_statute_assistant/tools/vector_store.py`:

```python
from __future__ import annotations

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from irish_statute_assistant.config import Config

_COLLECTION_NAME = "irish_statutes"


class VectorStore:
    def __init__(self, config: Config, embeddings=None) -> None:
        self._persist_directory = config.chroma_db_path
        if embeddings is None:
            embeddings = HuggingFaceEmbeddings(model_name=config.embedding_model)
        self._embedding_function = embeddings
        self._chroma = Chroma(
            collection_name=_COLLECTION_NAME,
            persist_directory=self._persist_directory,
            embedding_function=self._embedding_function,
        )

    def is_populated(self) -> bool:
        return self._chroma._collection.count() > 0

    def add_sections(self, sections: list[dict]) -> None:
        raise NotImplementedError

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        raise NotImplementedError
```

- [ ] **Step 4: Run test — must pass**

```bash
pytest tests/test_vector_store.py::test_is_populated_false_on_fresh_store -v
```
Expected: PASS.

- [ ] **Step 5: Write remaining 5 test cases (before implementing add_sections and search)**

Add to `tests/test_vector_store.py`:

```python
def test_is_populated_true_after_add_sections(tmp_path, monkeypatch):
    store = make_store(tmp_path, monkeypatch)
    store.add_sections([
        {"page_content": "An employer must give notice.", "title": "Employment Act 2001",
         "url": "https://example.com/act1", "section_index": 0},
    ])
    assert store.is_populated() is True


def test_is_populated_false_after_add_sections_empty_list(tmp_path, monkeypatch):
    store = make_store(tmp_path, monkeypatch)
    store.add_sections([
        {"page_content": "Some text.", "title": "Act A",
         "url": "https://example.com/a", "section_index": 0},
    ])
    store.add_sections([])  # wipe
    assert store.is_populated() is False


def test_search_returns_correct_keys(tmp_path, monkeypatch):
    store = make_store(tmp_path, monkeypatch)
    store.add_sections([
        {"page_content": "Tenant rights.", "title": "Housing Act 1992",
         "url": "https://example.com/housing", "section_index": 2},
    ])
    results = store.search("tenant", top_k=5)
    assert len(results) == 1
    assert results[0]["page_content"] == "Tenant rights."
    assert results[0]["title"] == "Housing Act 1992"
    assert results[0]["url"] == "https://example.com/housing"
    assert results[0]["section_index"] == 2


def test_search_returns_section_index_as_int(tmp_path, monkeypatch):
    store = make_store(tmp_path, monkeypatch)
    store.add_sections([
        {"page_content": "Section text.", "title": "Act B",
         "url": "https://example.com/b", "section_index": 7},
    ])
    results = store.search("text", top_k=5)
    assert isinstance(results[0]["section_index"], int)
    assert results[0]["section_index"] == 7


def test_add_sections_wipes_on_second_call(tmp_path, monkeypatch):
    store = make_store(tmp_path, monkeypatch)
    store.add_sections([
        {"page_content": "Old content.", "title": "Old Act",
         "url": "https://example.com/old", "section_index": 0},
    ])
    store.add_sections([
        {"page_content": "New content.", "title": "New Act",
         "url": "https://example.com/new", "section_index": 0},
    ])
    results = store.search("content", top_k=10)
    assert len(results) == 1
    assert results[0]["title"] == "New Act"
```

- [ ] **Step 6: Run remaining tests to verify they all fail**

```bash
pytest tests/test_vector_store.py -v
```
Expected: 5 tests FAIL with `NotImplementedError` (the `is_populated_false_on_fresh_store` test still passes).

- [ ] **Step 7: Implement add_sections and search**

Replace `add_sections` and `search` in `vector_store.py` with:

```python
    def add_sections(self, sections: list[dict]) -> None:
        try:
            self._chroma._client.delete_collection(_COLLECTION_NAME)
        except Exception:
            pass  # collection does not yet exist — nothing to delete
        self._chroma = Chroma(
            collection_name=_COLLECTION_NAME,
            persist_directory=self._persist_directory,
            embedding_function=self._embedding_function,
        )
        if sections:
            self._chroma.add_texts(
                texts=[s["page_content"] for s in sections],
                metadatas=[
                    {
                        "title": s["title"],
                        "url": s["url"],
                        "section_index": s["section_index"],
                    }
                    for s in sections
                ],
            )

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        docs = self._chroma.similarity_search(query, k=top_k)
        return [
            {
                "page_content": doc.page_content,
                "title": doc.metadata["title"],
                "url": doc.metadata["url"],
                "section_index": int(doc.metadata["section_index"]),
            }
            for doc in docs
        ]
```

- [ ] **Step 8: Run all vector store tests**

```bash
pytest tests/test_vector_store.py -v
```
Expected: all 6 tests PASS.

- [ ] **Step 9: Run full suite**

```bash
pytest tests/ -v
```
Expected: all tests pass.

- [ ] **Step 10: Commit**

```bash
git add src/irish_statute_assistant/tools/vector_store.py tests/test_vector_store.py
git commit -m "feat: add VectorStore ChromaDB wrapper with stub-embeddings tests"
```

---

## Chunk 2: Indexer and ResearcherAgent update

### Task 3: Indexer

**Files:**
- Create: `src/irish_statute_assistant/indexer.py`
- Create: `tests/test_indexer.py`

- [ ] **Step 1: Write failing indexer tests**

Create `tests/test_indexer.py`:

```python
from unittest.mock import MagicMock, patch

from irish_statute_assistant import indexer as indexer_module


def run_indexer(search_side_effect, fetch_side_effect, acts_per_category=5,
                index_categories=None):
    """Helper: runs indexer.main() with mocked dependencies.

    Returns (sections_passed_to_add_sections, mock_store).

    Uses module-level import + targeted patches so mock bindings survive the
    call (importlib.reload would re-execute the module's imports and overwrite
    the mocks).
    """
    if index_categories is None:
        index_categories = ["employment", "housing"]

    mock_store = MagicMock()
    captured = {}

    def capture_add(sections):
        captured["sections"] = sections

    mock_store.add_sections.side_effect = capture_add

    mock_config = MagicMock()
    mock_config.acts_per_category = acts_per_category
    mock_config.index_categories = index_categories

    with patch("irish_statute_assistant.indexer.search_statutes", side_effect=search_side_effect), \
         patch("irish_statute_assistant.indexer.fetch_act_sections", side_effect=fetch_side_effect), \
         patch("irish_statute_assistant.indexer.VectorStore", return_value=mock_store), \
         patch("irish_statute_assistant.indexer.Config", return_value=mock_config):
        indexer_module.main()

    return captured.get("sections", []), mock_store


def test_indexer_deduplicates_across_categories():
    # URL X appears in both categories — should only be fetched once
    search_results = [
        [{"title": "Act X", "url": "https://example.com/X"},
         {"title": "Act Y", "url": "https://example.com/Y"}],  # employment
        [{"title": "Act X", "url": "https://example.com/X"},
         {"title": "Act Z", "url": "https://example.com/Z"}],  # housing
    ]
    fetch_results = [
        ["Section 1 of X"],  # fetch for X
        ["Section 1 of Y"],  # fetch for Y
        ["Section 1 of Z"],  # fetch for Z
    ]
    sections, mock_store = run_indexer(search_results, fetch_results)
    urls = [s["url"] for s in sections]
    assert urls.count("https://example.com/X") == 1  # fetched once


def test_indexer_respects_acts_per_category():
    # acts_per_category=1: only first unique Act per category
    search_results = [
        [{"title": "Act A", "url": "https://example.com/A"},
         {"title": "Act B", "url": "https://example.com/B"}],  # employment: only A collected
        [{"title": "Act C", "url": "https://example.com/C"},
         {"title": "Act D", "url": "https://example.com/D"}],  # housing: only C collected
    ]
    fetch_results = [
        ["Section of A"],
        ["Section of C"],
    ]
    sections, _ = run_indexer(search_results, fetch_results, acts_per_category=1)
    collected_urls = {s["url"] for s in sections}
    assert "https://example.com/A" in collected_urls
    assert "https://example.com/C" in collected_urls
    assert "https://example.com/B" not in collected_urls
    assert "https://example.com/D" not in collected_urls


def test_indexer_calls_add_sections_once():
    search_results = [
        [{"title": "Act A", "url": "https://example.com/A"}],
        [{"title": "Act B", "url": "https://example.com/B"}],
    ]
    fetch_results = [["S1"], ["S2"]]
    _, mock_store = run_indexer(search_results, fetch_results)
    assert mock_store.add_sections.call_count == 1


def test_indexer_section_dict_shape():
    search_results = [
        [{"title": "Employment Act 2001", "url": "https://example.com/emp"}],
        [],  # housing has no results
    ]
    fetch_results = [["Workers get notice.", "Holiday entitlement applies."]]
    sections, _ = run_indexer(search_results, fetch_results)
    assert len(sections) == 2
    s0 = sections[0]
    assert s0["page_content"] == "Workers get notice."
    assert s0["title"] == "Employment Act 2001"
    assert s0["url"] == "https://example.com/emp"
    assert s0["section_index"] == 0
    assert sections[1]["section_index"] == 1
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/test_indexer.py -v
```
Expected: FAIL — `irish_statute_assistant.indexer` not importable.

- [ ] **Step 3: Implement indexer.py**

Create `src/irish_statute_assistant/indexer.py`:

```python
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
```

- [ ] **Step 4: Run indexer tests**

```bash
pytest tests/test_indexer.py -v
```
Expected: all 4 tests PASS.

- [ ] **Step 5: Run full suite**

```bash
pytest tests/ -v
```
Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/irish_statute_assistant/indexer.py tests/test_indexer.py
git commit -m "feat: add indexer script with category-based Act discovery"
```

---

### Task 4: Update ResearcherAgent

**Files:**
- Modify: `src/irish_statute_assistant/agents/researcher.py`
- Modify: `tests/test_researcher.py`

The new `ResearcherAgent` must:
1. Instantiate `VectorStore(config)` in `__init__`
2. In `run()`, check `_vector_store.is_populated()` first
3. If populated: use `_vector_store.search()`, group results, raise `ValueError` on empty
4. If not populated: log a warning, use the existing `_search` + `_fetch` path

- [ ] **Step 1: Write failing tests for the vector store path**

First, update the existing `make_researcher_with_mocks` helper at the top of `tests/test_researcher.py`. The updated `ResearcherAgent.__init__` will assign `self._vector_store`, so any test that uses `__new__` to bypass `__init__` must set `_vector_store` manually or it will get `AttributeError` when `run()` calls `self._vector_store.is_populated()`. Update the helper to add a mock store that always returns `False` for `is_populated` (forcing the live-fetch fallback, preserving the existing tests' behaviour):

```python
def make_researcher_with_mocks(search_results, sections_per_act):
    cache = SessionCache()
    agent = ResearcherAgent.__new__(ResearcherAgent)
    agent._cache = cache

    # _vector_store required by run() after the vector-store update.
    # Set is_populated=False so existing tests continue to exercise the live-fetch path.
    from unittest.mock import MagicMock
    mock_store = MagicMock()
    mock_store.is_populated.return_value = False
    agent._vector_store = mock_store

    mock_search = MagicMock(return_value=search_results)
    mock_fetch = MagicMock(return_value=sections_per_act)
    agent._search = mock_search
    agent._fetch = mock_fetch
    return agent
```

Then add new test cases at the end of `tests/test_researcher.py`:

```python
import logging
from unittest.mock import MagicMock, patch


def make_researcher_with_vector_store(mock_store, search_results=None, sections_per_act=None):
    """Create a ResearcherAgent with a mocked VectorStore and mocked live-fetch."""
    from irish_statute_assistant.config import Config
    from irish_statute_assistant.tools.session_cache import SessionCache
    import os
    os.environ.setdefault("ANTHROPIC_API_KEY", "test")

    with patch("irish_statute_assistant.agents.researcher.VectorStore", return_value=mock_store):
        config = Config()
        cache = SessionCache()
        agent = ResearcherAgent(config, cache)

    # Also wire live-fetch mocks for fallback tests
    if search_results is not None:
        agent._search = MagicMock(return_value=search_results)
    if sections_per_act is not None:
        agent._fetch = MagicMock(return_value=sections_per_act)
    return agent


def test_researcher_uses_vector_store_when_populated():
    mock_store = MagicMock()
    mock_store.is_populated.return_value = True
    mock_store.search.return_value = [
        {"page_content": "Workers must receive notice.", "title": "Employment Act 2001",
         "url": "https://example.com/emp", "section_index": 0},
        {"page_content": "Notice period is 4 weeks.", "title": "Employment Act 2001",
         "url": "https://example.com/emp", "section_index": 1},
    ]
    agent = make_researcher_with_vector_store(mock_store)
    result = agent.run("employment notice period")
    assert isinstance(result, ResearcherOutput)
    assert len(result.acts) == 1
    assert result.acts[0].title == "Employment Act 2001"
    assert result.acts[0].url == "https://example.com/emp"
    assert result.acts[0].sections[0] == "Workers must receive notice."
    assert result.acts[0].sections[1] == "Notice period is 4 weeks."


def test_researcher_sections_ordered_by_section_index():
    mock_store = MagicMock()
    mock_store.is_populated.return_value = True
    # Return results out of order
    mock_store.search.return_value = [
        {"page_content": "Second section.", "title": "Act A",
         "url": "https://example.com/a", "section_index": 1},
        {"page_content": "First section.", "title": "Act A",
         "url": "https://example.com/a", "section_index": 0},
    ]
    agent = make_researcher_with_vector_store(mock_store)
    result = agent.run("some query")
    assert result.acts[0].sections[0] == "First section."
    assert result.acts[0].sections[1] == "Second section."


def test_researcher_raises_on_empty_vector_results():
    mock_store = MagicMock()
    mock_store.is_populated.return_value = True
    mock_store.search.return_value = []
    agent = make_researcher_with_vector_store(mock_store)
    import pytest
    with pytest.raises(ValueError, match="No Acts found"):
        agent.run("unknown topic xyz")


def test_researcher_falls_back_when_store_not_populated(caplog):
    mock_store = MagicMock()
    mock_store.is_populated.return_value = False
    agent = make_researcher_with_vector_store(
        mock_store,
        search_results=[{"title": "Act A", "url": "https://example.com/a"}],
        sections_per_act=["Live section text."],
    )
    with caplog.at_level(logging.WARNING):
        result = agent.run("employment law")
    assert isinstance(result, ResearcherOutput)
    assert result.acts[0].sections[0] == "Live section text."
    assert any("vector store" in msg.lower() or "populated" in msg.lower()
               for msg in caplog.messages)
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/test_researcher.py::test_researcher_uses_vector_store_when_populated -v
```
Expected: FAIL — `ResearcherAgent.__init__` does not create a `VectorStore`.

- [ ] **Step 3: Update researcher.py**

Replace `src/irish_statute_assistant/agents/researcher.py` with:

```python
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
```

- [ ] **Step 4: Run all researcher tests**

```bash
pytest tests/test_researcher.py -v
```
Expected: all 7 tests PASS (3 original + 4 new).

**If `test_researcher_falls_back_when_store_not_populated` fails on caplog assertion:** The `ResearcherAgent` uses `logging.getLogger(__name__)` which is `irish_statute_assistant.agents.researcher`. Ensure the test uses `caplog.at_level(logging.WARNING, logger="irish_statute_assistant.agents.researcher")` if needed.

- [ ] **Step 5: Run full suite**

```bash
pytest tests/ -v
```
Expected: all tests pass. Note the exact count — it should be 51 (original) + config test (1) + vector store tests (6) + indexer tests (4) + researcher tests (4) = 66 tests.

- [ ] **Step 6: Commit**

```bash
git add src/irish_statute_assistant/agents/researcher.py tests/test_researcher.py
git commit -m "feat: update ResearcherAgent to query ChromaDB vector store with live-fetch fallback"
```

---

## Final verification

- [ ] **Run full test suite one more time**

```bash
pytest tests/ -v --tb=short
```
Expected: all tests pass.

- [ ] **Smoke-test the indexer (optional — requires network)**

```bash
python -m irish_statute_assistant.indexer
```
Expected output: `Indexed N sections from M Acts across 10 categories.`
