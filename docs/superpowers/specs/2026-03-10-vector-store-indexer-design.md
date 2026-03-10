# Vector Store Indexer — Design Doc

**Date:** 2026-03-10
**Feature:** Replace live HTTP fetching in ResearcherAgent with local ChromaDB vector search
**Scope:** Additive — no changes to Supervisor, Pipeline, CLI, or any other agent

---

## 1. Overview

Replace the `ResearcherAgent`'s live HTTP calls to irishstatutebook.ie with a local ChromaDB vector store. A separate indexer script crawls the statute book once (by legal category), embeds each Act's sections, and persists them to disk. Subsequent queries do a local similarity search — no network, no rate limiting, no latency.

The `ResearcherAgent`'s public interface (`run(query) -> ResearcherOutput`) is unchanged. Everything downstream (Supervisor, Pipeline, CLI) is unaffected.

---

## 2. New Components

### `src/irish_statute_assistant/tools/vector_store.py`

ChromaDB wrapper. Single responsibility: given a query string, return the most relevant statute sections as `list[dict]`.

**Interface:**
```python
class VectorStore:
    def __init__(self, config: Config, embeddings=None) -> None: ...
    def is_populated(self) -> bool: ...
    def add_sections(self, sections: list[dict]) -> None: ...
    def search(self, query: str, top_k: int = 10) -> list[dict]: ...
```

**Constructor:** Reads `config.chroma_db_path` and `config.embedding_model`. Stores both as instance attributes (`self._persist_directory` and `self._embedding_function`) so they can be reused during the wipe-and-recreate in `add_sections`. If `embeddings` is provided it is used as `self._embedding_function` directly; otherwise a `HuggingFaceEmbeddings(model_name=config.embedding_model)` instance is created. The optional `embeddings` parameter exists solely to allow test injection of a stub — production code never passes it.

The ChromaDB collection is opened eagerly on construction using:
```python
self._chroma = Chroma(
    collection_name="irish_statutes",
    persist_directory=self._persist_directory,
    embedding_function=self._embedding_function,
)
```

`config.chroma_db_path` is a `str` relative to the process working directory. Tests should pass `str(tmp_path / "chroma")` as the override value.

**`add_sections` input:** A list of dicts with keys:
- `page_content` (str) — the section text
- `title` (str) — Act title
- `url` (str) — Act URL
- `section_index` (int) — 0-based position of this section within the Act's section list (0–9 at most, since `fetch_act_sections` caps at 10 sections)

**`add_sections` behaviour:** Wipes the entire `"irish_statutes"` collection on every call by deleting and recreating it:
```python
try:
    self._chroma._client.delete_collection("irish_statutes")
except Exception:
    pass  # collection does not exist yet — nothing to delete
self._chroma = Chroma(
    collection_name="irish_statutes",
    persist_directory=self._persist_directory,
    embedding_function=self._embedding_function,
)
```
The `try/except` guards against ChromaDB versions that raise when the collection does not yet exist. If called with an empty list the collection is wiped and `is_populated()` returns `False`. The indexer collects all sections across all Acts into one list and calls `add_sections` exactly once at the end of the run.

**`search` return:** If the store contains fewer than `top_k` documents, all available documents are returned — no error is raised. `page_content` values are returned as-is (no filtering of empty strings). Each dict in the returned list contains:
- `page_content` (str) — section text (may be empty string if stored that way)
- `title` (str) — Act title (from metadata)
- `url` (str) — Act URL (from metadata)
- `section_index` (int) — 0-based section position cast explicitly to `int` from metadata (ChromaDB may return metadata values as strings; `search` must call `int(metadata["section_index"])`)

**`is_populated()`** returns `True` only if the `"irish_statutes"` collection contains at least one document (a freshly created or wiped collection returns `False`).

Uses `langchain_chroma.Chroma` with `langchain_huggingface.HuggingFaceEmbeddings` (model: `all-MiniLM-L6-v2`).

### `src/irish_statute_assistant/indexer.py`

Standalone CLI script. Discovers Acts by category via the site's Solr API, fetches their content using the existing `statute_fetcher`, and stores all sections in the vector store in a single call.

**Required imports:**
```python
from irish_statute_assistant.config import Config
from irish_statute_assistant.tools.session_cache import SessionCache
from irish_statute_assistant.tools.statute_fetcher import fetch_act_sections, search_statutes
from irish_statute_assistant.tools.vector_store import VectorStore
```

**Run:**
```bash
python -m irish_statute_assistant.indexer
```

**Flow:**
1. For each category in `config.index_categories`, call `search_statutes(category)`. This returns at most 5 Act results per call. `acts_per_category` has an effective ceiling of 5 — the field exists to allow collecting fewer than 5 when desired.
2. Deduplicate Acts by URL using a global seen-URL set. For each category, skip any URL already in the seen set; add each new URL to the seen set and count it against that category's `acts_per_category` limit. Stop when `acts_per_category` new unique URLs have been collected for the current category. Example: if category A returned URLs [X, Y] and category B's `search_statutes` returns [X, Z, W], then for category B: X is skipped, Z and W are new and counted against B's limit.
3. Fetch each unique Act's sections using `fetch_act_sections(url, cache)` with a single fresh `SessionCache` shared across all Acts. `fetch_act_sections` returns at most 10 sections per Act.
4. Collect all sections across all Acts into a single list. Each entry: `{"page_content": text, "title": title, "url": url, "section_index": i}` where `i` is the 0-based index of the section within that Act's section list.
5. Call `VectorStore.add_sections(all_sections)` once.
6. Print: `Indexed {n} sections from {m} Acts across {k} categories.` where `n` = total sections, `m` = total unique Acts fetched (after global deduplication), `k` = `len(config.index_categories)` (all configured categories, regardless of yield).

---

## 3. Updated Component

### `src/irish_statute_assistant/agents/researcher.py`

Replaces the primary search path with `VectorStore.search()`, with a live-fetch fallback.

**Constructor:** `__init__(self, config: Config, cache: SessionCache)` — signature is unchanged. The following instance attributes are set:
- `self._config = config`
- `self._cache = cache`
- `self._search = search_statutes` (retained unchanged for fallback)
- `self._fetch = fetch_act_sections` (retained unchanged for fallback)
- `self._vector_store = VectorStore(config)`

**Fallback:** If `self._vector_store.is_populated()` returns `False`, falls back to the existing `self._search(query)` + `self._fetch(url, self._cache)` path (identical to the current implementation) and logs a warning via `logging.warning(...)`. Acts for which `_fetch` returns `[]` are included in `ResearcherOutput` with `sections=[]` (matching current behaviour — no filtering). If `_search` returns an empty list, raises `ValueError("No Acts found for query: {query!r}")`.

**Zero results (vector path):** If `self._vector_store.search(query)` returns an empty list, raises `ValueError("No Acts found for query: {query!r}")` before attempting to group results or construct `ResearcherOutput`.

**Result grouping (vector path):** After the empty-results guard, search results are grouped by `url`. For each unique URL (preserving first-seen order), one `ActSection` is constructed with:
- `title` from the first result for that URL
- `url` from the result
- `sections` = list of `page_content` values for all results with that URL, ordered by `section_index` ascending (numeric, since `search` casts to `int`). Empty-string `page_content` values are included as-is.

**Test injection:** Tests use `unittest.mock.patch("irish_statute_assistant.agents.researcher.VectorStore")` to control what is instantiated, or set `researcher._vector_store` directly after construction.

---

## 4. Config Additions

Four new fields added to `Config` in `config.py`:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `chroma_db_path` | `str` | `"./data/chroma"` | Disk path relative to CWD for the ChromaDB collection |
| `embedding_model` | `str` | `"all-MiniLM-L6-v2"` | sentence-transformers model name |
| `index_categories` | `list[str]` | See below | Legal categories to auto-discover |
| `acts_per_category` | `int` | `5` | Max Acts collected per category (effective ceiling: 5, set by `search_statutes`) |

Default categories:
```python
["employment", "housing", "family", "criminal", "contract",
 "personal injury", "planning", "company", "tax", "consumer"]
```

---

## 5. New Dependencies

```
langchain-chroma>=0.1
langchain-huggingface>=0.1
sentence-transformers>=3.0
```

Exact minimum versions should be pinned once confirmed working during implementation. `data/chroma/` added to `.gitignore` (the index is generated, not committed).

---

## 6. Data Flow

### Indexing (one-time)
```
python -m irish_statute_assistant.indexer
  → Config loads index_categories, acts_per_category, chroma_db_path
  → For each category: search_statutes(category) → up to 5 Act URLs (deduplicated globally)
  → For each unique Act URL: fetch_act_sections() → up to 10 section texts
  → Collect all sections into one list with {page_content, title, url, section_index}
  → VectorStore.add_sections(all_sections)  ← called once, wipes then inserts
  → ChromaDB persists to data/chroma/
```

### Querying (every user query)
```
ResearcherAgent.run(query)
  → VectorStore.is_populated()?
      yes → VectorStore.search(query, top_k=10)
            → if empty: raise ValueError("No Acts found…")
            → group by URL, order sections by section_index (int)
            → ResearcherOutput(acts=[ActSection(...)])
      no  → log warning → existing live-fetch fallback
```

---

## 7. Testing

- `test_vector_store.py` — unit tests using a `tmp_path` fixture. Pass `str(tmp_path / "chroma")` as `config.chroma_db_path`. Inject a stub embeddings object via the `embeddings` constructor parameter. The stub must implement `embed_documents(texts: list[str]) -> list[list[float]]` and `embed_query(text: str) -> list[float]` returning fixed-length float lists. Includes:
  - `is_populated()` returns `False` for a fresh (never-written) collection
  - `is_populated()` returns `True` after `add_sections()` with at least one document
  - `is_populated()` returns `False` after `add_sections([])` (empty-list wipe)
  - `search()` returns dicts with keys `page_content`, `title`, `url`, `section_index`
  - `search()` returns `section_index` as `int` (not str)
  - `add_sections()` wipes existing documents before inserting new ones (second call replaces first)

- `test_indexer.py` — unit tests that mock `VectorStore` (using `unittest.mock.patch`) and mock `search_statutes` and `fetch_act_sections`. Includes:
  - Acts are deduplicated by URL across categories (a URL from category A is skipped in category B)
  - `acts_per_category` limit is respected per category
  - `VectorStore.add_sections` is called exactly once with all sections from all Acts
  - Each section dict has correct `page_content`, `title`, `url`, and `section_index` (0-based)

- `test_researcher.py` — tests use `unittest.mock.patch("irish_statute_assistant.agents.researcher.VectorStore")` to inject a mock store. New test cases:
  - Vector store path: populated store returns `ResearcherOutput` with grouped `ActSection` objects; sections are ordered by `section_index`
  - Zero results from vector store: raises `ValueError("No Acts found…")`
  - Fallback path: `is_populated() == False` triggers live-fetch and logs a warning

---

## 8. Files Changed / Created

| File | Action |
|------|--------|
| `src/irish_statute_assistant/tools/vector_store.py` | Create |
| `src/irish_statute_assistant/indexer.py` | Create |
| `src/irish_statute_assistant/config.py` | Modify (4 new fields) |
| `src/irish_statute_assistant/agents/researcher.py` | Modify (vector store path + fallback) |
| `requirements.txt` | Modify (3 new deps) |
| `.gitignore` | Modify (add `data/chroma/`) |
| `tests/test_vector_store.py` | Create |
| `tests/test_indexer.py` | Create |
| `tests/test_researcher.py` | Modify (new test cases) |
