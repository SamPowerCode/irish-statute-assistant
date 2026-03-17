# Irish Statute Research Assistant

A multi-agent AI system that answers natural language legal questions in plain English using Irish statute law from [irishstatutebook.ie](https://www.irishstatutebook.ie), retrieved via a vector store (ChromaDB locally or Qdrant Cloud).

Built with LangChain and Claude as a 7-week agentic AI capstone project (Gold level).

---

## How it works

Six agents orchestrated by a Supervisor:

| Agent | Role |
|-------|------|
| **Supervisor** | Routes queries, manages the refinement loop, enforces token budget |
| **Clarifier** | Asks one focused question when a query is too vague |
| **Legal Researcher** | Queries the vector store (built from irishstatutebook.ie), falls back to live HTTP fetch if the store is empty |
| **Legal Analyst** | Interprets statute text, identifies key clauses, assigns a confidence score |
| **Plain English Writer** | Produces a short answer (≤100 words) and a detailed breakdown |
| **Evaluator** | Scores the output and triggers a refinement loop if quality falls below threshold |

Session memory keeps the conversation coherent within a single run. The system asks a clarifying question when needed, and automatically retries with feedback when the Evaluator scores the answer too low.

### Reliability features

- **Typed exception hierarchy** — `StatuteNotFoundError`, `BudgetExceededError`, `ValidationRepairError`, and others for precise error handling
- **Validation retry** — agent calls that fail schema validation are retried up to `MAX_RETRIES` times before raising `ValidationRepairError`
- **Token budget enforcement** — `QueryContext` tracks token usage across agents per query and raises `BudgetExceededError` if `TOKEN_BUDGET_PER_QUERY` is exceeded
- **Config-wired rate limiting** — `RATE_LIMIT_DELAY` and `MAX_RETRIES` are injected into the statute fetcher at runtime

---

## Setup

**Requirements:** Python 3.10+, [uv](https://docs.astral.sh/uv/getting-started/installation/)

```bash
uv sync
cp .env.example .env
# Add your ANTHROPIC_API_KEY to .env
```

## Index statutes (one-time)

Build the vector store before running the assistant:

```bash
uv run python -m irish_statute_assistant.indexer
```

This crawls irishstatutebook.ie by legal category, embeds the statute sections, and persists them to the configured vector store. Re-run to refresh the index.

## Run

```bash
uv run python -m irish_statute_assistant.main
```

## Test

```bash
uv run pytest tests/ -v
```

---

## Vector store backends

The assistant supports two vector store backends, selected via `VECTOR_STORE_BACKEND`:

### ChromaDB (default — local)

No extra setup needed. The index is persisted to `CHROMA_DB_PATH` on disk.

```env
VECTOR_STORE_BACKEND=chroma
CHROMA_DB_PATH=./data/chroma
```

### Qdrant Cloud

Sign up for a free cluster at [qdrant.io](https://qdrant.io), then add to `.env`:

```env
VECTOR_STORE_BACKEND=qdrant
QDRANT_URL=https://your-cluster.qdrant.io
QDRANT_API_KEY=your-api-key
```

Leave `QDRANT_URL` empty (or unset) to use Qdrant in-memory mode, which is useful for local development and testing without a running server.

---

## Project structure

```
src/irish_statute_assistant/
  config.py          # Settings (API key, thresholds, rate limits, vector store)
  pipeline.py        # Top-level entry point; creates QueryContext per query
  main.py            # Interactive CLI with typed exception handling
  exceptions.py      # Typed exception hierarchy (IrishStatuteError and subclasses)
  context.py         # QueryContext — tracks token budget across a query
  retry.py           # run_with_retry helper for validation error retries
  agents/
    base_agent.py    # BaseAgent with token-usage tracking (TokenUsageCallback)
    supervisor.py    # Orchestration, refinement loop, budget/retry wiring
    clarifier.py     # Query disambiguation
    researcher.py    # Vector store search with live HTTP fallback
    analyst.py       # Legal interpretation
    writer.py        # Plain English output
    evaluator.py     # Quality scoring
  models/
    schemas.py       # Pydantic models for all agent I/O
  tools/
    statute_fetcher.py       # StatuteFetcher class (Solr API + HTML, config-wired retries)
    session_cache.py         # In-session URL cache
    vector_store.py          # ChromaDB backend + get_vector_store() factory
    qdrant_vector_store.py   # Qdrant backend (local or cloud)
  memory/
    session_memory.py    # Conversation history
  indexer.py             # One-time script to build the vector store
tests/               # 80 tests, all passing
```

---

## Configuration

All settings can be overridden in `.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | — | Required |
| `MODEL_NAME` | `claude-sonnet-4-6` | Claude model to use |
| `EVALUATOR_PASS_THRESHOLD` | `0.7` | Minimum score to accept an answer |
| `MAX_REFINEMENT_ROUNDS` | `2` | Max retry attempts on low-scoring answers |
| `MAX_RETRIES` | `3` | Retry attempts for transient/validation errors |
| `TOKEN_BUDGET_PER_QUERY` | `4000` | Token cap per query |
| `RATE_LIMIT_DELAY` | `1.0` | Seconds between statute book requests |
| `VECTOR_STORE_BACKEND` | `chroma` | Vector store backend: `chroma` or `qdrant` |
| `CHROMA_DB_PATH` | `./data/chroma` | Disk path for the ChromaDB vector store |
| `QDRANT_URL` | `` | Qdrant server URL (empty = in-memory) |
| `QDRANT_API_KEY` | `` | Qdrant Cloud API key |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | sentence-transformers model for embeddings |
| `INDEX_CATEGORIES` | 10 legal categories | Categories crawled by the indexer |
| `ACTS_PER_CATEGORY` | `5` | Max Acts collected per category during indexing |
