# Irish Statute Assistant

An AI assistant that answers plain-English questions about Irish law, powered by a
multi-agent pipeline that retrieves, analyses, and verifies statute text from
[irishstatutebook.ie](https://www.irishstatutebook.ie).

**[Full documentation →](https://irish-statute-assistant.readthedocs.io)**

---

## How it works

Eight agents orchestrated by a Supervisor:

| Agent | Role |
|---|---|
| **Clarifier** | Asks one focused question when a query is ambiguous |
| **Researcher** | Retrieves relevant Irish Acts from the vector store |
| **Analyst** | Identifies key clauses, assigns act/section citations, scores confidence |
| **Devil's Advocate** | Challenges the analyst's conclusions before the writer proceeds |
| **Writer** | Produces a short answer (≤100 words) and a detailed breakdown |
| **Grounding Checker** | Verifies every cited clause is traceable to retrieved statute text |
| **Evaluator** | Scores quality and triggers a refinement loop if below threshold |
| **Supervisor** | Orchestrates all agents, owns memory writes, detects user preferences |

Conversation history and user preferences persist across sessions in SQLite.
The assistant automatically retries with evaluator feedback when output quality
falls below threshold.

### Reliability features

- **Typed exception hierarchy** — `StatuteNotFoundError`, `BudgetExceededError`, `ValidationRepairError`
- **Validation retry** — failed schema validations are retried up to `MAX_RETRIES` times
- **Token budget** — `QueryContext` tracks usage per query and raises `BudgetExceededError` if exceeded
- **Multi-provider** — Anthropic, OpenAI, Google, Groq, and Ollama (local) supported

---

## Quick start

**Requirements:** Python 3.10+, [uv](https://docs.astral.sh/uv/getting-started/installation/)

```bash
git clone <repo-url>
cd langflow-learning-club
uv sync
cp .env.example .env
# Add your ANTHROPIC_API_KEY (or other provider key) to .env
```

**Index statutes (one-time):**

```bash
uv run python -m irish_statute_assistant.indexer
```

**Run:**

```bash
uv run python -m irish_statute_assistant.main
```

**Run (Streamlit UI):**

```bash
uv run --extra ui streamlit run app.py
```
Opens a browser UI at `http://localhost:8501` with a live pipeline trace sidebar.
(The `--extra ui` flag installs Streamlit on the fly if not already installed.)

**Test:**

```bash
uv run python -m pytest
```

---

## Vector store backends

| Backend | Setup | Best for |
|---|---|---|
| **ChromaDB** (default) | No extra setup — local files at `./data/chroma` | Development, local use |
| **Qdrant** | Set `VECTOR_STORE_BACKEND=qdrant`, `QDRANT_URL`, `QDRANT_API_KEY` | Production, cloud deployment |

---

## Configuration

Key settings (set in `.env` or as environment variables):

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `anthropic` | Provider: `anthropic`, `openai`, `google`, `groq`, `ollama` |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL. Only used when `LLM_PROVIDER=ollama` |
| `ANTHROPIC_API_KEY` | — | Required when `LLM_PROVIDER=anthropic`. Other providers need their own key — see full docs |
| `VECTOR_STORE_BACKEND` | `chroma` | `chroma` or `qdrant` |
| `EVALUATOR_PASS_THRESHOLD` | `0.7` | Minimum quality score to accept an answer |
| `MAX_REFINEMENT_ROUNDS` | `2` | Refinement retries before returning best attempt |
| `TOKEN_BUDGET_PER_QUERY` | `20000` | Token limit per query across all agents |

See the [full configuration reference](https://irish-statute-assistant.readthedocs.io/user-guide/configuration.html) for all settings.
