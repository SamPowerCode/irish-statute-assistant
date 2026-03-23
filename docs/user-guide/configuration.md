# Configuration Reference

All settings are read from environment variables or a `.env` file in the
project root. Variable names are the upper-cased versions of the field names
listed below.

## LLM Provider

| Setting | Type | Default | Description |
|---|---|---|---|
| `LLM_PROVIDER` | `anthropic` \| `openai` \| `google` \| `groq` \| `ollama` | `anthropic` | The LLM provider to use |
| `ANTHROPIC_API_KEY` | string | — | Required when `LLM_PROVIDER=anthropic` |
| `OPENAI_API_KEY` | string | — | Required when `LLM_PROVIDER=openai` |
| `GOOGLE_API_KEY` | string | — | Required when `LLM_PROVIDER=google` |
| `GROQ_API_KEY` | string | — | Required when `LLM_PROVIDER=groq` |
| `OLLAMA_BASE_URL` | string | `http://localhost:11434` | Ollama server URL. Only used when `LLM_PROVIDER=ollama` |
| `MODEL_NAME` | string | *(provider default)* | Override the model. Defaults: `claude-sonnet-4-6`, `gpt-4o`, `gemini-2.0-flash`, `llama-3.3-70b-versatile`. **Required when `LLM_PROVIDER=ollama`** (no default). |
| `TEMPERATURE` | float | `0.0` | LLM temperature. 0.0 gives deterministic output |

## Vector Store

| Setting | Type | Default | Description |
|---|---|---|---|
| `VECTOR_STORE_BACKEND` | `chroma` \| `qdrant` | `chroma` | Which vector store backend to use |
| `CHROMA_DB_PATH` | string | `./data/chroma` | Path for the local ChromaDB store |
| `QDRANT_URL` | string | — | Qdrant cluster URL (required when backend=qdrant) |
| `QDRANT_API_KEY` | string | — | Qdrant API key |
| `EMBEDDING_MODEL` | string | `all-MiniLM-L6-v2` | Sentence-transformers model used for embeddings |
| `INDEX_CATEGORIES` | list | *(10 categories)* | Legal categories to index. Default: employment, housing, family, criminal, contract, personal injury, planning, company, tax, consumer |
| `ACTS_PER_CATEGORY` | int | `5` | Number of Acts to index per category |
| `HF_TOKEN` | string | — | Hugging Face token for downloading embedding models. Required in some environments if the model requires authentication. |

## Memory

| Setting | Type | Default | Description |
|---|---|---|---|
| `CONVERSATIONS_DB_PATH` | string | `~/.irish_statute_assistant/conversations.db` | SQLite file for conversation history |
| `PREFERENCES_DB_PATH` | string | `~/.irish_statute_assistant/preferences.db` | SQLite file for user preferences |
| `CONVERSATION_HISTORY_LIMIT` | int | `20` | Maximum number of prior exchanges to load into context |

## Pipeline Tuning

| Setting | Type | Default | Description |
|---|---|---|---|
| `EVALUATOR_PASS_THRESHOLD` | float | `0.7` | Minimum quality score for an answer to be accepted |
| `MAX_REFINEMENT_ROUNDS` | int | `2` | Maximum refinement attempts before returning the best answer found |
| `MAX_RETRIES` | int | `3` | Maximum retries when an LLM call fails schema validation |
| `TOKEN_BUDGET_PER_QUERY` | int | `100000` | Maximum tokens to spend across all agents for a single query |
| `RATE_LIMIT_DELAY` | float | `1.0` | Seconds to wait between HTTP requests to irishstatutebook.ie |
| `LOG_LEVEL` | string | `INFO` | Logging verbosity. Standard Python level names: `DEBUG`, `INFO`, `WARNING`, `ERROR` |
