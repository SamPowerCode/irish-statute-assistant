# Installation

## Requirements

- Python 3.10 or later
- [uv](https://docs.astral.sh/uv/getting-started/installation/) — the package manager used by this project

## Install dependencies

```bash
git clone <repo-url>
cd langflow-learning-club
uv sync
```

## Configure your API key

Copy the example environment file and add your key:

```bash
cp .env.example .env
```

Open `.env` and set the key for your chosen LLM provider:

```
# Anthropic (default)
ANTHROPIC_API_KEY=sk-ant-...

# Or OpenAI
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...

# Or Google
LLM_PROVIDER=google
GOOGLE_API_KEY=...

# Or Groq
LLM_PROVIDER=groq
GROQ_API_KEY=...

# Hugging Face (for embedding model download, if required)
HF_TOKEN=hf_...
```

See [Configuration](configuration.md) for the full list of settings.

## Choose a vector store

The assistant retrieves statute text from a vector store. Two backends are supported:

**ChromaDB (default — local, no extra setup):**

No configuration needed. The store is created at `./data/chroma` on first use.

**Qdrant (cloud or self-hosted):**

```
VECTOR_STORE_BACKEND=qdrant
QDRANT_URL=https://your-cluster.qdrant.io
QDRANT_API_KEY=your-key
```

## Index statutes (one-time setup)

Before running the assistant for the first time, build the vector store:

```bash
uv run python -m irish_statute_assistant.indexer
```

This crawls irishstatutebook.ie by legal category, embeds the statute sections, and writes them to the configured vector store. Re-run this command to refresh the index.
