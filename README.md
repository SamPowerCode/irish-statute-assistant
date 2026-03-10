# Irish Statute Research Assistant

A multi-agent AI system that answers natural language legal questions in plain English using live Irish statute law from [irishstatutebook.ie](https://www.irishstatutebook.ie).

Built with LangChain and Claude as a 7-week agentic AI capstone project (Gold level).

---

## How it works

Six agents orchestrated by a Supervisor:

| Agent | Role |
|-------|------|
| **Supervisor** | Routes queries, manages the refinement loop |
| **Clarifier** | Asks one focused question when a query is too vague |
| **Legal Researcher** | Searches and fetches relevant Acts from irishstatutebook.ie |
| **Legal Analyst** | Interprets statute text, identifies key clauses, assigns a confidence score |
| **Plain English Writer** | Produces a short answer (≤100 words) and a detailed breakdown |
| **Evaluator** | Scores the output and triggers a refinement loop if quality falls below threshold |

Session memory keeps the conversation coherent within a single run. The system asks a clarifying question when needed, and automatically retries with feedback when the Evaluator scores the answer too low.

---

## Setup

**Requirements:** Python 3.10+

```bash
pip install -r requirements.txt
cp .env.example .env
# Add your ANTHROPIC_API_KEY to .env
```

## Run

```bash
python3 -m irish_statute_assistant.main
```

## Test

```bash
pytest tests/ -v
```

---

## Project structure

```
src/irish_statute_assistant/
  config.py          # Settings (API key, thresholds, rate limits)
  pipeline.py        # Top-level entry point
  main.py            # Interactive CLI
  agents/
    supervisor.py    # Orchestration and refinement loop
    clarifier.py     # Query disambiguation
    researcher.py    # Live statute retrieval
    analyst.py       # Legal interpretation
    writer.py        # Plain English output
    evaluator.py     # Quality scoring
  models/
    schemas.py       # Pydantic models for all agent I/O
  tools/
    statute_fetcher.py   # Fetches from irishstatutebook.ie (Solr API + HTML)
    session_cache.py     # In-session URL cache
  memory/
    session_memory.py    # Conversation history
tests/               # 51 tests, all passing
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
| `MAX_RETRIES` | `3` | HTTP retry attempts on transient errors |
| `TOKEN_BUDGET_PER_QUERY` | `4000` | Token cap per query |
| `RATE_LIMIT_DELAY` | `1.0` | Seconds between statute book requests |
