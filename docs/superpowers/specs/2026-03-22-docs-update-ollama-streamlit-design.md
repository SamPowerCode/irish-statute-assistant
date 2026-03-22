# Documentation Update: Ollama Provider + Streamlit UI — Design Spec

**Date:** 2026-03-22
**Status:** Approved

---

## Overview

Surgical updates to four user-facing documentation files to reflect two new features:
1. **Ollama provider** — local LLM support via `LLM_PROVIDER=ollama`, no API key required
2. **Streamlit UI** — optional web interface (`[ui]` extra) with live pipeline trace sidebar

---

## Scope

Four files only. No restructuring — add the missing content where it fits naturally.

| File | What changes |
|---|---|
| `README.md` | Provider list, config table, Streamlit run command |
| `docs/user-guide/installation.md` | Ollama config block, Streamlit optional install section |
| `docs/user-guide/running.md` | Streamlit UI section |
| `docs/user-guide/configuration.md` | `LLM_PROVIDER` options, `MODEL_NAME` note, `OLLAMA_BASE_URL` row |

---

## Per-file Changes

### `README.md`

1. **Reliability features — "Multi-provider" bullet** (line 35):

   From:
   ```
   - **Multi-provider** — Anthropic, OpenAI, Google, and Groq supported
   ```
   To:
   ```
   - **Multi-provider** — Anthropic, OpenAI, Google, Groq, and Ollama (local) supported
   ```

2. **Config table — `LLM_PROVIDER` row** (line 86):

   From:
   ```
   | `LLM_PROVIDER` | `anthropic` | Provider: `anthropic`, `openai`, `google`, `groq` |
   ```
   To:
   ```
   | `LLM_PROVIDER` | `anthropic` | Provider: `anthropic`, `openai`, `google`, `groq`, `ollama` |
   ```

3. **Config table — new `OLLAMA_BASE_URL` row** (add after `LLM_PROVIDER` row):

   ```
   | `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL. Only used when `LLM_PROVIDER=ollama` |
   ```

4. **Quick start — Streamlit run block** (add after the CLI `Run:` block):

   ```markdown
   **Run (Streamlit UI):**

   ```bash
   uv run --extra ui streamlit run app.py
   ```
   Opens a browser UI at `http://localhost:8501` with a live pipeline trace sidebar.
   ```

---

### `docs/user-guide/installation.md`

1. **"Configure your API key" section — Ollama block** (add after the Groq block, before the `HF_TOKEN` comment):

   ```
   # Or Ollama (local — no API key needed)
   LLM_PROVIDER=ollama
   MODEL_NAME=llama3.2          # required — no default
   OLLAMA_BASE_URL=http://localhost:11434   # optional, this is the default
   ```

2. **New section at end of file — "Optional: Streamlit UI"**:

   ```markdown
   ## Optional: Streamlit UI

   To use the browser-based interface, install the `ui` extra:

   ```bash
   uv sync --extra ui
   ```

   Then run:

   ```bash
   streamlit run app.py
   ```

   See [Running — Streamlit UI](running.md#streamlit-ui) for details.
   ```

---

### `docs/user-guide/running.md`

Add a new "Streamlit UI" section after the "Start" section (before "Example interaction"):

```markdown
## Streamlit UI

If you installed the `ui` extra (see [Installation](installation.md#optional-streamlit-ui)):

```bash
streamlit run app.py
```

Opens the assistant at `http://localhost:8501`. The interface has two panels:

- **Main area** — chat conversation: type your question, read the response
- **Sidebar** — live pipeline trace: shows each agent as it runs, with timing and key stats (acts found, confidence score, evaluator result, etc.)
```

---

### `docs/user-guide/configuration.md`

1. **`LLM_PROVIDER` row — add `ollama` to options** (line 11):

   From:
   ```
   | `LLM_PROVIDER` | `anthropic` \| `openai` \| `google` \| `groq` | `anthropic` | The LLM provider to use |
   ```
   To:
   ```
   | `LLM_PROVIDER` | `anthropic` \| `openai` \| `google` \| `groq` \| `ollama` | `anthropic` | The LLM provider to use |
   ```

2. **`MODEL_NAME` row — note Ollama has no default** (line 16):

   From:
   ```
   | `MODEL_NAME` | string | *(provider default)* | Override the model. Defaults: `claude-sonnet-4-6`, `gpt-4o`, `gemini-2.0-flash`, `llama-3.3-70b-versatile` |
   ```
   To:
   ```
   | `MODEL_NAME` | string | *(provider default)* | Override the model. Defaults: `claude-sonnet-4-6`, `gpt-4o`, `gemini-2.0-flash`, `llama-3.3-70b-versatile`. **Required when `LLM_PROVIDER=ollama`** (no default). |
   ```

3. **New `OLLAMA_BASE_URL` row** (add after `GROQ_API_KEY` row):

   ```
   | `OLLAMA_BASE_URL` | string | `http://localhost:11434` | Ollama server URL. Only used when `LLM_PROVIDER=ollama` |
   ```

---

## Non-Goals

- Developer guide / adding-a-provider.md — not in scope
- Architecture docs — not in scope
- API reference — not in scope
- Any restructuring of existing content
