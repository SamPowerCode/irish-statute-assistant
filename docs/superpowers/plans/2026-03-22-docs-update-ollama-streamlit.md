# Documentation Update: Ollama + Streamlit UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Update README, configuration reference, installation guide, running guide, and `.env.example` to document the Ollama provider and Streamlit UI.

**Architecture:** Five surgical text edits to five existing files. No new files. No code changes. Verify each edit with a targeted grep after applying, then commit.

**Tech Stack:** Markdown, bash grep for verification

---

## File Map

| File | Changes |
|---|---|
| `README.md` | 4 edits: provider list bullet, `LLM_PROVIDER` table row, new `OLLAMA_BASE_URL` row, Streamlit run block |
| `docs/user-guide/configuration.md` | 3 edits: `LLM_PROVIDER` options, `MODEL_NAME` note, new `OLLAMA_BASE_URL` row |
| `docs/user-guide/installation.md` | 2 edits: Ollama config block, new Streamlit UI section |
| `.env.example` | 1 edit: Ollama comment block |
| `docs/user-guide/running.md` | 1 edit: new Streamlit UI section |

---

## Task 1: Update README.md

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Update the Multi-provider reliability bullet**

In `README.md`, find:
```
- **Multi-provider** — Anthropic, OpenAI, Google, and Groq supported
```
Replace with:
```
- **Multi-provider** — Anthropic, OpenAI, Google, Groq, and Ollama (local) supported
```

- [ ] **Step 2: Update the `LLM_PROVIDER` config table row**

Find:
```
| `LLM_PROVIDER` | `anthropic` | Provider: `anthropic`, `openai`, `google`, `groq` |
```
Replace with:
```
| `LLM_PROVIDER` | `anthropic` | Provider: `anthropic`, `openai`, `google`, `groq`, `ollama` |
```

- [ ] **Step 3: Add `OLLAMA_BASE_URL` row to the config table**

> This step must run after Step 2, because the Find string matches the already-updated `LLM_PROVIDER` row.

Find (the updated `LLM_PROVIDER` row from Step 2, plus the `ANTHROPIC_API_KEY` row that follows it):
```
| `LLM_PROVIDER` | `anthropic` | Provider: `anthropic`, `openai`, `google`, `groq`, `ollama` |
| `ANTHROPIC_API_KEY` | — | Required when `LLM_PROVIDER=anthropic`. Other providers need their own key — see full docs |
```
Replace with (inserts `OLLAMA_BASE_URL` between them):
```
| `LLM_PROVIDER` | `anthropic` | Provider: `anthropic`, `openai`, `google`, `groq`, `ollama` |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL. Only used when `LLM_PROVIDER=ollama` |
| `ANTHROPIC_API_KEY` | — | Required when `LLM_PROVIDER=anthropic`. Other providers need their own key — see full docs |
```

- [ ] **Step 4: Add Streamlit run block to Quick start**

After the CLI run block:
```
**Run:**

```bash
uv run python -m irish_statute_assistant.main
```
```

Add immediately after:
```markdown

**Run (Streamlit UI):**

```bash
uv run --extra ui streamlit run app.py
```
Opens a browser UI at `http://localhost:8501` with a live pipeline trace sidebar.
(The `--extra ui` flag installs Streamlit on the fly if not already installed.)
```

- [ ] **Step 5: Verify all four changes are present**

```bash
grep -n "Ollama (local)" README.md
grep -n "ollama" README.md
grep -n "OLLAMA_BASE_URL" README.md
grep -n "Streamlit UI" README.md
```

Expected: each grep returns at least one match.

- [ ] **Step 6: Commit**

```bash
git add README.md
git commit -m "docs: add Ollama provider and Streamlit UI to README"
```

---

## Task 2: Update configuration.md

**Files:**
- Modify: `docs/user-guide/configuration.md`

- [ ] **Step 1: Add `ollama` to the `LLM_PROVIDER` options**

Find:
```
| `LLM_PROVIDER` | `anthropic` \| `openai` \| `google` \| `groq` | `anthropic` | The LLM provider to use |
```
Replace with:
```
| `LLM_PROVIDER` | `anthropic` \| `openai` \| `google` \| `groq` \| `ollama` | `anthropic` | The LLM provider to use |
```

- [ ] **Step 2: Update the `MODEL_NAME` row to note Ollama has no default**

Find:
```
| `MODEL_NAME` | string | *(provider default)* | Override the model. Defaults: `claude-sonnet-4-6`, `gpt-4o`, `gemini-2.0-flash`, `llama-3.3-70b-versatile` |
```
Replace with:
```
| `MODEL_NAME` | string | *(provider default)* | Override the model. Defaults: `claude-sonnet-4-6`, `gpt-4o`, `gemini-2.0-flash`, `llama-3.3-70b-versatile`. **Required when `LLM_PROVIDER=ollama`** (no default). |
```

- [ ] **Step 3: Add `OLLAMA_BASE_URL` row after `GROQ_API_KEY`**

Find:
```
| `GROQ_API_KEY` | string | — | Required when `LLM_PROVIDER=groq` |
```
Replace with:
```
| `GROQ_API_KEY` | string | — | Required when `LLM_PROVIDER=groq` |
| `OLLAMA_BASE_URL` | string | `http://localhost:11434` | Ollama server URL. Only used when `LLM_PROVIDER=ollama` |
```

- [ ] **Step 4: Verify all three changes**

```bash
grep -n "ollama" docs/user-guide/configuration.md
grep -n "OLLAMA_BASE_URL" docs/user-guide/configuration.md
grep -n "Required when.*ollama" docs/user-guide/configuration.md
```

Expected: each grep returns at least one match.

- [ ] **Step 5: Commit**

```bash
git add docs/user-guide/configuration.md
git commit -m "docs: add Ollama provider settings to configuration reference"
```

---

## Task 3: Update installation.md and .env.example

**Files:**
- Modify: `docs/user-guide/installation.md`
- Modify: `.env.example`

- [ ] **Step 1: Add Ollama config block to installation.md**

In the "Configure your API key" section, find the Groq block followed by the HF_TOKEN line:
```
# Or Groq
LLM_PROVIDER=groq
GROQ_API_KEY=...

# Hugging Face (for embedding model download, if required)
HF_TOKEN=hf_...
```
Replace with:
```
# Or Groq
LLM_PROVIDER=groq
GROQ_API_KEY=...

# Or Ollama (local — no API key needed)
LLM_PROVIDER=ollama
MODEL_NAME=llama3.2          # required — no default
OLLAMA_BASE_URL=http://localhost:11434   # optional, this is the default

# Hugging Face (for embedding model download, if required)
HF_TOKEN=hf_...
```

- [ ] **Step 2: Add Optional Streamlit UI section at end of installation.md**

Append to the end of `docs/user-guide/installation.md`:
```markdown

## Optional: Streamlit UI

To use the browser-based interface, install the `ui` extra:

```bash
uv sync --extra ui
```

Then run:

```bash
uv run streamlit run app.py
```

See [Running — Streamlit UI](running.md#streamlit-ui) for details.
```

- [ ] **Step 3: Add Ollama comment block to .env.example**

In `.env.example`, find:
```
HF_TOKEN=hf_your_token_here
```
Replace with:
```
# Or Ollama (local — no API key needed)
# LLM_PROVIDER=ollama
# MODEL_NAME=llama3.2
# OLLAMA_BASE_URL=http://localhost:11434

HF_TOKEN=hf_your_token_here
```

- [ ] **Step 4: Verify all changes**

```bash
grep -n "Ollama" docs/user-guide/installation.md
grep -n "Streamlit UI" docs/user-guide/installation.md
grep -n "ollama" .env.example
```

Expected: each grep returns at least one match.

- [ ] **Step 5: Commit**

```bash
git add docs/user-guide/installation.md .env.example
git commit -m "docs: add Ollama config block and Streamlit UI install to installation guide"
```

---

## Task 4: Update running.md

**Files:**
- Modify: `docs/user-guide/running.md`

- [ ] **Step 1: Add Streamlit UI section after the "Start" section**

In `docs/user-guide/running.md`, find:
```
## Example interaction
```
Insert the following block immediately before it:
```markdown
## Streamlit UI

If you installed the `ui` extra (see [Installation](installation.md#optional-streamlit-ui)):

```bash
uv run streamlit run app.py
```

Opens the assistant at `http://localhost:8501`. The interface has two panels:

- **Main area** — chat conversation: type your question, read the response
- **Sidebar** — live pipeline trace: shows each agent as it runs, with timing and key stats (acts found, confidence score, evaluator result, etc.)

---

```

- [ ] **Step 2: Verify**

```bash
grep -n "Streamlit UI" docs/user-guide/running.md
grep -n "localhost:8501" docs/user-guide/running.md
grep -n "pipeline trace" docs/user-guide/running.md
```

Expected: each grep returns at least one match.

- [ ] **Step 3: Commit**

```bash
git add docs/user-guide/running.md
git commit -m "docs: add Streamlit UI section to running guide"
```
