# uv Migration — Design Doc

**Date:** 2026-03-11
**Feature:** Replace pip + requirements.txt with uv
**Scope:** Packaging/tooling only — no changes to application code

---

## 1. Overview

Move all dependencies from `requirements.txt` into `pyproject.toml` under `[project] dependencies`. Keep the existing setuptools build backend. Run `uv sync` to generate `uv.lock`. Delete `requirements.txt`. Update README.

No application code, test code, or project structure changes.

---

## 2. Files Changed

| File | Action |
|------|--------|
| `pyproject.toml` | Add `dependencies` list under `[project]` |
| `requirements.txt` | Delete |
| `uv.lock` | Create (via `uv sync`) — commit to git |
| `README.md` | Replace `pip install -r requirements.txt` with `uv sync` |
| `.gitignore` | Add `.venv/` if not already present |

---

## 3. pyproject.toml Change

Add to the existing `[project]` section:

```toml
dependencies = [
    "langchain>=0.3",
    "langchain-anthropic>=0.3",
    "langchain-core>=0.3",
    "pydantic>=2.0",
    "pydantic-settings>=2.0",
    "httpx>=0.27",
    "beautifulsoup4>=4.12",
    "tenacity>=8.0",
    "pytest>=8.0",
    "pytest-httpx>=0.30",
    "python-dotenv>=1.0",
    "langchain-chroma>=0.1",
    "langchain-huggingface>=0.1",
    "sentence-transformers>=3.0",
]
```

Everything else in `pyproject.toml` (build system, pytest config, package discovery) is unchanged.

---

## 4. Workflow After Migration

```bash
uv sync                                          # install / update deps
uv run pytest tests/ -v                          # run tests
uv run python -m irish_statute_assistant.indexer # build vector store
uv run python -m irish_statute_assistant.main    # run app
```

`uv.lock` is committed to git for reproducible installs.

---

## 5. README Change

In the Setup section, replace:

```bash
pip install -r requirements.txt
```

with:

```bash
uv sync
```

---

## 6. Out of Scope

- Separating dev/test deps (keep everything together as one flat list)
- Switching build backend (keep setuptools)
- Any changes to application or test code
