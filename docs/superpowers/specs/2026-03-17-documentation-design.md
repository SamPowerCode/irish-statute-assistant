# Documentation Design — Irish Statute Assistant

**Date:** 2026-03-17
**Status:** Approved

---

## Overview

This document specifies the documentation system for the Irish Statute Assistant. The goal is a professional, audience-aware reference that serves three groups: end users running the CLI, developers extending the system, and learners studying the architecture.

The README is kept as a quick-start entry point. Sphinx provides the full reference, hosted on ReadTheDocs.

No capstone framing or tier labels appear anywhere in the user-facing documentation. The project is presented as a standalone application. Features are described by what they do.

---

## Approach

**Hybrid documentation model:**
- Hand-written Markdown pages for the user guide, architecture deep-dive, and developer guide
- `autodoc` generates the API reference from docstrings on public interfaces
- Mermaid diagrams embedded in the architecture section
- Theme: Furo (clean, modern, dark/light toggle)
- Hosted: ReadTheDocs

**README role:** Quick-start only. Covers installation, minimal configuration, and links to the full docs. The detailed configuration table moves to the user guide.

---

## File Structure

```
docs/
├── conf.py                        # Sphinx config: MyST, autodoc, Furo, Mermaid
├── index.md                       # Landing page with navigation
├── user-guide/
│   ├── installation.md
│   ├── running.md
│   └── configuration.md
├── architecture/
│   ├── overview.md
│   ├── agents.md
│   ├── schemas.md
│   ├── memory.md
│   └── flows.md
├── developer-guide/
│   ├── adding-an-agent.md
│   ├── adding-a-provider.md
│   └── testing.md
└── api-reference/
    ├── agents.md
    ├── schemas.md
    ├── config.md
    ├── memory.md
    └── pipeline.md
```

---

## Section Specifications

### User Guide

**`installation.md`**
Python version requirement, `uv` setup, API key configuration via `.env`, choosing a vector store backend (Chroma local vs Qdrant). Brief — links to README for the absolute basics.

**`running.md`**
How to start the CLI, the query/answer loop, handling clarifying questions, continuing a conversation across sessions. Annotated example of a real interaction showing each part of the output: short answer, key clauses with citations, caveats, grounding warnings, and the low-confidence notice (printed by `main.py` when `analyst_confidence < 0.5`).

**`configuration.md`**
Authoritative table of every `.env` setting: name, type, default, description. Grouped by concern: LLM provider, vector store, memory, tuning. This replaces the config table currently in the README.

---

### Architecture Deep Dive

**`overview.md`**
Mermaid pipeline diagram showing the full agent flow:
```
clarify → research → analyse → devil's advocate → write → ground-check → evaluate → (refinement loop)
```
Followed by a narrative covering each stage and how they connect.

**`agents.md`**
One section per agent:
- Role and responsibility
- Inputs and outputs (schema types)
- Design rationale

Covers the split-schema pattern (`AnalystLLMOutput` vs `AnalystOutput` — why `advocate_challenges` is excluded from the LLM schema), the confidence gate logic, and how the refinement loop carries `evaluator_flags` and `advocate_challenges` forward.

**`schemas.md`**
Explains the key schemas and the patterns behind them:
- `KeyClause` — structured citations enforcing `act` and `section` fields
- `ClarifierOutput` — drives the clarification flow (`needs_clarification`, `question`)
- `EvaluatorOutput` — central to the refinement loop; the `pass_` alias pattern (Pydantic alias for a Python reserved word)
- `WriterOutput` with `warnings` and `analyst_confidence`
- `AdvocateOutput` and `GroundingOutput`
- `ResearcherOutput` / `ActSection` — statute text passed through the pipeline
- Pydantic v2 patterns used throughout: `with_structured_output`, `Field` constraints, `model_validator`

**`memory.md`**
Explains `ConversationStore` and `UserPreferenceStore`: what each stores, why SQLite was chosen over in-memory state, how history limits work, and how preference detection operates (explicit keyword scan + inferred preference logic from repeated evaluator flags). Includes how `format_for_prompt()` is the integration point between the memory store and agent prompts — stored conversation history flows back into LLM calls via this method.

**`flows.md`**
Three Mermaid sequence diagrams:
1. Clarification flow — query arrives, clarifier asks a question, user responds, pipeline continues
2. Refinement loop — write → ground-check → evaluate → retry or return
3. Confidence gate — how low analyst confidence or major advocate severity triggers extended refinement in strict mode

---

### Developer Guide

**`adding-an-agent.md`**
Step-by-step: create the file in `agents/`, extend `BaseAgent`, define a Pydantic output schema in `schemas.py`, wire into `supervisor.py`. Includes a minimal stub agent example. Explains when to use `with_structured_output()` vs plain chain invocation, and the `_invoke_chain` pattern for token tracking.

**`adding-a-provider.md`**
How to add a new LLM provider: add to `_DEFAULT_MODELS` in `llm.py`, add the API key field in `config.py`, update `_PROVIDER_KEY_MAP`, add the lazy import branch in `get_llm()`. Explains the provider map sync assertion and why lazy imports are used.

**`testing.md`**
The project's testing approach: mocking at `_invoke_chain` level, using `__new__` to bypass `__init__` for agent tests, `tmp_path` fixtures for SQLite stores. Covers running the suite (`pytest`) and what each test file covers.

---

### API Reference

Auto-generated via `autodoc`. Docstrings are added to:
- All 8 agent classes and their `run()` methods
- All Pydantic schemas in `schemas.py`
- `Config`, `ConversationStore`, `UserPreferenceStore`
- `Pipeline` and `get_llm()`

Private methods and internal helpers are not documented. Only the public surface that developers would call.

Docstring style: Google format (readable in source, supported by `napoleon` Sphinx extension).

---

### README Update

Changes to `README.md`:
- Remove the detailed `.env` configuration table (now lives in `configuration.md`)
- Add a "Full documentation" link pointing to the ReadTheDocs URL
- Update provider/model defaults to reflect current values
- Remove the capstone framing line ("7-week agentic AI capstone project") and any tier labels — rewrite the opening description to present the project as a standalone application
- Update the agent list to reflect all 8 agents: clarifier, researcher, analyst, devil's advocate, writer, grounding checker, evaluator, supervisor
- Keep: installation steps, quick-start example, vector store backend overview

---

## Tooling

| Tool | Purpose |
|---|---|
| Sphinx | Documentation build system |
| MyST Parser | Write docs in Markdown instead of reStructuredText |
| autodoc + napoleon | Generate API reference from Google-style docstrings |
| sphinxcontrib-mermaid | Embed Mermaid diagrams in Markdown pages |
| Furo | Theme |
| ReadTheDocs | Hosting (auto-builds on push to `main`) |
| `.readthedocs.yaml` | Build configuration for ReadTheDocs; must pin Sphinx and all extensions (including `sphinxcontrib-mermaid`) to tested versions to ensure reproducible builds |

---

## Out of Scope

- Docstrings on private/internal methods
- Automated testing of documentation (link checking, doctests)
- Versioned documentation (single `latest` version only)
- Non-English documentation
