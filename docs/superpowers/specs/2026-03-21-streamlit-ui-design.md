# Streamlit Demo UI — Design Spec

**Date:** 2026-03-21
**Branch:** `feature/streamlit-ui`
**Audience:** Software engineers (live demo)

---

## Overview

A Streamlit web UI for the Irish Statute Research Assistant that shows the multi-agent pipeline executing in real time. Engineers watching the demo can see each agent step light up as it completes alongside the conversational answer.

---

## Goals

- Show the full agent pipeline executing live during a query, including the refinement loop
- Provide a clean chat interface for asking legal questions
- Run locally with a single command: `uv run streamlit run app.py`
- Live entirely on a separate branch (`feature/streamlit-ui`) — zero impact on `main`

---

## Non-Goals

- Production deployment or hosting
- Authentication or multi-user support
- Mobile layout
- Any changes to agent logic or pipeline behaviour
- Exposing per-query token counts (would require changes to `Pipeline.query()` return type)

---

## Architecture

### Branch & files

| Item | Location |
|---|---|
| Branch | `feature/streamlit-ui` |
| App entry point | `app.py` (project root) |
| Dependency | `streamlit>=1.35` in `pyproject.toml` `[ui]` optional extra |

No files under `src/` are created. `app.py` imports the existing `Pipeline` class.

### Pipeline flow (for context)

The actual Supervisor flow is not a flat sequence. The Devil's Advocate runs once before the refinement loop, then may re-run inside the loop on each failed evaluation round:

```
Clarifier → Researcher → Analyst → Devil's Advocate (initial)
  └─ Refinement loop (1–5 iterations):
       Writer → Grounding Checker → Evaluator
       └─ if failed: Devil's Advocate (retry, stricter mode) → next iteration
```

The progress panel must accommodate multiple Devil's Advocate rows (one per loop iteration that fails) and multiple Writer/Grounding/Evaluator rows when refinement rounds occur.

### Pipeline instrumentation

`Supervisor.run()` gains one optional parameter:

```python
progress_callback: Callable[[str, dict], None] | None = None
```

`Pipeline.query()` also gains this parameter and **forwards it to `Supervisor.run()`**, keeping `context` constructed internally as before:

```python
def query(self, user_query: str, progress_callback=None) -> WriterOutput | str:
    context = QueryContext(budget=self._config.token_budget_per_query)
    result = self._supervisor.run(query=user_query, context=context, progress_callback=progress_callback)
    logger.info("Query used %d/%d tokens", context.tokens_used, context.budget)
    return result
```

After each agent completes, the Supervisor calls:

```python
if progress_callback:
    progress_callback(agent_name, stats_dict)
```

`stats_dict` contains a small summary appropriate to each agent:

| Agent | Keys |
|---|---|
| Clarifier | `needs_clarification: bool`, `duration_s: float` |
| Researcher | `acts_found: int`, `source: str` ("vector store" or "live fetch"), `duration_s: float` |
| Analyst | `key_clauses: int` (len of `analyst_output.key_clauses`), `confidence: float`, `duration_s: float` |
| Devil's Advocate | `challenges: int` (len of `advocate_result.challenges`), `severity: str`, `duration_s: float` |
| Writer | `round: int`, `duration_s: float` |
| Grounding Checker | `grounding_passed: bool`, `ungrounded: int` (len of `grounding.ungrounded_claims`), `duration_s: float` |
| Evaluator | `score: float`, `passed: bool`, `flags: int` (len of `evaluation.flags`), `duration_s: float` |

The `duration_s` value is computed in the Supervisor using `time.time()` before and after each `run_with_retry()` call.

This is the only change to core code. All existing tests continue to pass unchanged; both parameters default to `None`.

### Streamlit app layout

Two-column layout via `st.columns([2, 1])`:

```
┌─────────────────────────────────────────────┐
│  🏛 Irish Statute Research Assistant         │
├──────────────────────────┬──────────────────┤
│                          │ Pipeline         │
│  [conversation thread]   │ ✓ Clarifier      │
│                          │ ✓ Researcher     │
│                          │ ✓ Analyst        │
│                          │ ⚔ Devil's Adv.  │
│                          │ ✓ Writer         │
│                          │ ✓ Grounding      │
│                          │ ✓ Evaluator      │
│                          │ ── 1 round ──    │
├──────────────────────────┴──────────────────┤
│  [chat input]                         Send  │
└─────────────────────────────────────────────┘
```

**Left column:** `st.chat_message` blocks rendered from `st.session_state.messages`. Each answer shows the short answer prominently, with statute references and caveats below. When the Clarifier returns a clarifying question (early exit), the question is shown as an assistant message and the pipeline panel shows only the Clarifier row with `? Needs clarification`.

**Right column:** A single `st.empty()` placeholder, re-rendered by the `progress_callback` each time an agent completes. Steps accumulate top-to-bottom during the query. When a new query starts, the step list resets. On a multi-round run, Writer/Grounding/Evaluator rows appear multiple times and Devil's Advocate may appear again — each occurrence is a separate row labelled with the round number (e.g. `⚔ Devil's Advocate (round 2)`).

**Footer summary row:** Shown after the Evaluator fires for the final time — total rounds completed and total wall-clock duration. No token counts (see Non-Goals).

### Visual treatment

- Clarifier, Researcher, Analyst, Writer, Grounding Checker, Evaluator rows: green `✓`
- Clarifier early-exit row: blue `?` with text `Needs clarification`
- Devil's Advocate row: amber `⚔` to visually distinguish the adversarial step
- Each row shows agent name + one-line summary (e.g. `3 clauses · confidence 0.88`) + duration
- In-progress agent shows a spinner until its callback fires

### Session state

| Key | Type | Purpose |
|---|---|---|
| `messages` | `list[dict]` | `{"role": "user"\|"assistant", "content": str}` — rendered in chat column |
| `pipeline_steps` | `list[dict]` | Accumulated agent step summaries for current query |
| `pipeline` | `Pipeline` | Single instance, shared across reruns |

`pipeline` is instantiated once via `@st.cache_resource`. Note: the `ConversationStore` inside `Pipeline` persists to SQLite and is shared across browser tabs. This is acceptable for a single-user local demo but means two open tabs will share conversation history. Opening a fresh tab mid-demo is not recommended.

### Error handling

The following exceptions from `Pipeline.query()` are caught and displayed as `st.error()` in the chat column. The pipeline step sidebar shows whatever completed before the error.

| Exception | User-facing message |
|---|---|
| `StatuteNotFoundError` | No relevant statutes found. Try rephrasing. |
| `BudgetExceededError` | Token budget exceeded. Increase `TOKEN_BUDGET_PER_QUERY` in `.env`. |
| `ValidationRepairError` | Could not produce a valid response. Try rephrasing. |
| `FatalError` | An unrecoverable error occurred. Check the terminal for details. |
| `Exception` (catch-all) | Something went wrong: `{e}` |

---

## Dependencies

```toml
[project.optional-dependencies]
ui = ["streamlit>=1.35"]
```

Install with: `uv sync --extra ui`
Run with: `uv run streamlit run app.py`

---

## Testing

The `progress_callback` parameter on `Supervisor.run()` and `Pipeline.query()` will be covered by unit tests:

- On a single-round successful run, callback is called exactly 7 times (once per agent in the happy path: Clarifier, Researcher, Analyst, Devil's Advocate, Writer, Grounding Checker, Evaluator)
- On a two-round run (one failed evaluation), callback is called 11 times (7 + Writer, Grounding Checker, Devil's Advocate, Evaluator for the second round)
- Callback receives correct agent name string and all expected stats keys for each agent
- `None` callback (default) does not affect pipeline behaviour or test outcomes

The Streamlit `app.py` itself is not unit-tested — it is validated by running the app and doing a live demo query.

---

## Out of scope

- Streaming the answer token-by-token (the pipeline returns a complete `WriterOutput`, not a stream)
- Showing the full agent output JSON (the sidebar shows a one-line summary per agent)
- Persisting conversation history across browser sessions (session state is in-memory; SQLite history persists but that is existing pipeline behaviour)
