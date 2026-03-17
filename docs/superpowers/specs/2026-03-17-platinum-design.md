# Platinum-Level Upgrade — Irish Statute Assistant

**Date:** 2026-03-17
**Status:** Approved

---

## Overview

This document specifies the changes required to bring the Irish Statute Assistant from Gold to Platinum level, as defined in `spec.md`. Platinum requires: debate-style reasoning, auto-revision, hallucination detection, advanced constraints, and multiple memory stores.

All changes follow the existing incremental-layering approach — new agents slot into the existing pipeline without restructuring the supervisor.

---

## 1. Architecture

### Updated Pipeline Order

```
clarify → research → analyse → devil's advocate → write → ground-check → evaluate → (refinement loop)
```

Two new agents are added:

- **`DevilsAdvocateAgent`** — between analyst and writer; challenges the analyst's output with specific legal objections
- **`GroundingCheckerAgent`** — between writer and evaluator; verifies each key clause is traceable to retrieved statute text

Two new persistent memory stores replace the existing in-memory `SessionMemory`:

- **`ConversationStore`** — SQLite-backed conversation history, survives process restarts
- **`UserPreferenceStore`** — SQLite-backed key-value store for user preferences

---

## 2. New Agents

### `DevilsAdvocateAgent` (`agents/devils_advocate.py`)

**Purpose:** Debate-style reasoning — challenges the analyst's interpretation before the writer proceeds.

**Input:** `AnalystOutput` + original query + retrieved statute text

**Output schema:**
```python
class AdvocateOutput(BaseModel):
    challenges: list[str]          # specific objections (1-3)
    severity: Literal["minor", "major"]
```

**Behaviour:**
- Identifies missing exceptions, overriding statutes, or claims not supported by the retrieved text
- `severity="major"` if the analyst's conclusion could be substantially wrong
- Challenges are passed to the writer as `advocate_challenges` context — the writer must acknowledge them in `caveats`
- Does not re-run the analyst; the debate is resolved by the writer's acknowledgement

**Confidence gate integration:**
- If `analyst.confidence < 0.5` OR `advocate.severity == "major"`, `effective_refinements = max_refinement_rounds * 2` and the advocate runs with a stricter prompt on retry

---

### `GroundingCheckerAgent` (`agents/grounding_checker.py`)

**Purpose:** Hallucination detection — verifies writer output is grounded in retrieved source text.

**Input:** `WriterOutput` + `ResearcherOutput`

**Output schema:**
```python
class GroundingOutput(BaseModel):
    ungrounded_claims: list[str]
    grounding_passed: bool
```

**Behaviour:**
- For each item in `key_clauses`, checks whether the claim is supported by the retrieved statute sections
- If `ungrounded_claims` is non-empty, they are attached to `WriterOutput.warnings` — the user sees them inline; the response is still returned
- `grounding_passed=False` adds a flag to the evaluator input so citation quality is scored lower

---

## 3. Schema Changes

### `KeyClause` (new model)
```python
class KeyClause(BaseModel):
    text: str
    act: str
    section: str
```
Replaces bare `str` in `key_clauses` — enforces citation structure at schema level.

### `WriterOutput` (updated)
```python
warnings: list[str] = []   # populated by GroundingCheckerAgent
```

### `AnalystOutput` (updated)
```python
advocate_challenges: list[str] = []   # populated by supervisor before passing to writer
```

---

## 4. Memory Stores

Both stores use Python's built-in `sqlite3`. Files are created at first use under `~/.irish_statute_assistant/`.

### `ConversationStore` (`memory/conversation_store.py`)

- Replaces `SessionMemory` with the same interface: `add_exchange()`, `format_for_prompt()`, `get_history()`
- Persists to `conversations_db_path` (default: `~/.irish_statute_assistant/conversations.db`)
- Loads the last `conversation_history_limit` exchanges on startup (default: 20)
- `Pipeline` swaps `SessionMemory` for `ConversationStore` — one line change

### `UserPreferenceStore` (`memory/user_preference_store.py`)

- Key-value interface: `set(key, value)`, `get(key, default)`, `all()`
- Persists to `preferences_db_path` (default: `~/.irish_statute_assistant/preferences.db`)
- Populated two ways:
  1. **Explicit:** supervisor detects preference signals in user query (e.g. "I'm a solicitor", "explain simply") via lightweight keyword scan — no extra LLM call
  2. **Inferred:** if evaluator repeatedly flags "plain English" failures, supervisor saves `preferred_language=technical`
- Preferences are injected into analyst and writer system prompts as a short context block

### `Config` additions
```
conversation_history_limit: int = 20
preferences_db_path: str = "~/.irish_statute_assistant/preferences.db"
conversations_db_path: str = "~/.irish_statute_assistant/conversations.db"
```

---

## 5. Supervisor Changes (`agents/supervisor.py`)

### Updated flow

```python
# 1. Clarify
# 2. Research
# 3. Analyse
analyst_result = run_with_retry(lambda: analyst.run(...))

# 4. Devil's advocate
advocate_result = run_with_retry(lambda: advocate.run(analyst_result, query, research))

# Confidence gate
if analyst_result.confidence < 0.5 or advocate_result.severity == "major":
    effective_refinements = max_refinement_rounds * 2
    advocate_mode = "strict"
else:
    effective_refinements = max_refinement_rounds
    advocate_mode = "standard"

# Inject challenges into analyst output for writer
analyst_result.advocate_challenges = advocate_result.challenges

# 5. Write
# 6. Ground-check
grounding = run_with_retry(lambda: grounding_checker.run(writer_result, research))
writer_result.warnings = grounding.ungrounded_claims

# 7. Evaluate (with grounding flag)
# 8. Refinement loop — carry forward advocate_challenges on retry
```

### Preference detection

After a successful `WriterOutput`, a keyword scan of `user_query` checks for preference signals and saves to `UserPreferenceStore`. This is intentionally a keyword check, not an LLM call.

---

## 6. `main.py` Output Changes

- Print `warnings` under `"--- Grounding warnings ---"` if non-empty
- Print `"Note: confidence in statute coverage was low for this query."` if analyst confidence was low

---

## 7. Error Handling

No new exception types needed:

- Both new agents are wrapped by `run_with_retry()` — validation failures retry, then raise `ValidationRepairError`
- SQLite errors in memory stores raise `FatalError`
- DB directories are auto-created on first use

---

## 8. Testing

Three new test files:

- `tests/test_devils_advocate.py` — challenges produced for weak analysis; clean pass on strong analysis; `severity=major` on serious gaps
- `tests/test_grounding_checker.py` — flags claims absent from source text; passes grounded claims; warnings attached correctly
- `tests/test_memory_stores.py` — persistence across instantiations; preference read/write; history limit enforced

Existing tests pass unchanged — schema changes are additive (new fields have defaults), and `ConversationStore` has the same interface as `SessionMemory`.

---

## 9. Platinum Criteria Mapping

| Platinum Requirement | Implementation |
|---|---|
| Debate-style reasoning | `DevilsAdvocateAgent` challenges analyst before writer proceeds |
| Auto-revision | Writer receives advocate challenges + evaluator flags; refinement loop carries challenges forward |
| Hallucination detection | `GroundingCheckerAgent` cross-references claims against source text; warnings surfaced to user |
| Advanced constraints | `KeyClause` schema enforces citations; confidence gate routes low-confidence queries to extended refinement |
| Multiple memory stores | `ConversationStore` (history) + `UserPreferenceStore` (preferences), both SQLite-persisted |
