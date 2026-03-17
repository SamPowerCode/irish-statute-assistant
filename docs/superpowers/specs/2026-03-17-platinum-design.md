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

The analyst runs **once before the refinement loop**. The refinement loop covers: write → ground-check → evaluate. On each loop iteration the writer receives the current `advocate_challenges` and any new `evaluator_flags`.

Two new agents:

- **`DevilsAdvocateAgent`** — between analyst and writer; challenges the analyst's output with specific legal objections
- **`GroundingCheckerAgent`** — between writer and evaluator; verifies each key clause is traceable to retrieved statute text

Two new persistent memory stores replace the existing in-memory `SessionMemory`:

- **`ConversationStore`** — SQLite-backed conversation history, survives process restarts
- **`UserPreferenceStore`** — SQLite-backed key-value store for user preferences

---

## 2. New Agents

### `DevilsAdvocateAgent` (`agents/devils_advocate.py`)

**Purpose:** Debate-style reasoning — challenges the analyst's interpretation before the writer proceeds.

**Signature:** `run(analyst_output: AnalystOutput, query: str, research: ResearcherOutput, mode: Literal["standard", "strict"] = "standard") -> AdvocateOutput`

- `mode="standard"`: find 1–3 meaningful challenges
- `mode="strict"`: prompt instructs the model to be adversarial — surface every possible exception, conflicting statute, or unsupported inference; aim for up to 5 challenges

**Output schema:**
```python
class AdvocateOutput(BaseModel):
    challenges: list[str] = Field(min_length=0, max_length=5)
    severity: Literal["minor", "major"]
```
The `challenges` length constraint is schema-enforced (0–5). An empty list with `severity="minor"` is valid and means the analyst's output was unchallenged.

**Behaviour:**
- Identifies missing exceptions, overriding statutes, or claims not supported by the retrieved text
- `severity="major"` if the analyst's conclusion could be substantially wrong
- Challenges are passed to the writer as `advocate_challenges` — the writer must address them in `caveats`
- The analyst is **not re-run**; the debate is resolved by the writer's acknowledgement

**Note on `advocate_challenges` field:** `AnalystOutput.advocate_challenges` is **not** populated by the analyst LLM. The prescribed pattern is a **split schema**: the LLM is bound to `AnalystLLMOutput` (which contains only `key_clauses`, `gaps`, and `confidence`), and the supervisor constructs a full `AnalystOutput` from it before injecting challenges:

```python
# In AnalystAgent — LLM sees only AnalystLLMOutput
llm = get_llm(config, max_tokens=512).with_structured_output(AnalystLLMOutput)

# In Supervisor — after advocate runs
analyst_output = AnalystOutput(**llm_result.model_dump(), advocate_challenges=advocate_result.challenges)
```

`AnalystLLMOutput` is defined in `schemas.py` alongside `AnalystOutput` and contains only the three fields the LLM populates. `AnalystOutput` extends it with `advocate_challenges: list[str] = []`.

---

### `GroundingCheckerAgent` (`agents/grounding_checker.py`)

**Purpose:** Hallucination detection — verifies writer output is grounded in retrieved source text.

**Signature:** `run(writer_output: WriterOutput, research: ResearcherOutput) -> GroundingOutput`

**Output schema:**
```python
class GroundingOutput(BaseModel):
    ungrounded_claims: list[str]
    grounding_passed: bool
```

**Behaviour:**
- For each `KeyClause` in `DetailedBreakdown.key_clauses`, checks whether the claim text is supported by the retrieved statute sections
- If `ungrounded_claims` is non-empty, they are attached to `WriterOutput.warnings` — the response is still returned
- `grounding_passed` is set to `False` if any ungrounded claims are found
- `grounding_passed` is passed to `EvaluatorAgent` via a new optional parameter (see Section 5)

---

## 3. Schema Changes

### `KeyClause` (new model)
```python
class KeyClause(BaseModel):
    text: str
    act: str
    section: str
```
Replaces bare `str` for `key_clauses` fields in both `AnalystOutput` and `DetailedBreakdown`.

### `AnalystOutput` (updated)
```python
key_clauses: list[KeyClause]          # was list[str]
advocate_challenges: list[str] = []   # supervisor-populated only; excluded from LLM schema
```

### `DetailedBreakdown` (updated)
```python
key_clauses: list[KeyClause]   # was list[str]
```

### `EvaluatorAgent` impact
The evaluator currently serialises `bd.key_clauses` with `"\n".join(bd.key_clauses)`. This must be updated to `"\n".join(f"{kc.text} ({kc.act}, {kc.section})" for kc in bd.key_clauses)`.

### `WriterOutput` (updated)
```python
warnings: list[str] = []          # populated by GroundingCheckerAgent
analyst_confidence: float = 1.0   # set by supervisor; used by main.py for low-confidence banner
```

### `EvaluatorOutput` — no changes needed; grounding is passed as a parameter (see Section 5).

---

## 4. Memory Stores

Both stores use Python's built-in `sqlite3`. The directory `~/.irish_statute_assistant/` is created automatically on first use. If SQLite raises an error (e.g. read-only filesystem), a `FatalError` is raised; `main.py`'s generic `except Exception` handler will catch it and print a meaningful message — no additional handler is needed.

### `ConversationStore` (`memory/conversation_store.py`)

**Constructor:** `ConversationStore(db_path: str, history_limit: int = 20)`

- `Pipeline` instantiates it as `ConversationStore(config.conversations_db_path, config.conversation_history_limit)` and passes it to `Supervisor` — `pipeline.py` no longer calls `add_exchange` directly; all memory write calls are owned by the supervisor
- Same public interface as `SessionMemory`: `add_exchange(user, assistant)`, `format_for_prompt()`, `get_history()`
- Loads the last `history_limit` exchanges from the DB on construction
- Each `add_exchange` call writes to the DB immediately (no batching)

### `UserPreferenceStore` (`memory/user_preference_store.py`)

**Constructor:** `UserPreferenceStore(db_path: str)`

- Key-value interface: `set(key: str, value: str)`, `get(key: str, default: str = "") -> str`, `all() -> dict[str, str]`
- Persists to `preferences_db_path`
- Preferences are injected into analyst and writer system prompts as a block: `"User preferences: {json.dumps(store.all())}"`

### Preference detection (in `supervisor.py`)

Preference detection runs on **every returned `WriterOutput`** (including low-confidence ones). It does not run if the pipeline returns a clarifying question string.

**Explicit keyword scan — recognised patterns and their stored keys/values:**

| Keyword / phrase | Key | Value |
|---|---|---|
| "i'm a solicitor" / "i am a solicitor" / "i'm a lawyer" | `user_type` | `solicitor` |
| "explain simply" / "plain english" / "non-lawyer" | `language_level` | `plain` |
| "use legal terms" / "technical" | `language_level` | `technical` |
| "brief" / "short answer" | `verbosity` | `brief` |
| "detailed" / "full explanation" | `verbosity` | `detailed` |

Matching is case-insensitive. If a keyword is detected, `UserPreferenceStore.set(key, value)` is called. Later preferences overwrite earlier ones for the same key.

**Inferred preference rule:** If `evaluator_flags` on the current query contain the string `"plain English"` (exact match, case-insensitive) **and** this is the second or subsequent query in the session where that flag appeared (tracked in-memory on the `Supervisor` instance as `_evaluator_flag_counts: dict[str, int]`), the supervisor saves `language_level=technical`. The counter resets when the session ends.

*Rationale:* A user who repeatedly receives answers that the evaluator flags as failing plain-English criteria is likely comfortable with legal terminology and does not need simplification. This avoids perpetually degrading responses for legally-trained users who never explicitly stated their background.

### `Config` additions
```
conversation_history_limit: int = 20
preferences_db_path: str = "~/.irish_statute_assistant/preferences.db"
conversations_db_path: str = "~/.irish_statute_assistant/conversations.db"
```

---

## 5. Supervisor Changes (`agents/supervisor.py`)

### Full updated flow

`Supervisor.__init__` now receives `memory: ConversationStore` and `preferences: UserPreferenceStore` (passed from `Pipeline`). `Pipeline` no longer calls `memory.add_exchange()` — all memory writes are owned by the supervisor.

```python
# 1. Clarify (unchanged)
history = memory.format_for_prompt()
clarifier_result = run_with_retry(lambda: clarifier.run(query, history))
if clarifier_result.needs_clarification:
    memory.add_exchange(user=query, assistant=clarifier_result.question)
    return clarifier_result.question

# 2. Research (unchanged)
research = run_with_retry(lambda: researcher.run(query))

# 3. Analyse (runs once, outside the refinement loop)
#    AnalystAgent uses with_structured_output(AnalystLLMOutput); supervisor wraps result
llm_analyst_result = run_with_retry(lambda: analyst.run(query, research))
analyst_output = AnalystOutput(**llm_analyst_result.model_dump(), advocate_challenges=[])

# 4. Devil's advocate (initial run, standard mode)
advocate_result = run_with_retry(lambda: advocate.run(analyst_output, query, research, mode="standard"))

# Confidence gate
if analyst_output.confidence < 0.5 or advocate_result.severity == "major":
    effective_refinements = max_refinement_rounds * 2
    advocate_mode_on_retry = "strict"
else:
    effective_refinements = max_refinement_rounds
    advocate_mode_on_retry = "standard"

# Inject challenges into analyst_output for writer
analyst_output.advocate_challenges = advocate_result.challenges

# 5–7. Refinement loop (write → ground-check → evaluate)
evaluator_flags: list[str] = []
best_output: WriterOutput | None = None

for _ in range(effective_refinements + 1):
    # 5. Write — receives analyst_output (including advocate_challenges) and evaluator_flags
    writer_result = run_with_retry(
        lambda: writer.run(query, analyst_output, research, evaluator_flags)
    )
    writer_result.analyst_confidence = analyst_output.confidence

    # 6. Ground-check
    grounding = run_with_retry(lambda: grounding_checker.run(writer_result, research))
    writer_result.warnings = grounding.ungrounded_claims

    # 7. Evaluate
    evaluation = run_with_retry(
        lambda: evaluator.run(query, writer_result, grounding_passed=grounding.grounding_passed)
    )

    best_output = writer_result

    if evaluation.pass_:
        _detect_and_save_preferences(query, evaluator_flags)
        memory.add_exchange(user=query, assistant=writer_result.short_answer)
        return writer_result

    evaluator_flags = evaluation.flags

    # On retry, re-run advocate in the mode determined by the confidence gate
    advocate_result = run_with_retry(
        lambda: advocate.run(analyst_output, query, research, mode=advocate_mode_on_retry)
    )
    analyst_output.advocate_challenges = advocate_result.challenges

# Exhausted refinements — return best attempt
_detect_and_save_preferences(query, evaluator_flags)
memory.add_exchange(user=query, assistant=best_output.short_answer)
return best_output
```

### `WriterAgent.run()` signature update

```python
def run(self, query: str, analysis: AnalystOutput, research: ResearcherOutput, evaluator_flags: list[str]) -> WriterOutput
```

The writer system prompt gains a new section that is injected when `analysis.advocate_challenges` is non-empty:

```
Challenges raised by the devil's advocate:
{chr(10).join(f"- {c}" for c in analysis.advocate_challenges)}

You must address each of these challenges in the caveats section.
```

### `EvaluatorAgent.run()` signature update

```python
def run(self, query: str, output: WriterOutput, grounding_passed: bool = True) -> EvaluatorOutput
```

If `grounding_passed=False`, the evaluator system prompt includes: `"Note: the grounding checker found unverified claims in this output. Score citation quality no higher than 0.4."`

### `_detect_and_save_preferences()` (private helper on `Supervisor`)

Called after every `WriterOutput` return (success or exhausted). Runs the keyword scan and inferred preference logic described in Section 4.

---

## 6. `main.py` Output Changes

- Print `warnings` under `"--- Grounding warnings ---"` section if `WriterOutput.warnings` is non-empty
- Print `"Note: confidence in statute coverage was low for this query."` if `result.analyst_confidence < 0.5` — read directly from `WriterOutput.analyst_confidence` (set by supervisor, see Section 5)

---

## 7. Error Handling

No new exception types needed:

- Both new agents are wrapped by `run_with_retry()` — validation failures retry up to `max_retries`, then raise `ValidationRepairError`
- SQLite errors in memory stores raise `FatalError`, caught by `main.py`'s generic `except Exception` handler
- DB directories are auto-created on first use via `os.makedirs(path, exist_ok=True)`

---

## 8. Testing

Five test files (three new, two existing updated):

- `tests/test_devils_advocate.py` — challenges produced for weak analysis; empty challenges + `severity=minor` on strong analysis; `severity=major` on serious gaps; `mode="strict"` produces more challenges than `mode="standard"`
- `tests/test_grounding_checker.py` — flags claims absent from source text; passes grounded claims; `warnings` attached correctly to `WriterOutput`; `grounding_passed=False` when ungrounded claims exist
- `tests/test_memory_stores.py` — `ConversationStore` persists across instantiations; history limit enforced; `UserPreferenceStore` read/write/overwrite; both stores handle missing DB file gracefully on first use
- `tests/test_supervisor.py` (update) — confidence gate sets `effective_refinements * 2`; advocate re-runs in `strict` mode on low-confidence retry; preference saved after successful `WriterOutput`; inferred preference test: simulate two queries where evaluator returns `"plain English"` flag, assert `language_level=technical` is written to `UserPreferenceStore` only after the second query
- `tests/test_schemas.py` (update) — `KeyClause` enforces `act` and `section` fields; `WriterOutput.warnings` defaults to `[]`; `AnalystLLMOutput` JSON schema does not contain `advocate_challenges`

Existing agent tests pass unchanged — `KeyClause` replaces `str` in `key_clauses` but all other fields have defaults.

---

## 9. Platinum Criteria Mapping

| Platinum Requirement | Implementation |
|---|---|
| Debate-style reasoning | `DevilsAdvocateAgent` challenges analyst before writer proceeds; advocate re-runs in strict mode on low-confidence queries |
| Auto-revision | Writer receives `advocate_challenges` + `evaluator_flags`; refinement loop carries both forward across iterations |
| Hallucination detection | `GroundingCheckerAgent` cross-references each `KeyClause` against source text; ungrounded claims surfaced as warnings; evaluator penalises citation quality |
| Advanced constraints | `KeyClause` schema enforces act + section citations; confidence gate routes low-confidence queries to extended refinement |
| Multiple memory stores | `ConversationStore` (history) + `UserPreferenceStore` (preferences), both SQLite-persisted and separately scoped |
