# Streamlit Demo UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Streamlit web UI that shows the Irish Statute Research Assistant's multi-agent pipeline executing live alongside a chat interface, running on a dedicated `feature/streamlit-ui` branch.

**Architecture:** A `progress_callback` parameter is added to `Supervisor.run()` and `Pipeline.query()` — the only change to core logic. `ResearcherAgent` gains a `last_source` attribute (like the existing `last_token_count`) so the Supervisor can report whether the vector store or live fetch was used. `app.py` at the project root creates a two-column Streamlit UI: chat on the left, a live pipeline trace on the right that updates as each agent fires its callback.

**Tech Stack:** Python, Streamlit ≥ 1.35, existing `Pipeline` / `Supervisor` classes, `uv` for dependency management.

---

## File Map

| File | Action | Purpose |
|---|---|---|
| `src/irish_statute_assistant/agents/researcher.py` | Modify | Add `last_source` attribute set to `"vector store"` or `"live fetch"` after each run |
| `src/irish_statute_assistant/agents/supervisor.py` | Modify | Add `progress_callback` param; fire it with timing stats after each agent |
| `src/irish_statute_assistant/pipeline.py` | Modify | Add `progress_callback` param; forward to `Supervisor.run()` |
| `pyproject.toml` | Modify | Add `[ui]` optional extra with `streamlit>=1.35` |
| `app.py` | Create | Streamlit app — layout, session state, callback wiring, error handling |
| `tests/test_researcher.py` | Modify | Add `last_source` tests |
| `tests/test_supervisor.py` | Modify | Add callback invocation and stats-key tests |
| `tests/test_pipeline.py` | Modify | Add callback forwarding test |

---

## Task 1: Create the branch and add the Streamlit dependency

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Create the feature branch**

```bash
git checkout -b feature/streamlit-ui
```

- [ ] **Step 2: Add the `[ui]` optional extra to `pyproject.toml`**

Open `pyproject.toml`. Find the `[project.optional-dependencies]` section (or create it after `[project]`). Add:

```toml
[project.optional-dependencies]
ui = ["streamlit>=1.35"]
```

- [ ] **Step 3: Install the new extra**

```bash
uv sync --extra ui
```

Expected: Streamlit and its dependencies download and install with no errors.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "chore: add streamlit as optional [ui] dependency"
```

---

## Task 2: Add `last_source` to `ResearcherAgent`

`ResearcherAgent.run()` chooses between the vector store and live HTTP fetch internally. To expose which path was taken (for the pipeline sidebar), we add a `last_source: str` attribute — the same pattern as `last_token_count` in `BaseAgent`.

**Files:**
- Modify: `src/irish_statute_assistant/agents/researcher.py`
- Modify: `tests/test_researcher.py`

- [ ] **Step 1: Write failing tests**

Open `tests/test_researcher.py` and add these two tests after the last existing test:

```python
def test_researcher_last_source_is_vector_store_when_populated(tmp_path, monkeypatch):
    from irish_statute_assistant.agents.researcher import ResearcherAgent
    from irish_statute_assistant.config import Config
    from irish_statute_assistant.tools.session_cache import SessionCache
    from unittest.mock import MagicMock, patch

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    config = Config(chroma_db_path=str(tmp_path / "chroma"))
    cache = SessionCache()
    fetcher = MagicMock()

    agent = ResearcherAgent(config, cache, fetcher)

    mock_vs = MagicMock()
    mock_vs.is_populated.return_value = True
    mock_vs.search.return_value = [
        {"page_content": "s1", "title": "Act A", "url": "https://x.com", "section_index": 0}
    ]
    agent._vector_store = mock_vs

    agent.run("test query")
    assert agent.last_source == "vector store"


def test_researcher_last_source_is_live_fetch_when_not_populated(tmp_path, monkeypatch):
    from irish_statute_assistant.agents.researcher import ResearcherAgent
    from irish_statute_assistant.config import Config
    from irish_statute_assistant.tools.session_cache import SessionCache
    from unittest.mock import MagicMock

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    config = Config(chroma_db_path=str(tmp_path / "chroma"))
    cache = SessionCache()

    fetcher = MagicMock()
    fetcher.search.return_value = [{"title": "Act A", "url": "https://x.com"}]
    fetcher.fetch.return_value = ["Section text"]

    agent = ResearcherAgent(config, cache, fetcher)

    mock_vs = MagicMock()
    mock_vs.is_populated.return_value = False
    agent._vector_store = mock_vs

    agent.run("test query")
    assert agent.last_source == "live fetch"
```

- [ ] **Step 2: Run to verify they fail**

```bash
uv run pytest tests/test_researcher.py::test_researcher_last_source_is_vector_store_when_populated tests/test_researcher.py::test_researcher_last_source_is_live_fetch_when_not_populated -v
```

Expected: FAILED — `AttributeError: 'ResearcherAgent' object has no attribute 'last_source'`

- [ ] **Step 3: Implement `last_source` in `researcher.py`**

In `src/irish_statute_assistant/agents/researcher.py`, add `last_source: str = "live fetch"` to `__init__`, and set it in `run()` before delegating:

```python
    def __init__(self, config: Config, cache: SessionCache, fetcher: StatuteFetcher) -> None:
        self._config = config
        self._cache = cache
        self._fetcher = fetcher
        self._vector_store = get_vector_store(config)
        self.last_source: str = "live fetch"

    def run(self, query: str) -> ResearcherOutput:
        """...(keep existing docstring)..."""
        if self._vector_store.is_populated():
            self.last_source = "vector store"
            return self._run_vector(query)
        self.last_source = "live fetch"
        logger.warning(
            "Vector store is not populated — falling back to live HTTP fetch. "
            "Run `python -m irish_statute_assistant.indexer` to build the index."
        )
        return self._run_live(query)
```

- [ ] **Step 4: Run the new tests**

```bash
uv run pytest tests/test_researcher.py::test_researcher_last_source_is_vector_store_when_populated tests/test_researcher.py::test_researcher_last_source_is_live_fetch_when_not_populated -v
```

Expected: 2 PASSED

- [ ] **Step 5: Run the full suite**

```bash
uv run pytest --tb=short -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/irish_statute_assistant/agents/researcher.py tests/test_researcher.py
git commit -m "feat: add last_source attribute to ResearcherAgent"
```

---

## Task 3: Add `progress_callback` to `Supervisor.run()`

This task instruments the Supervisor to fire a callback after each agent completes, with timing and a small stats dict.

**Files:**
- Modify: `src/irish_statute_assistant/agents/supervisor.py`
- Modify: `tests/test_supervisor.py`

- [ ] **Step 1: Write failing tests**

Add the following tests to `tests/test_supervisor.py` after the last existing test. The `make_supervisor` helper, `MagicMock`, and schema imports are already in that file:

```python
def test_supervisor_callback_called_for_each_agent_on_happy_path():
    research, analyst_llm, writer_out, eval_pass, advocate, grounding = make_defaults()
    sup = make_supervisor(
        clarifier_output=ClarifierOutput(needs_clarification=False),
        researcher_output=research,
        analyst_llm_output=analyst_llm,
        writer_output=writer_out,
        evaluator_output=eval_pass,
        advocate_output=advocate,
        grounding_output=grounding,
    )
    calls = []
    sup.run(query="Q", context=None, progress_callback=lambda name, stats: calls.append((name, stats)))
    agent_names = [name for name, _ in calls]
    assert agent_names == [
        "Clarifier", "Researcher", "Analyst", "Devil's Advocate",
        "Writer", "Grounding Checker", "Evaluator",
    ]


def test_supervisor_callback_stats_keys_per_agent():
    research, analyst_llm, writer_out, eval_pass, advocate, grounding = make_defaults()
    sup = make_supervisor(
        clarifier_output=ClarifierOutput(needs_clarification=False),
        researcher_output=research,
        analyst_llm_output=analyst_llm,
        writer_output=writer_out,
        evaluator_output=eval_pass,
        advocate_output=advocate,
        grounding_output=grounding,
    )
    stats_by_agent = {}
    sup.run(query="Q", context=None, progress_callback=lambda n, s: stats_by_agent.update({n: s}))

    assert {"needs_clarification", "duration_s"} <= stats_by_agent["Clarifier"].keys()
    assert {"acts_found", "source", "duration_s"} <= stats_by_agent["Researcher"].keys()
    assert {"key_clauses", "confidence", "duration_s"} <= stats_by_agent["Analyst"].keys()
    assert {"challenges", "severity", "round", "duration_s"} <= stats_by_agent["Devil's Advocate"].keys()
    assert {"round", "duration_s"} <= stats_by_agent["Writer"].keys()
    assert {"grounding_passed", "ungrounded", "duration_s"} <= stats_by_agent["Grounding Checker"].keys()
    assert {"score", "passed", "flags", "duration_s"} <= stats_by_agent["Evaluator"].keys()


def test_supervisor_callback_called_eleven_times_on_two_round_run():
    research, analyst_llm, writer_out, _, advocate, grounding = make_defaults()
    eval_fail = EvaluatorOutput(score=0.4, flags=["Bad"], **{"pass": False})
    eval_pass = EvaluatorOutput(score=0.85, flags=[], **{"pass": True})
    sup = make_supervisor(
        clarifier_output=ClarifierOutput(needs_clarification=False),
        researcher_output=research,
        analyst_llm_output=analyst_llm,
        writer_output=writer_out,
        evaluator_output=eval_fail,
        advocate_output=advocate,
        grounding_output=grounding,
    )
    sup._evaluator.run = MagicMock(side_effect=[eval_fail, eval_pass])
    calls = []
    sup.run(query="Q", context=None, progress_callback=lambda n, s: calls.append(n))
    # Clarifier, Researcher, Analyst, Advocate(initial) = 4
    # Round 1: Writer, Grounding Checker, Evaluator(fail), Advocate(retry) = 4
    # Round 2: Writer, Grounding Checker, Evaluator(pass) = 3
    assert len(calls) == 11


def test_supervisor_callback_fired_before_early_return_on_clarification():
    research, analyst_llm, writer_out, eval_pass, advocate, grounding = make_defaults()
    sup = make_supervisor(
        clarifier_output=ClarifierOutput(needs_clarification=True, question="Which court?"),
        researcher_output=research,
        analyst_llm_output=analyst_llm,
        writer_output=writer_out,
        evaluator_output=eval_pass,
        advocate_output=advocate,
        grounding_output=grounding,
    )
    calls = []
    result = sup.run(query="Q", context=None, progress_callback=lambda n, s: calls.append(n))
    assert result == "Which court?"
    assert calls == ["Clarifier"]
```

- [ ] **Step 1b: Add `last_source` to the `make_supervisor` test helper**

Open `tests/test_supervisor.py` and find the `make_supervisor` function. After the line `sup._researcher.last_token_count = 0`, add:

```python
    sup._researcher.last_source = "live fetch"
```

Without this, the callback tests will fail with `AttributeError` because the mock researcher has no `last_source` attribute.

- [ ] **Step 2: Run to verify they fail**

```bash
uv run pytest tests/test_supervisor.py::test_supervisor_callback_called_for_each_agent_on_happy_path tests/test_supervisor.py::test_supervisor_callback_stats_keys_per_agent tests/test_supervisor.py::test_supervisor_callback_called_eleven_times_on_two_round_run tests/test_supervisor.py::test_supervisor_callback_fired_before_early_return_on_clarification -v
```

Expected: 4 FAILED — `TypeError: run() got an unexpected keyword argument 'progress_callback'`

- [ ] **Step 3: Implement the callback in `supervisor.py`**

**3a.** Add these two imports at the top of `supervisor.py` (after the existing imports):

```python
import time
from collections.abc import Callable
```

**3b.** Change the `run()` signature:

```python
    def run(
        self,
        query: str,
        context: QueryContext | None = None,
        progress_callback: Callable[[str, dict], None] | None = None,
    ) -> WriterOutput | str:
```

**3c.** Replace the Clarifier block (currently lines 89–98) with:

```python
        # 1. Clarify
        _t0 = time.time()
        clarifier_result = run_with_retry(
            lambda: self._clarifier.run(query=query, history=history),
            self._max_retries,
        )
        if context:
            context.consume(self._clarifier.last_token_count)
        if progress_callback:
            progress_callback("Clarifier", {
                "needs_clarification": clarifier_result.needs_clarification,
                "duration_s": round(time.time() - _t0, 2),
            })
        if clarifier_result.needs_clarification:
            self._memory.add_exchange(user=query, assistant=clarifier_result.question)
            return clarifier_result.question
```

**3d.** Replace the Researcher block (currently lines 100–106) with:

```python
        # 2. Research
        _t0 = time.time()
        research = run_with_retry(
            lambda: self._researcher.run(query=query),
            self._max_retries,
        )
        if context:
            context.consume(self._researcher.last_token_count)
        if progress_callback:
            progress_callback("Researcher", {
                "acts_found": len(research.acts),
                "source": self._researcher.last_source,
                "duration_s": round(time.time() - _t0, 2),
            })
```

**3e.** Replace the Analyst block (currently lines 108–118) with:

```python
        # 3. Analyse (once, outside the loop)
        _t0 = time.time()
        llm_analyst_result = run_with_retry(
            lambda: self._analyst.run(query=query, research=research),
            self._max_retries,
        )
        if context:
            context.consume(self._analyst.last_token_count)
        analyst_output = AnalystOutput(
            **llm_analyst_result.model_dump(exclude={"advocate_challenges"}),
            advocate_challenges=[],
        )
        if progress_callback:
            progress_callback("Analyst", {
                "key_clauses": len(analyst_output.key_clauses),
                "confidence": analyst_output.confidence,
                "duration_s": round(time.time() - _t0, 2),
            })
```

**3f.** Replace the initial Devil's Advocate block (currently lines 120–131) with:

```python
        # 4. Devil's advocate (initial run)
        _t0 = time.time()
        advocate_result = run_with_retry(
            lambda: self._advocate.run(
                analyst_output=analyst_output,
                query=query,
                research=research,
                mode="standard",
            ),
            self._max_retries,
        )
        if context:
            context.consume(self._advocate.last_token_count)
        if progress_callback:
            progress_callback("Devil's Advocate", {
                "challenges": len(advocate_result.challenges),
                "severity": advocate_result.severity,
                "round": 0,
                "duration_s": round(time.time() - _t0, 2),
            })
```

**3g.** Replace the refinement loop (currently lines 140–203) with:

```python
        # 5–7. Refinement loop
        evaluator_flags: list[str] = []
        best_output: WriterOutput | None = None

        for round_num in range(1, effective_refinements + 2):
            _t0 = time.time()
            writer_result = run_with_retry(
                lambda: self._writer.run(
                    query=query,
                    analysis=analyst_output,
                    research=research,
                    evaluator_flags=evaluator_flags,
                    user_preferences=prefs,
                ),
                self._max_retries,
            )
            if context:
                context.consume(self._writer.last_token_count)
            writer_result.analyst_confidence = analyst_output.confidence
            if progress_callback:
                progress_callback("Writer", {
                    "round": round_num,
                    "duration_s": round(time.time() - _t0, 2),
                })

            _t0 = time.time()
            grounding = run_with_retry(
                lambda: self._grounding_checker.run(
                    writer_output=writer_result, research=research
                ),
                self._max_retries,
            )
            if context:
                context.consume(self._grounding_checker.last_token_count)
            writer_result.warnings = grounding.ungrounded_claims
            if progress_callback:
                progress_callback("Grounding Checker", {
                    "grounding_passed": grounding.grounding_passed,
                    "ungrounded": len(grounding.ungrounded_claims),
                    "duration_s": round(time.time() - _t0, 2),
                })

            _t0 = time.time()
            evaluation = run_with_retry(
                lambda: self._evaluator.run(
                    query=query,
                    output=writer_result,
                    grounding_passed=grounding.grounding_passed,
                ),
                self._max_retries,
            )
            if context:
                context.consume(self._evaluator.last_token_count)
            if progress_callback:
                progress_callback("Evaluator", {
                    "score": evaluation.score,
                    "passed": evaluation.pass_,
                    "flags": len(evaluation.flags),
                    "duration_s": round(time.time() - _t0, 2),
                })

            best_output = writer_result
            self._update_flag_counts(evaluation.flags)

            if evaluation.pass_:
                self._detect_and_save_preferences(query, evaluation.flags)
                self._memory.add_exchange(user=query, assistant=writer_result.short_answer)
                return writer_result

            evaluator_flags = evaluation.flags

            # Re-run advocate with potentially stricter mode
            _t0 = time.time()
            advocate_result = run_with_retry(
                lambda: self._advocate.run(
                    analyst_output=analyst_output,
                    query=query,
                    research=research,
                    mode=advocate_mode_on_retry,
                ),
                self._max_retries,
            )
            if context:
                context.consume(self._advocate.last_token_count)
            if progress_callback:
                progress_callback("Devil's Advocate", {
                    "challenges": len(advocate_result.challenges),
                    "severity": advocate_result.severity,
                    "round": round_num,
                    "duration_s": round(time.time() - _t0, 2),
                })
            analyst_output.advocate_challenges = advocate_result.challenges

        self._detect_and_save_preferences(query, evaluator_flags)
        self._memory.add_exchange(user=query, assistant=best_output.short_answer)
        return best_output
```

- [ ] **Step 4: Run the new tests**

```bash
uv run pytest tests/test_supervisor.py::test_supervisor_callback_called_for_each_agent_on_happy_path tests/test_supervisor.py::test_supervisor_callback_stats_keys_per_agent tests/test_supervisor.py::test_supervisor_callback_called_eleven_times_on_two_round_run tests/test_supervisor.py::test_supervisor_callback_fired_before_early_return_on_clarification -v
```

Expected: 4 PASSED

- [ ] **Step 5: Run the full suite**

```bash
uv run pytest --tb=short -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/irish_statute_assistant/agents/supervisor.py tests/test_supervisor.py
git commit -m "feat: add progress_callback to Supervisor.run() with per-agent timing stats"
```

---

## Task 4: Add `progress_callback` to `Pipeline.query()`

**Files:**
- Modify: `src/irish_statute_assistant/pipeline.py`
- Modify: `tests/test_pipeline.py`

- [ ] **Step 1: Write a failing test**

Open `tests/test_pipeline.py` and add this test after the last existing test:

```python
def test_pipeline_forwards_progress_callback_to_supervisor(monkeypatch):
    """Pipeline.query() must forward progress_callback to Supervisor.run()."""
    from unittest.mock import MagicMock, patch
    from irish_statute_assistant.pipeline import Pipeline
    from irish_statute_assistant.config import Config
    from irish_statute_assistant.models.schemas import WriterOutput, DetailedBreakdown, KeyClause

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    config = Config()
    pipeline = Pipeline(config)

    dummy_output = WriterOutput(
        short_answer="Test answer.",
        detailed_breakdown=DetailedBreakdown(
            summary="S", relevant_acts=["Act A"],
            key_clauses=[KeyClause(text="t", act="Act A", section="s.1")],
            caveats=["Seek advice"],
        ),
    )

    received_callback = {}

    def fake_run(query, context=None, progress_callback=None):
        received_callback["cb"] = progress_callback
        return dummy_output

    with patch.object(pipeline._supervisor, "run", side_effect=fake_run):
        cb = lambda name, stats: None
        pipeline.query("test query", progress_callback=cb)

    assert received_callback["cb"] is cb
```

- [ ] **Step 2: Run to verify it fails**

```bash
uv run pytest tests/test_pipeline.py::test_pipeline_forwards_progress_callback_to_supervisor -v
```

Expected: FAILED — `TypeError: query() got an unexpected keyword argument 'progress_callback'`

- [ ] **Step 3: Update `Pipeline.query()`**

Replace the `query` method in `src/irish_statute_assistant/pipeline.py`:

```python
    def query(self, user_query: str, progress_callback=None) -> WriterOutput | str:
        """Submit a user query and return the assistant's response.

        Args:
            user_query: The user's legal question.
            progress_callback: Optional callable fired after each agent completes.
                Receives (agent_name: str, stats: dict). Used by the Streamlit UI
                to update the pipeline trace panel in real time.

        Returns:
            A clarifying question string if the query is ambiguous,
            or a WriterOutput with the final answer.
        """
        context = QueryContext(budget=self._config.token_budget_per_query)
        result = self._supervisor.run(
            query=user_query,
            context=context,
            progress_callback=progress_callback,
        )
        logger.info(
            "Query used %d/%d tokens",
            context.tokens_used,
            context.budget,
        )
        return result
```

- [ ] **Step 4: Run the new test**

```bash
uv run pytest tests/test_pipeline.py::test_pipeline_forwards_progress_callback_to_supervisor -v
```

Expected: PASSED

- [ ] **Step 5: Run the full suite**

```bash
uv run pytest --tb=short -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/irish_statute_assistant/pipeline.py tests/test_pipeline.py
git commit -m "feat: add progress_callback to Pipeline.query(), forwarded to Supervisor"
```

---

## Task 5: Build `app.py`

**Files:**
- Create: `app.py` (project root)

No unit tests — validated by running the app manually (see Step 2).

- [ ] **Step 1: Create `app.py`**

Create `app.py` at the project root:

```python
"""Streamlit demo UI for the Irish Statute Research Assistant.

Two-column layout:
  Left  — chat conversation (questions + answers)
  Right — live pipeline trace (one row per agent, appears as each fires)

Run with:
    uv run streamlit run app.py
"""
from __future__ import annotations

import time

import streamlit as st

from irish_statute_assistant.config import Config
from irish_statute_assistant.exceptions import (
    BudgetExceededError,
    FatalError,
    StatuteNotFoundError,
    ValidationRepairError,
)
from irish_statute_assistant.models.schemas import WriterOutput
from irish_statute_assistant.pipeline import Pipeline

# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Irish Statute Research Assistant",
    page_icon="🏛",
    layout="wide",
)

# ── Pipeline (one instance, shared across reruns) ─────────────────────────────

@st.cache_resource
def get_pipeline() -> Pipeline:
    return Pipeline(Config())

pipeline = get_pipeline()

# ── Session state ─────────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []        # {"role": "user"|"assistant", "content": str}
if "pipeline_steps" not in st.session_state:
    st.session_state.pipeline_steps = []  # {"agent": str, "stats": dict}

# ── Helper: render one pipeline step row ──────────────────────────────────────

def _step_label(agent: str, stats: dict) -> str:
    """Return a markdown string for one completed pipeline step row."""
    if agent == "Devil's Advocate":
        icon = "⚔️"
        round_num = stats.get("round", 0)
        label = f"Devil's Advocate" + (f" (round {round_num})" if round_num else " (initial)")
    elif agent == "Clarifier" and stats.get("needs_clarification"):
        icon = "❓"
        label = "Clarifier"
    else:
        icon = "✅"
        label = agent

    dur = stats.get("duration_s", 0)

    if agent == "Clarifier":
        detail = "needs clarification" if stats.get("needs_clarification") else "no clarification needed"
    elif agent == "Researcher":
        detail = f"{stats.get('acts_found', 0)} acts · {stats.get('source', '')}"
    elif agent == "Analyst":
        detail = f"{stats.get('key_clauses', 0)} clauses · confidence {stats.get('confidence', 0):.2f}"
    elif agent == "Devil's Advocate":
        detail = f"{stats.get('challenges', 0)} challenge(s) · {stats.get('severity', '')}"
    elif agent == "Writer":
        detail = f"round {stats.get('round', 1)}"
    elif agent == "Grounding Checker":
        detail = "all claims grounded" if stats.get("grounding_passed") else f"{stats.get('ungrounded', 0)} ungrounded"
    elif agent == "Evaluator":
        outcome = "passed ✓" if stats.get("passed") else "failed ✗"
        detail = f"score {stats.get('score', 0):.2f} · {outcome}"
    else:
        detail = ""

    return f"{icon} **{label}** — {detail} *({dur}s)*"


def _render_pipeline(steps: list[dict], spinning: str | None = None) -> None:
    """Render pipeline steps into the current Streamlit column context."""
    if not steps and spinning is None:
        st.caption("Pipeline trace will appear here during a query.")
        return

    for step in steps:
        st.markdown(_step_label(step["agent"], step["stats"]))

    if spinning:
        st.markdown(f"⏳ **{spinning}**…")

    if steps and spinning is None:
        total_dur = sum(s["stats"].get("duration_s", 0) for s in steps)
        rounds = max(
            (s["stats"].get("round", 1) for s in steps if s["agent"] == "Writer"),
            default=1,
        )
        st.divider()
        st.caption(f"{rounds} round(s) · {total_dur:.1f}s total")


# ── Render existing conversation ───────────────────────────────────────────────

st.title("🏛 Irish Statute Research Assistant")
chat_col, pipe_col = st.columns([2, 1])

with chat_col:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

with pipe_col:
    st.subheader("Pipeline")
    pipe_placeholder = st.empty()
    with pipe_placeholder.container():
        _render_pipeline(st.session_state.pipeline_steps)

# ── Chat input ────────────────────────────────────────────────────────────────

user_input = st.chat_input("Ask a legal question…")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with chat_col:
        with st.chat_message("user"):
            st.markdown(user_input)

    st.session_state.pipeline_steps = []

    # Track next expected agent for the spinner
    _AGENT_ORDER = [
        "Clarifier", "Researcher", "Analyst", "Devil's Advocate",
        "Writer", "Grounding Checker", "Evaluator",
    ]
    next_agent: list[str] = ["Clarifier"]

    def on_step(agent_name: str, stats: dict) -> None:
        st.session_state.pipeline_steps.append({"agent": agent_name, "stats": stats})
        # Advance spinner to the next expected agent (rough heuristic)
        try:
            idx = _AGENT_ORDER.index(agent_name)
            next_agent[0] = _AGENT_ORDER[idx + 1] if idx + 1 < len(_AGENT_ORDER) else ""
        except ValueError:
            next_agent[0] = ""
        with pipe_placeholder.container():
            _render_pipeline(st.session_state.pipeline_steps, spinning=next_agent[0] or None)

    # Show initial spinner
    with pipe_placeholder.container():
        _render_pipeline([], spinning="Clarifier")

    result = None
    try:
        result = pipeline.query(user_input, progress_callback=on_step)
    except StatuteNotFoundError:
        with chat_col:
            st.error("No relevant statutes found. Please try rephrasing your question.")
    except BudgetExceededError:
        with chat_col:
            st.error(
                "This query exceeded the token budget. "
                "Increase `TOKEN_BUDGET_PER_QUERY` in your `.env` file."
            )
    except ValidationRepairError:
        with chat_col:
            st.error("Could not produce a valid response after several attempts. Please try rephrasing.")
    except FatalError as e:
        with chat_col:
            st.error(f"An unrecoverable error occurred. Check the terminal for details. ({e})")
    except Exception as e:
        with chat_col:
            st.error(f"Something went wrong: {e}")

    # Final render — no spinner
    with pipe_placeholder.container():
        _render_pipeline(st.session_state.pipeline_steps)

    if isinstance(result, str):
        # Clarifying question
        st.session_state.messages.append({"role": "assistant", "content": result})
        with chat_col:
            with st.chat_message("assistant"):
                st.markdown(result)
    elif isinstance(result, WriterOutput):
        bd = result.detailed_breakdown
        lines = [f"**{result.short_answer}**", ""]

        if bd.relevant_acts:
            lines.append("**Relevant Acts:** " + ", ".join(bd.relevant_acts))
            lines.append("")

        if bd.key_clauses:
            lines.append("**Key clauses:**")
            for kc in bd.key_clauses:
                lines.append(f"- {kc.text} *({kc.act}, {kc.section})*")
            lines.append("")

        if bd.caveats:
            lines.append("**Things to be aware of:**")
            for caveat in bd.caveats:
                lines.append(f"- {caveat}")

        if result.warnings:
            lines.append("")
            lines.append("⚠️ *Some claims could not be verified against source text.*")

        if result.analyst_confidence < 0.5:
            lines.append("")
            lines.append("*Note: confidence in statute coverage was low for this query.*")

        content = "\n".join(lines)
        st.session_state.messages.append({"role": "assistant", "content": content})
        with chat_col:
            with st.chat_message("assistant"):
                st.markdown(content)
```

- [ ] **Step 2: Run the app and verify it works**

```bash
uv run streamlit run app.py
```

Open `http://localhost:8501`. Ask: *"How long do I have to bring a personal injury claim in Ireland?"*

Verify:
- Pipeline sidebar populates row by row as agents complete
- Devil's Advocate row shows amber ⚔️ icon
- Final answer appears in the chat column with key clauses and caveats
- The pipeline footer shows round count and total duration
- Asking a second question resets the pipeline sidebar

- [ ] **Step 3: Commit**

```bash
git add app.py
git commit -m "feat: add Streamlit demo UI with live pipeline trace sidebar"
```

---

## Task 6: Final check

- [ ] **Step 1: Run the full test suite on the feature branch**

```bash
uv run pytest --tb=short -q
```

Expected: all tests pass.

- [ ] **Step 2: Verify the branch is clean and ahead of main**

```bash
git status
git log --oneline main..HEAD
```

Expected: no uncommitted changes; 5 commits ahead of `main`:
1. `chore: add streamlit as optional [ui] dependency`
2. `feat: add last_source attribute to ResearcherAgent`
3. `feat: add progress_callback to Supervisor.run() with per-agent timing stats`
4. `feat: add progress_callback to Pipeline.query(), forwarded to Supervisor`
5. `feat: add Streamlit demo UI with live pipeline trace sidebar`

(That is 5 commits total — one per task.)

- [ ] **Step 3: Done**

The feature branch `feature/streamlit-ui` is ready. Do **not** merge to `main` — this branch exists solely for the demo.
