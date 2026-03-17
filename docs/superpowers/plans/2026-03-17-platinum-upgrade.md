# Platinum Upgrade Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the Irish Statute Assistant to Platinum level by adding debate-style reasoning (DevilsAdvocateAgent), hallucination detection (GroundingCheckerAgent), structured citation enforcement (KeyClause schema), persistent dual memory stores (ConversationStore + UserPreferenceStore), and a confidence-gated refinement loop.

**Architecture:** New agents slot into the existing pipeline between analyst→writer and writer→evaluator. The analyst runs once before the refinement loop; the refinement loop covers write→ground-check→evaluate. The supervisor owns all memory writes; pipeline.py simply wires stores into the supervisor at construction time.

**Tech Stack:** Python 3.11+, LangChain, Pydantic v2, SQLite (stdlib `sqlite3`), pytest, unittest.mock

---

## File Map

### Create
- `src/irish_statute_assistant/agents/devils_advocate.py` — DevilsAdvocateAgent
- `src/irish_statute_assistant/agents/grounding_checker.py` — GroundingCheckerAgent
- `src/irish_statute_assistant/memory/conversation_store.py` — SQLite-backed conversation history
- `src/irish_statute_assistant/memory/user_preference_store.py` — SQLite-backed key-value preference store
- `tests/test_devils_advocate.py`
- `tests/test_grounding_checker.py`
- `tests/test_memory_stores.py`

### Modify
- `src/irish_statute_assistant/models/schemas.py` — add KeyClause, AnalystLLMOutput; update AnalystOutput, DetailedBreakdown, WriterOutput
- `src/irish_statute_assistant/config.py` — add temperature, conversation_history_limit, db path fields
- `src/irish_statute_assistant/agents/analyst.py` — use AnalystLLMOutput, remove evaluator_flags param
- `src/irish_statute_assistant/agents/writer.py` — KeyClause serialisation, advocate_challenges prompt injection
- `src/irish_statute_assistant/agents/evaluator.py` — grounding_passed param, KeyClause serialisation
- `src/irish_statute_assistant/agents/supervisor.py` — full rewrite per spec
- `src/irish_statute_assistant/pipeline.py` — wire ConversationStore + UserPreferenceStore into supervisor
- `src/irish_statute_assistant/main.py` — show grounding warnings and low-confidence banner
- `tests/test_schemas.py` — update for KeyClause, AnalystLLMOutput, new WriterOutput fields
- `tests/test_analyst.py` — update for removed evaluator_flags, AnalystLLMOutput return type
- `tests/test_writer.py` — update for KeyClause in key_clauses, advocate_challenges
- `tests/test_supervisor.py` — full rewrite for new supervisor interface

---

## Task 1: Schema Changes

**Files:**
- Modify: `src/irish_statute_assistant/models/schemas.py`
- Modify: `tests/test_schemas.py`

- [ ] **Step 1: Write failing tests for new and changed schema models**

Add to `tests/test_schemas.py`:

```python
from irish_statute_assistant.models.schemas import KeyClause, AnalystLLMOutput

# --- KeyClause ---

def test_key_clause_valid():
    kc = KeyClause(text="Actions must start within 6 years.", act="Statute of Limitations Act 1957", section="s.11")
    assert kc.text == "Actions must start within 6 years."

def test_key_clause_requires_act():
    with pytest.raises(ValidationError):
        KeyClause(text="something", section="s.1")

def test_key_clause_requires_section():
    with pytest.raises(ValidationError):
        KeyClause(text="something", act="Some Act")


# --- AnalystLLMOutput ---

def test_analyst_llm_output_valid():
    kc = KeyClause(text="6 year limit", act="Act A", section="s.1")
    data = AnalystLLMOutput(key_clauses=[kc], gaps=[], confidence=0.9)
    assert data.confidence == 0.9

def test_analyst_llm_output_has_no_advocate_challenges_field():
    """advocate_challenges must not appear in the JSON schema used by the LLM."""
    schema = AnalystLLMOutput.model_json_schema()
    assert "advocate_challenges" not in schema.get("properties", {})


# --- AnalystOutput (extends AnalystLLMOutput) ---

def test_analyst_output_has_advocate_challenges():
    kc = KeyClause(text="Clause", act="Act A", section="s.1")
    data = AnalystOutput(key_clauses=[kc], gaps=[], confidence=0.8, advocate_challenges=["Challenge 1"])
    assert data.advocate_challenges == ["Challenge 1"]

def test_analyst_output_advocate_challenges_defaults_empty():
    kc = KeyClause(text="Clause", act="Act A", section="s.1")
    data = AnalystOutput(key_clauses=[kc], gaps=[], confidence=0.8)
    assert data.advocate_challenges == []


# --- WriterOutput new fields ---

def test_writer_output_warnings_defaults_empty():
    kc = KeyClause(text="You must X", act="Act A", section="s.1")
    breakdown = DetailedBreakdown(
        summary="Summary", relevant_acts=["Act A"],
        key_clauses=[kc], caveats=["Seek advice"],
    )
    data = WriterOutput(short_answer="Short answer here.", detailed_breakdown=breakdown)
    assert data.warnings == []
    assert data.analyst_confidence == 1.0
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd /home/power/projects/ai/langflow-learning-club
python -m pytest tests/test_schemas.py -v 2>&1 | tail -20
```
Expected: ImportError or AttributeError — KeyClause, AnalystLLMOutput not found yet.

- [ ] **Step 3: Replace `schemas.py` with updated version**

```python
from __future__ import annotations

from typing import Literal, Optional
from pydantic import BaseModel, Field, field_validator, model_validator


class ActSection(BaseModel):
    title: str
    url: str
    sections: list[str]


class ResearcherOutput(BaseModel):
    acts: list[ActSection] = Field(min_length=1)


class KeyClause(BaseModel):
    text: str
    act: str
    section: str


class AnalystLLMOutput(BaseModel):
    """Schema fed to the LLM via with_structured_output — no supervisor-side fields."""
    key_clauses: list[KeyClause]
    gaps: list[str]
    confidence: float = Field(ge=0.0, le=1.0)


class AnalystOutput(AnalystLLMOutput):
    """Full analyst context passed through the pipeline.

    advocate_challenges is populated by the Supervisor after the DevilsAdvocateAgent
    runs — it is intentionally absent from AnalystLLMOutput so the LLM never sees it.
    """
    advocate_challenges: list[str] = []


class DetailedBreakdown(BaseModel):
    summary: str
    relevant_acts: list[str]
    key_clauses: list[KeyClause]
    caveats: list[str]


class WriterOutput(BaseModel):
    short_answer: str
    detailed_breakdown: DetailedBreakdown
    warnings: list[str] = []
    analyst_confidence: float = 1.0

    @field_validator("short_answer")
    @classmethod
    def short_answer_max_100_words(cls, v: str) -> str:
        word_count = len(v.split())
        if word_count > 100:
            raise ValueError(f"short_answer must be ≤100 words, got {word_count}")
        return v


class EvaluatorOutput(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    flags: list[str]
    pass_: bool = Field(alias="pass")

    model_config = {"populate_by_name": True}


class AdvocateOutput(BaseModel):
    challenges: list[str] = Field(default_factory=list, max_length=5)
    severity: Literal["minor", "major"]


class GroundingOutput(BaseModel):
    ungrounded_claims: list[str]
    grounding_passed: bool


class ClarifierOutput(BaseModel):
    needs_clarification: bool
    question: Optional[str] = None

    @model_validator(mode="after")
    def question_required_when_clarification_needed(self) -> "ClarifierOutput":
        if self.needs_clarification and self.question is None:
            raise ValueError("question must be provided when needs_clarification is True")
        return self
```

- [ ] **Step 4: Update existing tests in `test_schemas.py` that used `key_clauses=["string"]`**

Replace these existing tests (they used bare strings and now need `KeyClause`):

```python
# --- AnalystOutput ---

def test_analyst_output_valid():
    kc = KeyClause(text="Clause 1", act="Act A", section="s.1")
    data = AnalystOutput(key_clauses=[kc], gaps=[], confidence=0.85)
    assert data.confidence == 0.85


# --- WriterOutput ---

def test_writer_output_valid():
    kc = KeyClause(text="You must do X", act="Act A", section="s.1")
    breakdown = DetailedBreakdown(
        summary="Summary text",
        relevant_acts=["Act A"],
        key_clauses=[kc],
        caveats=["This may vary"],
    )
    data = WriterOutput(short_answer="You have two years to make a claim.", detailed_breakdown=breakdown)
    assert data.short_answer.startswith("You")


def test_writer_short_answer_over_100_words_rejected():
    long_answer = " ".join(["word"] * 101)
    kc = KeyClause(text="x", act="x", section="x")
    breakdown = DetailedBreakdown(summary="x", relevant_acts=[], key_clauses=[kc], caveats=[])
    with pytest.raises(ValidationError):
        WriterOutput(short_answer=long_answer, detailed_breakdown=breakdown)
```

- [ ] **Step 5: Run all schema tests**

```bash
python -m pytest tests/test_schemas.py -v 2>&1 | tail -30
```
Expected: All PASS.

- [ ] **Step 6: Commit**

```bash
git add src/irish_statute_assistant/models/schemas.py tests/test_schemas.py
git commit -m "feat: add KeyClause, AnalystLLMOutput, AdvocateOutput, GroundingOutput schemas"
```

---

## Task 2: ConversationStore

**Files:**
- Create: `src/irish_statute_assistant/memory/conversation_store.py`
- Create/Modify: `tests/test_memory_stores.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_memory_stores.py`:

```python
import os
import tempfile
import pytest
from irish_statute_assistant.memory.conversation_store import ConversationStore


def test_conversation_store_add_and_retrieve(tmp_path):
    db = str(tmp_path / "conv.db")
    store = ConversationStore(db_path=db, history_limit=20)
    store.add_exchange(user="Hello", assistant="Hi there")
    history = store.get_history()
    assert len(history) == 1
    assert history[0]["user"] == "Hello"
    assert history[0]["assistant"] == "Hi there"


def test_conversation_store_persists_across_instantiations(tmp_path):
    db = str(tmp_path / "conv.db")
    store1 = ConversationStore(db_path=db, history_limit=20)
    store1.add_exchange(user="Q1", assistant="A1")

    store2 = ConversationStore(db_path=db, history_limit=20)
    history = store2.get_history()
    assert len(history) == 1
    assert history[0]["user"] == "Q1"


def test_conversation_store_history_limit_enforced(tmp_path):
    db = str(tmp_path / "conv.db")
    store = ConversationStore(db_path=db, history_limit=3)
    for i in range(5):
        # Each instance loads up to limit; add via same instance to exceed
        pass
    # Write 5 exchanges to same DB, then load with limit=3
    store_write = ConversationStore(db_path=db, history_limit=10)
    for i in range(5):
        store_write.add_exchange(user=f"Q{i}", assistant=f"A{i}")

    store_read = ConversationStore(db_path=db, history_limit=3)
    history = store_read.get_history()
    assert len(history) == 3
    # Should be the 3 most recent
    assert history[0]["user"] == "Q2"
    assert history[2]["user"] == "Q4"


def test_conversation_store_format_for_prompt(tmp_path):
    db = str(tmp_path / "conv.db")
    store = ConversationStore(db_path=db)
    store.add_exchange(user="What are my rights?", assistant="That depends on context.")
    prompt = store.format_for_prompt()
    assert "User: What are my rights?" in prompt
    assert "Assistant: That depends on context." in prompt


def test_conversation_store_empty_format_for_prompt(tmp_path):
    db = str(tmp_path / "conv.db")
    store = ConversationStore(db_path=db)
    assert store.format_for_prompt() == ""


def test_conversation_store_creates_db_dir_on_first_use(tmp_path):
    nested = str(tmp_path / "a" / "b" / "conv.db")
    store = ConversationStore(db_path=nested)
    store.add_exchange(user="x", assistant="y")
    assert os.path.exists(nested)
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
python -m pytest tests/test_memory_stores.py -v 2>&1 | tail -15
```
Expected: ImportError — ConversationStore not found.

- [ ] **Step 3: Implement ConversationStore**

Create `src/irish_statute_assistant/memory/conversation_store.py`:

```python
from __future__ import annotations

import os
import sqlite3


class ConversationStore:
    """SQLite-backed conversation history. Drop-in replacement for SessionMemory."""

    def __init__(self, db_path: str, history_limit: int = 20) -> None:
        self._db_path = os.path.expanduser(db_path)
        self._history_limit = history_limit
        os.makedirs(os.path.dirname(self._db_path), exist_ok=True)
        self._init_db()
        self._history = self._load_history()

    def _init_db(self) -> None:
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS exchanges "
                "(id INTEGER PRIMARY KEY AUTOINCREMENT, user TEXT, assistant TEXT)"
            )

    def _load_history(self) -> list[dict[str, str]]:
        with sqlite3.connect(self._db_path) as conn:
            rows = conn.execute(
                "SELECT user, assistant FROM exchanges "
                "ORDER BY id DESC LIMIT ?",
                (self._history_limit,),
            ).fetchall()
        return [{"user": r[0], "assistant": r[1]} for r in reversed(rows)]

    def add_exchange(self, user: str, assistant: str) -> None:
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                "INSERT INTO exchanges (user, assistant) VALUES (?, ?)",
                (user, assistant),
            )
        self._history.append({"user": user, "assistant": assistant})
        if len(self._history) > self._history_limit:
            self._history = self._history[-self._history_limit:]

    def get_history(self) -> list[dict[str, str]]:
        return list(self._history)

    def format_for_prompt(self) -> str:
        if not self._history:
            return ""
        lines = []
        for exchange in self._history:
            lines.append(f"User: {exchange['user']}")
            lines.append(f"Assistant: {exchange['assistant']}")
        return "\n".join(lines)
```

- [ ] **Step 4: Run ConversationStore tests**

```bash
python -m pytest tests/test_memory_stores.py -k "conversation" -v 2>&1 | tail -20
```
Expected: All PASS.

- [ ] **Step 5: Commit**

```bash
git add src/irish_statute_assistant/memory/conversation_store.py tests/test_memory_stores.py
git commit -m "feat: add SQLite-backed ConversationStore"
```

---

## Task 3: UserPreferenceStore

**Files:**
- Create: `src/irish_statute_assistant/memory/user_preference_store.py`
- Modify: `tests/test_memory_stores.py`

- [ ] **Step 1: Add failing tests**

Append to `tests/test_memory_stores.py`:

```python
from irish_statute_assistant.memory.user_preference_store import UserPreferenceStore


def test_preference_store_set_and_get(tmp_path):
    db = str(tmp_path / "prefs.db")
    store = UserPreferenceStore(db_path=db)
    store.set("language_level", "plain")
    assert store.get("language_level") == "plain"


def test_preference_store_default_when_missing(tmp_path):
    db = str(tmp_path / "prefs.db")
    store = UserPreferenceStore(db_path=db)
    assert store.get("nonexistent", default="fallback") == "fallback"


def test_preference_store_overwrite(tmp_path):
    db = str(tmp_path / "prefs.db")
    store = UserPreferenceStore(db_path=db)
    store.set("verbosity", "brief")
    store.set("verbosity", "detailed")
    assert store.get("verbosity") == "detailed"


def test_preference_store_all(tmp_path):
    db = str(tmp_path / "prefs.db")
    store = UserPreferenceStore(db_path=db)
    store.set("user_type", "solicitor")
    store.set("language_level", "technical")
    prefs = store.all()
    assert prefs == {"user_type": "solicitor", "language_level": "technical"}


def test_preference_store_persists_across_instantiations(tmp_path):
    db = str(tmp_path / "prefs.db")
    UserPreferenceStore(db_path=db).set("user_type", "solicitor")
    assert UserPreferenceStore(db_path=db).get("user_type") == "solicitor"
```

- [ ] **Step 2: Run to confirm failure**

```bash
python -m pytest tests/test_memory_stores.py -k "preference" -v 2>&1 | tail -15
```
Expected: ImportError.

- [ ] **Step 3: Implement UserPreferenceStore**

Create `src/irish_statute_assistant/memory/user_preference_store.py`:

```python
from __future__ import annotations

import os
import sqlite3


class UserPreferenceStore:
    """SQLite-backed key-value store for user preferences."""

    def __init__(self, db_path: str) -> None:
        self._db_path = os.path.expanduser(db_path)
        os.makedirs(os.path.dirname(self._db_path), exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS preferences "
                "(key TEXT PRIMARY KEY, value TEXT)"
            )

    def set(self, key: str, value: str) -> None:
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                "INSERT INTO preferences (key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, value),
            )

    def get(self, key: str, default: str = "") -> str:
        with sqlite3.connect(self._db_path) as conn:
            row = conn.execute(
                "SELECT value FROM preferences WHERE key = ?", (key,)
            ).fetchone()
        return row[0] if row else default

    def all(self) -> dict[str, str]:
        with sqlite3.connect(self._db_path) as conn:
            rows = conn.execute("SELECT key, value FROM preferences").fetchall()
        return {k: v for k, v in rows}
```

- [ ] **Step 4: Run all memory store tests**

```bash
python -m pytest tests/test_memory_stores.py -v 2>&1 | tail -20
```
Expected: All PASS.

- [ ] **Step 5: Commit**

```bash
git add src/irish_statute_assistant/memory/user_preference_store.py tests/test_memory_stores.py
git commit -m "feat: add SQLite-backed UserPreferenceStore"
```

---

## Task 4: Config Updates

**Files:**
- Modify: `src/irish_statute_assistant/config.py`

- [ ] **Step 1: Run existing config tests to confirm baseline**

```bash
python -m pytest tests/test_config.py -v 2>&1 | tail -10
```
Expected: All PASS.

- [ ] **Step 2: Add new config fields**

In `config.py`, add after `token_budget_per_query`:

```python
    conversation_history_limit: int = 20
    conversations_db_path: str = "~/.irish_statute_assistant/conversations.db"
    preferences_db_path: str = "~/.irish_statute_assistant/preferences.db"
```

- [ ] **Step 3: Run config tests**

```bash
python -m pytest tests/test_config.py -v 2>&1 | tail -10
```
Expected: All PASS.

- [ ] **Step 4: Commit**

```bash
git add src/irish_statute_assistant/config.py
git commit -m "feat: add conversation_history_limit and db path config fields"
```

---

## Task 5: Update AnalystAgent

The analyst now returns `AnalystLLMOutput` (not `AnalystOutput`) and no longer takes `evaluator_flags` (it runs once before the refinement loop; flags belong to the writer).

**Files:**
- Modify: `src/irish_statute_assistant/agents/analyst.py`
- Modify: `tests/test_analyst.py`

- [ ] **Step 1: Update test_analyst.py**

Replace the full contents of `tests/test_analyst.py`:

```python
from unittest.mock import MagicMock
from irish_statute_assistant.agents.analyst import AnalystAgent
from irish_statute_assistant.models.schemas import (
    AnalystLLMOutput, KeyClause, ResearcherOutput, ActSection
)


def make_analyst(key_clauses, gaps, confidence):
    agent = AnalystAgent.__new__(AnalystAgent)
    mock_chain = MagicMock()
    mock_chain.invoke = MagicMock(return_value=AnalystLLMOutput(
        key_clauses=key_clauses, gaps=gaps, confidence=confidence
    ))
    agent._chain = mock_chain
    return agent


def sample_research():
    return ResearcherOutput(acts=[
        ActSection(
            title="Statute of Limitations Act 1957",
            url="https://example.com/1957",
            sections=["Actions must be brought within 6 years."]
        )
    ])


def sample_key_clause():
    return KeyClause(text="Bring action within 6 years", act="Statute of Limitations Act 1957", section="s.11")


def test_analyst_returns_analyst_llm_output():
    agent = make_analyst([sample_key_clause()], [], 0.9)
    result = agent.run(query="limitation period", research=sample_research())
    assert isinstance(result, AnalystLLMOutput)
    assert result.confidence == 0.9


def test_analyst_confidence_in_valid_range():
    agent = make_analyst([], [], 0.5)
    result = agent.run(query="Q", research=sample_research())
    assert 0.0 <= result.confidence <= 1.0


def test_analyst_run_does_not_accept_evaluator_flags():
    """Analyst runs once before the loop; evaluator_flags no longer belong here."""
    import inspect
    sig = inspect.signature(AnalystAgent.run)
    assert "evaluator_flags" not in sig.parameters
```

- [ ] **Step 2: Run to confirm failures**

```bash
python -m pytest tests/test_analyst.py -v 2>&1 | tail -15
```
Expected: Failures due to signature mismatch and wrong return type.

- [ ] **Step 3: Update analyst.py**

Replace `analyst.py` contents:

```python
from langchain_core.prompts import ChatPromptTemplate

from irish_statute_assistant.agents.base_agent import BaseAgent
from irish_statute_assistant.config import Config
from irish_statute_assistant.llm import get_llm
from irish_statute_assistant.models.schemas import AnalystLLMOutput, ResearcherOutput

SYSTEM_PROMPT = """You are a legal analyst. You have been given raw Irish statute text and
a user's question. Your job is to:
1. Identify the key clauses directly relevant to the question. For each clause include:
   - text: the rule in plain English
   - act: the full name of the Act (e.g. "Statute of Limitations Act 1957")
   - section: the section number (e.g. "s.11")
2. Note any gaps (things the user asked about that the statutes don't clearly address).
3. Assign a confidence score (0.0–1.0) for how well the statutes answer the question.
   - 0.9+ = question is fully and clearly answered
   - 0.5–0.89 = partially answered, some ambiguity
   - below 0.5 = statutes are unclear or not directly relevant
"""

HUMAN_PROMPT = """User question: {query}

Retrieved statute sections:
{statute_text}
"""


class AnalystAgent(BaseAgent):
    def __init__(self, config: Config) -> None:
        llm = get_llm(config, max_tokens=1024).with_structured_output(AnalystLLMOutput)
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", HUMAN_PROMPT),
        ])
        self._chain = prompt | llm

    def run(self, query: str, research: ResearcherOutput) -> AnalystLLMOutput:
        statute_text = self._format_research(research)
        return self._invoke_chain(self._chain, {
            "query": query,
            "statute_text": statute_text,
        })

    def _format_research(self, research: ResearcherOutput) -> str:
        parts = []
        for act in research.acts:
            parts.append(f"## {act.title}\nURL: {act.url}")
            for section in act.sections:
                parts.append(section)
        return "\n\n".join(parts)
```

- [ ] **Step 4: Run analyst tests**

```bash
python -m pytest tests/test_analyst.py -v 2>&1 | tail -15
```
Expected: All PASS.

- [ ] **Step 5: Commit**

```bash
git add src/irish_statute_assistant/agents/analyst.py tests/test_analyst.py
git commit -m "feat: analyst uses AnalystLLMOutput and no longer takes evaluator_flags"
```

---

## Task 6: Update WriterAgent

The writer now serialises `KeyClause` objects and injects `advocate_challenges` into its prompt.

**Files:**
- Modify: `src/irish_statute_assistant/agents/writer.py`
- Modify: `tests/test_writer.py`

- [ ] **Step 1: Update test_writer.py**

Replace full contents:

```python
from unittest.mock import MagicMock
from irish_statute_assistant.agents.writer import WriterAgent
from irish_statute_assistant.models.schemas import (
    WriterOutput, DetailedBreakdown, AnalystOutput,
    KeyClause, ResearcherOutput, ActSection
)


def sample_key_clause():
    return KeyClause(text="6 year limit", act="Statute of Limitations Act 1957", section="s.11")


def make_writer(short_answer, summary, relevant_acts, key_clauses, caveats):
    agent = WriterAgent.__new__(WriterAgent)
    mock_chain = MagicMock()
    mock_chain.invoke = MagicMock(return_value=WriterOutput(
        short_answer=short_answer,
        detailed_breakdown=DetailedBreakdown(
            summary=summary,
            relevant_acts=relevant_acts,
            key_clauses=key_clauses,
            caveats=caveats,
        )
    ))
    agent._chain = mock_chain
    return agent


def sample_analyst_output(advocate_challenges=None):
    return AnalystOutput(
        key_clauses=[sample_key_clause()],
        gaps=[],
        confidence=0.9,
        advocate_challenges=advocate_challenges or [],
    )


def sample_research():
    return ResearcherOutput(acts=[
        ActSection(title="Act A", url="https://example.com", sections=["Section 1"])
    ])


def test_writer_returns_writer_output():
    agent = make_writer(
        short_answer="You have six years to make a claim.",
        summary="The Statute of Limitations sets a 6-year window.",
        relevant_acts=["Statute of Limitations Act 1957"],
        key_clauses=[sample_key_clause()],
        caveats=["Personal injury cases have a shorter 2-year limit"],
    )
    result = agent.run(
        query="How long do I have to sue?",
        analysis=sample_analyst_output(),
        research=sample_research(),
        evaluator_flags=[],
    )
    assert isinstance(result, WriterOutput)
    assert len(result.short_answer.split()) <= 100


def test_writer_serialises_key_clauses_with_citations():
    agent = make_writer(
        short_answer="Short answer.",
        summary="S", relevant_acts=[], key_clauses=[sample_key_clause()], caveats=[],
    )
    agent.run(
        query="Q", analysis=sample_analyst_output(),
        research=sample_research(), evaluator_flags=[],
    )
    call_args = agent._chain.invoke.call_args[0][0]
    assert "Statute of Limitations Act 1957" in call_args["key_clauses"]
    assert "s.11" in call_args["key_clauses"]


def test_writer_injects_advocate_challenges_when_present():
    agent = make_writer(
        short_answer="Short answer.", summary="S", relevant_acts=[],
        key_clauses=[sample_key_clause()], caveats=[],
    )
    analysis = sample_analyst_output(advocate_challenges=["Road Traffic Act may override this."])
    agent.run(query="Q", analysis=analysis, research=sample_research(), evaluator_flags=[])
    call_args = agent._chain.invoke.call_args[0][0]
    assert "Road Traffic Act may override this." in call_args["advocate_challenges"]


def test_writer_advocate_challenges_empty_when_none():
    agent = make_writer(
        short_answer="Short answer.", summary="S", relevant_acts=[],
        key_clauses=[sample_key_clause()], caveats=[],
    )
    agent.run(query="Q", analysis=sample_analyst_output(), research=sample_research(), evaluator_flags=[])
    call_args = agent._chain.invoke.call_args[0][0]
    assert call_args["advocate_challenges"] == "None"


def test_writer_passes_evaluator_flags_in_prompt():
    agent = make_writer(
        short_answer="Short answer.", summary="S", relevant_acts=[],
        key_clauses=[sample_key_clause()], caveats=[],
    )
    agent.run(
        query="Q", analysis=sample_analyst_output(),
        research=sample_research(), evaluator_flags=["Add citation"],
    )
    call_args = agent._chain.invoke.call_args[0][0]
    assert "Add citation" in call_args["evaluator_flags"]
```

- [ ] **Step 2: Run to confirm failures**

```bash
python -m pytest tests/test_writer.py -v 2>&1 | tail -15
```
Expected: Failures on KeyClause serialisation and advocate_challenges tests.

- [ ] **Step 3: Update writer.py**

Replace full contents:

```python
from langchain_core.prompts import ChatPromptTemplate

from irish_statute_assistant.agents.base_agent import BaseAgent
from irish_statute_assistant.config import Config
from irish_statute_assistant.llm import get_llm
from irish_statute_assistant.models.schemas import AnalystOutput, ResearcherOutput, WriterOutput

SYSTEM_PROMPT = """You are a plain English legal writer. You explain Irish law to ordinary people
who have no legal training.

Rules:
- short_answer: one or two plain sentences (maximum 100 words). No legal jargon. Write as if
  explaining to a friend.
- detailed_breakdown.summary: 2-3 sentences summarising the legal position.
- detailed_breakdown.relevant_acts: list the Acts by name (e.g. "Statute of Limitations Act 1957").
- detailed_breakdown.key_clauses: the specific rules that answer the question. For each include
  the text, the act name, and the section reference.
- detailed_breakdown.caveats: important exceptions, edge cases, or "it depends" situations.
- Always include at least one caveat reminding the user to seek professional legal advice.
- Never make up law. Only use what the analyst found.

If there are evaluator flags, address them: {evaluator_flags}

Challenges raised by the devil's advocate (address each in caveats):
{advocate_challenges}
"""

HUMAN_PROMPT = """User question: {query}

Analyst findings:
Key clauses: {key_clauses}
Gaps: {gaps}
Confidence: {confidence}

Acts researched:
{act_titles}
"""


class WriterAgent(BaseAgent):
    def __init__(self, config: Config) -> None:
        llm = get_llm(config, max_tokens=2048).with_structured_output(WriterOutput)
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", HUMAN_PROMPT),
        ])
        self._chain = prompt | llm

    def run(
        self,
        query: str,
        analysis: AnalystOutput,
        research: ResearcherOutput,
        evaluator_flags: list[str],
    ) -> WriterOutput:
        flags_text = "\n".join(evaluator_flags) if evaluator_flags else "None"
        challenges_text = (
            "\n".join(f"- {c}" for c in analysis.advocate_challenges)
            if analysis.advocate_challenges
            else "None"
        )
        act_titles = "\n".join(act.title for act in research.acts)
        key_clauses_text = "\n".join(
            f"{kc.text} ({kc.act}, {kc.section})" for kc in analysis.key_clauses
        )
        return self._invoke_chain(self._chain, {
            "query": query,
            "key_clauses": key_clauses_text,
            "gaps": "\n".join(analysis.gaps) if analysis.gaps else "None",
            "confidence": analysis.confidence,
            "act_titles": act_titles,
            "evaluator_flags": flags_text,
            "advocate_challenges": challenges_text,
        })
```

- [ ] **Step 4: Run writer tests**

```bash
python -m pytest tests/test_writer.py -v 2>&1 | tail -15
```
Expected: All PASS.

- [ ] **Step 5: Commit**

```bash
git add src/irish_statute_assistant/agents/writer.py tests/test_writer.py
git commit -m "feat: writer serialises KeyClause citations and injects advocate_challenges"
```

---

## Task 7: Update EvaluatorAgent

Adds `grounding_passed` parameter and updates `key_clauses` serialisation.

**Files:**
- Modify: `src/irish_statute_assistant/agents/evaluator.py`
- Modify: `tests/test_evaluator.py`

- [ ] **Step 1: Read current test_evaluator.py**

```bash
cat tests/test_evaluator.py
```

- [ ] **Step 2: Update existing `sample_writer_output()` in `tests/test_evaluator.py`**

The existing helper uses `key_clauses=["string"]`. After the schema change, `DetailedBreakdown.key_clauses` requires `list[KeyClause]`. Update the import and helper at the top of the file:

```python
# Add to imports
from irish_statute_assistant.models.schemas import KeyClause

# Replace sample_writer_output() helper
def sample_writer_output():
    kc = KeyClause(text="Six year limit", act="Statute of Limitations Act 1957", section="s.11")
    return WriterOutput(
        short_answer="You have six years.",
        detailed_breakdown=DetailedBreakdown(
            summary="Six year limit applies.", relevant_acts=["Statute of Limitations Act 1957"],
            key_clauses=[kc], caveats=["Seek legal advice."],
        )
    )
```

- [ ] **Step 3: Run evaluator tests to confirm existing tests still pass with updated fixture**

```bash
python -m pytest tests/test_evaluator.py -v 2>&1 | tail -15
```
Expected: All PASS (same tests, just KeyClause instead of string).

- [ ] **Step 4: Add new evaluator tests**

Append to `tests/test_evaluator.py`:

```python
def test_evaluator_key_clauses_includes_citation():
    """Evaluator must serialise KeyClause objects with act and section."""
    from irish_statute_assistant.models.schemas import KeyClause, DetailedBreakdown, WriterOutput
    from irish_statute_assistant.agents.evaluator import EvaluatorAgent

    kc = KeyClause(text="6 year limit", act="Statute of Limitations Act 1957", section="s.11")
    writer_out = WriterOutput(
        short_answer="You have six years.",
        detailed_breakdown=DetailedBreakdown(
            summary="Six year limit.", relevant_acts=["Act A"],
            key_clauses=[kc], caveats=["Seek advice"],
        )
    )
    agent = EvaluatorAgent.__new__(EvaluatorAgent)
    from unittest.mock import MagicMock
    from irish_statute_assistant.models.schemas import EvaluatorOutput
    mock_chain = MagicMock()
    mock_chain.invoke = MagicMock(return_value=EvaluatorOutput(score=0.8, flags=[], **{"pass": True}))
    agent._chain = mock_chain
    agent._threshold = 0.7
    agent.run(query="Q", output=writer_out)

    call_args = agent._chain.invoke.call_args[0][0]
    assert "Statute of Limitations Act 1957" in call_args["key_clauses"]
    assert "s.11" in call_args["key_clauses"]


def test_evaluator_grounding_failed_lowers_citation_score():
    """When grounding_passed=False, system prompt must warn about unverified claims."""
    from irish_statute_assistant.models.schemas import KeyClause, DetailedBreakdown, WriterOutput
    from irish_statute_assistant.agents.evaluator import EvaluatorAgent
    from unittest.mock import MagicMock
    from irish_statute_assistant.models.schemas import EvaluatorOutput

    kc = KeyClause(text="Clause", act="Act A", section="s.1")
    writer_out = WriterOutput(
        short_answer="Short.",
        detailed_breakdown=DetailedBreakdown(
            summary="S.", relevant_acts=[], key_clauses=[kc], caveats=[]
        )
    )
    agent = EvaluatorAgent.__new__(EvaluatorAgent)
    mock_chain = MagicMock()
    mock_chain.invoke = MagicMock(return_value=EvaluatorOutput(score=0.8, flags=[], **{"pass": True}))
    agent._chain = mock_chain
    agent._threshold = 0.7

    agent.run(query="Q", output=writer_out, grounding_passed=False)
    call_args = agent._chain.invoke.call_args[0][0]
    assert "grounding" in call_args["threshold"].lower() or "unverified" in str(call_args)
```

- [ ] **Step 5: Run new tests to confirm they fail**

```bash
python -m pytest tests/test_evaluator.py -v 2>&1 | tail -15
```
Expected: New tests fail (grounding_note not yet in evaluator).

- [ ] **Step 6: Update evaluator.py**

Replace full contents:

```python
from langchain_core.prompts import ChatPromptTemplate

from irish_statute_assistant.agents.base_agent import BaseAgent
from irish_statute_assistant.config import Config
from irish_statute_assistant.llm import get_llm
from irish_statute_assistant.models.schemas import EvaluatorOutput, WriterOutput

SYSTEM_PROMPT = """You are a quality evaluator for an Irish legal research assistant.
Score the output on these four criteria (each 0.0–1.0, overall score is the average):

1. Accuracy: Does the answer correctly reflect what the statutes say?
2. Completeness: Does it answer the user's question fully?
3. Citation quality: Are relevant Acts named and referenced? {grounding_note}
4. Plain English: Is the short_answer genuinely understandable by a non-lawyer?

Set pass=true if overall score >= {threshold}.
List specific flags for anything that should be improved (even if passing).
Be strict. A vague or uncited answer should score below 0.7.
"""

HUMAN_PROMPT = """User question: {query}

Output to evaluate:
Short answer: {short_answer}
Summary: {summary}
Relevant Acts: {relevant_acts}
Key clauses: {key_clauses}
Caveats: {caveats}
"""


class EvaluatorAgent(BaseAgent):
    def __init__(self, config: Config) -> None:
        self._threshold = config.evaluator_pass_threshold
        llm = get_llm(config, max_tokens=512).with_structured_output(EvaluatorOutput)
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", HUMAN_PROMPT),
        ])
        self._chain = prompt | llm

    def run(self, query: str, output: WriterOutput, grounding_passed: bool = True) -> EvaluatorOutput:
        bd = output.detailed_breakdown
        grounding_note = (
            "Note: the grounding checker found unverified claims in this output. "
            "Score citation quality no higher than 0.4."
            if not grounding_passed
            else ""
        )
        result = self._invoke_chain(self._chain, {
            "threshold": self._threshold,
            "grounding_note": grounding_note,
            "query": query,
            "short_answer": output.short_answer,
            "summary": bd.summary,
            "relevant_acts": ", ".join(bd.relevant_acts),
            "key_clauses": "\n".join(
                f"{kc.text} ({kc.act}, {kc.section})" for kc in bd.key_clauses
            ),
            "caveats": "\n".join(bd.caveats),
        })
        return result.model_copy(update={"pass_": result.score >= self._threshold})
```

- [ ] **Step 7: Fix the grounding test assertion**

The test checks `call_args["threshold"]` but that key holds the float threshold, not the grounding note. Update the assertion in `test_evaluator_grounding_failed_lowers_citation_score`:

```python
    call_args = agent._chain.invoke.call_args[0][0]
    assert "unverified" in call_args["grounding_note"]
```

- [ ] **Step 8: Run all evaluator tests**

```bash
python -m pytest tests/test_evaluator.py -v 2>&1 | tail -15
```
Expected: All PASS.

- [ ] **Step 9: Commit**

```bash
git add src/irish_statute_assistant/agents/evaluator.py tests/test_evaluator.py
git commit -m "feat: evaluator serialises KeyClause citations and accepts grounding_passed flag"
```

---

## Task 8: DevilsAdvocateAgent

**Files:**
- Create: `src/irish_statute_assistant/agents/devils_advocate.py`
- Create: `tests/test_devils_advocate.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_devils_advocate.py`:

```python
from unittest.mock import MagicMock
import pytest
from irish_statute_assistant.agents.devils_advocate import DevilsAdvocateAgent
from irish_statute_assistant.models.schemas import (
    AdvocateOutput, AnalystOutput, KeyClause, ResearcherOutput, ActSection
)


def sample_analyst_output(confidence=0.9):
    kc = KeyClause(text="6 year limit", act="Statute of Limitations Act 1957", section="s.11")
    return AnalystOutput(key_clauses=[kc], gaps=[], confidence=confidence)


def sample_research():
    return ResearcherOutput(acts=[
        ActSection(title="Act A", url="https://example.com", sections=["Section text"])
    ])


def make_advocate(challenges, severity):
    agent = DevilsAdvocateAgent.__new__(DevilsAdvocateAgent)
    mock_chain = MagicMock()
    mock_chain.invoke = MagicMock(return_value=AdvocateOutput(
        challenges=challenges, severity=severity
    ))
    agent._chain_standard = mock_chain
    agent._chain_strict = mock_chain
    return agent


def test_advocate_returns_advocate_output_on_weak_analysis():
    agent = make_advocate(
        challenges=["Missing exception for minors."], severity="minor"
    )
    result = agent.run(
        analyst_output=sample_analyst_output(confidence=0.6),
        query="How long do I have to sue?",
        research=sample_research(),
        mode="standard",
    )
    assert isinstance(result, AdvocateOutput)
    assert len(result.challenges) >= 1


def test_advocate_returns_empty_challenges_on_strong_analysis():
    agent = make_advocate(challenges=[], severity="minor")
    result = agent.run(
        analyst_output=sample_analyst_output(confidence=0.95),
        query="Q",
        research=sample_research(),
        mode="standard",
    )
    assert result.challenges == []
    assert result.severity == "minor"


def test_advocate_severity_major_on_serious_gap():
    agent = make_advocate(
        challenges=["Road Traffic Acts override this entirely."], severity="major"
    )
    result = agent.run(
        analyst_output=sample_analyst_output(),
        query="Q",
        research=sample_research(),
    )
    assert result.severity == "major"


def test_advocate_uses_strict_chain_in_strict_mode():
    agent = DevilsAdvocateAgent.__new__(DevilsAdvocateAgent)
    standard_chain = MagicMock()
    strict_chain = MagicMock()
    strict_chain.invoke = MagicMock(return_value=AdvocateOutput(challenges=["c1", "c2", "c3"], severity="major"))
    standard_chain.invoke = MagicMock(return_value=AdvocateOutput(challenges=["c1"], severity="minor"))
    agent._chain_standard = standard_chain
    agent._chain_strict = strict_chain

    result = agent.run(
        analyst_output=sample_analyst_output(),
        query="Q",
        research=sample_research(),
        mode="strict",
    )
    strict_chain.invoke.assert_called_once()
    standard_chain.invoke.assert_not_called()
    assert len(result.challenges) == 3
```

- [ ] **Step 2: Run to confirm failures**

```bash
python -m pytest tests/test_devils_advocate.py -v 2>&1 | tail -15
```
Expected: ImportError.

- [ ] **Step 3: Implement DevilsAdvocateAgent**

Create `src/irish_statute_assistant/agents/devils_advocate.py`:

```python
from __future__ import annotations

from typing import Literal

from langchain_core.prompts import ChatPromptTemplate

from irish_statute_assistant.agents.base_agent import BaseAgent
from irish_statute_assistant.config import Config
from irish_statute_assistant.llm import get_llm
from irish_statute_assistant.models.schemas import (
    AdvocateOutput, AnalystOutput, ResearcherOutput
)

_SYSTEM_STANDARD = """You are a critical legal reviewer. You have been given an analyst's
findings about an Irish law question. Your job is to find 1-3 weaknesses in the analysis:
- Missing exceptions or special cases
- Statutes that may override or qualify the analyst's conclusion
- Claims that go beyond what the retrieved text actually says

If the analysis is solid and well-grounded, return an empty challenges list with severity=minor.
Set severity=major only if the analyst's conclusion could be substantially wrong.
"""

_SYSTEM_STRICT = """You are an adversarial legal reviewer. Find EVERY possible weakness
in this analysis — missing exceptions, conflicting statutes, unsupported inferences,
edge cases, anything. Aim for up to 5 challenges. Be aggressive.
Set severity=major if any challenge could substantially change the answer.
"""

_HUMAN_PROMPT = """User question: {query}

Analyst's key clauses:
{key_clauses}

Analyst's confidence: {confidence}

Retrieved statute text:
{statute_text}
"""


class DevilsAdvocateAgent(BaseAgent):
    def __init__(self, config: Config) -> None:
        llm = get_llm(config, max_tokens=512).with_structured_output(AdvocateOutput)
        self._chain_standard = (
            ChatPromptTemplate.from_messages([("system", _SYSTEM_STANDARD), ("human", _HUMAN_PROMPT)])
            | llm
        )
        self._chain_strict = (
            ChatPromptTemplate.from_messages([("system", _SYSTEM_STRICT), ("human", _HUMAN_PROMPT)])
            | llm
        )

    def run(
        self,
        analyst_output: AnalystOutput,
        query: str,
        research: ResearcherOutput,
        mode: Literal["standard", "strict"] = "standard",
    ) -> AdvocateOutput:
        chain = self._chain_strict if mode == "strict" else self._chain_standard
        key_clauses_text = "\n".join(
            f"{kc.text} ({kc.act}, {kc.section})" for kc in analyst_output.key_clauses
        )
        statute_text = "\n\n".join(
            f"## {act.title}\n" + "\n".join(act.sections)
            for act in research.acts
        )
        return self._invoke_chain(chain, {
            "query": query,
            "key_clauses": key_clauses_text,
            "confidence": analyst_output.confidence,
            "statute_text": statute_text,
        })
```

- [ ] **Step 4: Run advocate tests**

```bash
python -m pytest tests/test_devils_advocate.py -v 2>&1 | tail -15
```
Expected: All PASS.

- [ ] **Step 5: Commit**

```bash
git add src/irish_statute_assistant/agents/devils_advocate.py tests/test_devils_advocate.py
git commit -m "feat: add DevilsAdvocateAgent for debate-style reasoning"
```

---

## Task 9: GroundingCheckerAgent

**Files:**
- Create: `src/irish_statute_assistant/agents/grounding_checker.py`
- Create: `tests/test_grounding_checker.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_grounding_checker.py`:

```python
from unittest.mock import MagicMock
from irish_statute_assistant.agents.grounding_checker import GroundingCheckerAgent
from irish_statute_assistant.models.schemas import (
    GroundingOutput, WriterOutput, DetailedBreakdown, KeyClause, ResearcherOutput, ActSection
)


def sample_writer_output(key_clauses=None):
    kc = key_clauses or [KeyClause(text="6 year limit", act="Statute of Limitations Act 1957", section="s.11")]
    return WriterOutput(
        short_answer="You have six years.",
        detailed_breakdown=DetailedBreakdown(
            summary="Six year limit.", relevant_acts=["Act A"],
            key_clauses=kc, caveats=["Seek advice"],
        )
    )


def sample_research():
    return ResearcherOutput(acts=[
        ActSection(title="Act A", url="https://example.com", sections=["Actions must be within 6 years."])
    ])


def make_checker(ungrounded_claims, grounding_passed):
    agent = GroundingCheckerAgent.__new__(GroundingCheckerAgent)
    mock_chain = MagicMock()
    mock_chain.invoke = MagicMock(return_value=GroundingOutput(
        ungrounded_claims=ungrounded_claims,
        grounding_passed=grounding_passed,
    ))
    agent._chain = mock_chain
    return agent


def test_grounding_checker_passes_grounded_claims():
    agent = make_checker(ungrounded_claims=[], grounding_passed=True)
    result = agent.run(writer_output=sample_writer_output(), research=sample_research())
    assert isinstance(result, GroundingOutput)
    assert result.grounding_passed is True
    assert result.ungrounded_claims == []


def test_grounding_checker_flags_ungrounded_claims():
    agent = make_checker(
        ungrounded_claims=["Claim about road tax has no source in retrieved text."],
        grounding_passed=False,
    )
    result = agent.run(writer_output=sample_writer_output(), research=sample_research())
    assert result.grounding_passed is False
    assert len(result.ungrounded_claims) == 1


def test_grounding_checker_warnings_attached_to_writer_output():
    agent = make_checker(
        ungrounded_claims=["Unsupported claim."], grounding_passed=False
    )
    writer_out = sample_writer_output()
    grounding = agent.run(writer_output=writer_out, research=sample_research())
    # Supervisor attaches warnings — simulate that here
    writer_out.warnings = grounding.ungrounded_claims
    assert writer_out.warnings == ["Unsupported claim."]


def test_grounding_checker_grounding_passed_false_when_ungrounded():
    agent = make_checker(ungrounded_claims=["X"], grounding_passed=False)
    result = agent.run(writer_output=sample_writer_output(), research=sample_research())
    assert result.grounding_passed is False
```

- [ ] **Step 2: Run to confirm failures**

```bash
python -m pytest tests/test_grounding_checker.py -v 2>&1 | tail -15
```
Expected: ImportError.

- [ ] **Step 3: Implement GroundingCheckerAgent**

Create `src/irish_statute_assistant/agents/grounding_checker.py`:

```python
from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate

from irish_statute_assistant.agents.base_agent import BaseAgent
from irish_statute_assistant.config import Config
from irish_statute_assistant.llm import get_llm
from irish_statute_assistant.models.schemas import (
    GroundingOutput, ResearcherOutput, WriterOutput
)

SYSTEM_PROMPT = """You are a hallucination detector for an Irish legal research assistant.

You are given a list of key legal clauses from a writer's output, plus the raw statute text
that was retrieved. For each clause, check whether the claim is directly supported by the
retrieved text.

Return:
- ungrounded_claims: list of clauses (as strings) that are NOT supported by the retrieved text
- grounding_passed: true if all claims are grounded, false if any are not

Be strict: if a claim goes beyond what the text says, flag it.
"""

HUMAN_PROMPT = """Key clauses to verify:
{key_clauses}

Retrieved statute text:
{statute_text}
"""


class GroundingCheckerAgent(BaseAgent):
    def __init__(self, config: Config) -> None:
        llm = get_llm(config, max_tokens=512).with_structured_output(GroundingOutput)
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", HUMAN_PROMPT),
        ])
        self._chain = prompt | llm

    def run(self, writer_output: WriterOutput, research: ResearcherOutput) -> GroundingOutput:
        bd = writer_output.detailed_breakdown
        key_clauses_text = "\n".join(
            f"- {kc.text} ({kc.act}, {kc.section})" for kc in bd.key_clauses
        )
        statute_text = "\n\n".join(
            f"## {act.title}\n" + "\n".join(act.sections)
            for act in research.acts
        )
        return self._invoke_chain(self._chain, {
            "key_clauses": key_clauses_text,
            "statute_text": statute_text,
        })
```

- [ ] **Step 4: Run grounding tests**

```bash
python -m pytest tests/test_grounding_checker.py -v 2>&1 | tail -15
```
Expected: All PASS.

- [ ] **Step 5: Commit**

```bash
git add src/irish_statute_assistant/agents/grounding_checker.py tests/test_grounding_checker.py
git commit -m "feat: add GroundingCheckerAgent for hallucination detection"
```

---

## Task 10: Supervisor Rewrite

**Files:**
- Modify: `src/irish_statute_assistant/agents/supervisor.py`
- Modify: `tests/test_supervisor.py`

- [ ] **Step 1: Write the updated test_supervisor.py**

Replace the full file:

```python
import pytest
from unittest.mock import MagicMock
from irish_statute_assistant.agents.supervisor import Supervisor
from irish_statute_assistant.models.schemas import (
    ClarifierOutput, ResearcherOutput, ActSection,
    AnalystLLMOutput, AnalystOutput, KeyClause,
    WriterOutput, DetailedBreakdown,
    EvaluatorOutput, AdvocateOutput, GroundingOutput,
)


def make_key_clause():
    return KeyClause(text="6 year limit", act="Act A", section="s.1")


def make_defaults():
    research = ResearcherOutput(acts=[
        ActSection(title="Act A", url="https://example.com", sections=["Section 1"])
    ])
    kc = make_key_clause()
    analyst_llm = AnalystLLMOutput(key_clauses=[kc], gaps=[], confidence=0.9)
    writer_out = WriterOutput(
        short_answer="You have six years.",
        detailed_breakdown=DetailedBreakdown(
            summary="Six year limit.", relevant_acts=["Act A"],
            key_clauses=[kc], caveats=["Seek advice"],
        )
    )
    evaluator_pass = EvaluatorOutput(score=0.85, flags=[], **{"pass": True})
    advocate_minor = AdvocateOutput(challenges=[], severity="minor")
    grounding_pass = GroundingOutput(ungrounded_claims=[], grounding_passed=True)
    return research, analyst_llm, writer_out, evaluator_pass, advocate_minor, grounding_pass


def make_supervisor(
    clarifier_output, researcher_output, analyst_llm_output,
    writer_output, evaluator_output, advocate_output, grounding_output,
):
    sup = Supervisor.__new__(Supervisor)
    sup._max_refinements = 2
    sup._max_retries = 3
    sup._evaluator_flag_counts = {}

    sup._memory = MagicMock()
    sup._memory.format_for_prompt = MagicMock(return_value="")
    sup._memory.add_exchange = MagicMock()

    sup._preferences = MagicMock()
    sup._preferences.set = MagicMock()
    sup._preferences.get = MagicMock(return_value="")
    sup._preferences.all = MagicMock(return_value={})

    sup._clarifier = MagicMock()
    sup._clarifier.run = MagicMock(return_value=clarifier_output)
    sup._clarifier.last_token_count = 0

    sup._researcher = MagicMock()
    sup._researcher.run = MagicMock(return_value=researcher_output)
    sup._researcher.last_token_count = 0

    sup._analyst = MagicMock()
    sup._analyst.run = MagicMock(return_value=analyst_llm_output)
    sup._analyst.last_token_count = 0

    sup._advocate = MagicMock()
    sup._advocate.run = MagicMock(return_value=advocate_output)
    sup._advocate.last_token_count = 0

    sup._writer = MagicMock()
    sup._writer.run = MagicMock(return_value=writer_output)
    sup._writer.last_token_count = 0

    sup._grounding_checker = MagicMock()
    sup._grounding_checker.run = MagicMock(return_value=grounding_output)
    sup._grounding_checker.last_token_count = 0

    sup._evaluator = MagicMock()
    sup._evaluator.run = MagicMock(return_value=evaluator_output)
    sup._evaluator.last_token_count = 0

    return sup


def test_supervisor_returns_writer_output_when_clear_and_passes():
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
    result = sup.run(query="How long do I have to sue?", context=None)
    assert isinstance(result, WriterOutput)


def test_supervisor_returns_clarification_question_when_needed():
    research, analyst_llm, writer_out, eval_pass, advocate, grounding = make_defaults()
    sup = make_supervisor(
        clarifier_output=ClarifierOutput(needs_clarification=True, question="What type of case?"),
        researcher_output=research,
        analyst_llm_output=analyst_llm,
        writer_output=writer_out,
        evaluator_output=eval_pass,
        advocate_output=advocate,
        grounding_output=grounding,
    )
    result = sup.run(query="What are my rights?", context=None)
    assert result == "What type of case?"
    sup._memory.add_exchange.assert_called_once_with(
        user="What are my rights?", assistant="What type of case?"
    )


def test_supervisor_refinement_loop_retries_on_fail():
    research, analyst_llm, writer_out, _, advocate, grounding = make_defaults()
    eval_fail = EvaluatorOutput(score=0.5, flags=["Vague answer"], **{"pass": False})
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
    result = sup.run(query="How long?", context=None)
    assert isinstance(result, WriterOutput)
    assert sup._evaluator.run.call_count == 2


def test_supervisor_stops_after_max_refinements():
    research, analyst_llm, writer_out, _, advocate, grounding = make_defaults()
    eval_fail = EvaluatorOutput(score=0.4, flags=["Still bad"], **{"pass": False})

    sup = make_supervisor(
        clarifier_output=ClarifierOutput(needs_clarification=False),
        researcher_output=research,
        analyst_llm_output=analyst_llm,
        writer_output=writer_out,
        evaluator_output=eval_fail,
        advocate_output=advocate,
        grounding_output=grounding,
    )
    sup._evaluator.run = MagicMock(return_value=eval_fail)
    result = sup.run(query="How long?", context=None)
    assert isinstance(result, WriterOutput)
    assert sup._evaluator.run.call_count == sup._max_refinements + 1


def test_supervisor_confidence_gate_doubles_refinements_on_low_confidence():
    research, _, writer_out, eval_pass, _, grounding = make_defaults()
    kc = make_key_clause()
    low_conf_analyst = AnalystLLMOutput(key_clauses=[kc], gaps=[], confidence=0.3)
    advocate_minor = AdvocateOutput(challenges=[], severity="minor")

    sup = make_supervisor(
        clarifier_output=ClarifierOutput(needs_clarification=False),
        researcher_output=research,
        analyst_llm_output=low_conf_analyst,
        writer_output=writer_out,
        evaluator_output=eval_pass,
        advocate_output=advocate_minor,
        grounding_output=grounding,
    )
    # Capture effective_refinements by making evaluator always fail until exhausted
    eval_fail = EvaluatorOutput(score=0.4, flags=["bad"], **{"pass": False})
    sup._evaluator.run = MagicMock(return_value=eval_fail)
    sup.run(query="Q", context=None)
    # With low confidence, should run max_refinements*2 + 1 times
    assert sup._evaluator.run.call_count == sup._max_refinements * 2 + 1


def test_supervisor_confidence_gate_doubles_refinements_on_major_severity():
    research, analyst_llm, writer_out, eval_pass, _, grounding = make_defaults()
    advocate_major = AdvocateOutput(challenges=["Serious problem"], severity="major")

    sup = make_supervisor(
        clarifier_output=ClarifierOutput(needs_clarification=False),
        researcher_output=research,
        analyst_llm_output=analyst_llm,
        writer_output=writer_out,
        evaluator_output=eval_pass,
        advocate_output=advocate_major,
        grounding_output=grounding,
    )
    eval_fail = EvaluatorOutput(score=0.4, flags=["bad"], **{"pass": False})
    sup._evaluator.run = MagicMock(return_value=eval_fail)
    sup.run(query="Q", context=None)
    assert sup._evaluator.run.call_count == sup._max_refinements * 2 + 1


def test_supervisor_memory_add_exchange_called_on_success():
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
    sup.run(query="Q", context=None)
    sup._memory.add_exchange.assert_called_once()


def test_supervisor_explicit_preference_saved_on_solicitor_query():
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
    sup.run(query="I'm a solicitor — what are the rules on X?", context=None)
    sup._preferences.set.assert_any_call("user_type", "solicitor")


def test_supervisor_inferred_preference_saved_after_second_plain_english_flag():
    research, analyst_llm, writer_out, _, advocate, grounding = make_defaults()
    eval_with_flag = EvaluatorOutput(score=0.75, flags=["plain english"], **{"pass": True})

    sup = make_supervisor(
        clarifier_output=ClarifierOutput(needs_clarification=False),
        researcher_output=research,
        analyst_llm_output=analyst_llm,
        writer_output=writer_out,
        evaluator_output=eval_with_flag,
        advocate_output=advocate,
        grounding_output=grounding,
    )

    # First query — should NOT save preference
    sup.run(query="Q1", context=None)
    calls_after_first = [c for c in sup._preferences.set.call_args_list
                         if c[0] == ("language_level", "technical")]
    assert len(calls_after_first) == 0

    # Reset evaluator mock for second query
    sup._evaluator.run = MagicMock(return_value=eval_with_flag)
    sup._analyst.run = MagicMock(return_value=analyst_llm)
    sup._advocate.run = MagicMock(return_value=advocate)
    sup._grounding_checker.run = MagicMock(return_value=grounding)
    sup._writer.run = MagicMock(return_value=writer_out)
    sup._researcher.run = MagicMock(return_value=research)
    sup._clarifier.run = MagicMock(return_value=ClarifierOutput(needs_clarification=False))

    # Second query — should save preference
    sup.run(query="Q2", context=None)
    calls_after_second = [c for c in sup._preferences.set.call_args_list
                          if c[0] == ("language_level", "technical")]
    assert len(calls_after_second) == 1
```

- [ ] **Step 2: Run to confirm failures**

```bash
python -m pytest tests/test_supervisor.py -v 2>&1 | tail -20
```
Expected: Multiple failures — wrong signatures, missing agents.

- [ ] **Step 3: Rewrite supervisor.py**

Replace full contents:

```python
from __future__ import annotations

import logging
import re

from irish_statute_assistant.agents.analyst import AnalystAgent
from irish_statute_assistant.agents.clarifier import ClarifierAgent
from irish_statute_assistant.agents.devils_advocate import DevilsAdvocateAgent
from irish_statute_assistant.agents.evaluator import EvaluatorAgent
from irish_statute_assistant.agents.grounding_checker import GroundingCheckerAgent
from irish_statute_assistant.agents.researcher import ResearcherAgent
from irish_statute_assistant.agents.writer import WriterAgent
from irish_statute_assistant.config import Config
from irish_statute_assistant.context import QueryContext
from irish_statute_assistant.memory.conversation_store import ConversationStore
from irish_statute_assistant.memory.user_preference_store import UserPreferenceStore
from irish_statute_assistant.models.schemas import AnalystOutput, WriterOutput
from irish_statute_assistant.retry import run_with_retry
from irish_statute_assistant.tools.session_cache import SessionCache
from irish_statute_assistant.tools.statute_fetcher import StatuteFetcher

logger = logging.getLogger(__name__)

# Keyword patterns → (preference_key, preference_value)
_PREFERENCE_PATTERNS: list[tuple[str, str, str]] = [
    (r"i(?:'m| am) a (?:solicitor|lawyer)", "user_type", "solicitor"),
    (r"explain simply|plain english|non.?lawyer", "language_level", "plain"),
    (r"use legal terms|technical", "language_level", "technical"),
    (r"\bbrief\b|short answer", "verbosity", "brief"),
    (r"\bdetailed\b|full explanation", "verbosity", "detailed"),
]


class Supervisor:
    def __init__(
        self,
        config: Config,
        memory: ConversationStore,
        preferences: UserPreferenceStore,
    ) -> None:
        self._max_refinements = config.max_refinement_rounds
        self._max_retries = config.max_retries
        self._memory = memory
        self._preferences = preferences
        self._evaluator_flag_counts: dict[str, int] = {}

        cache = SessionCache()
        fetcher = StatuteFetcher(
            rate_limit_delay=config.rate_limit_delay,
            max_retries=config.max_retries,
        )
        self._clarifier = ClarifierAgent(config)
        self._researcher = ResearcherAgent(config, cache, fetcher)
        self._analyst = AnalystAgent(config)
        self._advocate = DevilsAdvocateAgent(config)
        self._writer = WriterAgent(config)
        self._grounding_checker = GroundingCheckerAgent(config)
        self._evaluator = EvaluatorAgent(config)

    def run(self, query: str, context: QueryContext | None = None) -> WriterOutput | str:
        history = self._memory.format_for_prompt()

        # 1. Clarify
        clarifier_result = run_with_retry(
            lambda: self._clarifier.run(query=query, history=history),
            self._max_retries,
        )
        if context:
            context.consume(self._clarifier.last_token_count)
        if clarifier_result.needs_clarification:
            self._memory.add_exchange(user=query, assistant=clarifier_result.question)
            return clarifier_result.question

        # 2. Research
        research = run_with_retry(
            lambda: self._researcher.run(query=query),
            self._max_retries,
        )
        if context:
            context.consume(self._researcher.last_token_count)

        # 3. Analyse (once, outside the loop)
        llm_analyst_result = run_with_retry(
            lambda: self._analyst.run(query=query, research=research),
            self._max_retries,
        )
        if context:
            context.consume(self._analyst.last_token_count)
        analyst_output = AnalystOutput(
            **llm_analyst_result.model_dump(), advocate_challenges=[]
        )

        # 4. Devil's advocate (initial run)
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

        # Confidence gate
        low_confidence = analyst_output.confidence < 0.5 or advocate_result.severity == "major"
        effective_refinements = self._max_refinements * 2 if low_confidence else self._max_refinements
        advocate_mode_on_retry = "strict" if low_confidence else "standard"

        analyst_output.advocate_challenges = advocate_result.challenges

        # 5–7. Refinement loop
        evaluator_flags: list[str] = []
        best_output: WriterOutput | None = None

        for _ in range(effective_refinements + 1):
            writer_result = run_with_retry(
                lambda: self._writer.run(
                    query=query,
                    analysis=analyst_output,
                    research=research,
                    evaluator_flags=evaluator_flags,
                ),
                self._max_retries,
            )
            if context:
                context.consume(self._writer.last_token_count)
            writer_result.analyst_confidence = analyst_output.confidence

            grounding = run_with_retry(
                lambda: self._grounding_checker.run(
                    writer_output=writer_result, research=research
                ),
                self._max_retries,
            )
            if context:
                context.consume(self._grounding_checker.last_token_count)
            writer_result.warnings = grounding.ungrounded_claims

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

            best_output = writer_result

            self._update_flag_counts(evaluation.flags)

            if evaluation.pass_:
                self._detect_and_save_preferences(query, evaluation.flags)
                self._memory.add_exchange(user=query, assistant=writer_result.short_answer)
                return writer_result

            evaluator_flags = evaluation.flags

            # Re-run advocate with potentially stricter mode
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
            analyst_output.advocate_challenges = advocate_result.challenges

        self._detect_and_save_preferences(query, evaluator_flags)
        self._memory.add_exchange(user=query, assistant=best_output.short_answer)
        return best_output

    def _detect_and_save_preferences(self, query: str, evaluator_flags: list[str]) -> None:
        query_lower = query.lower()
        for pattern, key, value in _PREFERENCE_PATTERNS:
            if re.search(pattern, query_lower):
                self._preferences.set(key, value)

        # Inferred: repeated "plain English" evaluator flag → user prefers technical
        for flag in evaluator_flags:
            if "plain english" in flag.lower():
                count = self._evaluator_flag_counts.get("plain english", 0)
                if count >= 1:
                    self._preferences.set("language_level", "technical")

    def _update_flag_counts(self, flags: list[str]) -> None:
        for flag in flags:
            key = flag.lower()
            self._evaluator_flag_counts[key] = self._evaluator_flag_counts.get(key, 0) + 1
```

- [ ] **Step 4: Run supervisor tests**

```bash
python -m pytest tests/test_supervisor.py -v 2>&1 | tail -25
```
Expected: All PASS.

- [ ] **Step 5: Commit**

```bash
git add src/irish_statute_assistant/agents/supervisor.py tests/test_supervisor.py
git commit -m "feat: rewrite supervisor with advocate, grounding checker, confidence gate, and preference detection"
```

---

## Task 11: Pipeline and main.py

**Files:**
- Modify: `src/irish_statute_assistant/pipeline.py`
- Modify: `src/irish_statute_assistant/main.py`
- Modify: `tests/test_pipeline.py` (check existing tests still pass)

- [ ] **Step 1: Check existing pipeline tests**

```bash
python -m pytest tests/test_pipeline.py -v 2>&1 | tail -15
```
Note which tests currently pass.

- [ ] **Step 2: Update pipeline.py**

Replace full contents:

```python
from __future__ import annotations

import logging

from irish_statute_assistant.agents.supervisor import Supervisor
from irish_statute_assistant.config import Config
from irish_statute_assistant.context import QueryContext
from irish_statute_assistant.memory.conversation_store import ConversationStore
from irish_statute_assistant.memory.user_preference_store import UserPreferenceStore
from irish_statute_assistant.models.schemas import WriterOutput

logger = logging.getLogger(__name__)


class Pipeline:
    def __init__(self, config: Config) -> None:
        self._config = config
        self._memory = ConversationStore(
            db_path=config.conversations_db_path,
            history_limit=config.conversation_history_limit,
        )
        self._preferences = UserPreferenceStore(db_path=config.preferences_db_path)
        self._supervisor = Supervisor(config, memory=self._memory, preferences=self._preferences)

    def query(self, user_query: str) -> WriterOutput | str:
        """
        Submit a query. Returns:
          - str: a clarifying question (supervisor writes to memory)
          - WriterOutput: the final answer (supervisor writes to memory)
        """
        context = QueryContext(budget=self._config.token_budget_per_query)
        result = self._supervisor.run(query=user_query, context=context)
        logger.info(
            "Query used %d/%d tokens",
            context.tokens_used,
            context.budget,
        )
        return result
```

- [ ] **Step 3: Update main.py format_output**

Replace the `format_output` function:

```python
def format_output(result: WriterOutput | str) -> str:
    if isinstance(result, str):
        return f"\nI need a bit more information:\n  {result}\n"

    bd = result.detailed_breakdown
    lines = [
        "",
        f"Answer: {result.short_answer}",
        "",
        "--- Detail ---",
        f"Summary: {bd.summary}",
        "",
        "Relevant Acts:",
        *[f"  - {act}" for act in bd.relevant_acts],
        "",
        "Key points:",
        *[f"  - {kc.text} ({kc.act}, {kc.section})" for kc in bd.key_clauses],
        "",
        "Things to be aware of:",
        *[f"  - {caveat}" for caveat in bd.caveats],
        "",
    ]

    if result.analyst_confidence < 0.5:
        lines.append("Note: confidence in statute coverage was low for this query.")
        lines.append("")

    if result.warnings:
        lines.append("--- Grounding warnings ---")
        lines.extend(f"  - {w}" for w in result.warnings)
        lines.append("")

    return "\n".join(lines)
```

- [ ] **Step 4: Update tests/test_pipeline.py**

The existing tests `test_pipeline_memory_updated_after_answer` and `test_pipeline_memory_not_updated_after_clarification` assert against `p._memory` directly. After the rewrite, memory writes are owned by the supervisor, so these assertions are stale. Replace them with tests that verify the supervisor is called correctly:

Read the current `tests/test_pipeline.py`, then replace any test that asserts `p._memory.get_history()` with a test that asserts `p._supervisor.run.call_count == 1` and that the return value is passed through correctly.

The `make_pipeline` helper uses `Supervisor.__new__` and mocks `_supervisor.run` — keep that pattern. Remove the two stale memory-assertion tests entirely; `test_memory_stores.py` covers persistence.

- [ ] **Step 5: Run pipeline tests**

```bash
python -m pytest tests/test_pipeline.py -v 2>&1 | tail -15
```
Expected: All PASS.

- [ ] **Step 6: Run the full test suite**

```bash
python -m pytest --tb=short 2>&1 | tail -30
```
Expected: All PASS (or only pre-existing failures unrelated to this feature).

- [ ] **Step 7: Commit**

```bash
git add src/irish_statute_assistant/pipeline.py src/irish_statute_assistant/main.py tests/test_pipeline.py
git commit -m "feat: wire ConversationStore and UserPreferenceStore into Pipeline; update main.py output"
```

---

## Task 12: Final Verification

- [ ] **Step 1: Run the full test suite one more time**

```bash
python -m pytest -v 2>&1 | tail -40
```
Expected: All PASS.

- [ ] **Step 2: Smoke test — start the assistant and ask a question**

```bash
python -m irish_statute_assistant.main
```
Type: `I'm a solicitor. How long do I have to bring a personal injury claim?`

Expected:
- Preferences DB written with `user_type=solicitor`
- Answer returned with key clauses showing `(Act Name, s.X)` format
- No `BudgetExceededError`

- [ ] **Step 3: Final commit**

```bash
git add -A
git commit -m "chore: platinum upgrade complete — all tests passing"
```
