# Documentation Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a complete Sphinx documentation site (hosted on ReadTheDocs) covering the user guide, architecture deep-dive, developer guide, and auto-generated API reference, plus update the README to be a clean quick-start.

**Architecture:** Hybrid model — hand-written Markdown narrative pages for guides and architecture, `autodoc` for the API reference generated from Google-style docstrings. MyST Parser throughout (Markdown, not reStructuredText). Furo theme. ReadTheDocs hosting.

**Tech Stack:** Python 3.11+, Sphinx 8, MyST Parser, sphinxcontrib-mermaid, Furo theme, ReadTheDocs

---

## File Map

### Create
- `docs/conf.py` — Sphinx configuration
- `docs/requirements.txt` — pinned doc dependencies
- `docs/index.md` — landing page with navigation
- `.readthedocs.yaml` — ReadTheDocs build config
- `docs/user-guide/installation.md`
- `docs/user-guide/running.md`
- `docs/user-guide/configuration.md`
- `docs/architecture/overview.md`
- `docs/architecture/agents.md`
- `docs/architecture/schemas.md`
- `docs/architecture/memory.md`
- `docs/architecture/flows.md`
- `docs/developer-guide/adding-an-agent.md`
- `docs/developer-guide/adding-a-provider.md`
- `docs/developer-guide/testing.md`
- `docs/api-reference/agents.md`
- `docs/api-reference/schemas.md`
- `docs/api-reference/config.md`
- `docs/api-reference/memory.md`
- `docs/api-reference/pipeline.md`

### Modify
- `src/irish_statute_assistant/agents/clarifier.py` — add docstrings
- `src/irish_statute_assistant/agents/researcher.py` — add docstrings
- `src/irish_statute_assistant/agents/analyst.py` — add docstrings
- `src/irish_statute_assistant/agents/devils_advocate.py` — add docstrings
- `src/irish_statute_assistant/agents/writer.py` — add docstrings
- `src/irish_statute_assistant/agents/grounding_checker.py` — add docstrings
- `src/irish_statute_assistant/agents/evaluator.py` — add docstrings
- `src/irish_statute_assistant/agents/supervisor.py` — add docstrings
- `src/irish_statute_assistant/models/schemas.py` — add docstrings
- `src/irish_statute_assistant/config.py` — add docstrings
- `src/irish_statute_assistant/memory/conversation_store.py` — add docstrings
- `src/irish_statute_assistant/memory/user_preference_store.py` — add docstrings
- `src/irish_statute_assistant/pipeline.py` — add docstrings
- `src/irish_statute_assistant/llm.py` — already has docstrings; verify
- `README.md` — update to quick-start only

---

## Task 1: Sphinx Infrastructure

**Files:**
- Create: `docs/conf.py`
- Create: `docs/requirements.txt`
- Create: `docs/index.md`
- Create: `.readthedocs.yaml`

- [ ] **Step 1: Create `docs/requirements.txt`**

```
sphinx==8.1.3
furo==2024.8.6
myst-parser==4.0.0
sphinxcontrib-mermaid==0.9.2
sphinx-autodoc-typehints==2.5.0
```

- [ ] **Step 2: Create `docs/conf.py`**

```python
"""Sphinx configuration for Irish Statute Assistant documentation."""
import os
import sys

# Make the src package importable for autodoc
sys.path.insert(0, os.path.abspath("../src"))

project = "Irish Statute Assistant"
copyright = "2026"
author = ""
release = "1.0"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinxcontrib.mermaid",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
]

html_theme = "furo"
html_title = "Irish Statute Assistant"

myst_enable_extensions = ["colon_fence"]

autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "private-members": False,
    "show-inheritance": True,
}

napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True

# Suppress noisy MyST warnings about heading levels
suppress_warnings = ["myst.header"]
```

- [ ] **Step 3: Create `docs/index.md`**

````markdown
# Irish Statute Assistant

An AI assistant that answers plain-English questions about Irish law, powered by a multi-agent pipeline that retrieves, analyses, and verifies statute text from [irishstatutebook.ie](https://www.irishstatutebook.ie).

```{toctree}
:maxdepth: 2
:caption: User Guide

user-guide/installation
user-guide/running
user-guide/configuration
```

```{toctree}
:maxdepth: 2
:caption: Architecture

architecture/overview
architecture/agents
architecture/schemas
architecture/memory
architecture/flows
```

```{toctree}
:maxdepth: 2
:caption: Developer Guide

developer-guide/adding-an-agent
developer-guide/adding-a-provider
developer-guide/testing
```

```{toctree}
:maxdepth: 2
:caption: API Reference

api-reference/agents
api-reference/schemas
api-reference/config
api-reference/memory
api-reference/pipeline
```
````

- [ ] **Step 4: Create `.readthedocs.yaml`**

```yaml
version: 2

build:
  os: ubuntu-22.04
  tools:
    python: "3.11"

sphinx:
  configuration: docs/conf.py
  fail_on_warning: true

python:
  install:
    - requirements: docs/requirements.txt
    - method: pip
      path: .
```

- [ ] **Step 5: Install doc dependencies and verify the build works**

```bash
cd /home/power/projects/ai/langflow-learning-club
uv pip install -r docs/requirements.txt
sphinx-build docs docs/_build/html -W --keep-going 2>&1 | tail -20
```

Expected: Build completes (may warn about missing content files — that is fine at this stage, as long as conf.py loads without errors).

- [ ] **Step 6: Commit**

```bash
git add docs/conf.py docs/requirements.txt docs/index.md .readthedocs.yaml
git commit -m "docs: add Sphinx infrastructure (conf.py, requirements, index, ReadTheDocs config)"
```

---

## Task 2: Source Code Docstrings

Add Google-style docstrings to all public interfaces. These feed the API Reference via `autodoc`.

**Files:** All 8 agent files, `schemas.py`, `config.py`, both memory files, `pipeline.py`

**Pattern for every agent class:**

```python
class SomeAgent(BaseAgent):
    """One-line summary of the agent's role.

    Longer description if needed — what it does, when it runs in the pipeline,
    what makes it notable.

    Args:
        config: Application configuration.
    """

    def __init__(self, config: Config) -> None:
        ...

    def run(self, ...) -> SomeOutput:
        """One-line summary.

        Args:
            query: The user's legal question.
            research: Retrieved statute sections from the researcher.
            ... (other params)

        Returns:
            SomeOutput with fields described.
        """
```

- [ ] **Step 1: Add docstrings to `src/irish_statute_assistant/agents/clarifier.py`**

Add to the `ClarifierAgent` class:

```python
class ClarifierAgent(BaseAgent):
    """Determines whether a query needs clarification before proceeding.

    Runs first in the pipeline. If the query is ambiguous, returns a single
    focused question for the user. Never asks about jurisdiction — the assistant
    always assumes Irish law.

    Args:
        config: Application configuration.
    """
```

Add to `ClarifierAgent.run`:

```python
    def run(self, query: str, history: str) -> ClarifierOutput:
        """Decide whether the query needs clarification.

        Args:
            query: The user's legal question.
            history: Formatted prior conversation from ConversationStore.

        Returns:
            ClarifierOutput with needs_clarification=True and a question,
            or needs_clarification=False to proceed.
        """
```

- [ ] **Step 2: Add docstrings to `src/irish_statute_assistant/agents/researcher.py`**

Add to `ResearcherAgent`:

```python
class ResearcherAgent(BaseAgent):
    """Retrieves relevant Irish statute sections for a query.

    Searches the vector store (ChromaDB or Qdrant) first. Falls back to a
    live HTTP fetch from irishstatutebook.ie if the store is empty or returns
    no results. Rate-limited and retried automatically.

    Args:
        config: Application configuration.
        cache: In-memory session cache to avoid duplicate fetches.
        fetcher: HTTP statute fetcher with rate limiting.
    """
```

Add to `ResearcherAgent.run`:

```python
    def run(self, query: str) -> ResearcherOutput:
        """Retrieve statute sections relevant to the query.

        Args:
            query: The user's legal question.

        Returns:
            ResearcherOutput containing a list of Acts with their sections.

        Raises:
            StatuteNotFoundError: If no relevant statutes are found.
        """
```

- [ ] **Step 3: Add docstrings to `src/irish_statute_assistant/agents/analyst.py`**

```python
class AnalystAgent(BaseAgent):
    """Interprets retrieved statute text and identifies key clauses.

    Runs once per query, before the refinement loop. Returns AnalystLLMOutput
    (without advocate_challenges) — the Supervisor wraps this into a full
    AnalystOutput after the devil's advocate has run.

    Args:
        config: Application configuration.
    """

    def run(self, query: str, research: ResearcherOutput) -> AnalystLLMOutput:
        """Analyse statute text and identify clauses relevant to the query.

        Args:
            query: The user's legal question.
            research: Retrieved statute sections from ResearcherAgent.

        Returns:
            AnalystLLMOutput with key_clauses (each citing act and section),
            gaps (things the statutes don't address), and a confidence score.
        """
```

- [ ] **Step 4: Add docstrings to `src/irish_statute_assistant/agents/devils_advocate.py`**

```python
class DevilsAdvocateAgent(BaseAgent):
    """Challenges the analyst's findings before the writer proceeds.

    Runs after the analyst and before the refinement loop. In standard mode,
    finds 1–3 weaknesses. In strict mode (used when confidence is low or on
    refinement retries after a low-confidence trigger), finds up to 5
    adversarial challenges. Challenges are injected into the writer's prompt
    via AnalystOutput.advocate_challenges.

    Args:
        config: Application configuration.
    """

    def run(
        self,
        analyst_output: AnalystOutput,
        query: str,
        research: ResearcherOutput,
        mode: Literal["standard", "strict"] = "standard",
    ) -> AdvocateOutput:
        """Challenge the analyst's interpretation.

        Args:
            analyst_output: The analyst's findings including key clauses.
            query: The user's legal question.
            research: Retrieved statute sections.
            mode: "standard" finds 1–3 challenges; "strict" finds up to 5.

        Returns:
            AdvocateOutput with a list of challenges (may be empty) and a
            severity of "minor" or "major". severity="major" means the
            analyst's conclusion could be substantially wrong.
        """
```

- [ ] **Step 5: Add docstrings to `src/irish_statute_assistant/agents/writer.py`**

```python
class WriterAgent(BaseAgent):
    """Produces a plain-English answer from the analyst's findings.

    Runs inside the refinement loop. Receives the analyst's key clauses
    (with act and section citations), any devil's advocate challenges, and
    any evaluator flags from a previous iteration. Must address all challenges
    in the caveats section.

    Args:
        config: Application configuration.
    """

    def run(
        self,
        query: str,
        analysis: AnalystOutput,
        research: ResearcherOutput,
        evaluator_flags: list[str],
    ) -> WriterOutput:
        """Write a plain-English answer to the query.

        Args:
            query: The user's legal question.
            analysis: Analyst output including key_clauses and advocate_challenges.
            research: Retrieved statute sections.
            evaluator_flags: Quality flags from the previous evaluation round,
                or an empty list on the first attempt.

        Returns:
            WriterOutput with a short_answer (≤100 words) and a detailed_breakdown.
        """
```

- [ ] **Step 6: Add docstrings to `src/irish_statute_assistant/agents/grounding_checker.py`**

```python
class GroundingCheckerAgent(BaseAgent):
    """Verifies that the writer's key clauses are traceable to retrieved statute text.

    Runs between the writer and the evaluator in the refinement loop. For each
    KeyClause in the writer's output, checks whether the claim is directly
    supported by the retrieved statute sections. Ungrounded claims are returned
    as warnings; the evaluator penalises citation quality when grounding fails.

    Args:
        config: Application configuration.
    """

    def run(self, writer_output: WriterOutput, research: ResearcherOutput) -> GroundingOutput:
        """Check each key clause against the retrieved statute text.

        Args:
            writer_output: The writer's answer including key_clauses to verify.
            research: Retrieved statute sections used as the ground truth.

        Returns:
            GroundingOutput with ungrounded_claims (empty if all pass) and
            grounding_passed=False if any claims could not be verified.
        """
```

- [ ] **Step 7: Add docstrings to `src/irish_statute_assistant/agents/evaluator.py`**

```python
class EvaluatorAgent(BaseAgent):
    """Scores the writer's output and decides whether to accept or retry.

    Runs at the end of each refinement loop iteration. Scores on four criteria:
    accuracy, completeness, citation quality, and plain English. If the overall
    score falls below evaluator_pass_threshold, returns flags for the writer to
    address in the next iteration.

    Args:
        config: Application configuration.
    """

    def run(
        self, query: str, output: WriterOutput, grounding_passed: bool = True
    ) -> EvaluatorOutput:
        """Score the writer's output.

        Args:
            query: The user's legal question.
            output: The writer's answer to evaluate.
            grounding_passed: If False, caps the citation quality score at 0.4.
                Set from GroundingCheckerAgent.grounding_passed.

        Returns:
            EvaluatorOutput with a score (0–1), flags for improvement, and
            pass_=True if score >= evaluator_pass_threshold.
        """
```

- [ ] **Step 8: Add docstrings to `src/irish_statute_assistant/agents/supervisor.py`**

```python
class Supervisor:
    """Orchestrates the full agent pipeline for a single query.

    Runs agents in order: clarify → research → analyse → devil's advocate →
    refinement loop (write → ground-check → evaluate). Owns all memory writes.
    Applies a confidence gate that doubles the refinement rounds when analyst
    confidence is low or the devil's advocate finds a major problem.
    Detects and persists user preferences from query text and evaluator signals.

    Args:
        config: Application configuration.
        memory: SQLite-backed conversation store.
        preferences: SQLite-backed user preference store.
    """

    def run(self, query: str, context: QueryContext | None = None) -> WriterOutput | str:
        """Process a single user query through the full pipeline.

        Args:
            query: The user's legal question.
            context: Optional token budget tracker. If provided, raises
                BudgetExceededError when the budget is exhausted.

        Returns:
            A clarifying question string if clarification is needed,
            or a WriterOutput with the final answer.
        """
```

- [ ] **Step 9: Add class docstrings to `src/irish_statute_assistant/models/schemas.py`**

Add a module docstring at the top, then docstrings to each key class:

```python
"""Pydantic v2 schemas for all agent inputs and outputs.

All LLM-facing schemas are used with LangChain's with_structured_output()
to constrain model output to valid JSON. The split between AnalystLLMOutput
and AnalystOutput is intentional: advocate_challenges is populated by the
Supervisor, not the LLM, so it must not appear in the schema the LLM sees.
"""
```

Per-class docstrings (add above each class):

```python
class KeyClause(BaseModel):
    """A specific legal rule with its source citation.

    Used in AnalystOutput.key_clauses and DetailedBreakdown.key_clauses.
    The act and section fields are required to enforce traceable citations.
    """

class AnalystLLMOutput(BaseModel):
    """Schema fed to the analyst LLM via with_structured_output.

    Does not include advocate_challenges — that field is populated by the
    Supervisor after the DevilsAdvocateAgent runs. See AnalystOutput.
    """

class AnalystOutput(AnalystLLMOutput):
    """Full analyst context passed through the pipeline.

    Extends AnalystLLMOutput with advocate_challenges, which is injected
    by the Supervisor and passed to the writer to address in caveats.
    """

class DetailedBreakdown(BaseModel):
    """The structured body of the writer's answer."""

class WriterOutput(BaseModel):
    """The writer's full answer, including grounding warnings set by the Supervisor."""

class EvaluatorOutput(BaseModel):
    """Quality score and flags from the evaluator.

    pass_ uses a Pydantic alias because 'pass' is a Python reserved word.
    Access it as result.pass_ in code; it serialises as 'pass' in JSON.
    """

class AdvocateOutput(BaseModel):
    """Challenges raised by the devil's advocate."""

class GroundingOutput(BaseModel):
    """Result of the grounding check — which claims are supported by retrieved text."""

class ClarifierOutput(BaseModel):
    """Decision from the clarifier — whether to ask a question or proceed."""

class ResearcherOutput(BaseModel):
    """Retrieved statute Acts and their sections."""

class ActSection(BaseModel):
    """A single Irish Act with its retrieved statute sections."""
```

- [ ] **Step 10: Add docstring to `Config` in `src/irish_statute_assistant/config.py`**

```python
class Config(BaseSettings):
    """Application configuration loaded from environment variables or .env file.

    All settings can be set via environment variables (upper-cased) or in a
    .env file in the project root. See the configuration reference in the docs
    for a full description of each setting.

    Example:
        LLM_PROVIDER=openai OPENAI_API_KEY=sk-... uv run python -m irish_statute_assistant.main
    """
```

- [ ] **Step 11: Add docstrings to `src/irish_statute_assistant/memory/conversation_store.py`**

```python
class ConversationStore:
    """SQLite-backed conversation history that persists across process restarts.

    Stores user/assistant exchange pairs. Loads the most recent history_limit
    exchanges on construction. Each add_exchange call writes to SQLite immediately.
    The DB directory is created automatically on first use.

    Args:
        db_path: Path to the SQLite database file. Supports ~ expansion.
        history_limit: Maximum number of exchanges to load and maintain in memory.

    Example:
        store = ConversationStore("~/.irish_statute_assistant/conversations.db")
        store.add_exchange(user="What are my rights?", assistant="That depends...")
        print(store.format_for_prompt())
    """

    def add_exchange(self, user: str, assistant: str) -> None:
        """Persist a user/assistant exchange and update the in-memory history.

        Args:
            user: The user's message.
            assistant: The assistant's response.
        """

    def get_history(self) -> list[dict[str, str]]:
        """Return a copy of the in-memory exchange history.

        Returns:
            List of dicts with 'user' and 'assistant' keys, oldest first.
        """

    def format_for_prompt(self) -> str:
        """Format conversation history for injection into an LLM prompt.

        Returns:
            A string of alternating "User: ..." and "Assistant: ..." lines,
            or an empty string if there is no history.
        """
```

- [ ] **Step 12: Add docstrings to `src/irish_statute_assistant/memory/user_preference_store.py`**

```python
class UserPreferenceStore:
    """SQLite-backed key-value store for user preferences.

    Preferences are detected from query text (e.g. "I'm a solicitor") and
    from repeated evaluator signals. They persist across sessions and are
    injected into analyst and writer prompts.

    Args:
        db_path: Path to the SQLite database file. Supports ~ expansion.

    Example:
        store = UserPreferenceStore("~/.irish_statute_assistant/preferences.db")
        store.set("language_level", "technical")
        print(store.all())  # {"language_level": "technical"}
    """

    def set(self, key: str, value: str) -> None:
        """Set a preference key to a value (upsert).

        Args:
            key: Preference key (e.g. "language_level", "verbosity", "user_type").
            value: Preference value (e.g. "plain", "brief", "solicitor").
        """

    def get(self, key: str, default: str = "") -> str:
        """Get a preference value by key.

        Args:
            key: Preference key to look up.
            default: Value to return if the key is not set.

        Returns:
            The stored value, or default if not found.
        """

    def all(self) -> dict[str, str]:
        """Return all stored preferences.

        Returns:
            Dict of all key-value pairs currently stored.
        """
```

- [ ] **Step 13: Add class docstring to `Pipeline` in `src/irish_statute_assistant/pipeline.py`**

```python
class Pipeline:
    """Top-level entry point for submitting queries to the assistant.

    Constructs the conversation store, preference store, and supervisor on
    initialisation. Each call to query() is independent — the supervisor
    owns all memory writes.

    Args:
        config: Application configuration.

    Example:
        config = Config()
        pipeline = Pipeline(config)
        result = pipeline.query("How long do I have to bring a personal injury claim?")
        if isinstance(result, str):
            print("Clarification needed:", result)
        else:
            print(result.short_answer)
    """
```

- [ ] **Step 14: Run existing tests to confirm docstrings didn't break anything**

```bash
cd /home/power/projects/ai/langflow-learning-club
python -m pytest --tb=short -q 2>&1 | tail -10
```

Expected: `143 passed`

- [ ] **Step 15: Build docs to verify autodoc picks up the docstrings**

```bash
sphinx-build docs docs/_build/html -W --keep-going 2>&1 | grep -E "(ERROR|WARNING|build succeeded)"
```

Expected: `build succeeded` (some warnings about missing content files are fine at this stage).

- [ ] **Step 16: Commit**

```bash
git add src/
git commit -m "docs: add Google-style docstrings to all public interfaces"
```

---

## Task 3: API Reference Pages

Five short files that tell `autodoc` which modules to document.

**Files:**
- Create: `docs/api-reference/agents.md`
- Create: `docs/api-reference/schemas.md`
- Create: `docs/api-reference/config.md`
- Create: `docs/api-reference/memory.md`
- Create: `docs/api-reference/pipeline.md`

- [ ] **Step 1: Create `docs/api-reference/agents.md`**

````markdown
# Agents

The eight agents that make up the pipeline. All extend `BaseAgent`.

## BaseAgent

```{eval-rst}
.. autoclass:: irish_statute_assistant.agents.base_agent.BaseAgent
   :members:
```

## ClarifierAgent

```{eval-rst}
.. autoclass:: irish_statute_assistant.agents.clarifier.ClarifierAgent
   :members:
```

## ResearcherAgent

```{eval-rst}
.. autoclass:: irish_statute_assistant.agents.researcher.ResearcherAgent
   :members:
```

## AnalystAgent

```{eval-rst}
.. autoclass:: irish_statute_assistant.agents.analyst.AnalystAgent
   :members:
```

## DevilsAdvocateAgent

```{eval-rst}
.. autoclass:: irish_statute_assistant.agents.devils_advocate.DevilsAdvocateAgent
   :members:
```

## WriterAgent

```{eval-rst}
.. autoclass:: irish_statute_assistant.agents.writer.WriterAgent
   :members:
```

## GroundingCheckerAgent

```{eval-rst}
.. autoclass:: irish_statute_assistant.agents.grounding_checker.GroundingCheckerAgent
   :members:
```

## EvaluatorAgent

```{eval-rst}
.. autoclass:: irish_statute_assistant.agents.evaluator.EvaluatorAgent
   :members:
```

## Supervisor

```{eval-rst}
.. autoclass:: irish_statute_assistant.agents.supervisor.Supervisor
   :members:
```
````

- [ ] **Step 2: Create `docs/api-reference/schemas.md`**

````markdown
# Schemas

Pydantic v2 models for all agent inputs and outputs.

```{eval-rst}
.. automodule:: irish_statute_assistant.models.schemas
   :members:
   :member-order: bysource
```
````

- [ ] **Step 3: Create `docs/api-reference/config.md`**

````markdown
# Configuration

```{eval-rst}
.. autoclass:: irish_statute_assistant.config.Config
   :members:
```
````

- [ ] **Step 4: Create `docs/api-reference/memory.md`**

````markdown
# Memory Stores

```{eval-rst}
.. autoclass:: irish_statute_assistant.memory.conversation_store.ConversationStore
   :members:
```

```{eval-rst}
.. autoclass:: irish_statute_assistant.memory.user_preference_store.UserPreferenceStore
   :members:
```
````

- [ ] **Step 5: Create `docs/api-reference/pipeline.md`**

````markdown
# Pipeline

```{eval-rst}
.. autoclass:: irish_statute_assistant.pipeline.Pipeline
   :members:
```

## LLM Factory

```{eval-rst}
.. autofunction:: irish_statute_assistant.llm.get_llm
```
````

- [ ] **Step 6: Build and verify**

```bash
sphinx-build docs docs/_build/html -W --keep-going 2>&1 | grep -E "(ERROR|api-reference|build succeeded)"
```

Expected: API reference pages build without errors.

- [ ] **Step 7: Commit**

```bash
git add docs/api-reference/
git commit -m "docs: add autodoc API reference pages"
```

---

## Task 4: User Guide

**Files:**
- Create: `docs/user-guide/installation.md`
- Create: `docs/user-guide/running.md`
- Create: `docs/user-guide/configuration.md`

- [ ] **Step 1: Create `docs/user-guide/installation.md`**

```markdown
# Installation

## Requirements

- Python 3.10 or later
- [uv](https://docs.astral.sh/uv/getting-started/installation/) — the package manager used by this project

## Install dependencies

```bash
git clone <repo-url>
cd langflow-learning-club
uv sync
```

## Configure your API key

Copy the example environment file and add your key:

```bash
cp .env.example .env
```

Open `.env` and set the key for your chosen LLM provider:

```
# Anthropic (default)
ANTHROPIC_API_KEY=sk-ant-...

# Or OpenAI
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...

# Or Google
LLM_PROVIDER=google
GOOGLE_API_KEY=...

# Or Groq
LLM_PROVIDER=groq
GROQ_API_KEY=...
```

See [Configuration](configuration.md) for the full list of settings.

## Choose a vector store

The assistant retrieves statute text from a vector store. Two backends are supported:

**ChromaDB (default — local, no extra setup):**

No configuration needed. The store is created at `./data/chroma` on first use.

**Qdrant (cloud or self-hosted):**

```
VECTOR_STORE_BACKEND=qdrant
QDRANT_URL=https://your-cluster.qdrant.io
QDRANT_API_KEY=your-key
```

## Index statutes (one-time setup)

Before running the assistant for the first time, build the vector store:

```bash
uv run python -m irish_statute_assistant.indexer
```

This crawls irishstatutebook.ie by legal category, embeds the statute sections, and writes them to the configured vector store. Re-run this command to refresh the index.
```

- [ ] **Step 2: Create `docs/user-guide/running.md`**

```markdown
# Running the Assistant

## Start

```bash
uv run python -m irish_statute_assistant.main
```

You will see a prompt:

```
Irish Statute Assistant
Type your question or 'quit' to exit.

Your question:
```

## Example interaction

```
Your question: How long do I have to bring a personal injury claim?

Answer: In Ireland you generally have two years from the date of the injury
to bring a personal injury claim, though this period may be extended in
certain circumstances such as when the injured person is a minor.

--- Detail ---
Summary: The Statute of Limitations Act 1957, as amended by the Civil
Liability and Courts Act 2004, sets a two-year limitation period for
personal injury claims.

Relevant Acts:
  - Statute of Limitations Act 1957
  - Civil Liability and Courts Act 2004

Key points:
  - Two-year limitation period for personal injury. (Statute of Limitations
    Act 1957, s.11)
  - Period runs from date of injury or date of knowledge. (Civil Liability
    and Courts Act 2004, s.7)

Things to be aware of:
  - The limitation period may be extended if the claimant was a minor at the
    time of injury.
  - Seek professional legal advice — this summary does not constitute legal
    advice.
```

### Parts of the output

| Section | What it means |
|---|---|
| **Answer** | A plain-English summary in 100 words or fewer |
| **Summary** | A 2–3 sentence overview of the legal position |
| **Relevant Acts** | The Irish Acts that apply to the question |
| **Key points** | Specific rules, each citing the Act name and section |
| **Things to be aware of** | Exceptions, edge cases, and reminders to seek advice |

### Additional notices

If the assistant is uncertain about statute coverage, you may see:

```
Note: confidence in statute coverage was low for this query.
```

If any key points could not be verified against retrieved statute text:

```
--- Grounding warnings ---
  - [description of unverified claim]
```

## Clarifying questions

If your question is ambiguous, the assistant will ask one clarifying question
before proceeding:

```
Your question: What are my rights?

I need a bit more information:
  What area of law are you asking about — for example, employment,
  housing, or consumer rights?

Your clarification: Employment rights
```

The assistant remembers the conversation within a session, so follow-up
questions are understood in context.

## Ending a session

Type `quit` or press `Ctrl+C` to exit. Conversation history is saved
automatically to `~/.irish_statute_assistant/conversations.db`.
```

- [ ] **Step 3: Create `docs/user-guide/configuration.md`**

```markdown
# Configuration Reference

All settings are read from environment variables or a `.env` file in the
project root. Variable names are the upper-cased versions of the field names
listed below.

## LLM Provider

| Setting | Type | Default | Description |
|---|---|---|---|
| `LLM_PROVIDER` | `anthropic` \| `openai` \| `google` \| `groq` | `anthropic` | The LLM provider to use |
| `ANTHROPIC_API_KEY` | string | — | Required when `LLM_PROVIDER=anthropic` |
| `OPENAI_API_KEY` | string | — | Required when `LLM_PROVIDER=openai` |
| `GOOGLE_API_KEY` | string | — | Required when `LLM_PROVIDER=google` |
| `GROQ_API_KEY` | string | — | Required when `LLM_PROVIDER=groq` |
| `MODEL_NAME` | string | *(provider default)* | Override the model. Defaults: `claude-sonnet-4-6`, `gpt-4o`, `gemini-2.0-flash`, `llama-3.3-70b-versatile` |
| `TEMPERATURE` | float | `0.0` | LLM temperature. 0.0 gives deterministic output |

## Vector Store

| Setting | Type | Default | Description |
|---|---|---|---|
| `VECTOR_STORE_BACKEND` | `chroma` \| `qdrant` | `chroma` | Which vector store backend to use |
| `CHROMA_DB_PATH` | string | `./data/chroma` | Path for the local ChromaDB store |
| `QDRANT_URL` | string | — | Qdrant cluster URL (required when backend=qdrant) |
| `QDRANT_API_KEY` | string | — | Qdrant API key |
| `EMBEDDING_MODEL` | string | `all-MiniLM-L6-v2` | Sentence-transformers model used for embeddings |
| `INDEX_CATEGORIES` | list | *(10 categories)* | Legal categories to index. Default: employment, housing, family, criminal, contract, personal injury, planning, company, tax, consumer |
| `ACTS_PER_CATEGORY` | int | `5` | Number of Acts to index per category |

## Memory

| Setting | Type | Default | Description |
|---|---|---|---|
| `CONVERSATIONS_DB_PATH` | string | `~/.irish_statute_assistant/conversations.db` | SQLite file for conversation history |
| `PREFERENCES_DB_PATH` | string | `~/.irish_statute_assistant/preferences.db` | SQLite file for user preferences |
| `CONVERSATION_HISTORY_LIMIT` | int | `20` | Maximum number of prior exchanges to load into context |

## Pipeline Tuning

| Setting | Type | Default | Description |
|---|---|---|---|
| `EVALUATOR_PASS_THRESHOLD` | float | `0.7` | Minimum quality score for an answer to be accepted |
| `MAX_REFINEMENT_ROUNDS` | int | `2` | Maximum refinement attempts before returning the best answer found |
| `MAX_RETRIES` | int | `3` | Maximum retries when an LLM call fails schema validation |
| `TOKEN_BUDGET_PER_QUERY` | int | `100000` | Maximum tokens to spend across all agents for a single query |
| `RATE_LIMIT_DELAY` | float | `1.0` | Seconds to wait between HTTP requests to irishstatutebook.ie |
```

- [ ] **Step 4: Build and verify**

```bash
sphinx-build docs docs/_build/html -W --keep-going 2>&1 | grep -E "(ERROR|user-guide|build succeeded)"
```

- [ ] **Step 5: Commit**

```bash
git add docs/user-guide/
git commit -m "docs: add user guide (installation, running, configuration)"
```

---

## Task 5: Architecture — Overview and Flows

**Files:**
- Create: `docs/architecture/overview.md`
- Create: `docs/architecture/flows.md`

- [ ] **Step 1: Create `docs/architecture/overview.md`**

````markdown
# Architecture Overview

The Irish Statute Assistant uses a multi-agent pipeline. Each agent has a single
responsibility and communicates through typed Pydantic schemas. The pipeline runs
sequentially — there is no parallelism.

## Pipeline

```{mermaid}
flowchart TD
    A([User Query]) --> B[Clarifier]
    B -->|needs clarification| C([Clarifying Question → User])
    B -->|clear| D[Researcher]
    D --> E[Analyst]
    E --> F[Devil's Advocate]
    F --> G{Confidence Gate}
    G -->|low confidence or major challenge| H["Refinement Loop\n(doubled rounds, strict mode)"]
    G -->|normal| I["Refinement Loop\n(standard rounds)"]
    H --> J[Writer]
    I --> J
    J --> K[Grounding Checker]
    K --> L[Evaluator]
    L -->|pass| M([Answer → User])
    L -->|fail, retries remain| J
    L -->|fail, exhausted| M
```

## Agents at a glance

| Agent | Runs | Purpose |
|---|---|---|
| **Clarifier** | Once, first | Asks one question if the query is ambiguous |
| **Researcher** | Once | Retrieves relevant Acts from the vector store |
| **Analyst** | Once | Identifies key clauses and scores confidence |
| **Devil's Advocate** | Once + on each retry | Challenges the analyst's conclusions |
| **Writer** | Each loop iteration | Produces the plain-English answer |
| **Grounding Checker** | Each loop iteration | Verifies citations against retrieved text |
| **Evaluator** | Each loop iteration | Scores quality and decides whether to retry |
| **Supervisor** | Orchestrator | Runs all agents, owns memory writes, detects preferences |

## Refinement loop

The writer, grounding checker, and evaluator form a refinement loop. If the
evaluator's score falls below `EVALUATOR_PASS_THRESHOLD` (default 0.7), the
writer runs again with the evaluator's flags as feedback. This repeats up to
`MAX_REFINEMENT_ROUNDS` times, after which the best attempt is returned.

## Confidence gate

After the analyst runs, the devil's advocate evaluates its findings. If the
analyst's confidence is below 0.5, or if the devil's advocate rates the
problem as "major", the refinement rounds are doubled and the devil's advocate
switches to strict mode on retries. See [Flows](flows.md) for a sequence diagram.

## Memory

Two SQLite stores persist state across sessions:

- **ConversationStore** — stores every user/assistant exchange; injected into
  the clarifier's prompt so the assistant maintains context across questions
- **UserPreferenceStore** — stores detected preferences (language level,
  verbosity, user type); injected into the analyst and writer prompts

See [Memory](memory.md) for details.
````

- [ ] **Step 2: Create `docs/architecture/flows.md`**

````markdown
# Detailed Flows

Sequence diagrams for the three key flows in the pipeline.

## Clarification flow

When a query is ambiguous, the clarifier asks one focused question. The exchange
is stored in memory so the follow-up answer has context.

```{mermaid}
sequenceDiagram
    participant U as User
    participant P as Pipeline
    participant Cl as Clarifier
    participant M as ConversationStore

    U->>P: query("What are my rights?")
    P->>Cl: run(query, history="")
    Cl-->>P: ClarifierOutput(needs_clarification=True,<br/>question="What area of law?")
    P->>M: add_exchange(user=query, assistant=question)
    P-->>U: "What area of law?"

    U->>P: query("Employment rights")
    P->>Cl: run("Employment rights", history)
    Cl-->>P: ClarifierOutput(needs_clarification=False)
    Note over P: Pipeline continues to Researcher
```

## Refinement loop

The writer, grounding checker, and evaluator iterate until quality passes or
retries are exhausted. Each retry carries the evaluator's flags and the latest
devil's advocate challenges into the next writer call.

```{mermaid}
sequenceDiagram
    participant S as Supervisor
    participant W as Writer
    participant G as GroundingChecker
    participant E as Evaluator

    loop Up to effective_refinements + 1 times
        S->>W: run(query, analyst_output, research, evaluator_flags)
        W-->>S: WriterOutput
        S->>G: run(writer_output, research)
        G-->>S: GroundingOutput(ungrounded_claims, grounding_passed)
        S->>S: writer_result.warnings = grounding.ungrounded_claims
        S->>E: run(query, writer_result, grounding_passed)
        E-->>S: EvaluatorOutput(score, flags, pass_)

        alt pass_ is True
            S->>S: memory.add_exchange()
            S-->>S: return WriterOutput
        else retries remain
            S->>S: evaluator_flags = flags
            Note over S: Re-run devil's advocate, inject new challenges
        end
    end
    Note over S: Exhausted — return best attempt
```

## Confidence gate

The confidence gate runs after the analyst and the first devil's advocate call.
It determines how many refinement rounds to allow and whether the devil's
advocate uses standard or strict mode on retries.

```{mermaid}
sequenceDiagram
    participant S as Supervisor
    participant An as Analyst
    participant Ad as DevilsAdvocate

    S->>An: run(query, research)
    An-->>S: AnalystLLMOutput(key_clauses, gaps, confidence)
    S->>S: analyst_output = AnalystOutput(**result, advocate_challenges=[])

    S->>Ad: run(analyst_output, query, research, mode="standard")
    Ad-->>S: AdvocateOutput(challenges, severity)

    alt confidence < 0.5 OR severity == "major"
        Note over S: effective_refinements = max_refinements × 2
        Note over S: advocate_mode_on_retry = "strict"
    else
        Note over S: effective_refinements = max_refinements
        Note over S: advocate_mode_on_retry = "standard"
    end

    S->>S: analyst_output.advocate_challenges = challenges
    Note over S: Enters refinement loop
```
````

- [ ] **Step 3: Build and verify Mermaid renders**

```bash
sphinx-build docs docs/_build/html -W --keep-going 2>&1 | grep -E "(ERROR|mermaid|architecture|build succeeded)"
```

Expected: Build succeeds. Open `docs/_build/html/architecture/overview.html` and confirm the flowchart renders.

- [ ] **Step 4: Commit**

```bash
git add docs/architecture/overview.md docs/architecture/flows.md
git commit -m "docs: add architecture overview and flow diagrams"
```

---

## Task 6: Architecture — Agents, Schemas, Memory

**Files:**
- Create: `docs/architecture/agents.md`
- Create: `docs/architecture/schemas.md`
- Create: `docs/architecture/memory.md`

- [ ] **Step 1: Create `docs/architecture/agents.md`**

```markdown
# Agents

## ClarifierAgent

Runs first, before any statute retrieval. Reads conversation history from
`ConversationStore.format_for_prompt()` and decides in one LLM call whether
the query is specific enough to proceed. If not, it returns a single focused
question — never asks about jurisdiction (always Irish law).

**Input:** `query: str`, `history: str`
**Output:** `ClarifierOutput(needs_clarification: bool, question: str | None)`

## ResearcherAgent

Searches the vector store for statute sections relevant to the query. Uses the
configured embedding model (`all-MiniLM-L6-v2` by default) to find semantically
similar Act sections. Falls back to a live HTTP fetch from irishstatutebook.ie
if the store is empty, with rate limiting and retry logic.

**Input:** `query: str`
**Output:** `ResearcherOutput(acts: list[ActSection])`

## AnalystAgent

Reads the retrieved statute text and produces structured findings. Returns
`AnalystLLMOutput` (not `AnalystOutput`) — the Supervisor wraps the result into
a full `AnalystOutput` after the devil's advocate runs. This split is intentional:
the `advocate_challenges` field must not appear in the JSON schema the LLM sees.

**Input:** `query: str`, `research: ResearcherOutput`
**Output:** `AnalystLLMOutput(key_clauses: list[KeyClause], gaps: list[str], confidence: float)`

## DevilsAdvocateAgent

Challenges the analyst's findings before the writer proceeds. Two modes:

- **Standard** — finds 1–3 weaknesses: missing exceptions, overriding statutes,
  or claims not supported by the retrieved text
- **Strict** — adversarial; aims for up to 5 challenges. Used when analyst
  confidence is below 0.5, or when re-running after a major-severity result

Challenges are injected into `AnalystOutput.advocate_challenges` and passed to
the writer, which must address them in caveats.

**Input:** `analyst_output`, `query`, `research`, `mode: "standard" | "strict"`
**Output:** `AdvocateOutput(challenges: list[str], severity: "minor" | "major")`

## WriterAgent

Produces the user-facing answer from the analyst's key clauses, devil's advocate
challenges, and any evaluator flags from a previous iteration. The short answer
must be 100 words or fewer. Each key clause is formatted with its act and section
citation.

**Input:** `query`, `analysis: AnalystOutput`, `research`, `evaluator_flags: list[str]`
**Output:** `WriterOutput`

## GroundingCheckerAgent

Cross-references each `KeyClause` in the writer's output against the retrieved
statute text. Any claim that cannot be traced to the source text is added to
`ungrounded_claims`. The Supervisor attaches these to `WriterOutput.warnings`.
When `grounding_passed=False`, the evaluator penalises citation quality.

**Input:** `writer_output: WriterOutput`, `research: ResearcherOutput`
**Output:** `GroundingOutput(ungrounded_claims: list[str], grounding_passed: bool)`

## EvaluatorAgent

Scores the writer's output on four criteria: accuracy, completeness, citation
quality, and plain English. If `grounding_passed=False`, citation quality is
capped at 0.4 in the scoring prompt. Returns `pass_=True` if the overall score
meets `EVALUATOR_PASS_THRESHOLD`. Flags are passed back to the writer on retry.

**Input:** `query`, `output: WriterOutput`, `grounding_passed: bool`
**Output:** `EvaluatorOutput(score: float, flags: list[str], pass_: bool)`

## Supervisor

Orchestrates all agents in the correct order and owns all memory writes. Key
responsibilities:

- Constructs `AnalystOutput` from the analyst's `AnalystLLMOutput` result
- Runs the confidence gate to set `effective_refinements` and `advocate_mode_on_retry`
- Runs `_update_flag_counts` before checking `evaluation.pass_` (so counts are
  accurate for both passing and failing evaluations)
- Calls `_detect_and_save_preferences` on every returned `WriterOutput` (both
  early success and exhausted-refinements paths)
- Detects user preferences from query text and from repeated evaluator signals

## BaseAgent

All agents extend `BaseAgent`, which provides `_invoke_chain(chain, inputs)`.
This method wraps every LLM call with a `TokenUsageCallback` that reads
`usage_metadata` from the response message — compatible with all four supported
providers. The token count is exposed as `agent.last_token_count` and consumed
by the Supervisor into the `QueryContext` budget tracker.
```

- [ ] **Step 2: Create `docs/architecture/schemas.md`**

```markdown
# Schemas

All agent inputs and outputs are typed Pydantic v2 models defined in
`src/irish_statute_assistant/models/schemas.py`.

## KeyClause

```python
class KeyClause(BaseModel):
    text: str     # the rule in plain English
    act: str      # full Act name, e.g. "Statute of Limitations Act 1957"
    section: str  # section reference, e.g. "s.11"
```

`KeyClause` enforces structured citations throughout the pipeline. Every key
point in the analyst's output, the writer's breakdown, and the grounding check
must carry an act name and section reference. This makes it possible to verify
claims against source text and to present traceable citations to the user.

## Split-schema pattern: AnalystLLMOutput and AnalystOutput

```python
class AnalystLLMOutput(BaseModel):
    key_clauses: list[KeyClause]
    gaps: list[str]
    confidence: float  # 0.0–1.0

class AnalystOutput(AnalystLLMOutput):
    advocate_challenges: list[str] = []  # populated by Supervisor, not the LLM
```

`AnalystLLMOutput` is what the analyst LLM is bound to via
`with_structured_output(AnalystLLMOutput)`. The LLM never sees `advocate_challenges`
in its JSON schema. After the devil's advocate runs, the Supervisor constructs a
full `AnalystOutput`:

```python
analyst_output = AnalystOutput(**llm_result.model_dump(), advocate_challenges=[])
# ... later, after advocate runs:
analyst_output.advocate_challenges = advocate_result.challenges
```

## EvaluatorOutput and the `pass_` alias

```python
class EvaluatorOutput(BaseModel):
    score: float
    flags: list[str]
    pass_: bool = Field(alias="pass")

    model_config = {"populate_by_name": True}
```

`pass` is a Python reserved word. Pydantic's alias mechanism maps the JSON key
`"pass"` to the Python attribute `pass_`. When constructing in tests:

```python
EvaluatorOutput(score=0.85, flags=[], **{"pass": True})
```

## WriterOutput

```python
class WriterOutput(BaseModel):
    short_answer: str              # ≤100 words, validated
    detailed_breakdown: DetailedBreakdown
    warnings: list[str] = []      # ungrounded claims, set by Supervisor
    analyst_confidence: float = 1.0  # set by Supervisor from AnalystOutput
```

`warnings` and `analyst_confidence` are not populated by the writer LLM — they
are set by the Supervisor after the grounding check and before returning the result.

## AdvocateOutput

```python
class AdvocateOutput(BaseModel):
    challenges: list[str] = Field(default_factory=list, max_length=5)
    severity: Literal["minor", "major"]
```

An empty `challenges` list with `severity="minor"` is valid — it means the
analyst's output was unchallenged.

## GroundingOutput

```python
class GroundingOutput(BaseModel):
    ungrounded_claims: list[str]
    grounding_passed: bool
```

## ClarifierOutput

```python
class ClarifierOutput(BaseModel):
    needs_clarification: bool
    question: str | None = None
```

A `model_validator` enforces that `question` is present when
`needs_clarification=True`.
```

- [ ] **Step 3: Create `docs/architecture/memory.md`**

```markdown
# Memory Stores

Two SQLite-backed stores provide persistence across sessions. Both are
constructed by `Pipeline` and passed to the `Supervisor`.

## ConversationStore

Stores every user/assistant exchange. On construction, loads the most recent
`CONVERSATION_HISTORY_LIMIT` exchanges from the database.

```python
store = ConversationStore(
    db_path="~/.irish_statute_assistant/conversations.db",
    history_limit=20,
)
store.add_exchange(user="How long do I have to sue?", assistant="Six years.")
```

### Integration with agent prompts

`format_for_prompt()` is the bridge between stored history and LLM calls:

```python
history = memory.format_for_prompt()
# Returns:
# "User: How long do I have to sue?
#  Assistant: Six years."
```

The Supervisor passes this string to the Clarifier, which uses it to understand
follow-up questions in context. All writes to `ConversationStore` are owned by
the Supervisor — `Pipeline` never calls `add_exchange()` directly.

### When exchanges are written

- After a clarifying question is returned to the user
- After a successful `WriterOutput` (evaluator passed)
- After the refinement loop is exhausted (best attempt returned)

## UserPreferenceStore

A key-value store for persistent user preferences, detected automatically from
queries and evaluator signals.

```python
store = UserPreferenceStore(db_path="~/.irish_statute_assistant/preferences.db")
store.set("language_level", "plain")
store.get("language_level")   # "plain"
store.all()                   # {"language_level": "plain"}
```

### Detected preferences

**Explicit keyword scan** (case-insensitive, on every query):

| Phrase | Key | Value |
|---|---|---|
| "I'm a solicitor" / "I am a lawyer" | `user_type` | `solicitor` |
| "explain simply" / "plain English" / "non-lawyer" | `language_level` | `plain` |
| "use legal terms" / "technical" | `language_level` | `technical` |
| "brief" / "short answer" | `verbosity` | `brief` |
| "detailed" / "full explanation" | `verbosity` | `detailed` |

**Inferred preference**: If the evaluator returns a `"plain english"` flag on
two or more queries in the same session, the Supervisor saves
`language_level=technical`. The reasoning: a user who repeatedly receives
answers the evaluator flags for failing plain-English criteria is likely
comfortable with legal terminology.

## Why SQLite

Both stores use Python's stdlib `sqlite3` — no extra dependencies, no running
server, works on all platforms, and survives process restarts. The database
directory (`~/.irish_statute_assistant/`) is created automatically on first use.
```

- [ ] **Step 4: Build and verify**

```bash
sphinx-build docs docs/_build/html -W --keep-going 2>&1 | grep -E "(ERROR|architecture|build succeeded)"
```

- [ ] **Step 5: Commit**

```bash
git add docs/architecture/agents.md docs/architecture/schemas.md docs/architecture/memory.md
git commit -m "docs: add architecture pages (agents, schemas, memory)"
```

---

## Task 7: Developer Guide

**Files:**
- Create: `docs/developer-guide/adding-an-agent.md`
- Create: `docs/developer-guide/adding-a-provider.md`
- Create: `docs/developer-guide/testing.md`

- [ ] **Step 1: Create `docs/developer-guide/adding-an-agent.md`**

````markdown
# Adding an Agent

This walkthrough adds a hypothetical `SummarizerAgent` that produces a one-sentence
summary of the writer's output. Use it as a template for any new agent.

## 1. Define an output schema

Add your schema to `src/irish_statute_assistant/models/schemas.py`:

```python
class SummarizerOutput(BaseModel):
    """One-sentence summary produced by the SummarizerAgent."""
    one_liner: str
```

## 2. Create the agent file

Create `src/irish_statute_assistant/agents/summarizer.py`:

```python
from langchain_core.prompts import ChatPromptTemplate

from irish_statute_assistant.agents.base_agent import BaseAgent
from irish_statute_assistant.config import Config
from irish_statute_assistant.llm import get_llm
from irish_statute_assistant.models.schemas import SummarizerOutput, WriterOutput

SYSTEM_PROMPT = "You produce one-sentence summaries of legal answers."
HUMAN_PROMPT = "Summarise this answer in one sentence: {short_answer}"


class SummarizerAgent(BaseAgent):
    """Produces a one-sentence summary of the writer's output.

    Args:
        config: Application configuration.
    """

    def __init__(self, config: Config) -> None:
        llm = get_llm(config, max_tokens=128).with_structured_output(SummarizerOutput)
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", HUMAN_PROMPT),
        ])
        self._chain = prompt | llm

    def run(self, writer_output: WriterOutput) -> SummarizerOutput:
        """Summarise the writer's short answer.

        Args:
            writer_output: The writer's answer.

        Returns:
            SummarizerOutput with a one-liner summary.
        """
        return self._invoke_chain(self._chain, {
            "short_answer": writer_output.short_answer,
        })
```

Always use `self._invoke_chain(chain, inputs)` — never `chain.invoke(inputs)` directly.
This ensures token usage is tracked via `BaseAgent.last_token_count`.

## 3. Wire into the Supervisor

In `src/irish_statute_assistant/agents/supervisor.py`:

```python
from irish_statute_assistant.agents.summarizer import SummarizerAgent

class Supervisor:
    def __init__(self, config, memory, preferences):
        ...
        self._summarizer = SummarizerAgent(config)

    def run(self, query, context=None):
        ...
        # After the writer runs:
        summary = run_with_retry(
            lambda: self._summarizer.run(writer_result),
            self._max_retries,
        )
        if context:
            context.consume(self._summarizer.last_token_count)
        writer_result.one_liner = summary.one_liner  # add field to WriterOutput if needed
```

## 4. Write tests

Follow the project pattern — bypass `__init__` using `__new__` and mock `_invoke_chain`:

```python
from unittest.mock import MagicMock
from irish_statute_assistant.agents.summarizer import SummarizerAgent
from irish_statute_assistant.models.schemas import SummarizerOutput, WriterOutput, DetailedBreakdown, KeyClause

def make_summarizer(one_liner):
    agent = SummarizerAgent.__new__(SummarizerAgent)
    agent._invoke_chain = MagicMock(return_value=SummarizerOutput(one_liner=one_liner))
    return agent

def test_summarizer_returns_output():
    agent = make_summarizer("You have two years to claim.")
    kc = KeyClause(text="Two year limit", act="Act A", section="s.1")
    writer_out = WriterOutput(
        short_answer="You have two years.",
        detailed_breakdown=DetailedBreakdown(
            summary="S.", relevant_acts=[], key_clauses=[kc], caveats=[]
        )
    )
    result = agent.run(writer_output=writer_out)
    assert result.one_liner == "You have two years to claim."
    agent._invoke_chain.assert_called_once()
```

## When to use `with_structured_output`

Always use `with_structured_output(YourSchema)` on the LLM when your agent
must return structured data. This constrains the model to valid JSON matching
the schema, and enables validation retry via `run_with_retry`.

If you only need free-text output (unusual), use a plain chain without
`with_structured_output` and return a string.
````

- [ ] **Step 2: Create `docs/developer-guide/adding-a-provider.md`**

```markdown
# Adding an LLM Provider

The provider map is maintained across two files that must stay in sync.
An assertion in `config.py` catches any mismatch at import time.

## 1. Add the default model to `llm.py`

In `src/irish_statute_assistant/llm.py`, add an entry to `_DEFAULT_MODELS`:

```python
_DEFAULT_MODELS: dict[str, str] = {
    "anthropic": "claude-sonnet-4-6",
    "openai":    "gpt-4o",
    "google":    "gemini-2.0-flash",
    "groq":      "llama-3.3-70b-versatile",
    "mistral":   "mistral-large-latest",   # new entry
}
```

## 2. Add the API key field to `config.py`

In `src/irish_statute_assistant/config.py`:

```python
# Add the field
mistral_api_key: str = ""

# Add to _PROVIDER_KEY_MAP (at the top of the file, before Config)
_PROVIDER_KEY_MAP = {
    "anthropic": "anthropic_api_key",
    "openai":    "openai_api_key",
    "google":    "google_api_key",
    "groq":      "groq_api_key",
    "mistral":   "mistral_api_key",   # new entry
}

# Update the Literal type annotation on llm_provider
llm_provider: Literal["anthropic", "openai", "google", "groq", "mistral"] = "anthropic"
```

## 3. Add the lazy import to `get_llm()` in `llm.py`

```python
elif config.llm_provider == "mistral":
    from langchain_mistralai import ChatMistralAI
    return ChatMistralAI(
        model=config.model_name,
        api_key=config.mistral_api_key,
        max_tokens=max_tokens,
        temperature=config.temperature,
    )
```

Imports are lazy (inside the `elif` branch) so the package is only required
when that provider is actually selected.

## 4. Install the LangChain package

```bash
uv add langchain-mistralai
```

## 5. Verify the sync assertion

```bash
python -c "from irish_statute_assistant.config import Config"
```

If `_PROVIDER_KEY_MAP` and `_DEFAULT_MODELS` are out of sync, this raises an
`AssertionError` immediately. Fix by ensuring both dicts have exactly the same keys.

## 6. Test

Run the full suite to confirm no regressions:

```bash
python -m pytest --tb=short -q
```
```

- [ ] **Step 3: Create `docs/developer-guide/testing.md`**

```markdown
# Testing

## Running the tests

```bash
# All tests
python -m pytest

# With verbose output
python -m pytest -v

# A specific file
python -m pytest tests/test_supervisor.py -v

# A specific test
python -m pytest tests/test_supervisor.py::test_supervisor_returns_writer_output_when_clear_and_passes -v
```

Expected: `143 passed`

## Test file overview

| File | What it covers |
|---|---|
| `test_schemas.py` | Pydantic validation: required fields, constraints, aliases |
| `test_analyst.py` | Analyst returns `AnalystLLMOutput`; no `evaluator_flags` param |
| `test_writer.py` | `KeyClause` serialisation; `advocate_challenges` injection |
| `test_evaluator.py` | Citation serialisation; `grounding_passed` flag effect |
| `test_devils_advocate.py` | Chain routing by mode; challenge/severity behaviour |
| `test_grounding_checker.py` | Grounded vs ungrounded claims; input formatting |
| `test_supervisor.py` | Full pipeline flow; confidence gate; memory writes; preferences |
| `test_pipeline.py` | Pipeline wiring; QueryContext forwarding |
| `test_memory_stores.py` | SQLite persistence; history limits; preference read/write |
| `test_config.py` | Config loading from env; provider key validation |
| `test_adversarial.py` | Error propagation; budget exceeded; statute not found |

## Testing patterns

### Bypass `__init__` with `__new__`

Agent tests bypass the constructor (which needs a real `Config` and LLM) using
`__new__`, then inject mock attributes directly:

```python
agent = WriterAgent.__new__(WriterAgent)
agent._invoke_chain = MagicMock(return_value=WriterOutput(...))
```

### Mock `_invoke_chain`, not `_chain.invoke`

Mock at `_invoke_chain` level so the test is sensitive to whether the agent
correctly delegates through `BaseAgent`:

```python
agent._invoke_chain = MagicMock(return_value=SomeOutput(...))
result = agent.run(...)
agent._invoke_chain.assert_called_once()
```

### SQLite fixtures with `tmp_path`

Memory store tests use pytest's `tmp_path` fixture for isolated temporary databases:

```python
def test_conversation_store_persists(tmp_path):
    db = str(tmp_path / "conv.db")
    store1 = ConversationStore(db_path=db)
    store1.add_exchange(user="Q", assistant="A")
    store2 = ConversationStore(db_path=db)
    assert store2.get_history()[0]["user"] == "Q"
```

### Supervisor tests use `Supervisor.__new__`

The Supervisor test helper creates a fully-wired mock supervisor without
touching any infrastructure:

```python
sup = Supervisor.__new__(Supervisor)
sup._max_refinements = 2
sup._max_retries = 3
sup._evaluator_flag_counts = {}
sup._memory = MagicMock()
sup._clarifier = MagicMock()
# ... etc
```
```

- [ ] **Step 4: Build and verify**

```bash
sphinx-build docs docs/_build/html -W --keep-going 2>&1 | grep -E "(ERROR|developer-guide|build succeeded)"
```

- [ ] **Step 5: Commit**

```bash
git add docs/developer-guide/
git commit -m "docs: add developer guide (adding-an-agent, adding-a-provider, testing)"
```

---

## Task 8: README Update

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Read the current README**

```bash
cat README.md
```

- [ ] **Step 2: Replace `README.md` with the updated version**

Replace the full contents with:

```markdown
# Irish Statute Assistant

An AI assistant that answers plain-English questions about Irish law, powered by a
multi-agent pipeline that retrieves, analyses, and verifies statute text from
[irishstatutebook.ie](https://www.irishstatutebook.ie).

**[Full documentation →](https://irish-statute-assistant.readthedocs.io)**

---

## How it works

Eight agents orchestrated by a Supervisor:

| Agent | Role |
|---|---|
| **Clarifier** | Asks one focused question when a query is ambiguous |
| **Researcher** | Retrieves relevant Irish Acts from the vector store |
| **Analyst** | Identifies key clauses, assigns act/section citations, scores confidence |
| **Devil's Advocate** | Challenges the analyst's conclusions before the writer proceeds |
| **Writer** | Produces a short answer (≤100 words) and a detailed breakdown |
| **Grounding Checker** | Verifies every cited clause is traceable to retrieved statute text |
| **Evaluator** | Scores quality and triggers a refinement loop if below threshold |
| **Supervisor** | Orchestrates all agents, owns memory writes, detects user preferences |

Conversation history and user preferences persist across sessions in SQLite.
The assistant automatically retries with evaluator feedback when output quality
falls below threshold.

### Reliability features

- **Typed exception hierarchy** — `StatuteNotFoundError`, `BudgetExceededError`, `ValidationRepairError`
- **Validation retry** — failed schema validations are retried up to `MAX_RETRIES` times
- **Token budget** — `QueryContext` tracks usage per query and raises `BudgetExceededError` if exceeded
- **Multi-provider** — Anthropic, OpenAI, Google, and Groq supported

---

## Quick start

**Requirements:** Python 3.10+, [uv](https://docs.astral.sh/uv/getting-started/installation/)

```bash
git clone <repo-url>
cd langflow-learning-club
uv sync
cp .env.example .env
# Add your ANTHROPIC_API_KEY (or other provider key) to .env
```

**Index statutes (one-time):**

```bash
uv run python -m irish_statute_assistant.indexer
```

**Run:**

```bash
uv run python -m irish_statute_assistant.main
```

**Test:**

```bash
uv run python -m pytest
```

---

## Vector store backends

| Backend | Setup | Best for |
|---|---|---|
| **ChromaDB** (default) | No extra setup — local files at `./data/chroma` | Development, local use |
| **Qdrant** | Set `VECTOR_STORE_BACKEND=qdrant`, `QDRANT_URL`, `QDRANT_API_KEY` | Production, cloud deployment |

---

## Configuration

Key settings (set in `.env` or as environment variables):

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `anthropic` | Provider: `anthropic`, `openai`, `google`, `groq` |
| `ANTHROPIC_API_KEY` | — | Required for Anthropic |
| `VECTOR_STORE_BACKEND` | `chroma` | `chroma` or `qdrant` |
| `EVALUATOR_PASS_THRESHOLD` | `0.7` | Minimum quality score to accept an answer |
| `MAX_REFINEMENT_ROUNDS` | `2` | Refinement retries before returning best attempt |
| `TOKEN_BUDGET_PER_QUERY` | `100000` | Token limit per query across all agents |

See the [full configuration reference](https://irish-statute-assistant.readthedocs.io/user-guide/configuration.html) for all settings.
```

- [ ] **Step 3: Verify the project still builds and tests pass**

```bash
python -m pytest -q 2>&1 | tail -5
```

Expected: `143 passed`

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "docs: update README to quick-start with link to full docs"
```

---

## Task 9: Full Build Verification

- [ ] **Step 1: Run the full Sphinx build with warnings-as-errors**

```bash
cd /home/power/projects/ai/langflow-learning-club
sphinx-build docs docs/_build/html -W --keep-going 2>&1 | tail -30
```

Expected: `build succeeded` with 0 errors.

If there are warnings about missing references, fix the offending file (usually a typo in a `toctree` entry or an `autoclass` path).

- [ ] **Step 2: Run all tests to confirm no regressions**

```bash
python -m pytest -q 2>&1 | tail -5
```

Expected: `143 passed`

- [ ] **Step 3: Spot-check key pages in the built HTML**

```bash
# Check architecture overview rendered
grep -l "mermaid" docs/_build/html/architecture/overview.html && echo "Mermaid present"

# Check API reference has agent content
grep "ClarifierAgent" docs/_build/html/api-reference/agents.html && echo "Autodoc working"

# Check config table in user guide
grep "EVALUATOR_PASS_THRESHOLD" docs/_build/html/user-guide/configuration.html && echo "Config table present"
```

- [ ] **Step 4: Final commit if any fixes were needed**

```bash
git add -A
git commit -m "docs: fix any build warnings from full Sphinx verification"
```
