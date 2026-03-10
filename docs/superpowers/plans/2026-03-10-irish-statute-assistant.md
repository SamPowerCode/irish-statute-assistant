# Irish Statute Research Assistant Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a six-agent LangChain system that answers natural language Irish legal questions in plain English, backed by live data from irishstatutebook.ie.

**Architecture:** A Supervisor agent orchestrates five specialist agents (Clarifier, Legal Researcher, Legal Analyst, Plain English Writer, Evaluator) in a sequential pipeline with a refinement loop. All agents return Pydantic-validated JSON. Session memory tracks conversation history within a single run.

**Tech Stack:** Python 3.11+, LangChain 0.3+, langchain-anthropic, Pydantic v2, pydantic-settings, httpx, BeautifulSoup4, tenacity, pytest, pytest-httpx

---

## File Structure

```
src/
  irish_statute_assistant/
    __init__.py
    config.py                  # Settings: API keys, thresholds, rate limits
    models/
      __init__.py
      schemas.py               # Pydantic models for all agent I/O
    tools/
      __init__.py
      statute_fetcher.py       # Search and fetch from irishstatutebook.ie
      session_cache.py         # In-session dict cache keyed by URL
    memory/
      __init__.py
      session_memory.py        # ConversationBufferMemory wrapper
    agents/
      __init__.py
      clarifier.py             # Decides if clarification needed, generates question
      researcher.py            # Searches statute book, returns Acts + sections
      analyst.py               # Interprets statutes, returns key clauses + confidence
      writer.py                # Produces short answer + detailed breakdown
      evaluator.py             # Scores output, flags issues, returns pass/fail
      supervisor.py            # Orchestrates pipeline, manages refinement loop
    pipeline.py                # Top-level entry: takes query, returns final output
    main.py                    # CLI: read query from stdin, print response
tests/
  conftest.py                  # Shared fixtures (mock LLM, mock HTTP)
  test_schemas.py
  test_statute_fetcher.py
  test_session_cache.py
  test_session_memory.py
  test_clarifier.py
  test_researcher.py
  test_analyst.py
  test_writer.py
  test_evaluator.py
  test_supervisor.py
  test_pipeline.py
requirements.txt
.env.example
```

---

## Chunk 1: Foundation

### Task 1: Project Setup

**Files:**
- Create: `requirements.txt`
- Create: `.env.example`
- Create: `src/irish_statute_assistant/__init__.py`
- Create: `src/irish_statute_assistant/models/__init__.py`
- Create: `src/irish_statute_assistant/tools/__init__.py`
- Create: `src/irish_statute_assistant/memory/__init__.py`
- Create: `src/irish_statute_assistant/agents/__init__.py`
- Create: `tests/conftest.py`

- [ ] **Step 1: Create requirements.txt**

```
langchain>=0.3
langchain-anthropic>=0.3
langchain-core>=0.3
pydantic>=2.0
pydantic-settings>=2.0
httpx>=0.27
beautifulsoup4>=4.12
tenacity>=8.0
pytest>=8.0
pytest-httpx>=0.30
python-dotenv>=1.0
```

- [ ] **Step 2: Install dependencies**

Run: `pip install -r requirements.txt`
Expected: All packages install without error.

- [ ] **Step 3: Create .env.example**

```
ANTHROPIC_API_KEY=your-key-here
MODEL_NAME=claude-sonnet-4-6
EVALUATOR_PASS_THRESHOLD=0.7
MAX_REFINEMENT_ROUNDS=2
MAX_RETRIES=3
TOKEN_BUDGET_PER_QUERY=4000
RATE_LIMIT_DELAY=1.0
```

- [ ] **Step 4: Create all `__init__.py` files**

Each file is empty. Create:
- `src/irish_statute_assistant/__init__.py`
- `src/irish_statute_assistant/models/__init__.py`
- `src/irish_statute_assistant/tools/__init__.py`
- `src/irish_statute_assistant/memory/__init__.py`
- `src/irish_statute_assistant/agents/__init__.py`

- [ ] **Step 5: Create tests/conftest.py**

```python
import pytest
import httpx
from unittest.mock import MagicMock


@pytest.fixture
def mock_llm():
    """A mock LangChain chat model that returns a preset string."""
    llm = MagicMock()
    llm.invoke = MagicMock(return_value=MagicMock(content="{}"))
    return llm


@pytest.fixture
def sample_html_search_results():
    """Minimal HTML mimicking irishstatutebook.ie search results."""
    return """
    <html><body>
      <ul class="searchresults">
        <li class="result">
          <a href="/eli/2004/act/24/enacted/en/html">
            Civil Liability and Courts Act 2004
          </a>
        </li>
      </ul>
    </body></html>
    """


@pytest.fixture
def sample_html_act_page():
    """Minimal HTML mimicking an Act page on irishstatutebook.ie."""
    return """
    <html><body>
      <div class="section">
        <h3>Section 1</h3>
        <p>A person who suffers a personal injury shall bring a claim within two years.</p>
      </div>
      <div class="section">
        <h3>Section 2</h3>
        <p>The court may extend this period in exceptional circumstances.</p>
      </div>
    </body></html>
    """
```

- [ ] **Step 6: Commit**

```bash
git add requirements.txt .env.example src/ tests/conftest.py
git commit -m "feat: project scaffolding and dependencies"
```

---

### Task 2: Configuration

**Files:**
- Create: `src/irish_statute_assistant/config.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_config.py`:

```python
import pytest
from irish_statute_assistant.config import Config


def test_config_loads_defaults():
    config = Config(anthropic_api_key="test-key")
    assert config.model_name == "claude-sonnet-4-6"
    assert config.evaluator_pass_threshold == 0.7
    assert config.max_refinement_rounds == 2
    assert config.max_retries == 3
    assert config.token_budget_per_query == 4000
    assert config.rate_limit_delay == 1.0


def test_config_requires_api_key():
    import os
    os.environ.pop("ANTHROPIC_API_KEY", None)
    with pytest.raises(Exception):
        Config()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError` or `ImportError`

- [ ] **Step 3: Implement config.py**

```python
from pydantic_settings import BaseSettings


class Config(BaseSettings):
    anthropic_api_key: str
    model_name: str = "claude-sonnet-4-6"
    evaluator_pass_threshold: float = 0.7
    max_refinement_rounds: int = 2
    max_retries: int = 3
    token_budget_per_query: int = 4000
    rate_limit_delay: float = 1.0

    model_config = {"env_file": ".env"}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_config.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/irish_statute_assistant/config.py tests/test_config.py
git commit -m "feat: configuration with pydantic-settings"
```

---

### Task 3: Pydantic Schemas

**Files:**
- Create: `src/irish_statute_assistant/models/schemas.py`
- Create: `tests/test_schemas.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_schemas.py`:

```python
import pytest
from pydantic import ValidationError
from irish_statute_assistant.models.schemas import (
    ActSection,
    ResearcherOutput,
    AnalystOutput,
    DetailedBreakdown,
    WriterOutput,
    EvaluatorOutput,
    ClarifierOutput,
)


# --- ResearcherOutput ---

def test_researcher_output_valid():
    data = ResearcherOutput(acts=[
        ActSection(title="Act A", url="https://example.com/act-a", sections=["Section 1 text"])
    ])
    assert len(data.acts) == 1


def test_researcher_output_empty_acts_rejected():
    with pytest.raises(ValidationError):
        ResearcherOutput(acts=[])


# --- AnalystOutput ---

def test_analyst_output_valid():
    data = AnalystOutput(key_clauses=["Clause 1"], gaps=[], confidence=0.85)
    assert data.confidence == 0.85


def test_analyst_confidence_above_1_rejected():
    with pytest.raises(ValidationError):
        AnalystOutput(key_clauses=[], gaps=[], confidence=1.5)


def test_analyst_confidence_below_0_rejected():
    with pytest.raises(ValidationError):
        AnalystOutput(key_clauses=[], gaps=[], confidence=-0.1)


# --- WriterOutput ---

def test_writer_output_valid():
    breakdown = DetailedBreakdown(
        summary="Summary text",
        relevant_acts=["Act A"],
        key_clauses=["You must do X"],
        caveats=["This may vary"],
    )
    data = WriterOutput(short_answer="You have two years to make a claim.", detailed_breakdown=breakdown)
    assert data.short_answer.startswith("You")


def test_writer_short_answer_over_100_words_rejected():
    long_answer = " ".join(["word"] * 101)
    breakdown = DetailedBreakdown(summary="x", relevant_acts=[], key_clauses=[], caveats=[])
    with pytest.raises(ValidationError):
        WriterOutput(short_answer=long_answer, detailed_breakdown=breakdown)


# --- EvaluatorOutput ---

def test_evaluator_output_pass():
    data = EvaluatorOutput(score=0.8, flags=[], **{"pass": True})
    assert data.pass_ is True


def test_evaluator_score_above_1_rejected():
    with pytest.raises(ValidationError):
        EvaluatorOutput(score=1.1, flags=[], **{"pass": True})


# --- ClarifierOutput ---

def test_clarifier_needs_clarification():
    data = ClarifierOutput(needs_clarification=True, question="Which county are you in?")
    assert data.question is not None


def test_clarifier_no_clarification_needed():
    data = ClarifierOutput(needs_clarification=False)
    assert data.question is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_schemas.py -v`
Expected: FAIL — `ImportError`

- [ ] **Step 3: Implement schemas.py**

```python
from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field, field_validator


class ActSection(BaseModel):
    title: str
    url: str
    sections: list[str]


class ResearcherOutput(BaseModel):
    acts: list[ActSection] = Field(min_length=1)


class AnalystOutput(BaseModel):
    key_clauses: list[str]
    gaps: list[str]
    confidence: float = Field(ge=0.0, le=1.0)


class DetailedBreakdown(BaseModel):
    summary: str
    relevant_acts: list[str]
    key_clauses: list[str]
    caveats: list[str]


class WriterOutput(BaseModel):
    short_answer: str
    detailed_breakdown: DetailedBreakdown

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


class ClarifierOutput(BaseModel):
    needs_clarification: bool
    question: Optional[str] = None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_schemas.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/irish_statute_assistant/models/schemas.py tests/test_schemas.py
git commit -m "feat: pydantic schemas for all agent I/O with semantic validation"
```

---

### Task 4: Session Cache

**Files:**
- Create: `src/irish_statute_assistant/tools/session_cache.py`
- Create: `tests/test_session_cache.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_session_cache.py`:

```python
from irish_statute_assistant.tools.session_cache import SessionCache


def test_cache_miss_returns_none():
    cache = SessionCache()
    assert cache.get("https://example.com/page") is None


def test_cache_hit_returns_stored_value():
    cache = SessionCache()
    cache.set("https://example.com/page", ["section 1", "section 2"])
    assert cache.get("https://example.com/page") == ["section 1", "section 2"]


def test_cache_is_isolated_per_instance():
    cache_a = SessionCache()
    cache_b = SessionCache()
    cache_a.set("https://example.com/page", ["data"])
    assert cache_b.get("https://example.com/page") is None


def test_cache_overwrite():
    cache = SessionCache()
    cache.set("https://example.com/page", ["old"])
    cache.set("https://example.com/page", ["new"])
    assert cache.get("https://example.com/page") == ["new"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_session_cache.py -v`
Expected: FAIL — `ImportError`

- [ ] **Step 3: Implement session_cache.py**

```python
from typing import Optional


class SessionCache:
    """In-session dict cache keyed by URL. Not shared across instances."""

    def __init__(self) -> None:
        self._store: dict[str, list[str]] = {}

    def get(self, url: str) -> Optional[list[str]]:
        return self._store.get(url)

    def set(self, url: str, sections: list[str]) -> None:
        self._store[url] = sections
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_session_cache.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/irish_statute_assistant/tools/session_cache.py tests/test_session_cache.py
git commit -m "feat: in-session URL cache for statute fetcher"
```

---

### Task 5: Statute Fetcher

**Files:**
- Create: `src/irish_statute_assistant/tools/statute_fetcher.py`
- Create: `tests/test_statute_fetcher.py`

> **Note:** Before implementing, run a quick manual inspection of irishstatutebook.ie to verify HTML selectors. The site's search URL is `https://www.irishstatutebook.ie/eli/ResultsTitle.html?q=<query>&types=act`. Inspect the HTML of a search results page and an Act page, and update the `SEARCH_RESULT_SELECTOR` and `SECTION_SELECTOR` constants below to match the real selectors.

- [ ] **Step 1: Inspect the live site to identify selectors**

Run this in a Python REPL:
```python
import httpx
from bs4 import BeautifulSoup

r = httpx.get("https://www.irishstatutebook.ie/eli/ResultsTitle.html?q=limitation+of+actions&types=act")
soup = BeautifulSoup(r.text, "html.parser")
print(soup.prettify()[:3000])
```

Note the CSS selectors for:
- Individual search result items (each Act link)
- The title text within each result
- The href of each Act link

Then fetch an Act page and identify the selector for individual sections:
```python
r2 = httpx.get("https://www.irishstatutebook.ie/eli/1957/act/6/enacted/en/html")
soup2 = BeautifulSoup(r2.text, "html.parser")
print(soup2.prettify()[:3000])
```

Update `SEARCH_RESULT_SELECTOR` and `SECTION_SELECTOR` in the implementation below before running tests.

- [ ] **Step 2: Write the failing tests**

Create `tests/test_statute_fetcher.py`:

```python
import pytest
import httpx
from pytest_httpx import HTTPXMock
from irish_statute_assistant.tools.statute_fetcher import search_statutes, fetch_act_sections
from irish_statute_assistant.tools.session_cache import SessionCache


SEARCH_URL = "https://www.irishstatutebook.ie/eli/ResultsTitle.html"
ACT_URL = "https://www.irishstatutebook.ie/eli/2004/act/24/enacted/en/html"


def test_search_statutes_returns_results(httpx_mock: HTTPXMock, sample_html_search_results):
    httpx_mock.add_response(
        url=f"{SEARCH_URL}?q=personal+injury&types=act",
        text=sample_html_search_results,
    )
    results = search_statutes("personal injury")
    assert len(results) >= 1
    assert "title" in results[0]
    assert "url" in results[0]


def test_search_statutes_returns_at_most_5(httpx_mock: HTTPXMock):
    # Build a page with 10 results
    items = "".join(
        f'<li class="result"><a href="/eli/200{i}/act/{i}/enacted/en/html">Act {i}</a></li>'
        for i in range(10)
    )
    html = f"<html><body><ul class='searchresults'>{items}</ul></body></html>"
    httpx_mock.add_response(url=f"{SEARCH_URL}?q=test&types=act", text=html)
    results = search_statutes("test")
    assert len(results) <= 5


def test_fetch_act_sections_returns_sections(httpx_mock: HTTPXMock, sample_html_act_page):
    httpx_mock.add_response(url=ACT_URL, text=sample_html_act_page)
    cache = SessionCache()
    sections = fetch_act_sections(ACT_URL, cache)
    assert len(sections) >= 1
    assert isinstance(sections[0], str)


def test_fetch_act_sections_uses_cache(httpx_mock: HTTPXMock, sample_html_act_page):
    """Second call should not make an HTTP request."""
    httpx_mock.add_response(url=ACT_URL, text=sample_html_act_page)
    cache = SessionCache()
    fetch_act_sections(ACT_URL, cache)      # first call — hits network
    fetch_act_sections(ACT_URL, cache)      # second call — from cache
    assert len(httpx_mock.get_requests()) == 1


def test_search_statutes_retries_on_failure(httpx_mock: HTTPXMock, sample_html_search_results):
    httpx_mock.add_response(url=f"{SEARCH_URL}?q=test&types=act", status_code=500)
    httpx_mock.add_response(url=f"{SEARCH_URL}?q=test&types=act", text=sample_html_search_results)
    results = search_statutes("test")
    assert len(results) >= 1
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/test_statute_fetcher.py -v`
Expected: FAIL — `ImportError`

- [ ] **Step 4: Implement statute_fetcher.py**

```python
import time
from typing import Optional
import httpx
from bs4 import BeautifulSoup
from tenacity import retry, stop_after_attempt, wait_exponential

from irish_statute_assistant.tools.session_cache import SessionCache

BASE_URL = "https://www.irishstatutebook.ie"
SEARCH_URL = f"{BASE_URL}/eli/ResultsTitle.html"

# Update these after inspecting the live site (see Task 5, Step 1)
SEARCH_RESULT_SELECTOR = "li.result a"   # selector for Act links in search results
SECTION_SELECTOR = "div.section"         # selector for sections in an Act page

RATE_LIMIT_DELAY = 1.0  # seconds; override via config if needed


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
def search_statutes(query: str) -> list[dict]:
    """Search irishstatutebook.ie and return up to 5 results as {title, url}."""
    params = {"q": query, "types": "act"}
    response = httpx.get(SEARCH_URL, params=params, timeout=30)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    results = []
    for link in soup.select(SEARCH_RESULT_SELECTOR)[:5]:
        href = link.get("href", "")
        url = href if href.startswith("http") else BASE_URL + href
        results.append({"title": link.get_text(strip=True), "url": url})

    return results


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
def fetch_act_sections(url: str, cache: SessionCache) -> list[str]:
    """Fetch an Act page and return up to 10 section texts. Uses cache."""
    cached = cache.get(url)
    if cached is not None:
        return cached

    time.sleep(RATE_LIMIT_DELAY)
    response = httpx.get(url, timeout=30)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    sections = []
    for section in soup.select(SECTION_SELECTOR)[:10]:
        text = section.get_text(strip=True)
        if text:
            sections.append(text[:2000])

    cache.set(url, sections)
    return sections
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_statute_fetcher.py -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add src/irish_statute_assistant/tools/statute_fetcher.py tests/test_statute_fetcher.py
git commit -m "feat: statute fetcher with retry, rate limiting, and session cache"
```

---

### Task 6: Session Memory

**Files:**
- Create: `src/irish_statute_assistant/memory/session_memory.py`
- Create: `tests/test_session_memory.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_session_memory.py`:

```python
from irish_statute_assistant.memory.session_memory import SessionMemory


def test_empty_memory_returns_empty_history():
    memory = SessionMemory()
    assert memory.get_history() == []


def test_add_exchange_stores_user_and_assistant():
    memory = SessionMemory()
    memory.add_exchange(user="What is the law on X?", assistant="The law says Y.")
    history = memory.get_history()
    assert len(history) == 1
    assert history[0]["user"] == "What is the law on X?"
    assert history[0]["assistant"] == "The law says Y."


def test_multiple_exchanges_ordered():
    memory = SessionMemory()
    memory.add_exchange(user="Q1", assistant="A1")
    memory.add_exchange(user="Q2", assistant="A2")
    history = memory.get_history()
    assert len(history) == 2
    assert history[0]["user"] == "Q1"
    assert history[1]["user"] == "Q2"


def test_format_for_prompt_returns_string():
    memory = SessionMemory()
    memory.add_exchange(user="Q1", assistant="A1")
    formatted = memory.format_for_prompt()
    assert "Q1" in formatted
    assert "A1" in formatted
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_session_memory.py -v`
Expected: FAIL — `ImportError`

- [ ] **Step 3: Implement session_memory.py**

```python
class SessionMemory:
    """Stores conversation exchanges for a single session."""

    def __init__(self) -> None:
        self._history: list[dict[str, str]] = []

    def add_exchange(self, user: str, assistant: str) -> None:
        self._history.append({"user": user, "assistant": assistant})

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

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_session_memory.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/irish_statute_assistant/memory/session_memory.py tests/test_session_memory.py
git commit -m "feat: session memory for in-conversation history"
```

---

## Chunk 2: Agents

### Task 7: Clarifier Agent

**Files:**
- Create: `src/irish_statute_assistant/agents/clarifier.py`
- Create: `tests/test_clarifier.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_clarifier.py`:

```python
import pytest
from unittest.mock import MagicMock, patch
from irish_statute_assistant.agents.clarifier import ClarifierAgent
from irish_statute_assistant.models.schemas import ClarifierOutput


def make_clarifier(needs_clarification: bool, question: str | None = None):
    agent = ClarifierAgent.__new__(ClarifierAgent)
    mock_chain = MagicMock()
    mock_chain.invoke = MagicMock(return_value=ClarifierOutput(
        needs_clarification=needs_clarification,
        question=question,
    ))
    agent._chain = mock_chain
    return agent


def test_clarifier_returns_clarifier_output_when_ambiguous():
    agent = make_clarifier(needs_clarification=True, question="Which county do you live in?")
    result = agent.run(query="What are my rights?", history="")
    assert result.needs_clarification is True
    assert result.question is not None


def test_clarifier_returns_no_clarification_when_clear():
    agent = make_clarifier(needs_clarification=False)
    result = agent.run(query="What is the statute of limitations for personal injury in Ireland?", history="")
    assert result.needs_clarification is False
    assert result.question is None


def test_clarifier_passes_history_to_chain():
    agent = make_clarifier(needs_clarification=False)
    agent.run(query="Some question", history="User: previous\nAssistant: answer")
    call_args = agent._chain.invoke.call_args[0][0]
    assert "previous" in call_args["history"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_clarifier.py -v`
Expected: FAIL — `ImportError`

- [ ] **Step 3: Implement clarifier.py**

```python
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate

from irish_statute_assistant.config import Config
from irish_statute_assistant.models.schemas import ClarifierOutput

SYSTEM_PROMPT = """You are a helpful legal assistant intake agent. Your job is to decide
whether a user's legal question is clear enough to research, or whether you need to ask
one focused clarifying question first.

Rules:
- If the question clearly identifies a legal topic (e.g. "statute of limitations for injury"),
  return needs_clarification=false.
- If the question is too vague (e.g. "what are my rights?"), return needs_clarification=true
  and ask exactly ONE short, specific question that will make the topic researchable.
- Never ask more than one question.
- Keep the question simple enough for a non-lawyer to understand.

Conversation so far:
{history}
"""

HUMAN_PROMPT = "User's question: {query}"


class ClarifierAgent:
    def __init__(self, config: Config) -> None:
        llm = ChatAnthropic(
            model=config.model_name,
            api_key=config.anthropic_api_key,
            max_tokens=256,
        ).with_structured_output(ClarifierOutput)
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", HUMAN_PROMPT),
        ])
        self._chain = prompt | llm

    def run(self, query: str, history: str) -> ClarifierOutput:
        return self._chain.invoke({"query": query, "history": history})
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_clarifier.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/irish_statute_assistant/agents/clarifier.py tests/test_clarifier.py
git commit -m "feat: clarifier agent with structured output"
```

---

### Task 8: Legal Researcher Agent

**Files:**
- Create: `src/irish_statute_assistant/agents/researcher.py`
- Create: `tests/test_researcher.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_researcher.py`:

```python
from unittest.mock import MagicMock, patch
from irish_statute_assistant.agents.researcher import ResearcherAgent
from irish_statute_assistant.models.schemas import ResearcherOutput, ActSection
from irish_statute_assistant.tools.session_cache import SessionCache


def make_researcher_with_mocks(search_results, sections_per_act):
    cache = SessionCache()
    agent = ResearcherAgent.__new__(ResearcherAgent)
    agent._cache = cache

    mock_search = MagicMock(return_value=search_results)
    mock_fetch = MagicMock(return_value=sections_per_act)
    agent._search = mock_search
    agent._fetch = mock_fetch
    return agent


def test_researcher_returns_researcher_output():
    agent = make_researcher_with_mocks(
        search_results=[{"title": "Act A", "url": "https://example.com/act-a"}],
        sections_per_act=["Section 1 text"],
    )
    result = agent.run(query="personal injury time limit")
    assert isinstance(result, ResearcherOutput)
    assert len(result.acts) >= 1


def test_researcher_includes_title_and_url():
    agent = make_researcher_with_mocks(
        search_results=[{"title": "Statute of Limitations Act 1957", "url": "https://example.com/1957"}],
        sections_per_act=["You must bring an action within 6 years."],
    )
    result = agent.run(query="limitation of actions")
    assert result.acts[0].title == "Statute of Limitations Act 1957"
    assert result.acts[0].url == "https://example.com/1957"


def test_researcher_raises_on_no_results():
    agent = make_researcher_with_mocks(search_results=[], sections_per_act=[])
    import pytest
    with pytest.raises(ValueError, match="No Acts found"):
        agent.run(query="completely unknown legal topic xyz")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_researcher.py -v`
Expected: FAIL — `ImportError`

- [ ] **Step 3: Implement researcher.py**

```python
from irish_statute_assistant.config import Config
from irish_statute_assistant.models.schemas import ActSection, ResearcherOutput
from irish_statute_assistant.tools.session_cache import SessionCache
from irish_statute_assistant.tools.statute_fetcher import fetch_act_sections, search_statutes


class ResearcherAgent:
    def __init__(self, config: Config, cache: SessionCache) -> None:
        self._cache = cache
        self._search = search_statutes
        self._fetch = fetch_act_sections

    def run(self, query: str) -> ResearcherOutput:
        results = self._search(query)
        if not results:
            raise ValueError(f"No Acts found for query: {query!r}")

        acts = []
        for result in results:
            sections = self._fetch(result["url"], self._cache)
            acts.append(ActSection(
                title=result["title"],
                url=result["url"],
                sections=sections,
            ))

        return ResearcherOutput(acts=acts)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_researcher.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/irish_statute_assistant/agents/researcher.py tests/test_researcher.py
git commit -m "feat: legal researcher agent wrapping statute fetcher"
```

---

### Task 9: Legal Analyst Agent

**Files:**
- Create: `src/irish_statute_assistant/agents/analyst.py`
- Create: `tests/test_analyst.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_analyst.py`:

```python
from unittest.mock import MagicMock
from irish_statute_assistant.agents.analyst import AnalystAgent
from irish_statute_assistant.models.schemas import AnalystOutput, ResearcherOutput, ActSection


def make_analyst(key_clauses, gaps, confidence):
    agent = AnalystAgent.__new__(AnalystAgent)
    mock_chain = MagicMock()
    mock_chain.invoke = MagicMock(return_value=AnalystOutput(
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


def test_analyst_returns_analyst_output():
    agent = make_analyst(["Bring action within 6 years"], [], 0.9)
    result = agent.run(query="limitation period", research=sample_research(), evaluator_flags=[])
    assert isinstance(result, AnalystOutput)
    assert result.confidence == 0.9


def test_analyst_passes_evaluator_flags():
    agent = make_analyst(["Clause"], [], 0.8)
    agent.run(query="Q", research=sample_research(), evaluator_flags=["Missing citation"])
    call_args = agent._chain.invoke.call_args[0][0]
    assert "Missing citation" in call_args["evaluator_flags"]


def test_analyst_confidence_in_valid_range():
    agent = make_analyst([], [], 0.5)
    result = agent.run(query="Q", research=sample_research(), evaluator_flags=[])
    assert 0.0 <= result.confidence <= 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_analyst.py -v`
Expected: FAIL — `ImportError`

- [ ] **Step 3: Implement analyst.py**

```python
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate

from irish_statute_assistant.config import Config
from irish_statute_assistant.models.schemas import AnalystOutput, ResearcherOutput

SYSTEM_PROMPT = """You are a legal analyst. You have been given raw Irish statute text and
a user's question. Your job is to:
1. Identify the key clauses directly relevant to the question.
2. Note any gaps (things the user asked about that the statutes don't clearly address).
3. Assign a confidence score (0.0–1.0) for how well the statutes answer the question.
   - 0.9+ = question is fully and clearly answered
   - 0.5–0.89 = partially answered, some ambiguity
   - below 0.5 = statutes are unclear or not directly relevant

If there are evaluator flags from a previous attempt, address them in this analysis.

Evaluator flags from previous attempt (if any): {evaluator_flags}
"""

HUMAN_PROMPT = """User question: {query}

Retrieved statute sections:
{statute_text}
"""


class AnalystAgent:
    def __init__(self, config: Config) -> None:
        llm = ChatAnthropic(
            model=config.model_name,
            api_key=config.anthropic_api_key,
            max_tokens=1024,
        ).with_structured_output(AnalystOutput)
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", HUMAN_PROMPT),
        ])
        self._chain = prompt | llm

    def run(self, query: str, research: ResearcherOutput, evaluator_flags: list[str]) -> AnalystOutput:
        statute_text = self._format_research(research)
        flags_text = "\n".join(evaluator_flags) if evaluator_flags else "None"
        return self._chain.invoke({
            "query": query,
            "statute_text": statute_text,
            "evaluator_flags": flags_text,
        })

    def _format_research(self, research: ResearcherOutput) -> str:
        parts = []
        for act in research.acts:
            parts.append(f"## {act.title}\nURL: {act.url}")
            for section in act.sections:
                parts.append(section)
        return "\n\n".join(parts)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_analyst.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/irish_statute_assistant/agents/analyst.py tests/test_analyst.py
git commit -m "feat: legal analyst agent with confidence scoring"
```

---

### Task 10: Plain English Writer Agent

**Files:**
- Create: `src/irish_statute_assistant/agents/writer.py`
- Create: `tests/test_writer.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_writer.py`:

```python
from unittest.mock import MagicMock
from irish_statute_assistant.agents.writer import WriterAgent
from irish_statute_assistant.models.schemas import (
    WriterOutput, DetailedBreakdown, AnalystOutput, ResearcherOutput, ActSection
)


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


def sample_analyst_output():
    return AnalystOutput(key_clauses=["6 year limit"], gaps=[], confidence=0.9)


def sample_research():
    return ResearcherOutput(acts=[
        ActSection(title="Act A", url="https://example.com", sections=["Section 1"])
    ])


def test_writer_returns_writer_output():
    agent = make_writer(
        short_answer="You have six years to make a claim.",
        summary="The Statute of Limitations sets a 6-year window.",
        relevant_acts=["Statute of Limitations Act 1957"],
        key_clauses=["6 year limit from date of cause of action"],
        caveats=["Personal injury cases have a shorter 2-year limit"],
    )
    result = agent.run(query="How long do I have to sue?", analysis=sample_analyst_output(), research=sample_research(), evaluator_flags=[])
    assert isinstance(result, WriterOutput)
    assert len(result.short_answer.split()) <= 100


def test_writer_short_answer_is_plain_english():
    agent = make_writer(
        short_answer="You have six years to make a claim in most cases.",
        summary="Summary", relevant_acts=[], key_clauses=[], caveats=[],
    )
    result = agent.run(query="Q", analysis=sample_analyst_output(), research=sample_research(), evaluator_flags=[])
    # Plain English check: no Latin phrases
    assert "sub judice" not in result.short_answer
    assert "inter alia" not in result.short_answer


def test_writer_passes_evaluator_flags_in_prompt():
    agent = make_writer(
        short_answer="Short answer.", summary="S", relevant_acts=[], key_clauses=[], caveats=[],
    )
    agent.run(query="Q", analysis=sample_analyst_output(), research=sample_research(), evaluator_flags=["Add citation"])
    call_args = agent._chain.invoke.call_args[0][0]
    assert "Add citation" in call_args["evaluator_flags"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_writer.py -v`
Expected: FAIL — `ImportError`

- [ ] **Step 3: Implement writer.py**

```python
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate

from irish_statute_assistant.config import Config
from irish_statute_assistant.models.schemas import AnalystOutput, ResearcherOutput, WriterOutput

SYSTEM_PROMPT = """You are a plain English legal writer. You explain Irish law to ordinary people
who have no legal training.

Rules:
- short_answer: one or two plain sentences (maximum 100 words). No legal jargon. Write as if
  explaining to a friend.
- detailed_breakdown.summary: 2-3 sentences summarising the legal position.
- detailed_breakdown.relevant_acts: list the Acts by name (e.g. "Statute of Limitations Act 1957").
- detailed_breakdown.key_clauses: the specific rules that answer the question, in plain English.
- detailed_breakdown.caveats: important exceptions, edge cases, or "it depends" situations.
- Always include at least one caveat reminding the user to seek professional legal advice.
- Never make up law. Only use what the analyst found.

If there are evaluator flags, address them: {evaluator_flags}
"""

HUMAN_PROMPT = """User question: {query}

Analyst findings:
Key clauses: {key_clauses}
Gaps: {gaps}
Confidence: {confidence}

Acts researched:
{act_titles}
"""


class WriterAgent:
    def __init__(self, config: Config) -> None:
        llm = ChatAnthropic(
            model=config.model_name,
            api_key=config.anthropic_api_key,
            max_tokens=1024,
        ).with_structured_output(WriterOutput)
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
        act_titles = "\n".join(act.title for act in research.acts)
        return self._chain.invoke({
            "query": query,
            "key_clauses": "\n".join(analysis.key_clauses),
            "gaps": "\n".join(analysis.gaps) if analysis.gaps else "None",
            "confidence": analysis.confidence,
            "act_titles": act_titles,
            "evaluator_flags": flags_text,
        })
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_writer.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/irish_statute_assistant/agents/writer.py tests/test_writer.py
git commit -m "feat: plain english writer agent with structured breakdown"
```

---

### Task 11: Evaluator Agent

**Files:**
- Create: `src/irish_statute_assistant/agents/evaluator.py`
- Create: `tests/test_evaluator.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_evaluator.py`:

```python
from unittest.mock import MagicMock
from irish_statute_assistant.agents.evaluator import EvaluatorAgent
from irish_statute_assistant.models.schemas import EvaluatorOutput, WriterOutput, DetailedBreakdown


def make_evaluator(score, flags, passed):
    agent = EvaluatorAgent.__new__(EvaluatorAgent)
    mock_chain = MagicMock()
    mock_chain.invoke = MagicMock(return_value=EvaluatorOutput(
        score=score, flags=flags, **{"pass": passed}
    ))
    agent._chain = mock_chain
    agent._threshold = 0.7
    return agent


def sample_writer_output():
    return WriterOutput(
        short_answer="You have six years to make a claim.",
        detailed_breakdown=DetailedBreakdown(
            summary="The law gives you six years.",
            relevant_acts=["Statute of Limitations Act 1957"],
            key_clauses=["6 year limit"],
            caveats=["Seek legal advice"],
        )
    )


def test_evaluator_passes_high_score():
    agent = make_evaluator(score=0.9, flags=[], passed=True)
    result = agent.run(query="How long do I have?", output=sample_writer_output())
    assert result.pass_ is True
    assert result.score == 0.9


def test_evaluator_fails_low_score():
    agent = make_evaluator(score=0.5, flags=["Missing citation", "Answer too vague"], passed=False)
    result = agent.run(query="How long do I have?", output=sample_writer_output())
    assert result.pass_ is False
    assert len(result.flags) > 0


def test_evaluator_score_in_valid_range():
    agent = make_evaluator(score=0.75, flags=[], passed=True)
    result = agent.run(query="Q", output=sample_writer_output())
    assert 0.0 <= result.score <= 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_evaluator.py -v`
Expected: FAIL — `ImportError`

- [ ] **Step 3: Implement evaluator.py**

```python
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate

from irish_statute_assistant.config import Config
from irish_statute_assistant.models.schemas import EvaluatorOutput, WriterOutput

SYSTEM_PROMPT = """You are a quality evaluator for an Irish legal research assistant.
Score the output on these four criteria (each 0.0–1.0, overall score is the average):

1. Accuracy: Does the answer correctly reflect what the statutes say?
2. Completeness: Does it answer the user's question fully?
3. Citation quality: Are relevant Acts named and referenced?
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


class EvaluatorAgent:
    def __init__(self, config: Config) -> None:
        self._threshold = config.evaluator_pass_threshold
        llm = ChatAnthropic(
            model=config.model_name,
            api_key=config.anthropic_api_key,
            max_tokens=512,
        ).with_structured_output(EvaluatorOutput)
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", HUMAN_PROMPT),
        ])
        self._chain = prompt | llm

    def run(self, query: str, output: WriterOutput) -> EvaluatorOutput:
        bd = output.detailed_breakdown
        return self._chain.invoke({
            "threshold": self._threshold,
            "query": query,
            "short_answer": output.short_answer,
            "summary": bd.summary,
            "relevant_acts": ", ".join(bd.relevant_acts),
            "key_clauses": "\n".join(bd.key_clauses),
            "caveats": "\n".join(bd.caveats),
        })
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_evaluator.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/irish_statute_assistant/agents/evaluator.py tests/test_evaluator.py
git commit -m "feat: evaluator agent with score, flags, and pass/fail"
```

---

## Chunk 3: Orchestration

### Task 12: Supervisor Agent

**Files:**
- Create: `src/irish_statute_assistant/agents/supervisor.py`
- Create: `tests/test_supervisor.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_supervisor.py`:

```python
import pytest
from unittest.mock import MagicMock
from irish_statute_assistant.agents.supervisor import Supervisor
from irish_statute_assistant.models.schemas import (
    ClarifierOutput, ResearcherOutput, ActSection,
    AnalystOutput, WriterOutput, DetailedBreakdown, EvaluatorOutput
)


def make_supervisor(
    clarifier_output: ClarifierOutput,
    researcher_output: ResearcherOutput,
    analyst_output: AnalystOutput,
    writer_output: WriterOutput,
    evaluator_output: EvaluatorOutput,
):
    sup = Supervisor.__new__(Supervisor)
    sup._max_refinements = 2

    sup._clarifier = MagicMock()
    sup._clarifier.run = MagicMock(return_value=clarifier_output)

    sup._researcher = MagicMock()
    sup._researcher.run = MagicMock(return_value=researcher_output)

    sup._analyst = MagicMock()
    sup._analyst.run = MagicMock(return_value=analyst_output)

    sup._writer = MagicMock()
    sup._writer.run = MagicMock(return_value=writer_output)

    sup._evaluator = MagicMock()
    sup._evaluator.run = MagicMock(return_value=evaluator_output)

    return sup


def make_defaults():
    research = ResearcherOutput(acts=[
        ActSection(title="Act A", url="https://example.com", sections=["Section 1"])
    ])
    analysis = AnalystOutput(key_clauses=["6 year limit"], gaps=[], confidence=0.9)
    writer_out = WriterOutput(
        short_answer="You have six years.",
        detailed_breakdown=DetailedBreakdown(
            summary="Six year limit.", relevant_acts=["Act A"],
            key_clauses=["6 year limit"], caveats=["Seek advice"],
        )
    )
    evaluator_pass = EvaluatorOutput(score=0.85, flags=[], **{"pass": True})
    return research, analysis, writer_out, evaluator_pass


def test_supervisor_returns_writer_output_when_clear_and_passes():
    research, analysis, writer_out, evaluator_pass = make_defaults()
    sup = make_supervisor(
        clarifier_output=ClarifierOutput(needs_clarification=False),
        researcher_output=research,
        analyst_output=analysis,
        writer_output=writer_out,
        evaluator_output=evaluator_pass,
    )
    result = sup.run(query="How long do I have to sue?", history="")
    assert isinstance(result, WriterOutput)


def test_supervisor_returns_clarification_question_when_needed():
    research, analysis, writer_out, evaluator_pass = make_defaults()
    sup = make_supervisor(
        clarifier_output=ClarifierOutput(needs_clarification=True, question="What type of case?"),
        researcher_output=research,
        analyst_output=analysis,
        writer_output=writer_out,
        evaluator_output=evaluator_pass,
    )
    result = sup.run(query="What are my rights?", history="")
    assert result == "What type of case?"


def test_supervisor_refinement_loop_retries_on_fail():
    research, analysis, writer_out, _ = make_defaults()
    evaluator_fail = EvaluatorOutput(score=0.5, flags=["Vague answer"], **{"pass": False})
    evaluator_pass = EvaluatorOutput(score=0.85, flags=[], **{"pass": True})

    sup = make_supervisor(
        clarifier_output=ClarifierOutput(needs_clarification=False),
        researcher_output=research,
        analyst_output=analysis,
        writer_output=writer_out,
        evaluator_output=evaluator_fail,
    )
    # Override evaluator to fail once then pass
    sup._evaluator.run = MagicMock(side_effect=[evaluator_fail, evaluator_pass])

    result = sup.run(query="How long?", history="")
    assert isinstance(result, WriterOutput)
    assert sup._evaluator.run.call_count == 2


def test_supervisor_stops_after_max_refinements():
    research, analysis, writer_out, _ = make_defaults()
    evaluator_fail = EvaluatorOutput(score=0.4, flags=["Still bad"], **{"pass": False})

    sup = make_supervisor(
        clarifier_output=ClarifierOutput(needs_clarification=False),
        researcher_output=research,
        analyst_output=analysis,
        writer_output=writer_out,
        evaluator_output=evaluator_fail,
    )
    # Always fails
    sup._evaluator.run = MagicMock(return_value=evaluator_fail)

    result = sup.run(query="How long?", history="")
    # After max_refinements, returns best available output
    assert isinstance(result, WriterOutput)
    assert sup._evaluator.run.call_count == sup._max_refinements + 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_supervisor.py -v`
Expected: FAIL — `ImportError`

- [ ] **Step 3: Implement supervisor.py**

```python
from irish_statute_assistant.agents.analyst import AnalystAgent
from irish_statute_assistant.agents.clarifier import ClarifierAgent
from irish_statute_assistant.agents.evaluator import EvaluatorAgent
from irish_statute_assistant.agents.researcher import ResearcherAgent
from irish_statute_assistant.agents.writer import WriterAgent
from irish_statute_assistant.config import Config
from irish_statute_assistant.models.schemas import WriterOutput
from irish_statute_assistant.tools.session_cache import SessionCache


class Supervisor:
    def __init__(self, config: Config) -> None:
        self._max_refinements = config.max_refinement_rounds
        cache = SessionCache()
        self._clarifier = ClarifierAgent(config)
        self._researcher = ResearcherAgent(config, cache)
        self._analyst = AnalystAgent(config)
        self._writer = WriterAgent(config)
        self._evaluator = EvaluatorAgent(config)

    def run(self, query: str, history: str) -> WriterOutput | str:
        """
        Returns:
          - str: a clarifying question if the query is ambiguous
          - WriterOutput: the final answer if the query is clear
        """
        clarifier_result = self._clarifier.run(query=query, history=history)
        if clarifier_result.needs_clarification:
            return clarifier_result.question

        research = self._researcher.run(query=query)
        evaluator_flags: list[str] = []
        best_output: WriterOutput | None = None

        for _ in range(self._max_refinements + 1):
            analysis = self._analyst.run(query=query, research=research, evaluator_flags=evaluator_flags)
            output = self._writer.run(query=query, analysis=analysis, research=research, evaluator_flags=evaluator_flags)
            evaluation = self._evaluator.run(query=query, output=output)
            best_output = output

            if evaluation.pass_:
                return output

            evaluator_flags = evaluation.flags

        return best_output
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_supervisor.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/irish_statute_assistant/agents/supervisor.py tests/test_supervisor.py
git commit -m "feat: supervisor agent with clarification routing and refinement loop"
```

---

### Task 13: Pipeline

**Files:**
- Create: `src/irish_statute_assistant/pipeline.py`
- Create: `tests/test_pipeline.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_pipeline.py`:

```python
from unittest.mock import MagicMock
from irish_statute_assistant.pipeline import Pipeline
from irish_statute_assistant.models.schemas import WriterOutput, DetailedBreakdown


def make_pipeline(supervisor_returns):
    p = Pipeline.__new__(Pipeline)
    p._supervisor = MagicMock()
    p._supervisor.run = MagicMock(return_value=supervisor_returns)
    from irish_statute_assistant.memory.session_memory import SessionMemory
    p._memory = SessionMemory()
    return p


def writer_output():
    return WriterOutput(
        short_answer="You have six years.",
        detailed_breakdown=DetailedBreakdown(
            summary="Six year limit.", relevant_acts=["Act A"],
            key_clauses=["6 year limit"], caveats=["Seek advice"],
        )
    )


def test_pipeline_returns_writer_output():
    p = make_pipeline(writer_output())
    result = p.query("How long do I have?")
    assert isinstance(result, WriterOutput)


def test_pipeline_returns_clarification_string():
    p = make_pipeline("Can you be more specific?")
    result = p.query("What are my rights?")
    assert isinstance(result, str)
    assert result == "Can you be more specific?"


def test_pipeline_memory_updated_after_answer():
    p = make_pipeline(writer_output())
    p.query("How long do I have?")
    history = p._memory.get_history()
    assert len(history) == 1
    assert history[0]["user"] == "How long do I have?"


def test_pipeline_memory_not_updated_after_clarification():
    p = make_pipeline("Can you be more specific?")
    p.query("What are my rights?")
    history = p._memory.get_history()
    assert len(history) == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_pipeline.py -v`
Expected: FAIL — `ImportError`

- [ ] **Step 3: Implement pipeline.py**

```python
from irish_statute_assistant.agents.supervisor import Supervisor
from irish_statute_assistant.config import Config
from irish_statute_assistant.memory.session_memory import SessionMemory
from irish_statute_assistant.models.schemas import WriterOutput


class Pipeline:
    def __init__(self, config: Config) -> None:
        self._supervisor = Supervisor(config)
        self._memory = SessionMemory()

    def query(self, user_query: str) -> WriterOutput | str:
        """
        Submit a query. Returns:
          - str: a clarifying question (do not update memory)
          - WriterOutput: the final answer (update memory)
        """
        history = self._memory.format_for_prompt()
        result = self._supervisor.run(query=user_query, history=history)

        if isinstance(result, WriterOutput):
            self._memory.add_exchange(
                user=user_query,
                assistant=result.short_answer,
            )

        return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_pipeline.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/irish_statute_assistant/pipeline.py tests/test_pipeline.py
git commit -m "feat: pipeline orchestrator with session memory integration"
```

---

### Task 14: CLI Entry Point

**Files:**
- Create: `src/irish_statute_assistant/main.py`

- [ ] **Step 1: Implement main.py**

```python
import sys
from irish_statute_assistant.config import Config
from irish_statute_assistant.models.schemas import WriterOutput
from irish_statute_assistant.pipeline import Pipeline


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
        *[f"  - {clause}" for clause in bd.key_clauses],
        "",
        "Things to be aware of:",
        *[f"  - {caveat}" for caveat in bd.caveats],
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    config = Config()
    pipeline = Pipeline(config)

    print("Irish Statute Research Assistant")
    print("Type your legal question, or 'quit' to exit.\n")

    while True:
        try:
            user_input = input("Your question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            sys.exit(0)

        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye.")
            break

        if not user_input:
            continue

        try:
            result = pipeline.query(user_input)
            print(format_output(result))
        except Exception as e:
            print(f"\nSomething went wrong: {e}\nPlease try again or rephrase your question.\n")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Test the CLI manually**

Create a `.env` file from `.env.example` and add your Anthropic API key. Then run:

```bash
python -m irish_statute_assistant.main
```

Try: `What is the statute of limitations for a personal injury claim in Ireland?`

Expected: A plain English answer with relevant Acts cited.

- [ ] **Step 3: Commit**

```bash
git add src/irish_statute_assistant/main.py
git commit -m "feat: CLI entry point for interactive legal Q&A"
```

---

### Task 15: Full Test Suite Run

- [ ] **Step 1: Run all tests**

Run: `pytest tests/ -v`
Expected: All tests PASS

- [ ] **Step 2: If any tests fail, fix the underlying issue**

Do not skip or mock around failures — diagnose and fix root cause.

- [ ] **Step 3: Run tests with coverage (optional)**

Run: `pytest tests/ -v --tb=short`
Expected: Clear summary with no failures.

- [ ] **Step 4: Final commit**

```bash
git add -A
git commit -m "feat: complete Irish Statute Research Assistant — all tests passing"
```

---

## Running the System

```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY

# Run the CLI
python -m irish_statute_assistant.main

# Run tests
pytest tests/ -v
```
