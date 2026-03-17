# Multi-Provider LLM Support Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Allow users to select Anthropic, OpenAI, Google, or Groq as their LLM provider via a single `LLM_PROVIDER` env var, with sensible model defaults and per-provider API key validation.

**Architecture:** A new `llm.py` module holds a `get_llm(config, max_tokens)` factory and the canonical `_DEFAULT_MODELS` dict. `config.py` imports `_DEFAULT_MODELS` from `llm.py` (using `TYPE_CHECKING` in `llm.py` to avoid a circular import). The four LLM-using agents each swap one line. Token counting in `BaseAgent` is fixed to read from `message.usage_metadata`, which all four providers populate.

**Tech Stack:** LangChain (`langchain-anthropic`, `langchain-openai`, `langchain-google-genai`, `langchain-groq`), pydantic-settings v2, pytest

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `src/irish_statute_assistant/llm.py` | **Create** | `_DEFAULT_MODELS` dict + `get_llm()` factory |
| `src/irish_statute_assistant/config.py` | **Modify** | Add `llm_provider`, API key fields, model validator |
| `src/irish_statute_assistant/agents/base_agent.py` | **Modify** | Fix `on_llm_end` to use `message.usage_metadata` |
| `src/irish_statute_assistant/agents/analyst.py` | **Modify** | Replace `ChatAnthropic` with `get_llm()` |
| `src/irish_statute_assistant/agents/clarifier.py` | **Modify** | Replace `ChatAnthropic` with `get_llm()` |
| `src/irish_statute_assistant/agents/evaluator.py` | **Modify** | Replace `ChatAnthropic` with `get_llm()` |
| `src/irish_statute_assistant/agents/writer.py` | **Modify** | Replace `ChatAnthropic` with `get_llm()` |
| `pyproject.toml` | **Modify** | Add three LangChain provider packages |
| `tests/test_llm.py` | **Create** | Factory tests (one per provider) |
| `tests/test_config.py` | **Modify** | Update defaults test; add provider validation tests |
| `README.md` | **Modify** | Add LLM providers section |

---

## Chunk 1: `llm.py` factory + dependencies

### Task 1: Add provider packages to `pyproject.toml` and sync

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add the three new packages**

In `pyproject.toml`, add to `dependencies` (after the existing `langchain-chroma` line):

```toml
    "langchain-openai>=0.2",
    "langchain-google-genai>=2.0",
    "langchain-groq>=0.2",
```

- [ ] **Step 2: Sync the environment**

```bash
uv sync
```

Expected: packages install without errors.

- [ ] **Step 3: Verify packages are importable**

```bash
uv run python -c "import langchain_openai, langchain_google_genai, langchain_groq; print('ok')"
```

Expected: `ok`

---

### Task 2: Write failing tests for `get_llm()` factory

**Files:**
- Create: `tests/test_llm.py`

- [ ] **Step 1: Write the tests**

```python
"""Tests for the get_llm() provider factory."""
import pytest
from unittest.mock import MagicMock, patch


def make_config(provider: str, api_key: str, model: str = ""):
    import os
    os.environ.setdefault("ANTHROPIC_API_KEY", "placeholder")
    from irish_statute_assistant.config import Config

    key_map = {
        "anthropic": {"anthropic_api_key": api_key},
        "openai":    {"openai_api_key": api_key},
        "google":    {"google_api_key": api_key},
        "groq":      {"groq_api_key": api_key},
    }
    kwargs = {"llm_provider": provider, **key_map[provider]}
    if model:
        kwargs["model_name"] = model
    return Config(**kwargs)


@pytest.mark.parametrize("provider,patch_target,expected_kwarg", [
    ("anthropic", "langchain_anthropic.ChatAnthropic",          "max_tokens"),
    ("openai",    "langchain_openai.ChatOpenAI",                "max_tokens"),
    ("google",    "langchain_google_genai.ChatGoogleGenerativeAI", "max_output_tokens"),
    ("groq",      "langchain_groq.ChatGroq",                    "max_tokens"),
])
def test_get_llm_returns_correct_class(provider, patch_target, expected_kwarg):
    from irish_statute_assistant.llm import get_llm
    config = make_config(provider, "test-key")
    with patch(patch_target) as MockLLM:
        MockLLM.return_value = MagicMock()
        result = get_llm(config, max_tokens=256)
    assert MockLLM.called
    assert result is MockLLM.return_value
    call_kwargs = MockLLM.call_args.kwargs
    assert expected_kwarg in call_kwargs
    assert call_kwargs[expected_kwarg] == 256


def test_get_llm_passes_model_name():
    from irish_statute_assistant.llm import get_llm
    config = make_config("openai", "test-key", model="gpt-4-turbo")
    with patch("langchain_openai.ChatOpenAI") as MockLLM:
        MockLLM.return_value = MagicMock()
        get_llm(config, max_tokens=100)
    assert MockLLM.call_args.kwargs["model"] == "gpt-4-turbo"


def test_get_llm_passes_api_key():
    from irish_statute_assistant.llm import get_llm
    config = make_config("groq", "my-groq-key")
    with patch("langchain_groq.ChatGroq") as MockLLM:
        MockLLM.return_value = MagicMock()
        get_llm(config, max_tokens=100)
    assert MockLLM.call_args.kwargs["api_key"] == "my-groq-key"
```

- [ ] **Step 2: Run tests to confirm they fail (file not found)**

```bash
uv run pytest tests/test_llm.py -v
```

Expected: `ImportError` or `ModuleNotFoundError` — `llm.py` doesn't exist yet.

---

### Task 3: Implement `llm.py`

**Files:**
- Create: `src/irish_statute_assistant/llm.py`

- [ ] **Step 1: Create the file**

```python
"""LLM provider factory for Irish Statute Assistant."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from irish_statute_assistant.config import Config

# Single source of truth for default models per provider.
# Imported by config.py — do not duplicate this dict.
_DEFAULT_MODELS: dict[str, str] = {
    "anthropic": "claude-sonnet-4-6",
    "openai":    "gpt-4o",
    "google":    "gemini-2.0-flash",
    "groq":      "llama-3.3-70b-versatile",
}


def get_llm(config: Config, max_tokens: int) -> Any:
    """Return the LangChain chat model for the configured provider.

    Imports are lazy so missing optional packages only fail if that
    provider is actually selected.
    """
    if config.llm_provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=config.model_name,
            api_key=config.openai_api_key,
            max_tokens=max_tokens,
        )
    elif config.llm_provider == "google":
        # Google uses max_output_tokens, not max_tokens
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=config.model_name,
            google_api_key=config.google_api_key,
            max_output_tokens=max_tokens,
        )
    elif config.llm_provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(
            model=config.model_name,
            api_key=config.groq_api_key,
            max_tokens=max_tokens,
        )
    else:  # anthropic (default)
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=config.model_name,
            api_key=config.anthropic_api_key,
            max_tokens=max_tokens,
        )
```

- [ ] **Step 2: Run the factory tests**

```bash
uv run pytest tests/test_llm.py -v
```

Expected: all 6 tests pass.

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml uv.lock src/irish_statute_assistant/llm.py tests/test_llm.py
git commit -m "feat: add get_llm() provider factory and langchain provider dependencies"
```

---

## Chunk 2: Config changes

### Task 4: Write failing config tests

**Files:**
- Modify: `tests/test_config.py`

- [ ] **Step 1: Add the new tests** (append to the existing file)

```python
import pytest
from pydantic import ValidationError
from irish_statute_assistant.config import Config
from irish_statute_assistant.llm import _DEFAULT_MODELS


# --- existing tests remain unchanged above this line ---


@pytest.mark.parametrize("provider,key_kwarg,expected_model", [
    ("anthropic", "anthropic_api_key", "claude-sonnet-4-6"),
    ("openai",    "openai_api_key",    "gpt-4o"),
    ("google",    "google_api_key",    "gemini-2.0-flash"),
    ("groq",      "groq_api_key",      "llama-3.3-70b-versatile"),
])
def test_config_default_model_per_provider(provider, key_kwarg, expected_model, monkeypatch):
    monkeypatch.delenv("MODEL_NAME", raising=False)
    config = Config(**{"llm_provider": provider, key_kwarg: "test-key"}, _env_file=None)
    assert config.model_name == expected_model
    assert config.model_name == _DEFAULT_MODELS[provider]


@pytest.mark.parametrize("provider,key_kwarg", [
    ("anthropic", "anthropic_api_key"),
    ("openai",    "openai_api_key"),
    ("google",    "google_api_key"),
    ("groq",      "groq_api_key"),
])
def test_config_missing_provider_key_raises(provider, key_kwarg, monkeypatch):
    monkeypatch.delenv(key_kwarg.upper(), raising=False)
    with pytest.raises(ValidationError, match=key_kwarg.upper()):
        Config(llm_provider=provider, _env_file=None)


def test_config_explicit_model_name_not_overridden(monkeypatch):
    """When MODEL_NAME is set explicitly, the validator must not overwrite it."""
    config = Config(
        anthropic_api_key="test",
        model_name="claude-opus-4-6",
        _env_file=None,
    )
    assert config.model_name == "claude-opus-4-6"
```

- [ ] **Step 2: Run the new tests to confirm they fail**

```bash
uv run pytest tests/test_config.py::test_config_default_model_per_provider \
              tests/test_config.py::test_config_missing_provider_key_raises \
              tests/test_config.py::test_config_explicit_model_name_not_overridden -v
```

Expected: failures — the new fields and validator don't exist yet.

---

### Task 5: Implement config changes

**Files:**
- Modify: `src/irish_statute_assistant/config.py`

- [ ] **Step 1: Replace the file contents**

```python
from typing import Literal

from pydantic import model_validator
from pydantic_settings import BaseSettings

from irish_statute_assistant.llm import _DEFAULT_MODELS

_PROVIDER_KEY_MAP = {
    "anthropic": "anthropic_api_key",
    "openai":    "openai_api_key",
    "google":    "google_api_key",
    "groq":      "groq_api_key",
}


class Config(BaseSettings):
    anthropic_api_key: str = ""
    model_name: str = ""
    llm_provider: Literal["anthropic", "openai", "google", "groq"] = "anthropic"
    openai_api_key: str = ""
    google_api_key: str = ""
    groq_api_key: str = ""
    evaluator_pass_threshold: float = 0.7
    max_refinement_rounds: int = 2
    max_retries: int = 3
    token_budget_per_query: int = 4000
    rate_limit_delay: float = 1.0
    # Vector store
    vector_store_backend: Literal["chroma", "qdrant"] = "chroma"
    chroma_db_path: str = "./data/chroma"
    qdrant_url: str = ""
    qdrant_api_key: str = ""
    embedding_model: str = "all-MiniLM-L6-v2"
    index_categories: list[str] = [
        "employment", "housing", "family", "criminal", "contract",
        "personal injury", "planning", "company", "tax", "consumer",
    ]
    acts_per_category: int = 5

    model_config = {"env_file": ".env"}

    @model_validator(mode="after")
    def check_provider_and_set_model(self) -> "Config":
        # Set default model if not explicitly provided
        if not self.model_name:
            self.model_name = _DEFAULT_MODELS[self.llm_provider]
        # Validate that the required API key for the chosen provider is set
        key_field = _PROVIDER_KEY_MAP[self.llm_provider]
        if not getattr(self, key_field):
            raise ValueError(
                f"{key_field.upper()} is required when LLM_PROVIDER={self.llm_provider!r}"
            )
        return self
```

- [ ] **Step 2: Run the full test suite**

```bash
uv run pytest tests/ -v
```

Expected: all tests pass. The existing `test_config_loads_defaults` passes because `Config(anthropic_api_key="test-key")` triggers the validator which sets `model_name = "claude-sonnet-4-6"`.

If `test_config_loads_defaults` fails, check: it currently asserts `config.model_name == "claude-sonnet-4-6"` — this will still be true because the validator sets it from `_DEFAULT_MODELS["anthropic"]`.

- [ ] **Step 3: Commit**

```bash
git add src/irish_statute_assistant/config.py tests/test_config.py
git commit -m "feat: add multi-provider config fields and model validator"
```

---

## Chunk 3: Token counting fix + agent updates

### Task 6: Write failing test for `TokenUsageCallback`

**Files:**
- Create: `tests/test_base_agent.py`

- [ ] **Step 1: Write the test**

```python
"""Tests for BaseAgent token counting via TokenUsageCallback."""
from unittest.mock import MagicMock
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from irish_statute_assistant.agents.base_agent import TokenUsageCallback


def _make_result(total_tokens: int) -> LLMResult:
    """Build a minimal LLMResult with usage_metadata on the AIMessage."""
    msg = AIMessage(
        content="test response",
        usage_metadata={"input_tokens": 10, "output_tokens": total_tokens - 10, "total_tokens": total_tokens},
    )
    gen = ChatGeneration(message=msg)
    return LLMResult(generations=[[gen]])


def test_callback_counts_tokens_from_usage_metadata():
    cb = TokenUsageCallback()
    result = _make_result(total_tokens=42)
    cb.on_llm_end(result)
    assert cb.total_tokens == 42


def test_callback_accumulates_across_calls():
    cb = TokenUsageCallback()
    cb.on_llm_end(_make_result(total_tokens=30))
    cb.on_llm_end(_make_result(total_tokens=20))
    assert cb.total_tokens == 50


def test_callback_returns_zero_when_no_metadata():
    """No usage_metadata → total_tokens stays 0."""
    cb = TokenUsageCallback()
    msg = AIMessage(content="test")  # no usage_metadata
    gen = ChatGeneration(message=msg)
    result = LLMResult(generations=[[gen]])
    cb.on_llm_end(result)
    assert cb.total_tokens == 0


def test_last_token_count_property():
    from irish_statute_assistant.agents.base_agent import BaseAgent
    from unittest.mock import MagicMock, patch

    agent = BaseAgent()
    mock_chain = MagicMock()

    # Simulate _invoke_chain with token callback returning 55 tokens
    with patch.object(BaseAgent, "_invoke_chain", wraps=agent._invoke_chain):
        # Patch the callback inside _invoke_chain by patching TokenUsageCallback
        with patch("irish_statute_assistant.agents.base_agent.TokenUsageCallback") as MockCB:
            mock_cb = MagicMock()
            mock_cb.total_tokens = 55
            MockCB.return_value = mock_cb
            mock_chain.invoke.return_value = "result"
            agent._invoke_chain(mock_chain, {"key": "value"})

    assert agent.last_token_count == 55
```

- [ ] **Step 2: Run to confirm failure**

```bash
uv run pytest tests/test_base_agent.py -v
```

Expected: `test_callback_counts_tokens_from_usage_metadata` and `test_callback_accumulates_across_calls` fail — current implementation reads `generation_info`, not `message.usage_metadata`.

---

### Task 7: Fix `TokenUsageCallback.on_llm_end`

**Files:**
- Modify: `src/irish_statute_assistant/agents/base_agent.py`

- [ ] **Step 1: Update `base_agent.py`**

Replace the entire file:

```python
"""BaseAgent with token-usage tracking via LangChain callback."""
from __future__ import annotations

import logging
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from langchain_core.runnables import Runnable, RunnableConfig

logger = logging.getLogger(__name__)


class TokenUsageCallback(BaseCallbackHandler):
    """Tracks token usage from LLM responses across all supported providers.

    All four providers (Anthropic, OpenAI, Groq, Google) set usage_metadata
    on the AIMessage inside each ChatGeneration with normalised keys:
    input_tokens, output_tokens, total_tokens.
    """

    def __init__(self) -> None:
        super().__init__()
        self.total_tokens: int = 0

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        for generation in response.generations:
            for gen in generation:
                msg = getattr(gen, "message", None)
                meta = getattr(msg, "usage_metadata", None) if msg else None
                if meta:
                    self.total_tokens += meta.get("total_tokens", 0)
        if self.total_tokens == 0:
            logger.debug(
                "TokenUsageCallback: no token counts found in response "
                "— budget enforcement inactive for this call"
            )


class BaseAgent:
    _last_token_count: int = 0

    def _invoke_chain(self, chain: Runnable, inputs: dict) -> Any:
        """Invoke chain, capture token usage, return result."""
        tracker = TokenUsageCallback()
        result = chain.invoke(inputs, config=RunnableConfig(callbacks=[tracker]))
        self._last_token_count = tracker.total_tokens
        return result

    @property
    def last_token_count(self) -> int:
        return self._last_token_count
```

- [ ] **Step 2: Run the base agent tests**

```bash
uv run pytest tests/test_base_agent.py -v
```

Expected: all 4 tests pass.

- [ ] **Step 3: Run the full suite to check no regressions**

```bash
uv run pytest tests/ -v
```

Expected: all tests pass.

- [ ] **Step 4: Commit**

```bash
git add src/irish_statute_assistant/agents/base_agent.py tests/test_base_agent.py
git commit -m "fix: correct token counting — read from message.usage_metadata for all providers"
```

---

### Task 8: Update the four agents to use `get_llm()`

**Files:**
- Modify: `src/irish_statute_assistant/agents/analyst.py`
- Modify: `src/irish_statute_assistant/agents/clarifier.py`
- Modify: `src/irish_statute_assistant/agents/evaluator.py`
- Modify: `src/irish_statute_assistant/agents/writer.py`

The four existing agent tests mock `self._chain` directly, so they are unaffected by this change.

- [ ] **Step 1: Update `analyst.py`**

Replace the import and LLM construction lines:

```python
# Remove:
from langchain_anthropic import ChatAnthropic

# Add:
from irish_statute_assistant.llm import get_llm
```

And in `__init__`:
```python
# Remove:
llm = ChatAnthropic(
    model=config.model_name,
    api_key=config.anthropic_api_key,
    max_tokens=1024,
).with_structured_output(AnalystOutput)

# Replace with:
llm = get_llm(config, max_tokens=1024).with_structured_output(AnalystOutput)
```

- [ ] **Step 2: Update `clarifier.py`**

Same pattern:
```python
# Remove:
from langchain_anthropic import ChatAnthropic
# Add:
from irish_statute_assistant.llm import get_llm
```
```python
# Remove:
llm = ChatAnthropic(
    model=config.model_name,
    api_key=config.anthropic_api_key,
    max_tokens=256,
).with_structured_output(ClarifierOutput)
# Replace with:
llm = get_llm(config, max_tokens=256).with_structured_output(ClarifierOutput)
```

- [ ] **Step 3: Update `evaluator.py`**

Same pattern:
```python
# Remove:
from langchain_anthropic import ChatAnthropic
# Add:
from irish_statute_assistant.llm import get_llm
```
```python
# Remove:
llm = ChatAnthropic(
    model=config.model_name,
    api_key=config.anthropic_api_key,
    max_tokens=512,
).with_structured_output(EvaluatorOutput)
# Replace with:
llm = get_llm(config, max_tokens=512).with_structured_output(EvaluatorOutput)
```

- [ ] **Step 4: Update `writer.py`**

Same pattern:
```python
# Remove:
from langchain_anthropic import ChatAnthropic
# Add:
from irish_statute_assistant.llm import get_llm
```
```python
# Remove:
llm = ChatAnthropic(
    model=config.model_name,
    api_key=config.anthropic_api_key,
    max_tokens=2048,
).with_structured_output(WriterOutput)
# Replace with:
llm = get_llm(config, max_tokens=2048).with_structured_output(WriterOutput)
```

- [ ] **Step 5: Run the full test suite**

```bash
uv run pytest tests/ -v
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/irish_statute_assistant/agents/analyst.py \
        src/irish_statute_assistant/agents/clarifier.py \
        src/irish_statute_assistant/agents/evaluator.py \
        src/irish_statute_assistant/agents/writer.py
git commit -m "feat: replace ChatAnthropic with get_llm() in all agents"
```

---

## Chunk 4: README

### Task 9: Update README

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Add the LLM providers section**

Add the following section to `README.md` after the "Vector store backends" section and before "Project structure":

```markdown
## LLM providers

The assistant supports four LLM providers, selected via `LLM_PROVIDER`:

| Provider | `LLM_PROVIDER` value | Default model | Required env var |
|----------|---------------------|---------------|-----------------|
| Anthropic (default) | `anthropic` | `claude-sonnet-4-6` | `ANTHROPIC_API_KEY` |
| OpenAI | `openai` | `gpt-4o` | `OPENAI_API_KEY` |
| Google | `google` | `gemini-2.0-flash` | `GOOGLE_API_KEY` |
| Groq | `groq` | `llama-3.3-70b-versatile` | `GROQ_API_KEY` |

Set `MODEL_NAME` to override the default model for any provider.

**Example: switch to OpenAI**
```env
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
# MODEL_NAME=gpt-4o  # optional — gpt-4o is the default
```

**Example: switch to Groq**
```env
LLM_PROVIDER=groq
GROQ_API_KEY=gsk_...
MODEL_NAME=llama-3.1-8b-instant  # optional override
```

All providers support structured output (`.with_structured_output()`) which this system relies on. Use capable models (GPT-4+, Gemini 1.5+, Llama 70B+) — smaller models may produce unreliable structured output.
```

Also update the Configuration table to add:

| `LLM_PROVIDER` | `anthropic` | LLM backend: `anthropic`, `openai`, `google`, or `groq` |
| `OPENAI_API_KEY` | `` | Required when `LLM_PROVIDER=openai` |
| `GOOGLE_API_KEY` | `` | Required when `LLM_PROVIDER=google` |
| `GROQ_API_KEY` | `` | Required when `LLM_PROVIDER=groq` |

And update the `ANTHROPIC_API_KEY` row description to: "Required when `LLM_PROVIDER=anthropic` (default)"

- [ ] **Step 2: Final test run**

```bash
uv run pytest tests/ -q
```

Expected: all tests pass.

- [ ] **Step 3: Commit and push**

```bash
git add README.md
git commit -m "docs: add multi-provider LLM documentation to README"
git push
```
