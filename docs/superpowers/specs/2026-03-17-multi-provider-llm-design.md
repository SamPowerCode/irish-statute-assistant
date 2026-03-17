# Multi-Provider LLM Support Design

**Date:** 2026-03-17
**Project:** Irish Statute Assistant
**Goal:** Allow users to choose between Anthropic (default), OpenAI, Google, and Groq as their LLM backend via a single config field.

---

## Context

All five agents currently hardcode `ChatAnthropic`. Adding provider choice requires:
- A factory function that returns the right LangChain chat model
- Config fields for provider selection and per-provider API keys
- Fixed and extended token counting (the existing `TokenUsageCallback` silently counts zero tokens for all providers including Anthropic — this spec fixes that bug and extends it to all four providers)

---

## Approach: `get_llm()` Factory (Option A)

A single `get_llm(config, max_tokens)` function in a new `llm.py` file. Each agent replaces its `ChatAnthropic(...)` call with `get_llm(config, max_tokens=N)`. All other agent logic is unchanged.

---

## Section 1: Config

**File:** `src/irish_statute_assistant/config.py`

### New fields

```python
llm_provider: Literal["anthropic", "openai", "google", "groq"] = "anthropic"
openai_api_key: str = ""
google_api_key: str = ""
groq_api_key: str = ""
```

`anthropic_api_key` changes from required to `str = ""`.

`model_name` changes from `str = "claude-sonnet-4-6"` to `str = ""`. The `model_validator` always sets it from `_DEFAULT_MODELS` when empty, so the effective default per provider is clear and there is no misleading class-level default.

### Default models

`_DEFAULT_MODELS` is the single canonical source of truth, defined in `llm.py` and imported by `config.py`:

| Provider  | Default model              |
|-----------|---------------------------|
| anthropic | `claude-sonnet-4-6`        |
| openai    | `gpt-4o`                   |
| google    | `gemini-2.0-flash`         |
| groq      | `llama-3.3-70b-versatile`  |

### Key validation and model defaulting

```python
from irish_statute_assistant.llm import _DEFAULT_MODELS

_PROVIDER_KEY_MAP = {
    "anthropic": "anthropic_api_key",
    "openai":    "openai_api_key",
    "google":    "google_api_key",
    "groq":      "groq_api_key",
}

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

The validator uses `self.model_name` (empty string = not set) rather than `model_fields_set`, which is simpler and equally correct given the field default is now `""`.

---

## Section 2: `get_llm()` Factory

**New file:** `src/irish_statute_assistant/llm.py`

`_DEFAULT_MODELS` lives here and is imported by `config.py` — single source of truth.

`llm.py` uses `TYPE_CHECKING` to avoid a circular import (`config.py` imports `_DEFAULT_MODELS` from `llm.py`; `llm.py` needs `Config` for type annotations). At runtime, `Config` is never imported by `llm.py`; the annotation is only evaluated by type checkers.

```python
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from irish_statute_assistant.config import Config

_DEFAULT_MODELS = {
    "anthropic": "claude-sonnet-4-6",
    "openai":    "gpt-4o",
    "google":    "gemini-2.0-flash",
    "groq":      "llama-3.3-70b-versatile",
}


def get_llm(config: Config, max_tokens: int):
    """Return the LangChain chat model for the configured provider."""
    if config.llm_provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=config.model_name,
            api_key=config.openai_api_key,
            max_tokens=max_tokens,
        )
    elif config.llm_provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        # Note: Google uses max_output_tokens, not max_tokens
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

Provider imports are lazy (inside the `if` branches) so missing optional packages only fail at runtime if that provider is actually selected.

---

## Section 3: Agent Changes

Five agents need one-line changes each:

| File | Change |
|------|--------|
| `agents/analyst.py` | Remove `ChatAnthropic` import; use `get_llm(config, max_tokens=1024)` |
| `agents/clarifier.py` | Remove `ChatAnthropic` import; use `get_llm(config, max_tokens=256)` |
| `agents/evaluator.py` | Remove `ChatAnthropic` import; use `get_llm(config, max_tokens=512)` |
| `agents/writer.py` | Remove `ChatAnthropic` import; use `get_llm(config, max_tokens=2048)` |
| `agents/researcher.py` | No LLM — no change needed |

Pattern (same for all four):
```python
# Before
from langchain_anthropic import ChatAnthropic
llm = ChatAnthropic(model=config.model_name, api_key=config.anthropic_api_key, max_tokens=N)

# After
from irish_statute_assistant.llm import get_llm
llm = get_llm(config, max_tokens=N)
```

`.with_structured_output(Schema)` is chained identically after the factory call — no change to the chain construction.

---

## Section 4: Token Counting

**File:** `src/irish_statute_assistant/agents/base_agent.py`

### Existing bug

The current `TokenUsageCallback.on_llm_end` reads from `generation_info["input_tokens"]` / `generation_info["output_tokens"]`. This is incorrect — `ChatAnthropic` places token counts on `message.usage_metadata`, not in `generation_info`. As a result, the existing callback silently counts zero tokens for every call, meaning `QueryContext` budget enforcement never actually fires. This spec fixes that bug.

### Correct approach: `message.usage_metadata`

All four providers in scope (Anthropic, OpenAI, Groq, Google) set `usage_metadata` on the `AIMessage` inside each `ChatGeneration`. LangChain normalises the keys to `input_tokens`, `output_tokens`, and `total_tokens` across providers. Reading `total_tokens` from `gen.message.usage_metadata` is the universal approach:

```python
def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
    for generation in response.generations:
        for gen in generation:
            msg = getattr(gen, "message", None)
            meta = getattr(msg, "usage_metadata", None) if msg else None
            if meta:
                self.total_tokens += meta.get("total_tokens", 0)
    if self.total_tokens == 0:
        logger.debug("TokenUsageCallback: no token counts found in response — budget enforcement inactive for this call")
```

The `logger.debug` line makes silent zero-counting observable without being noisy in production.

---

## Section 5: Dependencies

**File:** `pyproject.toml`

Three new packages added to `dependencies`:

```toml
"langchain-openai>=0.2",
"langchain-google-genai>=2.0",
"langchain-groq>=0.2",
```

---

## Section 6: Tests

### `tests/test_config.py`

- `test_config_loads_defaults` — **update** to use `Config(anthropic_api_key="test-key")` which still works (provider defaults to `"anthropic"`, key is set); also assert `model_name == "claude-sonnet-4-6"` (set by validator from `_DEFAULT_MODELS`)
- `test_config_default_model_per_provider` — new: verify `model_name` is set to the correct default for each provider when `MODEL_NAME` env var is not given
- `test_config_missing_provider_key_raises` — new: verify `ValidationError` is raised when the chosen provider's key is empty (parametrised over all four providers)
- `test_config_requires_api_key` — already updated to use `Config(_env_file=None)`; remains valid since Anthropic is the default provider and no Anthropic key is set

### `tests/test_llm.py` (new)

- `test_get_llm_returns_correct_class_for_each_provider` — for each provider, construct a minimal `Config` with the provider's key set, call `get_llm()`, assert it returns the right LangChain class. Patch the four LangChain constructors to avoid importing uninstalled packages.

### Existing tests

All existing agent, supervisor, pipeline, researcher, statute fetcher, and vector store tests are unaffected — they mock `_chain` directly.

---

## Section 7: README

New "LLM providers" section documenting:
- The four supported providers with their default models
- Required env var for each
- Example `.env` snippets for switching providers

---

## Files Changed

| File | Change |
|------|--------|
| `src/irish_statute_assistant/llm.py` | **New** — `get_llm()` factory and `_DEFAULT_MODELS` |
| `src/irish_statute_assistant/config.py` | Add provider + API key fields; model validator; import `_DEFAULT_MODELS` from `llm.py` |
| `src/irish_statute_assistant/agents/base_agent.py` | Fix token counting bug; extend to all four providers via `message.usage_metadata` |
| `src/irish_statute_assistant/agents/analyst.py` | Use `get_llm()` |
| `src/irish_statute_assistant/agents/clarifier.py` | Use `get_llm()` |
| `src/irish_statute_assistant/agents/evaluator.py` | Use `get_llm()` |
| `src/irish_statute_assistant/agents/writer.py` | Use `get_llm()` |
| `pyproject.toml` | Add three LangChain provider packages |
| `tests/test_config.py` | Update `test_config_loads_defaults`; add provider validation tests |
| `tests/test_llm.py` | **New** — factory tests |
| `README.md` | Add LLM providers section |

## What Does Not Change

- All agent `run()` signatures
- All Pydantic schemas
- All vector store code
- All supervisor/pipeline/retry/budget logic
- All existing tests (agent, supervisor, pipeline, researcher, statute fetcher, vector store)

## Known Limitations

- Token budget enforcement was silently broken before this spec (zero counts were recorded for all calls). This spec fixes it. Any existing deployments relying on the budget not firing will see it start enforcing after this change.
- If a future provider does not set `usage_metadata` on its `AIMessage`, token counting will silently return 0 for that provider. A `logger.debug` message is emitted in that case.
