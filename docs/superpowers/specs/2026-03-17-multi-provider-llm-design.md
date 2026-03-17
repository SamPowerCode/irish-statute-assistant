# Multi-Provider LLM Support Design

**Date:** 2026-03-17
**Project:** Irish Statute Assistant
**Goal:** Allow users to choose between Anthropic (default), OpenAI, Google, and Groq as their LLM backend via a single config field.

---

## Context

All five agents currently hardcode `ChatAnthropic`. Adding provider choice requires:
- A factory function that returns the right LangChain chat model
- Config fields for provider selection and per-provider API keys
- Updated token counting to handle each provider's response format

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

### Default models

A `model_validator` sets `model_name` to a provider-appropriate default when the user has not set `MODEL_NAME` explicitly:

| Provider  | Default model              |
|-----------|---------------------------|
| anthropic | `claude-sonnet-4-6`        |
| openai    | `gpt-4o`                   |
| google    | `gemini-2.0-flash`         |
| groq      | `llama-3.3-70b-versatile`  |

### Key validation

The same `model_validator` raises `ValueError` (surfaced as `ValidationError`) if the API key for the chosen provider is empty:

```python
_PROVIDER_KEY_MAP = {
    "anthropic": "anthropic_api_key",
    "openai":    "openai_api_key",
    "google":    "google_api_key",
    "groq":      "groq_api_key",
}

@model_validator(mode="after")
def check_provider_key(self) -> "Config":
    key_field = _PROVIDER_KEY_MAP[self.llm_provider]
    if not getattr(self, key_field):
        raise ValueError(
            f"{key_field.upper()} is required when LLM_PROVIDER={self.llm_provider!r}"
        )
    return self
```

---

## Section 2: `get_llm()` Factory

**New file:** `src/irish_statute_assistant/llm.py`

```python
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

`TokenUsageCallback.on_llm_end` currently only reads Anthropic's token format. Updated to try all four provider formats:

| Provider | Location | Keys |
|----------|----------|------|
| OpenAI / Groq | `response.llm_output["token_usage"]` | `prompt_tokens`, `completion_tokens` |
| Anthropic | `generation_info` | `input_tokens`, `output_tokens` |
| Google | `generation_info["usage_metadata"]` | `total_token_count` |

```python
def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
    # OpenAI / Groq: token_usage in llm_output
    if response.llm_output:
        usage = response.llm_output.get("token_usage", {})
        if usage:
            self.total_tokens += usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)
            return
    # Anthropic / Google: check generation_info
    for generation in response.generations:
        for gen in generation:
            info = getattr(gen, "generation_info", {}) or {}
            self.total_tokens += info.get("input_tokens", 0) + info.get("output_tokens", 0)
            meta = info.get("usage_metadata", {}) or {}
            self.total_tokens += meta.get("total_token_count", 0)
```

If no format matches, `total_tokens` stays 0 — budget enforcement silently skips. Nothing breaks.

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

New tests:
- `test_config_default_model_per_provider` — verify default model is set correctly for each provider when `MODEL_NAME` is not given
- `test_config_missing_provider_key_raises` — verify `ValidationError` is raised when the chosen provider's key is empty
- `test_config_anthropic_still_works` — verify existing Anthropic config behaviour is unchanged

### `tests/test_llm.py` (new)

- `test_get_llm_returns_correct_class_for_each_provider` — mock config per provider, assert `get_llm()` returns the right LangChain class
- No real API calls — patch the LangChain constructors

### Existing tests

No changes required. Agent tests mock `_chain` directly and are unaffected by what built it.

---

## Section 7: README

New "LLM providers" section documenting:
- The four supported providers
- Default model for each
- Required env var for each
- Example `.env` snippets for switching providers

---

## Files Changed

| File | Change |
|------|--------|
| `src/irish_statute_assistant/llm.py` | **New** — `get_llm()` factory |
| `src/irish_statute_assistant/config.py` | Add provider + API key fields; model validator |
| `src/irish_statute_assistant/agents/base_agent.py` | Multi-provider token counting |
| `src/irish_statute_assistant/agents/analyst.py` | Use `get_llm()` |
| `src/irish_statute_assistant/agents/clarifier.py` | Use `get_llm()` |
| `src/irish_statute_assistant/agents/evaluator.py` | Use `get_llm()` |
| `src/irish_statute_assistant/agents/writer.py` | Use `get_llm()` |
| `pyproject.toml` | Add three LangChain provider packages |
| `tests/test_config.py` | Add provider validation tests |
| `tests/test_llm.py` | **New** — factory tests |
| `README.md` | Add LLM providers section |

## What Does Not Change

- All agent `run()` signatures
- All Pydantic schemas
- All vector store code
- All supervisor/pipeline/retry/budget logic
- All existing tests (agent, supervisor, pipeline, researcher, statute fetcher, vector store)
