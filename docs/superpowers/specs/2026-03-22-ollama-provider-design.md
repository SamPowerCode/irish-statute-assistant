# Ollama LLM Provider — Design Spec

**Date:** 2026-03-22
**Status:** Approved

---

## Overview

Add Ollama as a supported LLM provider so the Irish Statute Assistant can run entirely locally using models served by a local Ollama instance. No API key is required; the user specifies a model name and optionally a base URL.

---

## Goals

- Support `LLM_PROVIDER=ollama` alongside the existing providers (Anthropic, OpenAI, Google, Groq)
- Allow the Ollama base URL to be configured via `OLLAMA_BASE_URL` (default: `http://localhost:11434`)
- Require the user to explicitly set `MODEL_NAME` (no built-in default, since Ollama models vary per installation)
- Raise a clear error if `MODEL_NAME` is not set when using the Ollama provider

## Non-Goals

- Advanced Ollama options (keep-alive, timeouts, GPU layers) — not in scope
- A UI or auto-discovery of locally available models

---

## Design

### Files Changed

#### `src/irish_statute_assistant/llm.py`

- Add `"ollama": ""` to `_DEFAULT_MODELS` (empty string signals "no default")
- Add an `ollama` branch to `get_llm()`:

```python
elif config.llm_provider == "ollama":
    from langchain_ollama import ChatOllama
    return ChatOllama(
        model=config.model_name,
        base_url=config.ollama_base_url,
        temperature=config.temperature,
        num_predict=max_tokens,
    )
```

#### `src/irish_statute_assistant/config.py`

- Add `"ollama"` to the `llm_provider` Literal type
- Add `"ollama": ""` to `_PROVIDER_KEY_MAP` (empty string = no key field)
- Add `ollama_base_url: str = "http://localhost:11434"`
- Update `check_provider_and_set_model` validator:
  - If `llm_provider == "ollama"` and `model_name` is empty, raise `ValueError` with a clear message
  - Skip the default-model fallback and API key check for Ollama
  - For all other providers, existing logic is unchanged

#### `pyproject.toml`

- Add `langchain-ollama` to the project dependencies

### Validation Logic (updated)

```python
@model_validator(mode="after")
def check_provider_and_set_model(self) -> "Config":
    if self.llm_provider == "ollama":
        if not self.model_name:
            raise ValueError(
                "MODEL_NAME is required when LLM_PROVIDER=ollama "
                "(e.g. MODEL_NAME=llama3.2)"
            )
        return self
    # Non-Ollama providers: set default model and validate API key
    if not self.model_name:
        self.model_name = _DEFAULT_MODELS[self.llm_provider]
    key_field = _PROVIDER_KEY_MAP[self.llm_provider]
    if not getattr(self, key_field):
        raise ValueError(
            f"{key_field.upper()} is required when LLM_PROVIDER={self.llm_provider!r}"
        )
    return self
```

---

## Usage

```bash
# Basic usage
LLM_PROVIDER=ollama MODEL_NAME=llama3.2 uv run python -m irish_statute_assistant.main

# Custom Ollama host
LLM_PROVIDER=ollama MODEL_NAME=mistral OLLAMA_BASE_URL=http://192.168.1.10:11434 \
  uv run python -m irish_statute_assistant.main
```

Or in `.env`:
```
LLM_PROVIDER=ollama
MODEL_NAME=llama3.2
OLLAMA_BASE_URL=http://localhost:11434
```

---

## Testing

Update existing provider tests in `tests/` to cover:

- `Config` with `llm_provider="ollama"` and a model name set — should pass validation
- `Config` with `llm_provider="ollama"` and no model name — should raise `ValueError`
- `get_llm()` with Ollama config — mock `langchain_ollama.ChatOllama`, assert called with correct `model`, `base_url`, `temperature`, `num_predict`
- `Config` with `llm_provider="ollama"` and a custom `OLLAMA_BASE_URL` — assert stored correctly

---

## Dependencies

| Package | Purpose |
|---|---|
| `langchain-ollama` | LangChain integration for Ollama (`ChatOllama`) |
