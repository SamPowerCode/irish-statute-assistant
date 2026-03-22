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

- Add `"ollama": "",  # no default; MODEL_NAME required` to `_DEFAULT_MODELS` (empty string signals "no default"; the inline comment prevents future readers from thinking `""` is a valid model name)
- Add an `ollama` branch to `get_llm()` as a new `elif` before the final provider
- Replace the final `else: # anthropic (default)` catch-all with an explicit `elif config.llm_provider == "anthropic"` followed by `raise ValueError(f"Unknown provider: {config.llm_provider!r}")` to prevent silent fallback on misconfiguration

```python
elif config.llm_provider == "ollama":
    from langchain_ollama import ChatOllama
    return ChatOllama(
        model=config.model_name,
        base_url=config.ollama_base_url,
        temperature=config.temperature,
        num_predict=max_tokens,
    )
elif config.llm_provider == "anthropic":
    from langchain_anthropic import ChatAnthropic
    return ChatAnthropic(
        model=config.model_name,
        api_key=config.anthropic_api_key,
        max_tokens=max_tokens,
        temperature=config.temperature,
    )
else:
    raise ValueError(f"Unknown provider: {config.llm_provider!r}")
```

#### `src/irish_statute_assistant/config.py`

- Add `"ollama"` to the `llm_provider` Literal type
- Add `"ollama": None` to `_PROVIDER_KEY_MAP` (`None` = no API key required; avoids the `getattr(self, "")` trap)
- Add `ollama_base_url: str = "http://localhost:11434"`
- Update `check_provider_and_set_model` validator:
  - If `llm_provider == "ollama"` and `model_name` is empty, raise `ValueError` with a clear message
  - Skip the default-model fallback and API key check for Ollama (early return)
  - For all other providers, guard the `getattr` call with `if key_field is not None`

#### `pyproject.toml`

- Add `langchain-ollama>=0.3` to the project dependencies
- After adding, run the full test suite to verify no `langchain-core` version conflict with existing `langchain>=0.3` constraint

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
    if key_field is not None and not getattr(self, key_field):
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

- `test_config_ollama_valid` — `Config` with `llm_provider="ollama"` and `model_name="llama3.2"` passes validation
- `test_config_ollama_no_model` — `Config` with `llm_provider="ollama"` and no `model_name` raises `ValueError`
- `test_config_ollama_defaults` — `Config` with `llm_provider="ollama"` and `model_name` set; assert `ollama_base_url == "http://localhost:11434"`
- `test_config_ollama_custom_url` — `Config` with `llm_provider="ollama"`, `model_name` set, and `ollama_base_url="http://192.168.1.10:11434"`; assert stored correctly
- `test_get_llm_ollama` — mock `langchain_ollama.ChatOllama`; call `get_llm()` with an Ollama config; assert `ChatOllama` called with correct `model`, `base_url`, `temperature`, `num_predict`

**Note on `make_config` helper:** The existing `make_config` helper in `test_llm.py` constructs configs with an API key argument. For Ollama tests, either extend `make_config` to handle the no-key case or construct `Config(...)` inline (no API key arg, `model_name` required).

---

## Dependencies

| Package | Min Version | Purpose |
|---|---|---|
| `langchain-ollama` | `>=0.3` | LangChain integration for Ollama (`ChatOllama`) |

**Compatibility note:** After adding `langchain-ollama>=0.3`, run `uv lock` and verify no `langchain-core` version conflict with the existing `langchain>=0.3` constraint in `pyproject.toml`. Run the full 66-test suite before committing the dependency change.
