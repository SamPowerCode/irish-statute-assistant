# Ollama LLM Provider Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Ollama as a locally-hosted LLM provider option, selectable via `LLM_PROVIDER=ollama`.

**Architecture:** Three small changes in three files — `pyproject.toml` gets the new dependency, `config.py` gains the `ollama` provider and `ollama_base_url` setting, and `llm.py` gains the Ollama branch in `get_llm()` plus a safety fix to the Anthropic catch-all. All changes are covered by tests before they are written.

**Tech Stack:** `langchain-ollama>=0.3` (`ChatOllama`), `pydantic-settings`, pytest

---

## File Map

| File | Change |
|---|---|
| `pyproject.toml` | Add `langchain-ollama>=0.3` dependency |
| `src/irish_statute_assistant/config.py` | Add `"ollama"` to Literal, `None` sentinel in `_PROVIDER_KEY_MAP`, `ollama_base_url` field, update validator |
| `src/irish_statute_assistant/llm.py` | Add `"ollama": ""` to `_DEFAULT_MODELS`, Ollama branch in `get_llm()`, fix Anthropic catch-all |
| `tests/test_config.py` | Four new Ollama config tests |
| `tests/test_llm.py` | One new Ollama `get_llm()` test, update `make_config` helper |

---

## Task 1: Add the dependency

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add `langchain-ollama>=0.3` to dependencies**

In `pyproject.toml`, add to the `dependencies` list after `langchain-groq>=0.2`:

```toml
"langchain-ollama>=0.3",
```

- [ ] **Step 2: Install and verify no conflicts**

```bash
uv pip install langchain-ollama
```

Expected: installs without `langchain-core` version conflicts. If a conflict is reported, try `langchain-ollama>=0.2` instead and note the version used.

- [ ] **Step 3: Run the full test suite to confirm nothing broke**

```bash
pytest --tb=short -q
```

Expected: all 66 tests pass.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml
git commit -m "feat: add langchain-ollama dependency"
```

---

## Task 2: Config + `_DEFAULT_MODELS` sync (TDD)

**Files:**
- Modify: `tests/test_config.py`
- Modify: `src/irish_statute_assistant/config.py`
- Modify: `src/irish_statute_assistant/llm.py` (only `_DEFAULT_MODELS` — `get_llm()` changes come in Task 3)

> **Important:** `config.py` contains a module-level assertion `assert set(_PROVIDER_KEY_MAP) == set(_DEFAULT_MODELS)`. Both dicts must be updated together in the same commit, or every test will fail at collection time with `AssertionError`.

### Step 1 — Write the four failing tests

- [ ] **Step 1: Add Ollama tests to `tests/test_config.py`**

Add after the last existing test:

```python
# --- Ollama provider ---

def test_config_ollama_valid():
    """Ollama with a model name set passes validation; no API key needed."""
    config = Config(
        llm_provider="ollama",
        model_name="llama3.2",
        _env_file=None,
    )
    assert config.model_name == "llama3.2"


def test_config_ollama_no_model_raises():
    """Ollama without a model name raises a clear ValidationError."""
    with pytest.raises(ValidationError, match="MODEL_NAME is required"):
        Config(llm_provider="ollama", _env_file=None)


def test_config_ollama_defaults():
    """Ollama base URL defaults to localhost:11434."""
    config = Config(
        llm_provider="ollama",
        model_name="llama3.2",
        _env_file=None,
    )
    assert config.ollama_base_url == "http://localhost:11434"


def test_config_ollama_custom_url():
    """OLLAMA_BASE_URL env var is picked up correctly."""
    config = Config(
        llm_provider="ollama",
        model_name="mistral",
        ollama_base_url="http://192.168.1.10:11434",
        _env_file=None,
    )
    assert config.ollama_base_url == "http://192.168.1.10:11434"
```

- [ ] **Step 2: Run the new tests to confirm they fail**

```bash
pytest tests/test_config.py::test_config_ollama_valid \
       tests/test_config.py::test_config_ollama_no_model_raises \
       tests/test_config.py::test_config_ollama_defaults \
       tests/test_config.py::test_config_ollama_custom_url \
       -v
```

Expected: all four FAIL (e.g. `"ollama" is not a valid Literal value`).

### Step 2 — Implement config changes

- [ ] **Step 3a: Add `"ollama": ""` to `_DEFAULT_MODELS` in `src/irish_statute_assistant/llm.py`**

Change:
```python
_DEFAULT_MODELS: dict[str, str] = {
    "anthropic": "claude-sonnet-4-6",
    "openai":    "gpt-4o",
    "google":    "gemini-2.0-flash",
    "groq":      "llama-3.3-70b-versatile",
}
```
To:
```python
_DEFAULT_MODELS: dict[str, str] = {
    "anthropic": "claude-sonnet-4-6",
    "openai":    "gpt-4o",
    "google":    "gemini-2.0-flash",
    "groq":      "llama-3.3-70b-versatile",
    "ollama":    "",  # no default; MODEL_NAME required
}
```

- [ ] **Step 3b: Update `_PROVIDER_KEY_MAP` in `src/irish_statute_assistant/config.py`**

Change:
```python
_PROVIDER_KEY_MAP = {
    "anthropic": "anthropic_api_key",
    "openai":    "openai_api_key",
    "google":    "google_api_key",
    "groq":      "groq_api_key",
}
```
To:
```python
_PROVIDER_KEY_MAP = {
    "anthropic": "anthropic_api_key",
    "openai":    "openai_api_key",
    "google":    "google_api_key",
    "groq":      "groq_api_key",
    "ollama":    None,  # no API key required for local Ollama
}
```

- [ ] **Step 4: Update the `llm_provider` Literal and add `ollama_base_url` field**

Change:
```python
llm_provider: Literal["anthropic", "openai", "google", "groq"] = "anthropic"
```
To:
```python
llm_provider: Literal["anthropic", "openai", "google", "groq", "ollama"] = "anthropic"
```

Add after `groq_api_key`:
```python
ollama_base_url: str = "http://localhost:11434"
```

- [ ] **Step 5: Update the `check_provider_and_set_model` validator**

Replace the entire validator method with:
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

- [ ] **Step 6: Run the four new tests — expect pass**

```bash
pytest tests/test_config.py::test_config_ollama_valid \
       tests/test_config.py::test_config_ollama_no_model_raises \
       tests/test_config.py::test_config_ollama_defaults \
       tests/test_config.py::test_config_ollama_custom_url \
       -v
```

Expected: all four PASS.

- [ ] **Step 7: Run the full config test file — no regressions**

```bash
pytest tests/test_config.py -v
```

Expected: all tests pass.

- [ ] **Step 8: Commit (both dicts in the same commit to keep the sync assertion happy)**

```bash
git add src/irish_statute_assistant/config.py src/irish_statute_assistant/llm.py tests/test_config.py
git commit -m "feat: add ollama provider to Config and _DEFAULT_MODELS"
```

---

## Task 3: LLM factory changes (TDD)

**Files:**
- Modify: `tests/test_llm.py`
- Modify: `src/irish_statute_assistant/llm.py`

### Step 1 — Write the failing test

- [ ] **Step 1: Update `make_config` helper and add Ollama test in `tests/test_llm.py`**

Replace the `make_config` function with:
```python
def make_config(provider: str, api_key: str = "", model: str = ""):
    from irish_statute_assistant.config import Config

    key_map = {
        "anthropic": {"anthropic_api_key": api_key},
        "openai":    {"openai_api_key": api_key},
        "google":    {"google_api_key": api_key},
        "groq":      {"groq_api_key": api_key},
        "ollama":    {},  # no API key
    }
    kwargs = {"llm_provider": provider, **key_map[provider]}
    if model:
        kwargs["model_name"] = model
    return Config(**kwargs, _env_file=None)
```

Add after the last existing test:
```python
def test_get_llm_ollama():
    from irish_statute_assistant.llm import get_llm
    config = make_config("ollama", model="llama3.2")
    with patch("langchain_ollama.ChatOllama") as MockLLM:
        MockLLM.return_value = MagicMock()
        result = get_llm(config, max_tokens=512)
    assert MockLLM.called
    assert result is MockLLM.return_value
    call_kwargs = MockLLM.call_args.kwargs
    assert call_kwargs["model"] == "llama3.2"
    assert call_kwargs["base_url"] == "http://localhost:11434"
    assert call_kwargs["temperature"] == config.temperature
    assert call_kwargs["num_predict"] == 512
```

- [ ] **Step 2: Run the new test to confirm it fails**

```bash
pytest tests/test_llm.py::test_get_llm_ollama -v
```

Expected: FAIL (no Ollama branch in `get_llm()` yet).

### Step 2 — Implement `llm.py` changes

- [ ] **Step 3: Add the Ollama branch and fix the Anthropic catch-all in `get_llm()`**

> `_DEFAULT_MODELS` was already updated in Task 2. Only `get_llm()` needs changing here.

Replace the end of `get_llm()` — specifically the final `else:` block:

From:
```python
    else:  # anthropic (default)
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=config.model_name,
            api_key=config.anthropic_api_key,
            max_tokens=max_tokens,
            temperature=config.temperature,
        )
```

To:
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

- [ ] **Step 4: Run the Ollama test — expect pass**

```bash
pytest tests/test_llm.py::test_get_llm_ollama -v
```

Expected: PASS.

- [ ] **Step 5: Run the full `test_llm.py` — no regressions**

```bash
pytest tests/test_llm.py -v
```

Expected: all tests pass.

- [ ] **Step 6: Run the full test suite**

```bash
pytest --tb=short -q
```

Expected: all tests pass (66 existing + 5 new = 71 total).

- [ ] **Step 7: Commit**

```bash
git add src/irish_statute_assistant/llm.py tests/test_llm.py
git commit -m "feat: add ollama branch to get_llm()"
```
