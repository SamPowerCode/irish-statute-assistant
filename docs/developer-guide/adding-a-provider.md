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
python -c "import irish_statute_assistant.config"
```

If `_PROVIDER_KEY_MAP` and `_DEFAULT_MODELS` are out of sync, this raises an
`AssertionError` immediately. Fix by ensuring both dicts have exactly the same keys.

## 6. Test

Run the full suite to confirm no regressions:

```bash
python -m pytest --tb=short -q
```
