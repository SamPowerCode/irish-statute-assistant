import pytest
from pydantic import ValidationError

from irish_statute_assistant.config import Config
from irish_statute_assistant.llm import _DEFAULT_MODELS


def test_config_loads_defaults():
    config = Config(anthropic_api_key="test-key")
    assert config.model_name == "claude-sonnet-4-6"
    assert config.evaluator_pass_threshold == 0.7
    assert config.max_refinement_rounds == 2
    assert config.max_retries == 3
    assert config.token_budget_per_query == 4000
    assert config.rate_limit_delay == 1.0


def test_config_requires_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ValidationError):
        Config(_env_file=None)  # ignore .env so only env vars are checked


def test_config_chroma_defaults(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    config = Config()
    assert config.chroma_db_path == "./data/chroma"
    assert config.embedding_model == "all-MiniLM-L6-v2"
    assert config.acts_per_category == 5
    assert "employment" in config.index_categories
    assert "housing" in config.index_categories


# --- new tests for multi-provider config ---


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
