"""Tests for the get_llm() provider factory."""
from unittest.mock import MagicMock, patch

import pytest


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


@pytest.mark.parametrize("provider,patch_target,expected_kwarg", [
    ("anthropic", "langchain_anthropic.ChatAnthropic",             "max_tokens"),
    ("openai",    "langchain_openai.ChatOpenAI",                   "max_tokens"),
    ("google",    "langchain_google_genai.ChatGoogleGenerativeAI", "max_output_tokens"),
    ("groq",      "langchain_groq.ChatGroq",                       "max_tokens"),
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
