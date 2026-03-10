import pytest
from pydantic import ValidationError

from irish_statute_assistant.config import Config


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
        Config()
