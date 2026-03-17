"""Tests for BaseAgent token counting via TokenUsageCallback."""
from unittest.mock import MagicMock, patch

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

    agent = BaseAgent()
    mock_chain = MagicMock()

    with patch("irish_statute_assistant.agents.base_agent.TokenUsageCallback") as MockCB:
        mock_cb = MagicMock()
        mock_cb.total_tokens = 55
        MockCB.return_value = mock_cb
        mock_chain.invoke.return_value = "result"
        agent._invoke_chain(mock_chain, {"key": "value"})

    assert agent.last_token_count == 55
