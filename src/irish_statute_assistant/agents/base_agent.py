"""BaseAgent with token-usage tracking via LangChain callback."""
from __future__ import annotations

from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from langchain_core.runnables import Runnable, RunnableConfig


class TokenUsageCallback(BaseCallbackHandler):
    """LangChain callback that captures token counts from Anthropic responses."""

    def __init__(self) -> None:
        super().__init__()
        self.total_tokens: int = 0

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        for generation in response.generations:
            for gen in generation:
                usage = getattr(gen, "generation_info", {}) or {}
                self.total_tokens += usage.get("input_tokens", 0)
                self.total_tokens += usage.get("output_tokens", 0)


class BaseAgent:
    _last_token_count: int = 0

    def _invoke_chain(self, chain: Runnable, inputs: dict) -> Any:
        """Invoke chain, capture token usage, return result."""
        tracker = TokenUsageCallback()
        result = chain.invoke(inputs, config=RunnableConfig(callbacks=[tracker]))
        self._last_token_count = tracker.total_tokens
        return result

    @property
    def last_token_count(self) -> int:
        return self._last_token_count
