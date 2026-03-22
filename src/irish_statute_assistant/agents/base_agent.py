"""BaseAgent with token-usage tracking via LangChain callback."""
from __future__ import annotations

import json
import logging
from collections.abc import Callable
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from langchain_core.runnables import Runnable, RunnableConfig

logger = logging.getLogger(__name__)

# Optional hook called after every agent invocation with (agent_name, inputs, output_json).
# Set via set_message_hook() before running the pipeline; cleared afterwards.
_message_hook: Callable[[str, dict, str], None] | None = None


def set_message_hook(fn: Callable[[str, dict, str], None] | None) -> None:
    global _message_hook
    _message_hook = fn


class TokenUsageCallback(BaseCallbackHandler):
    """Tracks token usage from LLM responses across all supported providers.

    All four providers (Anthropic, OpenAI, Groq, Google) set usage_metadata
    on the AIMessage inside each ChatGeneration with normalised keys:
    input_tokens, output_tokens, total_tokens.
    """

    def __init__(self) -> None:
        super().__init__()
        self.total_tokens: int = 0

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        for generation in response.generations:
            for gen in generation:
                msg = getattr(gen, "message", None)
                meta = getattr(msg, "usage_metadata", None) if msg else None
                if meta:
                    self.total_tokens += meta.get("total_tokens", 0)
        if self.total_tokens == 0:
            logger.debug(
                "TokenUsageCallback: no token counts found in response "
                "— budget enforcement inactive for this call"
            )


class BaseAgent:
    _last_token_count: int = 0

    def _invoke_chain(self, chain: Runnable, inputs: dict) -> Any:
        """Invoke chain, capture token usage, return result."""
        agent_name = type(self).__name__
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "%s → LLM input:\n%s",
                agent_name,
                json.dumps(inputs, indent=2, default=str),
            )
        tracker = TokenUsageCallback()
        result = chain.invoke(inputs, config=RunnableConfig(callbacks=[tracker]))
        self._last_token_count = tracker.total_tokens
        try:
            result_json = result.model_dump_json(indent=2)
        except AttributeError:
            result_json = json.dumps(result, indent=2, default=str)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("%s → LLM input:\n%s", agent_name, json.dumps(inputs, indent=2, default=str))
            logger.debug("%s ← LLM output:\n%s", agent_name, result_json)
        if _message_hook is not None:
            _message_hook(agent_name, inputs, result_json)
        return result

    @property
    def last_token_count(self) -> int:
        return self._last_token_count
