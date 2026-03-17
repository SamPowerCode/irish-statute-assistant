from __future__ import annotations

import logging

from irish_statute_assistant.agents.supervisor import Supervisor
from irish_statute_assistant.config import Config
from irish_statute_assistant.context import QueryContext
from irish_statute_assistant.memory.session_memory import SessionMemory
from irish_statute_assistant.models.schemas import WriterOutput

logger = logging.getLogger(__name__)


class Pipeline:
    def __init__(self, config: Config) -> None:
        self._config = config
        self._supervisor = Supervisor(config)
        self._memory = SessionMemory()

    def query(self, user_query: str) -> WriterOutput | str:
        """
        Submit a query. Returns:
          - str: a clarifying question (memory not updated)
          - WriterOutput: the final answer (memory updated)
        """
        history = self._memory.format_for_prompt()
        context = QueryContext(budget=self._config.token_budget_per_query)
        result = self._supervisor.run(query=user_query, history=history, context=context)
        logger.info(
            "Query %s used %d/%d tokens",
            context.query_id,
            context.tokens_used,
            context.budget,
        )

        if isinstance(result, WriterOutput):
            self._memory.add_exchange(
                user=user_query,
                assistant=result.short_answer,
            )

        return result
