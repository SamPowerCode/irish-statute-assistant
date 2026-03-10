from __future__ import annotations

from irish_statute_assistant.agents.supervisor import Supervisor
from irish_statute_assistant.config import Config
from irish_statute_assistant.memory.session_memory import SessionMemory
from irish_statute_assistant.models.schemas import WriterOutput


class Pipeline:
    def __init__(self, config: Config) -> None:
        self._supervisor = Supervisor(config)
        self._memory = SessionMemory()

    def query(self, user_query: str) -> WriterOutput | str:
        """
        Submit a query. Returns:
          - str: a clarifying question (memory not updated)
          - WriterOutput: the final answer (memory updated)
        """
        history = self._memory.format_for_prompt()
        result = self._supervisor.run(query=user_query, history=history)

        if isinstance(result, WriterOutput):
            self._memory.add_exchange(
                user=user_query,
                assistant=result.short_answer,
            )

        return result
