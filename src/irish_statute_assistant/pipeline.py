from __future__ import annotations

import logging

from irish_statute_assistant.agents.supervisor import Supervisor
from irish_statute_assistant.config import Config
from irish_statute_assistant.context import QueryContext
from irish_statute_assistant.memory.conversation_store import ConversationStore
from irish_statute_assistant.memory.user_preference_store import UserPreferenceStore
from irish_statute_assistant.models.schemas import WriterOutput

logger = logging.getLogger(__name__)


class Pipeline:
    """Top-level entry point for submitting queries to the assistant.

    Constructs the conversation store, preference store, and supervisor on
    initialisation. Each call to query() is independent — the supervisor
    owns all memory writes.

    Args:
        config: Application configuration.

    Example::

        config = Config()
        pipeline = Pipeline(config)
        result = pipeline.query("How long do I have to bring a personal injury claim?")
        if isinstance(result, str):
            print("Clarification needed:", result)
        else:
            print(result.short_answer)
    """

    def __init__(self, config: Config) -> None:
        self._config = config
        self._memory = ConversationStore(
            db_path=config.conversations_db_path,
            history_limit=config.conversation_history_limit,
        )
        self._preferences = UserPreferenceStore(db_path=config.preferences_db_path)
        self._supervisor = Supervisor(config, memory=self._memory, preferences=self._preferences)

    def query(self, user_query: str) -> WriterOutput | str:
        """Submit a user query and return the assistant's response.

        Args:
            user_query: The user's legal question.

        Returns:
            A clarifying question string if the query is ambiguous,
            or a WriterOutput with the final answer.
        """
        context = QueryContext(budget=self._config.token_budget_per_query)
        result = self._supervisor.run(query=user_query, context=context)
        logger.info(
            "Query used %d/%d tokens",
            context.tokens_used,
            context.budget,
        )
        return result
