from __future__ import annotations

from irish_statute_assistant.agents.analyst import AnalystAgent
from irish_statute_assistant.agents.clarifier import ClarifierAgent
from irish_statute_assistant.agents.evaluator import EvaluatorAgent
from irish_statute_assistant.agents.researcher import ResearcherAgent
from irish_statute_assistant.agents.writer import WriterAgent
from irish_statute_assistant.config import Config
from irish_statute_assistant.context import QueryContext
from irish_statute_assistant.models.schemas import WriterOutput
from irish_statute_assistant.retry import run_with_retry
from irish_statute_assistant.tools.session_cache import SessionCache
from irish_statute_assistant.tools.statute_fetcher import StatuteFetcher


class Supervisor:
    def __init__(self, config: Config) -> None:
        self._max_refinements = config.max_refinement_rounds
        self._max_retries = config.max_retries
        cache = SessionCache()
        fetcher = StatuteFetcher(
            rate_limit_delay=config.rate_limit_delay,
            max_retries=config.max_retries,
        )
        self._clarifier = ClarifierAgent(config)
        self._researcher = ResearcherAgent(config, cache, fetcher)
        self._analyst = AnalystAgent(config)
        self._writer = WriterAgent(config)
        self._evaluator = EvaluatorAgent(config)

    def run(
        self, query: str, history: str, context: QueryContext | None = None
    ) -> WriterOutput | str:
        """
        Returns:
          - str: a clarifying question if the query is ambiguous
          - WriterOutput: the final answer if the query is clear
        """
        clarifier_result = run_with_retry(
            lambda: self._clarifier.run(query=query, history=history),
            self._max_retries,
        )
        if context:
            context.consume(self._clarifier.last_token_count)

        if clarifier_result.needs_clarification:
            return clarifier_result.question

        research = run_with_retry(
            lambda: self._researcher.run(query=query),
            self._max_retries,
        )
        if context:
            context.consume(self._researcher.last_token_count)

        evaluator_flags: list[str] = []
        best_output: WriterOutput | None = None

        for _ in range(self._max_refinements + 1):
            analysis = run_with_retry(
                lambda: self._analyst.run(query=query, research=research, evaluator_flags=evaluator_flags),
                self._max_retries,
            )
            if context:
                context.consume(self._analyst.last_token_count)

            output = run_with_retry(
                lambda: self._writer.run(query=query, analysis=analysis, research=research, evaluator_flags=evaluator_flags),
                self._max_retries,
            )
            if context:
                context.consume(self._writer.last_token_count)

            evaluation = run_with_retry(
                lambda: self._evaluator.run(query=query, output=output),
                self._max_retries,
            )
            if context:
                context.consume(self._evaluator.last_token_count)

            best_output = output

            if evaluation.pass_:
                return output

            evaluator_flags = evaluation.flags

        return best_output
