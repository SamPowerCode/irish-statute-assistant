from irish_statute_assistant.agents.analyst import AnalystAgent
from irish_statute_assistant.agents.clarifier import ClarifierAgent
from irish_statute_assistant.agents.evaluator import EvaluatorAgent
from irish_statute_assistant.agents.researcher import ResearcherAgent
from irish_statute_assistant.agents.writer import WriterAgent
from irish_statute_assistant.config import Config
from irish_statute_assistant.models.schemas import WriterOutput
from irish_statute_assistant.tools.session_cache import SessionCache


class Supervisor:
    def __init__(self, config: Config) -> None:
        self._max_refinements = config.max_refinement_rounds
        cache = SessionCache()
        self._clarifier = ClarifierAgent(config)
        self._researcher = ResearcherAgent(config, cache)
        self._analyst = AnalystAgent(config)
        self._writer = WriterAgent(config)
        self._evaluator = EvaluatorAgent(config)

    def run(self, query: str, history: str) -> WriterOutput | str:
        """
        Returns:
          - str: a clarifying question if the query is ambiguous
          - WriterOutput: the final answer if the query is clear
        """
        clarifier_result = self._clarifier.run(query=query, history=history)
        if clarifier_result.needs_clarification:
            return clarifier_result.question

        research = self._researcher.run(query=query)
        evaluator_flags: list[str] = []
        best_output: WriterOutput | None = None

        for _ in range(self._max_refinements + 1):
            analysis = self._analyst.run(query=query, research=research, evaluator_flags=evaluator_flags)
            output = self._writer.run(query=query, analysis=analysis, research=research, evaluator_flags=evaluator_flags)
            evaluation = self._evaluator.run(query=query, output=output)
            best_output = output

            if evaluation.pass_:
                return output

            evaluator_flags = evaluation.flags

        return best_output
