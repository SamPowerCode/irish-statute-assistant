from __future__ import annotations

import logging
import re

from irish_statute_assistant.agents.analyst import AnalystAgent
from irish_statute_assistant.agents.clarifier import ClarifierAgent
from irish_statute_assistant.agents.devils_advocate import DevilsAdvocateAgent
from irish_statute_assistant.agents.evaluator import EvaluatorAgent
from irish_statute_assistant.agents.grounding_checker import GroundingCheckerAgent
from irish_statute_assistant.agents.researcher import ResearcherAgent
from irish_statute_assistant.agents.writer import WriterAgent
from irish_statute_assistant.config import Config
from irish_statute_assistant.context import QueryContext
from irish_statute_assistant.memory.conversation_store import ConversationStore
from irish_statute_assistant.memory.user_preference_store import UserPreferenceStore
from irish_statute_assistant.models.schemas import AnalystOutput, WriterOutput
from irish_statute_assistant.retry import run_with_retry
from irish_statute_assistant.tools.session_cache import SessionCache
from irish_statute_assistant.tools.statute_fetcher import StatuteFetcher

logger = logging.getLogger(__name__)

# Keyword patterns → (preference_key, preference_value)
_PREFERENCE_PATTERNS: list[tuple[str, str, str]] = [
    (r"i(?:'m| am) a (?:solicitor|lawyer)", "user_type", "solicitor"),
    (r"explain simply|plain english|non.?lawyer", "language_level", "plain"),
    (r"use legal terms|technical", "language_level", "technical"),
    (r"\bbrief\b|short answer", "verbosity", "brief"),
    (r"\bdetailed\b|full explanation", "verbosity", "detailed"),
]


class Supervisor:
    def __init__(
        self,
        config: Config,
        memory: ConversationStore,
        preferences: UserPreferenceStore,
    ) -> None:
        self._max_refinements = config.max_refinement_rounds
        self._max_retries = config.max_retries
        self._memory = memory
        self._preferences = preferences
        self._evaluator_flag_counts: dict[str, int] = {}

        cache = SessionCache()
        fetcher = StatuteFetcher(
            rate_limit_delay=config.rate_limit_delay,
            max_retries=config.max_retries,
        )
        self._clarifier = ClarifierAgent(config)
        self._researcher = ResearcherAgent(config, cache, fetcher)
        self._analyst = AnalystAgent(config)
        self._advocate = DevilsAdvocateAgent(config)
        self._writer = WriterAgent(config)
        self._grounding_checker = GroundingCheckerAgent(config)
        self._evaluator = EvaluatorAgent(config)

    def run(self, query: str, context: QueryContext | None = None) -> WriterOutput | str:
        history = self._memory.format_for_prompt()

        # 1. Clarify
        clarifier_result = run_with_retry(
            lambda: self._clarifier.run(query=query, history=history),
            self._max_retries,
        )
        if context:
            context.consume(self._clarifier.last_token_count)
        if clarifier_result.needs_clarification:
            self._memory.add_exchange(user=query, assistant=clarifier_result.question)
            return clarifier_result.question

        # 2. Research
        research = run_with_retry(
            lambda: self._researcher.run(query=query),
            self._max_retries,
        )
        if context:
            context.consume(self._researcher.last_token_count)

        # 3. Analyse (once, outside the loop)
        llm_analyst_result = run_with_retry(
            lambda: self._analyst.run(query=query, research=research),
            self._max_retries,
        )
        if context:
            context.consume(self._analyst.last_token_count)
        analyst_output = AnalystOutput(
            **llm_analyst_result.model_dump(exclude={"advocate_challenges"}),
            advocate_challenges=[],
        )

        # 4. Devil's advocate (initial run)
        advocate_result = run_with_retry(
            lambda: self._advocate.run(
                analyst_output=analyst_output,
                query=query,
                research=research,
                mode="standard",
            ),
            self._max_retries,
        )
        if context:
            context.consume(self._advocate.last_token_count)

        # Confidence gate
        low_confidence = analyst_output.confidence < 0.5 or advocate_result.severity == "major"
        effective_refinements = self._max_refinements * 2 if low_confidence else self._max_refinements
        advocate_mode_on_retry = "strict" if low_confidence else "standard"

        analyst_output.advocate_challenges = advocate_result.challenges

        # 5–7. Refinement loop
        evaluator_flags: list[str] = []
        best_output: WriterOutput | None = None

        for _ in range(effective_refinements + 1):
            writer_result = run_with_retry(
                lambda: self._writer.run(
                    query=query,
                    analysis=analyst_output,
                    research=research,
                    evaluator_flags=evaluator_flags,
                ),
                self._max_retries,
            )
            if context:
                context.consume(self._writer.last_token_count)
            writer_result.analyst_confidence = analyst_output.confidence

            grounding = run_with_retry(
                lambda: self._grounding_checker.run(
                    writer_output=writer_result, research=research
                ),
                self._max_retries,
            )
            if context:
                context.consume(self._grounding_checker.last_token_count)
            writer_result.warnings = grounding.ungrounded_claims

            evaluation = run_with_retry(
                lambda: self._evaluator.run(
                    query=query,
                    output=writer_result,
                    grounding_passed=grounding.grounding_passed,
                ),
                self._max_retries,
            )
            if context:
                context.consume(self._evaluator.last_token_count)

            best_output = writer_result

            self._update_flag_counts(evaluation.flags)

            if evaluation.pass_:
                self._detect_and_save_preferences(query, evaluation.flags)
                self._memory.add_exchange(user=query, assistant=writer_result.short_answer)
                return writer_result

            evaluator_flags = evaluation.flags

            # Re-run advocate with potentially stricter mode
            advocate_result = run_with_retry(
                lambda: self._advocate.run(
                    analyst_output=analyst_output,
                    query=query,
                    research=research,
                    mode=advocate_mode_on_retry,
                ),
                self._max_retries,
            )
            if context:
                context.consume(self._advocate.last_token_count)
            analyst_output.advocate_challenges = advocate_result.challenges

        self._detect_and_save_preferences(query, evaluator_flags)
        self._memory.add_exchange(user=query, assistant=best_output.short_answer)
        return best_output

    def _detect_and_save_preferences(self, query: str, evaluator_flags: list[str]) -> None:
        query_lower = query.lower()
        for pattern, key, value in _PREFERENCE_PATTERNS:
            if re.search(pattern, query_lower):
                self._preferences.set(key, value)

        # Inferred: repeated "plain English" evaluator flag → user prefers technical
        for flag in evaluator_flags:
            if "plain english" in flag.lower():
                flag_key = flag.lower()
                count = self._evaluator_flag_counts.get(flag_key, 0)
                if count >= 2:
                    self._preferences.set("language_level", "technical")

    def _update_flag_counts(self, flags: list[str]) -> None:
        for flag in flags:
            key = flag.lower()
            self._evaluator_flag_counts[key] = self._evaluator_flag_counts.get(key, 0) + 1
