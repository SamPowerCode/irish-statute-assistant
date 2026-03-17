from __future__ import annotations

from typing import Literal

from langchain_core.prompts import ChatPromptTemplate

from irish_statute_assistant.agents.base_agent import BaseAgent
from irish_statute_assistant.config import Config
from irish_statute_assistant.llm import get_llm
from irish_statute_assistant.models.schemas import (
    AdvocateOutput, AnalystOutput, ResearcherOutput
)

_SYSTEM_STANDARD = """You are a critical legal reviewer. You have been given an analyst's
findings about an Irish law question. Your job is to find 1-3 weaknesses in the analysis:
- Missing exceptions or special cases
- Statutes that may override or qualify the analyst's conclusion
- Claims that go beyond what the retrieved text actually says

If the analysis is solid and well-grounded, return an empty challenges list with severity=minor.
Set severity=major only if the analyst's conclusion could be substantially wrong.
"""

_SYSTEM_STRICT = """You are an adversarial legal reviewer. Find EVERY possible weakness
in this analysis — missing exceptions, conflicting statutes, unsupported inferences,
edge cases, anything. Aim for up to 5 challenges. Be aggressive.
Set severity=major if any challenge could substantially change the answer.
"""

_HUMAN_PROMPT = """User question: {query}

Analyst's key clauses:
{key_clauses}

Analyst's confidence: {confidence}

Retrieved statute text:
{statute_text}
"""


class DevilsAdvocateAgent(BaseAgent):
    def __init__(self, config: Config) -> None:
        llm = get_llm(config, max_tokens=512).with_structured_output(AdvocateOutput)
        self._chain_standard = (
            ChatPromptTemplate.from_messages([("system", _SYSTEM_STANDARD), ("human", _HUMAN_PROMPT)])
            | llm
        )
        self._chain_strict = (
            ChatPromptTemplate.from_messages([("system", _SYSTEM_STRICT), ("human", _HUMAN_PROMPT)])
            | llm
        )

    def run(
        self,
        analyst_output: AnalystOutput,
        query: str,
        research: ResearcherOutput,
        mode: Literal["standard", "strict"] = "standard",
    ) -> AdvocateOutput:
        chain = self._chain_strict if mode == "strict" else self._chain_standard
        key_clauses_text = "\n".join(
            f"{kc.text} ({kc.act}, {kc.section})" for kc in analyst_output.key_clauses
        )
        statute_text = "\n\n".join(
            f"## {act.title}\n" + "\n".join(act.sections)
            for act in research.acts
        )
        return self._invoke_chain(chain, {
            "query": query,
            "key_clauses": key_clauses_text,
            "confidence": analyst_output.confidence,
            "statute_text": statute_text,
        })
