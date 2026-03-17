from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate

from irish_statute_assistant.agents.base_agent import BaseAgent
from irish_statute_assistant.config import Config
from irish_statute_assistant.llm import get_llm
from irish_statute_assistant.models.schemas import (
    GroundingOutput, ResearcherOutput, WriterOutput
)

SYSTEM_PROMPT = """You are a hallucination detector for an Irish legal research assistant.

You are given a list of key legal clauses from a writer's output, plus the raw statute text
that was retrieved. For each clause, check whether the claim is directly supported by the
retrieved text.

Return:
- ungrounded_claims: list of clauses (as strings) that are NOT supported by the retrieved text
- grounding_passed: true if all claims are grounded, false if any are not

Be strict: if a claim goes beyond what the text says, flag it.
"""

HUMAN_PROMPT = """Key clauses to verify:
{key_clauses}

Retrieved statute text:
{statute_text}
"""


class GroundingCheckerAgent(BaseAgent):
    def __init__(self, config: Config) -> None:
        llm = get_llm(config, max_tokens=512).with_structured_output(GroundingOutput)
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", HUMAN_PROMPT),
        ])
        self._chain = prompt | llm

    def run(self, writer_output: WriterOutput, research: ResearcherOutput) -> GroundingOutput:
        bd = writer_output.detailed_breakdown
        key_clauses_text = "\n".join(
            f"- {kc.text} ({kc.act}, {kc.section})" for kc in bd.key_clauses
        )
        statute_text = "\n\n".join(
            f"## {act.title}\n" + "\n".join(act.sections)
            for act in research.acts
        )
        return self._invoke_chain(self._chain, {
            "key_clauses": key_clauses_text,
            "statute_text": statute_text,
        })
