from langchain_core.prompts import ChatPromptTemplate

from irish_statute_assistant.agents.base_agent import BaseAgent
from irish_statute_assistant.config import Config
from irish_statute_assistant.llm import get_llm
from irish_statute_assistant.models.schemas import AnalystOutput, ResearcherOutput, WriterOutput

SYSTEM_PROMPT = """You are a plain English legal writer. You explain Irish law to ordinary people
who have no legal training.

Rules:
- short_answer: one or two plain sentences (maximum 100 words). No legal jargon. Write as if
  explaining to a friend.
- detailed_breakdown.summary: 2-3 sentences summarising the legal position.
- detailed_breakdown.relevant_acts: list the Acts by name (e.g. "Statute of Limitations Act 1957").
- detailed_breakdown.key_clauses: the specific rules that answer the question, in plain English.
- detailed_breakdown.caveats: important exceptions, edge cases, or "it depends" situations.
- Always include at least one caveat reminding the user to seek professional legal advice.
- Never make up law. Only use what the analyst found.

If there are evaluator flags, address them: {evaluator_flags}
"""

HUMAN_PROMPT = """User question: {query}

Analyst findings:
Key clauses: {key_clauses}
Gaps: {gaps}
Confidence: {confidence}

Acts researched:
{act_titles}
"""


class WriterAgent(BaseAgent):
    def __init__(self, config: Config) -> None:
        llm = get_llm(config, max_tokens=2048).with_structured_output(WriterOutput)
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", HUMAN_PROMPT),
        ])
        self._chain = prompt | llm

    def run(
        self,
        query: str,
        analysis: AnalystOutput,
        research: ResearcherOutput,
        evaluator_flags: list[str],
    ) -> WriterOutput:
        flags_text = "\n".join(evaluator_flags) if evaluator_flags else "None"
        act_titles = "\n".join(act.title for act in research.acts)
        return self._invoke_chain(self._chain, {
            "query": query,
            "key_clauses": "\n".join(analysis.key_clauses),
            "gaps": "\n".join(analysis.gaps) if analysis.gaps else "None",
            "confidence": analysis.confidence,
            "act_titles": act_titles,
            "evaluator_flags": flags_text,
        })
