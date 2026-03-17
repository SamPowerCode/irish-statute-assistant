from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate

from irish_statute_assistant.agents.base_agent import BaseAgent
from irish_statute_assistant.config import Config
from irish_statute_assistant.models.schemas import AnalystOutput, ResearcherOutput

SYSTEM_PROMPT = """You are a legal analyst. You have been given raw Irish statute text and
a user's question. Your job is to:
1. Identify the key clauses directly relevant to the question.
2. Note any gaps (things the user asked about that the statutes don't clearly address).
3. Assign a confidence score (0.0–1.0) for how well the statutes answer the question.
   - 0.9+ = question is fully and clearly answered
   - 0.5–0.89 = partially answered, some ambiguity
   - below 0.5 = statutes are unclear or not directly relevant

If there are evaluator flags from a previous attempt, address them in this analysis.

Evaluator flags from previous attempt (if any): {evaluator_flags}
"""

HUMAN_PROMPT = """User question: {query}

Retrieved statute sections:
{statute_text}
"""


class AnalystAgent(BaseAgent):
    def __init__(self, config: Config) -> None:
        llm = ChatAnthropic(
            model=config.model_name,
            api_key=config.anthropic_api_key,
            max_tokens=1024,
        ).with_structured_output(AnalystOutput)
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", HUMAN_PROMPT),
        ])
        self._chain = prompt | llm

    def run(self, query: str, research: ResearcherOutput, evaluator_flags: list[str]) -> AnalystOutput:
        statute_text = self._format_research(research)
        flags_text = "\n".join(evaluator_flags) if evaluator_flags else "None"
        return self._invoke_chain(self._chain, {
            "query": query,
            "statute_text": statute_text,
            "evaluator_flags": flags_text,
        })

    def _format_research(self, research: ResearcherOutput) -> str:
        parts = []
        for act in research.acts:
            parts.append(f"## {act.title}\nURL: {act.url}")
            for section in act.sections:
                parts.append(section)
        return "\n\n".join(parts)
