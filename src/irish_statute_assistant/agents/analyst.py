from langchain_core.prompts import ChatPromptTemplate

from irish_statute_assistant.agents.base_agent import BaseAgent
from irish_statute_assistant.config import Config
from irish_statute_assistant.llm import get_llm
from irish_statute_assistant.models.schemas import AnalystLLMOutput, ResearcherOutput

SYSTEM_PROMPT = """You are a legal analyst. You have been given raw Irish statute text and
a user's question. Your job is to:
1. Identify the key clauses directly relevant to the question. For each clause include:
   - text: the rule in plain English
   - act: the full name of the Act (e.g. "Statute of Limitations Act 1957")
   - section: the section number (e.g. "s.11")
2. Note any gaps (things the user asked about that the statutes don't clearly address).
3. Assign a confidence score (0.0–1.0) for how well the statutes answer the question.
   - 0.9+ = question is fully and clearly answered
   - 0.5–0.89 = partially answered, some ambiguity
   - below 0.5 = statutes are unclear or not directly relevant
"""

HUMAN_PROMPT = """User question: {query}

Retrieved statute sections:
{statute_text}
"""


class AnalystAgent(BaseAgent):
    def __init__(self, config: Config) -> None:
        llm = get_llm(config, max_tokens=1024).with_structured_output(AnalystLLMOutput)
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", HUMAN_PROMPT),
        ])
        self._chain = prompt | llm

    def run(self, query: str, research: ResearcherOutput) -> AnalystLLMOutput:
        statute_text = self._format_research(research)
        return self._invoke_chain(self._chain, {
            "query": query,
            "statute_text": statute_text,
        })

    def _format_research(self, research: ResearcherOutput) -> str:
        parts = []
        for act in research.acts:
            parts.append(f"## {act.title}\nURL: {act.url}")
            for section in act.sections:
                parts.append(section)
        return "\n\n".join(parts)
