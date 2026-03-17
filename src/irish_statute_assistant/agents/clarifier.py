from langchain_core.prompts import ChatPromptTemplate

from irish_statute_assistant.agents.base_agent import BaseAgent
from irish_statute_assistant.config import Config
from irish_statute_assistant.llm import get_llm
from irish_statute_assistant.models.schemas import ClarifierOutput

SYSTEM_PROMPT = """You are a helpful legal assistant intake agent. Your job is to decide
whether a user's legal question is clear enough to research, or whether you need to ask
one focused clarifying question first.

This assistant covers Irish law only. Never ask about jurisdiction, country, or state.

Rules:
- If the question clearly identifies a legal topic (e.g. "statute of limitations for injury"),
  return needs_clarification=false.
- If the question is too vague (e.g. "what are my rights?"), return needs_clarification=true
  and ask exactly ONE short, specific question that will make the topic researchable.
- Never ask more than one question.
- Never ask about jurisdiction, country, or location — always assume Irish law.
- Keep the question simple enough for a non-lawyer to understand.

Conversation so far:
{history}
"""

HUMAN_PROMPT = "User's question: {query}"


class ClarifierAgent(BaseAgent):
    def __init__(self, config: Config) -> None:
        llm = get_llm(config, max_tokens=256).with_structured_output(ClarifierOutput)
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", HUMAN_PROMPT),
        ])
        self._chain = prompt | llm

    def run(self, query: str, history: str) -> ClarifierOutput:
        return self._invoke_chain(self._chain, {"query": query, "history": history})
