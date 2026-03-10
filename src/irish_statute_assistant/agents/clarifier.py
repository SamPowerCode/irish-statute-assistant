from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate

from irish_statute_assistant.config import Config
from irish_statute_assistant.models.schemas import ClarifierOutput

SYSTEM_PROMPT = """You are a helpful legal assistant intake agent. Your job is to decide
whether a user's legal question is clear enough to research, or whether you need to ask
one focused clarifying question first.

Rules:
- If the question clearly identifies a legal topic (e.g. "statute of limitations for injury"),
  return needs_clarification=false.
- If the question is too vague (e.g. "what are my rights?"), return needs_clarification=true
  and ask exactly ONE short, specific question that will make the topic researchable.
- Never ask more than one question.
- Keep the question simple enough for a non-lawyer to understand.

Conversation so far:
{history}
"""

HUMAN_PROMPT = "User's question: {query}"


class ClarifierAgent:
    def __init__(self, config: Config) -> None:
        llm = ChatAnthropic(
            model=config.model_name,
            api_key=config.anthropic_api_key,
            max_tokens=256,
        ).with_structured_output(ClarifierOutput)
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", HUMAN_PROMPT),
        ])
        self._chain = prompt | llm

    def run(self, query: str, history: str) -> ClarifierOutput:
        return self._chain.invoke({"query": query, "history": history})
