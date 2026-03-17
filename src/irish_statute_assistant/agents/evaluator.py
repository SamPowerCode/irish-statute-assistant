from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate

from irish_statute_assistant.agents.base_agent import BaseAgent
from irish_statute_assistant.config import Config
from irish_statute_assistant.models.schemas import EvaluatorOutput, WriterOutput

SYSTEM_PROMPT = """You are a quality evaluator for an Irish legal research assistant.
Score the output on these four criteria (each 0.0–1.0, overall score is the average):

1. Accuracy: Does the answer correctly reflect what the statutes say?
2. Completeness: Does it answer the user's question fully?
3. Citation quality: Are relevant Acts named and referenced?
4. Plain English: Is the short_answer genuinely understandable by a non-lawyer?

Set pass=true if overall score >= {threshold}.
List specific flags for anything that should be improved (even if passing).
Be strict. A vague or uncited answer should score below 0.7.
"""

HUMAN_PROMPT = """User question: {query}

Output to evaluate:
Short answer: {short_answer}
Summary: {summary}
Relevant Acts: {relevant_acts}
Key clauses: {key_clauses}
Caveats: {caveats}
"""


class EvaluatorAgent(BaseAgent):
    def __init__(self, config: Config) -> None:
        self._threshold = config.evaluator_pass_threshold
        llm = ChatAnthropic(
            model=config.model_name,
            api_key=config.anthropic_api_key,
            max_tokens=512,
        ).with_structured_output(EvaluatorOutput)
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", HUMAN_PROMPT),
        ])
        self._chain = prompt | llm

    def run(self, query: str, output: WriterOutput) -> EvaluatorOutput:
        bd = output.detailed_breakdown
        result = self._invoke_chain(self._chain, {
            "threshold": self._threshold,
            "query": query,
            "short_answer": output.short_answer,
            "summary": bd.summary,
            "relevant_acts": ", ".join(bd.relevant_acts),
            "key_clauses": "\n".join(bd.key_clauses),
            "caveats": "\n".join(bd.caveats),
        })
        # Enforce threshold locally — the LLM's pass_ may be inconsistent with its score
        return result.model_copy(update={"pass_": result.score >= self._threshold})
