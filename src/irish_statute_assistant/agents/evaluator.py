from langchain_core.prompts import ChatPromptTemplate

from irish_statute_assistant.agents.base_agent import BaseAgent
from irish_statute_assistant.config import Config
from irish_statute_assistant.llm import get_llm
from irish_statute_assistant.models.schemas import EvaluatorOutput, WriterOutput

SYSTEM_PROMPT = """You are a quality evaluator for an Irish legal research assistant.
Score the output on these four criteria (each 0.0–1.0, overall score is the average):

1. Accuracy: Does the answer correctly reflect what the statutes say?
2. Completeness: Does it answer the user's question fully?
3. Citation quality: Are relevant Acts named and referenced? {grounding_note}
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
        llm = get_llm(config, max_tokens=512).with_structured_output(EvaluatorOutput)
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", HUMAN_PROMPT),
        ])
        self._chain = prompt | llm

    def run(self, query: str, output: WriterOutput, grounding_passed: bool = True) -> EvaluatorOutput:
        bd = output.detailed_breakdown
        grounding_note = (
            "Note: the grounding checker found unverified claims in this output. "
            "Score citation quality no higher than 0.4."
            if not grounding_passed
            else ""
        )
        result = self._invoke_chain(self._chain, {
            "threshold": self._threshold,
            "grounding_note": grounding_note,
            "query": query,
            "short_answer": output.short_answer,
            "summary": bd.summary,
            "relevant_acts": ", ".join(bd.relevant_acts),
            "key_clauses": "\n".join(
                f"{kc.text} ({kc.act}, {kc.section})" for kc in bd.key_clauses
            ),
            "caveats": "\n".join(bd.caveats),
        })
        return result.model_copy(update={"pass_": result.score >= self._threshold})
