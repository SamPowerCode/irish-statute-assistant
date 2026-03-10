from unittest.mock import MagicMock
from irish_statute_assistant.agents.evaluator import EvaluatorAgent
from irish_statute_assistant.models.schemas import EvaluatorOutput, WriterOutput, DetailedBreakdown


def make_evaluator(score, flags, passed):
    agent = EvaluatorAgent.__new__(EvaluatorAgent)
    mock_chain = MagicMock()
    mock_chain.invoke = MagicMock(return_value=EvaluatorOutput(
        score=score, flags=flags, **{"pass": passed}
    ))
    agent._chain = mock_chain
    agent._threshold = 0.7
    return agent


def sample_writer_output():
    return WriterOutput(
        short_answer="You have six years to make a claim.",
        detailed_breakdown=DetailedBreakdown(
            summary="The law gives you six years.",
            relevant_acts=["Statute of Limitations Act 1957"],
            key_clauses=["6 year limit"],
            caveats=["Seek legal advice"],
        )
    )


def test_evaluator_passes_high_score():
    agent = make_evaluator(score=0.9, flags=[], passed=True)
    result = agent.run(query="How long do I have?", output=sample_writer_output())
    assert result.pass_ is True
    assert result.score == 0.9


def test_evaluator_fails_low_score():
    agent = make_evaluator(score=0.5, flags=["Missing citation", "Answer too vague"], passed=False)
    result = agent.run(query="How long do I have?", output=sample_writer_output())
    assert result.pass_ is False
    assert len(result.flags) > 0


def test_evaluator_score_in_valid_range():
    agent = make_evaluator(score=0.75, flags=[], passed=True)
    result = agent.run(query="Q", output=sample_writer_output())
    assert 0.0 <= result.score <= 1.0
