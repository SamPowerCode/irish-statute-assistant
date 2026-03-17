from unittest.mock import MagicMock
from irish_statute_assistant.agents.evaluator import EvaluatorAgent
from irish_statute_assistant.models.schemas import EvaluatorOutput, WriterOutput, DetailedBreakdown, KeyClause


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
    kc = KeyClause(text="Six year limit", act="Statute of Limitations Act 1957", section="s.11")
    return WriterOutput(
        short_answer="You have six years to make a claim.",
        detailed_breakdown=DetailedBreakdown(
            summary="The law gives you six years.",
            relevant_acts=["Statute of Limitations Act 1957"],
            key_clauses=[kc],
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


def test_evaluator_key_clauses_includes_citation():
    """Evaluator must serialise KeyClause objects with act and section."""
    from irish_statute_assistant.models.schemas import KeyClause, DetailedBreakdown, WriterOutput
    from irish_statute_assistant.agents.evaluator import EvaluatorAgent
    from unittest.mock import MagicMock
    from irish_statute_assistant.models.schemas import EvaluatorOutput

    kc = KeyClause(text="6 year limit", act="Statute of Limitations Act 1957", section="s.11")
    writer_out = WriterOutput(
        short_answer="You have six years.",
        detailed_breakdown=DetailedBreakdown(
            summary="Six year limit.", relevant_acts=["Act A"],
            key_clauses=[kc], caveats=["Seek advice"],
        )
    )
    agent = EvaluatorAgent.__new__(EvaluatorAgent)
    mock_chain = MagicMock()
    mock_chain.invoke = MagicMock(return_value=EvaluatorOutput(score=0.8, flags=[], **{"pass": True}))
    agent._chain = mock_chain
    agent._threshold = 0.7
    agent.run(query="Q", output=writer_out)
    call_args = agent._chain.invoke.call_args[0][0]
    assert "Statute of Limitations Act 1957" in call_args["key_clauses"]
    assert "s.11" in call_args["key_clauses"]


def test_evaluator_grounding_failed_lowers_citation_score():
    """When grounding_passed=False, system prompt must warn about unverified claims."""
    from irish_statute_assistant.models.schemas import KeyClause, DetailedBreakdown, WriterOutput
    from irish_statute_assistant.agents.evaluator import EvaluatorAgent
    from unittest.mock import MagicMock
    from irish_statute_assistant.models.schemas import EvaluatorOutput

    kc = KeyClause(text="Clause", act="Act A", section="s.1")
    writer_out = WriterOutput(
        short_answer="Short.",
        detailed_breakdown=DetailedBreakdown(
            summary="S.", relevant_acts=[], key_clauses=[kc], caveats=[]
        )
    )
    agent = EvaluatorAgent.__new__(EvaluatorAgent)
    mock_chain = MagicMock()
    mock_chain.invoke = MagicMock(return_value=EvaluatorOutput(score=0.8, flags=[], **{"pass": True}))
    agent._chain = mock_chain
    agent._threshold = 0.7
    agent.run(query="Q", output=writer_out, grounding_passed=False)
    call_args = agent._chain.invoke.call_args[0][0]
    assert "unverified" in call_args["grounding_note"]
