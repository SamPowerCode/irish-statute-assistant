from unittest.mock import MagicMock
from irish_statute_assistant.agents.writer import WriterAgent
from irish_statute_assistant.models.schemas import (
    WriterOutput, DetailedBreakdown, AnalystOutput, ResearcherOutput, ActSection
)


def make_writer(short_answer, summary, relevant_acts, key_clauses, caveats):
    agent = WriterAgent.__new__(WriterAgent)
    mock_chain = MagicMock()
    mock_chain.invoke = MagicMock(return_value=WriterOutput(
        short_answer=short_answer,
        detailed_breakdown=DetailedBreakdown(
            summary=summary,
            relevant_acts=relevant_acts,
            key_clauses=key_clauses,
            caveats=caveats,
        )
    ))
    agent._chain = mock_chain
    return agent


def sample_analyst_output():
    return AnalystOutput(key_clauses=["6 year limit"], gaps=[], confidence=0.9)


def sample_research():
    return ResearcherOutput(acts=[
        ActSection(title="Act A", url="https://example.com", sections=["Section 1"])
    ])


def test_writer_returns_writer_output():
    agent = make_writer(
        short_answer="You have six years to make a claim.",
        summary="The Statute of Limitations sets a 6-year window.",
        relevant_acts=["Statute of Limitations Act 1957"],
        key_clauses=["6 year limit from date of cause of action"],
        caveats=["Personal injury cases have a shorter 2-year limit"],
    )
    result = agent.run(query="How long do I have to sue?", analysis=sample_analyst_output(), research=sample_research(), evaluator_flags=[])
    assert isinstance(result, WriterOutput)
    assert len(result.short_answer.split()) <= 100


def test_writer_short_answer_is_plain_english():
    agent = make_writer(
        short_answer="You have six years to make a claim in most cases.",
        summary="Summary", relevant_acts=[], key_clauses=[], caveats=[],
    )
    result = agent.run(query="Q", analysis=sample_analyst_output(), research=sample_research(), evaluator_flags=[])
    assert "sub judice" not in result.short_answer
    assert "inter alia" not in result.short_answer


def test_writer_passes_evaluator_flags_in_prompt():
    agent = make_writer(
        short_answer="Short answer.", summary="S", relevant_acts=[], key_clauses=[], caveats=[],
    )
    agent.run(query="Q", analysis=sample_analyst_output(), research=sample_research(), evaluator_flags=["Add citation"])
    call_args = agent._chain.invoke.call_args[0][0]
    assert "Add citation" in call_args["evaluator_flags"]
