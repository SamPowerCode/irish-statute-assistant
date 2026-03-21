from unittest.mock import MagicMock
from irish_statute_assistant.agents.writer import WriterAgent
from irish_statute_assistant.models.schemas import (
    WriterOutput, DetailedBreakdown, AnalystOutput,
    KeyClause, ResearcherOutput, ActSection
)


def sample_key_clause():
    return KeyClause(text="6 year limit", act="Statute of Limitations Act 1957", section="s.11")


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


def sample_analyst_output(advocate_challenges=None):
    return AnalystOutput(
        key_clauses=[sample_key_clause()],
        gaps=[],
        confidence=0.9,
        advocate_challenges=advocate_challenges or [],
    )


def sample_research():
    return ResearcherOutput(acts=[
        ActSection(title="Act A", url="https://example.com", sections=["Section 1"])
    ])


def test_writer_returns_writer_output():
    agent = make_writer(
        short_answer="You have six years to make a claim.",
        summary="The Statute of Limitations sets a 6-year window.",
        relevant_acts=["Statute of Limitations Act 1957"],
        key_clauses=[sample_key_clause()],
        caveats=["Personal injury cases have a shorter 2-year limit"],
    )
    result = agent.run(
        query="How long do I have to sue?",
        analysis=sample_analyst_output(),
        research=sample_research(),
        evaluator_flags=[],
    )
    assert isinstance(result, WriterOutput)
    assert len(result.short_answer.split()) <= 100


def test_writer_serialises_key_clauses_with_citations():
    agent = make_writer(
        short_answer="Short answer.",
        summary="S", relevant_acts=[], key_clauses=[sample_key_clause()], caveats=[],
    )
    agent.run(
        query="Q", analysis=sample_analyst_output(),
        research=sample_research(), evaluator_flags=[],
    )
    call_args = agent._chain.invoke.call_args[0][0]
    assert "Statute of Limitations Act 1957" in call_args["key_clauses"]
    assert "s.11" in call_args["key_clauses"]


def test_writer_injects_advocate_challenges_when_present():
    agent = make_writer(
        short_answer="Short answer.", summary="S", relevant_acts=[],
        key_clauses=[sample_key_clause()], caveats=[],
    )
    analysis = sample_analyst_output(advocate_challenges=["Road Traffic Act may override this."])
    agent.run(query="Q", analysis=analysis, research=sample_research(), evaluator_flags=[])
    call_args = agent._chain.invoke.call_args[0][0]
    assert "Road Traffic Act may override this." in call_args["advocate_challenges"]


def test_writer_advocate_challenges_empty_when_none():
    agent = make_writer(
        short_answer="Short answer.", summary="S", relevant_acts=[],
        key_clauses=[sample_key_clause()], caveats=[],
    )
    agent.run(query="Q", analysis=sample_analyst_output(), research=sample_research(), evaluator_flags=[])
    call_args = agent._chain.invoke.call_args[0][0]
    assert call_args["advocate_challenges"] == "None"


def test_writer_passes_evaluator_flags_in_prompt():
    agent = make_writer(
        short_answer="Short answer.", summary="S", relevant_acts=[],
        key_clauses=[sample_key_clause()], caveats=[],
    )
    agent.run(
        query="Q", analysis=sample_analyst_output(),
        research=sample_research(), evaluator_flags=["Add citation"],
    )
    call_args = agent._chain.invoke.call_args[0][0]
    assert "Add citation" in call_args["evaluator_flags"]


def test_writer_injects_user_preferences_when_set():
    agent = make_writer(
        short_answer="Short answer.", summary="S", relevant_acts=[],
        key_clauses=[sample_key_clause()], caveats=[],
    )
    agent.run(
        query="Q", analysis=sample_analyst_output(),
        research=sample_research(), evaluator_flags=[],
        user_preferences={"language_level": "technical", "user_type": "solicitor"},
    )
    call_args = agent._chain.invoke.call_args[0][0]
    assert "technical" in call_args["user_preferences"]
    assert "solicitor" in call_args["user_preferences"]


def test_writer_user_preferences_none_when_empty():
    agent = make_writer(
        short_answer="Short answer.", summary="S", relevant_acts=[],
        key_clauses=[sample_key_clause()], caveats=[],
    )
    agent.run(
        query="Q", analysis=sample_analyst_output(),
        research=sample_research(), evaluator_flags=[],
        user_preferences={},
    )
    call_args = agent._chain.invoke.call_args[0][0]
    assert call_args["user_preferences"] == "None"
