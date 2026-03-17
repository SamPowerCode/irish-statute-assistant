from unittest.mock import MagicMock
from irish_statute_assistant.agents.analyst import AnalystAgent
from irish_statute_assistant.models.schemas import (
    AnalystLLMOutput, KeyClause, ResearcherOutput, ActSection
)


def make_analyst(key_clauses, gaps, confidence):
    agent = AnalystAgent.__new__(AnalystAgent)
    mock_chain = MagicMock()
    mock_chain.invoke = MagicMock(return_value=AnalystLLMOutput(
        key_clauses=key_clauses, gaps=gaps, confidence=confidence
    ))
    agent._chain = mock_chain
    return agent


def sample_research():
    return ResearcherOutput(acts=[
        ActSection(
            title="Statute of Limitations Act 1957",
            url="https://example.com/1957",
            sections=["Actions must be brought within 6 years."]
        )
    ])


def sample_key_clause():
    return KeyClause(text="Bring action within 6 years", act="Statute of Limitations Act 1957", section="s.11")


def test_analyst_returns_analyst_llm_output():
    agent = make_analyst([sample_key_clause()], [], 0.9)
    result = agent.run(query="limitation period", research=sample_research())
    assert isinstance(result, AnalystLLMOutput)
    assert result.confidence == 0.9


def test_analyst_confidence_in_valid_range():
    agent = make_analyst([], [], 0.5)
    result = agent.run(query="Q", research=sample_research())
    assert 0.0 <= result.confidence <= 1.0


def test_analyst_run_does_not_accept_evaluator_flags():
    """Analyst runs once before the loop; evaluator_flags no longer belong here."""
    import inspect
    sig = inspect.signature(AnalystAgent.run)
    assert "evaluator_flags" not in sig.parameters
