from unittest.mock import MagicMock
from irish_statute_assistant.agents.analyst import AnalystAgent
from irish_statute_assistant.models.schemas import AnalystOutput, ResearcherOutput, ActSection


def make_analyst(key_clauses, gaps, confidence):
    agent = AnalystAgent.__new__(AnalystAgent)
    mock_chain = MagicMock()
    mock_chain.invoke = MagicMock(return_value=AnalystOutput(
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


def test_analyst_returns_analyst_output():
    agent = make_analyst(["Bring action within 6 years"], [], 0.9)
    result = agent.run(query="limitation period", research=sample_research(), evaluator_flags=[])
    assert isinstance(result, AnalystOutput)
    assert result.confidence == 0.9


def test_analyst_passes_evaluator_flags():
    agent = make_analyst(["Clause"], [], 0.8)
    agent.run(query="Q", research=sample_research(), evaluator_flags=["Missing citation"])
    call_args = agent._chain.invoke.call_args[0][0]
    assert "Missing citation" in call_args["evaluator_flags"]


def test_analyst_confidence_in_valid_range():
    agent = make_analyst([], [], 0.5)
    result = agent.run(query="Q", research=sample_research(), evaluator_flags=[])
    assert 0.0 <= result.confidence <= 1.0
