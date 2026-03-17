from unittest.mock import MagicMock
import pytest
from irish_statute_assistant.agents.devils_advocate import DevilsAdvocateAgent
from irish_statute_assistant.models.schemas import (
    AdvocateOutput, AnalystOutput, KeyClause, ResearcherOutput, ActSection
)


def sample_analyst_output(confidence=0.9):
    kc = KeyClause(text="6 year limit", act="Statute of Limitations Act 1957", section="s.11")
    return AnalystOutput(key_clauses=[kc], gaps=[], confidence=confidence)


def sample_research():
    return ResearcherOutput(acts=[
        ActSection(title="Act A", url="https://example.com", sections=["Section text"])
    ])


def make_advocate(challenges, severity):
    agent = DevilsAdvocateAgent.__new__(DevilsAdvocateAgent)
    return_value = AdvocateOutput(challenges=challenges, severity=severity)
    standard_mock = MagicMock()
    standard_mock.invoke = MagicMock(return_value=return_value)
    strict_mock = MagicMock()
    strict_mock.invoke = MagicMock(return_value=return_value)
    agent._chain_standard = standard_mock
    agent._chain_strict = strict_mock
    return agent


def test_advocate_returns_advocate_output_on_weak_analysis():
    agent = make_advocate(
        challenges=["Missing exception for minors."], severity="minor"
    )
    result = agent.run(
        analyst_output=sample_analyst_output(confidence=0.6),
        query="How long do I have to sue?",
        research=sample_research(),
        mode="standard",
    )
    assert isinstance(result, AdvocateOutput)
    assert len(result.challenges) >= 1
    agent._chain_standard.invoke.assert_called_once()
    agent._chain_strict.invoke.assert_not_called()


def test_advocate_returns_empty_challenges_on_strong_analysis():
    agent = make_advocate(challenges=[], severity="minor")
    result = agent.run(
        analyst_output=sample_analyst_output(confidence=0.95),
        query="Q",
        research=sample_research(),
        mode="standard",
    )
    assert result.challenges == []
    assert result.severity == "minor"
    agent._chain_standard.invoke.assert_called_once()
    agent._chain_strict.invoke.assert_not_called()


def test_advocate_severity_major_on_serious_gap():
    agent = make_advocate(
        challenges=["Road Traffic Acts override this entirely."], severity="major"
    )
    result = agent.run(
        analyst_output=sample_analyst_output(),
        query="Q",
        research=sample_research(),
    )
    assert result.severity == "major"
    agent._chain_standard.invoke.assert_called_once()
    agent._chain_strict.invoke.assert_not_called()


def test_advocate_uses_strict_chain_in_strict_mode():
    agent = DevilsAdvocateAgent.__new__(DevilsAdvocateAgent)
    standard_chain = MagicMock()
    strict_chain = MagicMock()
    strict_chain.invoke = MagicMock(return_value=AdvocateOutput(challenges=["c1", "c2", "c3"], severity="major"))
    standard_chain.invoke = MagicMock(return_value=AdvocateOutput(challenges=["c1"], severity="minor"))
    agent._chain_standard = standard_chain
    agent._chain_strict = strict_chain

    result = agent.run(
        analyst_output=sample_analyst_output(),
        query="Q",
        research=sample_research(),
        mode="strict",
    )
    strict_chain.invoke.assert_called_once()
    standard_chain.invoke.assert_not_called()
    assert len(result.challenges) == 3
