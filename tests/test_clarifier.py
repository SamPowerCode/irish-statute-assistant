import pytest
from unittest.mock import MagicMock, patch
from irish_statute_assistant.agents.clarifier import ClarifierAgent
from irish_statute_assistant.models.schemas import ClarifierOutput


def make_clarifier(needs_clarification: bool, question: str | None = None):
    agent = ClarifierAgent.__new__(ClarifierAgent)
    mock_chain = MagicMock()
    mock_chain.invoke = MagicMock(return_value=ClarifierOutput(
        needs_clarification=needs_clarification,
        question=question,
    ))
    agent._chain = mock_chain
    return agent


def test_clarifier_returns_clarifier_output_when_ambiguous():
    agent = make_clarifier(needs_clarification=True, question="Which county do you live in?")
    result = agent.run(query="What are my rights?", history="")
    assert result.needs_clarification is True
    assert result.question is not None


def test_clarifier_returns_no_clarification_when_clear():
    agent = make_clarifier(needs_clarification=False)
    result = agent.run(query="What is the statute of limitations for personal injury in Ireland?", history="")
    assert result.needs_clarification is False
    assert result.question is None


def test_clarifier_passes_history_to_chain():
    agent = make_clarifier(needs_clarification=False)
    agent.run(query="Some question", history="User: previous\nAssistant: answer")
    call_args = agent._chain.invoke.call_args[0][0]
    assert "previous" in call_args["history"]
