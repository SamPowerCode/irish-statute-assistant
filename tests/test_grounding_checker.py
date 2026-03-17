from unittest.mock import MagicMock
from irish_statute_assistant.agents.grounding_checker import GroundingCheckerAgent
from irish_statute_assistant.models.schemas import (
    GroundingOutput, WriterOutput, DetailedBreakdown, KeyClause, ResearcherOutput, ActSection
)


def sample_writer_output(key_clauses=None):
    kc = key_clauses or [KeyClause(text="6 year limit", act="Statute of Limitations Act 1957", section="s.11")]
    return WriterOutput(
        short_answer="You have six years.",
        detailed_breakdown=DetailedBreakdown(
            summary="Six year limit.", relevant_acts=["Act A"],
            key_clauses=kc, caveats=["Seek advice"],
        )
    )


def sample_research():
    return ResearcherOutput(acts=[
        ActSection(title="Act A", url="https://example.com", sections=["Actions must be within 6 years."])
    ])


def make_checker(ungrounded_claims, grounding_passed):
    agent = GroundingCheckerAgent.__new__(GroundingCheckerAgent)
    agent._chain = MagicMock()
    agent._invoke_chain = MagicMock(return_value=GroundingOutput(
        ungrounded_claims=ungrounded_claims,
        grounding_passed=grounding_passed,
    ))
    return agent


def test_grounding_checker_passes_grounded_claims():
    agent = make_checker(ungrounded_claims=[], grounding_passed=True)
    result = agent.run(writer_output=sample_writer_output(), research=sample_research())
    assert isinstance(result, GroundingOutput)
    assert result.grounding_passed is True
    assert result.ungrounded_claims == []
    agent._invoke_chain.assert_called_once()


def test_grounding_checker_flags_ungrounded_claims():
    agent = make_checker(
        ungrounded_claims=["Claim about road tax has no source in retrieved text."],
        grounding_passed=False,
    )
    result = agent.run(writer_output=sample_writer_output(), research=sample_research())
    assert result.grounding_passed is False
    assert len(result.ungrounded_claims) == 1
    agent._invoke_chain.assert_called_once()


def test_grounding_checker_warnings_attached_to_writer_output():
    agent = make_checker(
        ungrounded_claims=["Unsupported claim."], grounding_passed=False
    )
    writer_out = sample_writer_output()
    grounding = agent.run(writer_output=writer_out, research=sample_research())
    # Supervisor attaches warnings — simulate that here
    writer_out.warnings = grounding.ungrounded_claims
    assert writer_out.warnings == ["Unsupported claim."]


def test_grounding_checker_grounding_passed_false_when_ungrounded():
    agent = make_checker(ungrounded_claims=["X"], grounding_passed=False)
    result = agent.run(writer_output=sample_writer_output(), research=sample_research())
    assert result.grounding_passed is False


def test_grounding_checker_passes_key_clauses_and_statute_text_to_chain():
    agent = make_checker(ungrounded_claims=[], grounding_passed=True)
    kc = KeyClause(text="6 year limit", act="Statute of Limitations Act 1957", section="s.11")
    agent.run(writer_output=sample_writer_output([kc]), research=sample_research())
    # _invoke_chain is called as: _invoke_chain(chain, inputs_dict)
    call_args = agent._invoke_chain.call_args[0]
    inputs = call_args[1]  # second positional arg is the inputs dict
    assert "Statute of Limitations Act 1957" in inputs["key_clauses"]
    assert "s.11" in inputs["key_clauses"]
    assert "Act A" in inputs["statute_text"]
