import pytest
from unittest.mock import MagicMock
from irish_statute_assistant.agents.supervisor import Supervisor
from irish_statute_assistant.models.schemas import (
    ClarifierOutput, ResearcherOutput, ActSection,
    AnalystOutput, WriterOutput, DetailedBreakdown, EvaluatorOutput
)


def make_supervisor(
    clarifier_output: ClarifierOutput,
    researcher_output: ResearcherOutput,
    analyst_output: AnalystOutput,
    writer_output: WriterOutput,
    evaluator_output: EvaluatorOutput,
):
    sup = Supervisor.__new__(Supervisor)
    sup._max_refinements = 2
    sup._max_retries = 3

    sup._clarifier = MagicMock()
    sup._clarifier.run = MagicMock(return_value=clarifier_output)
    sup._clarifier.last_token_count = 0

    sup._researcher = MagicMock()
    sup._researcher.run = MagicMock(return_value=researcher_output)
    sup._researcher.last_token_count = 0

    sup._analyst = MagicMock()
    sup._analyst.run = MagicMock(return_value=analyst_output)
    sup._analyst.last_token_count = 0

    sup._writer = MagicMock()
    sup._writer.run = MagicMock(return_value=writer_output)
    sup._writer.last_token_count = 0

    sup._evaluator = MagicMock()
    sup._evaluator.run = MagicMock(return_value=evaluator_output)
    sup._evaluator.last_token_count = 0

    return sup


def make_defaults():
    research = ResearcherOutput(acts=[
        ActSection(title="Act A", url="https://example.com", sections=["Section 1"])
    ])
    analysis = AnalystOutput(key_clauses=["6 year limit"], gaps=[], confidence=0.9)
    writer_out = WriterOutput(
        short_answer="You have six years.",
        detailed_breakdown=DetailedBreakdown(
            summary="Six year limit.", relevant_acts=["Act A"],
            key_clauses=["6 year limit"], caveats=["Seek advice"],
        )
    )
    evaluator_pass = EvaluatorOutput(score=0.85, flags=[], **{"pass": True})
    return research, analysis, writer_out, evaluator_pass


def test_supervisor_returns_writer_output_when_clear_and_passes():
    research, analysis, writer_out, evaluator_pass = make_defaults()
    sup = make_supervisor(
        clarifier_output=ClarifierOutput(needs_clarification=False),
        researcher_output=research,
        analyst_output=analysis,
        writer_output=writer_out,
        evaluator_output=evaluator_pass,
    )
    result = sup.run(query="How long do I have to sue?", history="")
    assert isinstance(result, WriterOutput)


def test_supervisor_returns_clarification_question_when_needed():
    research, analysis, writer_out, evaluator_pass = make_defaults()
    sup = make_supervisor(
        clarifier_output=ClarifierOutput(needs_clarification=True, question="What type of case?"),
        researcher_output=research,
        analyst_output=analysis,
        writer_output=writer_out,
        evaluator_output=evaluator_pass,
    )
    result = sup.run(query="What are my rights?", history="")
    assert result == "What type of case?"


def test_supervisor_refinement_loop_retries_on_fail():
    research, analysis, writer_out, _ = make_defaults()
    evaluator_fail = EvaluatorOutput(score=0.5, flags=["Vague answer"], **{"pass": False})
    evaluator_pass = EvaluatorOutput(score=0.85, flags=[], **{"pass": True})

    sup = make_supervisor(
        clarifier_output=ClarifierOutput(needs_clarification=False),
        researcher_output=research,
        analyst_output=analysis,
        writer_output=writer_out,
        evaluator_output=evaluator_fail,
    )
    sup._evaluator.run = MagicMock(side_effect=[evaluator_fail, evaluator_pass])

    result = sup.run(query="How long?", history="")
    assert isinstance(result, WriterOutput)
    assert sup._evaluator.run.call_count == 2


def test_supervisor_stops_after_max_refinements():
    research, analysis, writer_out, _ = make_defaults()
    evaluator_fail = EvaluatorOutput(score=0.4, flags=["Still bad"], **{"pass": False})

    sup = make_supervisor(
        clarifier_output=ClarifierOutput(needs_clarification=False),
        researcher_output=research,
        analyst_output=analysis,
        writer_output=writer_out,
        evaluator_output=evaluator_fail,
    )
    sup._evaluator.run = MagicMock(return_value=evaluator_fail)

    result = sup.run(query="How long?", history="")
    assert isinstance(result, WriterOutput)
    assert sup._evaluator.run.call_count == sup._max_refinements + 1
