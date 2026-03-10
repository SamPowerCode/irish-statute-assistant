import pytest
from pydantic import ValidationError
from irish_statute_assistant.models.schemas import (
    ActSection,
    ResearcherOutput,
    AnalystOutput,
    DetailedBreakdown,
    WriterOutput,
    EvaluatorOutput,
    ClarifierOutput,
)


# --- ResearcherOutput ---

def test_researcher_output_valid():
    data = ResearcherOutput(acts=[
        ActSection(title="Act A", url="https://example.com/act-a", sections=["Section 1 text"])
    ])
    assert len(data.acts) == 1


def test_researcher_output_empty_acts_rejected():
    with pytest.raises(ValidationError):
        ResearcherOutput(acts=[])


# --- AnalystOutput ---

def test_analyst_output_valid():
    data = AnalystOutput(key_clauses=["Clause 1"], gaps=[], confidence=0.85)
    assert data.confidence == 0.85


def test_analyst_confidence_above_1_rejected():
    with pytest.raises(ValidationError):
        AnalystOutput(key_clauses=[], gaps=[], confidence=1.5)


def test_analyst_confidence_below_0_rejected():
    with pytest.raises(ValidationError):
        AnalystOutput(key_clauses=[], gaps=[], confidence=-0.1)


# --- WriterOutput ---

def test_writer_output_valid():
    breakdown = DetailedBreakdown(
        summary="Summary text",
        relevant_acts=["Act A"],
        key_clauses=["You must do X"],
        caveats=["This may vary"],
    )
    data = WriterOutput(short_answer="You have two years to make a claim.", detailed_breakdown=breakdown)
    assert data.short_answer.startswith("You")


def test_writer_short_answer_over_100_words_rejected():
    long_answer = " ".join(["word"] * 101)
    breakdown = DetailedBreakdown(summary="x", relevant_acts=[], key_clauses=[], caveats=[])
    with pytest.raises(ValidationError):
        WriterOutput(short_answer=long_answer, detailed_breakdown=breakdown)


# --- EvaluatorOutput ---

def test_evaluator_output_pass():
    data = EvaluatorOutput(score=0.8, flags=[], **{"pass": True})
    assert data.pass_ is True


def test_evaluator_score_above_1_rejected():
    with pytest.raises(ValidationError):
        EvaluatorOutput(score=1.1, flags=[], **{"pass": True})


# --- ClarifierOutput ---

def test_clarifier_needs_clarification():
    data = ClarifierOutput(needs_clarification=True, question="Which county are you in?")
    assert data.question is not None


def test_clarifier_no_clarification_needed():
    data = ClarifierOutput(needs_clarification=False)
    assert data.question is None


def test_clarifier_question_required_when_needs_clarification_true():
    with pytest.raises(ValidationError):
        ClarifierOutput(needs_clarification=True, question=None)
