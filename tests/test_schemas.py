import pytest
from pydantic import ValidationError
from irish_statute_assistant.models.schemas import (
    ActSection,
    ResearcherOutput,
    KeyClause,
    AnalystLLMOutput,
    AnalystOutput,
    DetailedBreakdown,
    WriterOutput,
    EvaluatorOutput,
    ClarifierOutput,
    AdvocateOutput,
    GroundingOutput,
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


# --- KeyClause ---

def test_key_clause_valid():
    kc = KeyClause(text="Actions must start within 6 years.", act="Statute of Limitations Act 1957", section="s.11")
    assert kc.text == "Actions must start within 6 years."

def test_key_clause_requires_act():
    with pytest.raises(ValidationError):
        KeyClause(text="something", section="s.1")

def test_key_clause_requires_section():
    with pytest.raises(ValidationError):
        KeyClause(text="something", act="Some Act")


# --- AnalystLLMOutput ---

def test_analyst_llm_output_valid():
    kc = KeyClause(text="6 year limit", act="Act A", section="s.1")
    data = AnalystLLMOutput(key_clauses=[kc], gaps=[], confidence=0.9)
    assert data.confidence == 0.9

def test_analyst_llm_output_has_no_advocate_challenges_field():
    """advocate_challenges must not appear in the JSON schema used by the LLM."""
    schema = AnalystLLMOutput.model_json_schema()
    assert "advocate_challenges" not in schema.get("properties", {})


# --- AnalystOutput ---

def test_analyst_output_valid():
    kc = KeyClause(text="Clause 1", act="Act A", section="s.1")
    data = AnalystOutput(key_clauses=[kc], gaps=[], confidence=0.85)
    assert data.confidence == 0.85


def test_analyst_output_has_advocate_challenges():
    kc = KeyClause(text="Clause", act="Act A", section="s.1")
    data = AnalystOutput(key_clauses=[kc], gaps=[], confidence=0.8, advocate_challenges=["Challenge 1"])
    assert data.advocate_challenges == ["Challenge 1"]

def test_analyst_output_advocate_challenges_defaults_empty():
    kc = KeyClause(text="Clause", act="Act A", section="s.1")
    data = AnalystOutput(key_clauses=[kc], gaps=[], confidence=0.8)
    assert data.advocate_challenges == []


def test_analyst_confidence_above_1_rejected():
    with pytest.raises(ValidationError):
        AnalystOutput(key_clauses=[], gaps=[], confidence=1.5)


def test_analyst_confidence_below_0_rejected():
    with pytest.raises(ValidationError):
        AnalystOutput(key_clauses=[], gaps=[], confidence=-0.1)


# --- WriterOutput ---

def test_writer_output_valid():
    kc = KeyClause(text="You must do X", act="Act A", section="s.1")
    breakdown = DetailedBreakdown(
        summary="Summary text",
        relevant_acts=["Act A"],
        key_clauses=[kc],
        caveats=["This may vary"],
    )
    data = WriterOutput(short_answer="You have two years to make a claim.", detailed_breakdown=breakdown)
    assert data.short_answer.startswith("You")


def test_writer_short_answer_over_100_words_rejected():
    long_answer = " ".join(["word"] * 101)
    kc = KeyClause(text="x", act="x", section="x")
    breakdown = DetailedBreakdown(summary="x", relevant_acts=[], key_clauses=[kc], caveats=[])
    with pytest.raises(ValidationError):
        WriterOutput(short_answer=long_answer, detailed_breakdown=breakdown)


def test_writer_output_warnings_defaults_empty():
    kc = KeyClause(text="You must X", act="Act A", section="s.1")
    breakdown = DetailedBreakdown(
        summary="Summary", relevant_acts=["Act A"],
        key_clauses=[kc], caveats=["Seek advice"],
    )
    data = WriterOutput(short_answer="Short answer here.", detailed_breakdown=breakdown)
    assert data.warnings == []
    assert data.analyst_confidence == 1.0


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


# --- AdvocateOutput ---

def test_advocate_output_valid():
    data = AdvocateOutput(challenges=["Challenge 1", "Challenge 2"], severity="minor")
    assert len(data.challenges) == 2
    assert data.severity == "minor"


def test_advocate_output_challenges_max_length_exceeded():
    with pytest.raises(ValidationError):
        AdvocateOutput(challenges=["c1", "c2", "c3", "c4", "c5", "c6"], severity="major")


def test_advocate_output_invalid_severity():
    with pytest.raises(ValidationError):
        AdvocateOutput(challenges=[], severity="critical")


# --- GroundingOutput ---

def test_grounding_output_valid():
    data = GroundingOutput(ungrounded_claims=[], grounding_passed=True)
    assert data.grounding_passed is True
    assert data.ungrounded_claims == []


def test_grounding_output_grounding_passed_false():
    data = GroundingOutput(ungrounded_claims=["Claim A is unsupported"], grounding_passed=False)
    assert data.grounding_passed is False
