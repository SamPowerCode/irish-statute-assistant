"""Pydantic v2 schemas for all agent inputs and outputs.

All LLM-facing schemas are used with LangChain's with_structured_output()
to constrain model output to valid JSON. The split between AnalystLLMOutput
and AnalystOutput is intentional: advocate_challenges is populated by the
Supervisor, not the LLM, so it must not appear in the schema the LLM sees.
"""
from __future__ import annotations

from typing import Literal, Optional
from pydantic import BaseModel, Field, field_validator, model_validator


class ActSection(BaseModel):
    """A single Irish Act with its retrieved statute sections."""

    title: str
    url: str
    sections: list[str]


class ResearcherOutput(BaseModel):
    """Retrieved statute Acts and their sections."""

    acts: list[ActSection] = Field(min_length=1)


class KeyClause(BaseModel):
    """A specific legal rule with its source citation.

    Used in AnalystOutput.key_clauses and DetailedBreakdown.key_clauses.
    The act and section fields are required to enforce traceable citations.
    """

    text: str = Field(min_length=1)
    act: str = Field(min_length=1)
    section: str = Field(min_length=1)


class AnalystLLMOutput(BaseModel):
    """Schema fed to the analyst LLM via with_structured_output.

    Does not include advocate_challenges — that field is populated by the
    Supervisor after the DevilsAdvocateAgent runs. See AnalystOutput.
    """

    key_clauses: list[KeyClause]
    gaps: list[str]
    confidence: float = Field(ge=0.0, le=1.0)


class AnalystOutput(AnalystLLMOutput):
    """Full analyst context passed through the pipeline.

    Extends AnalystLLMOutput with advocate_challenges, which is injected
    by the Supervisor and passed to the writer to address in caveats.
    """

    advocate_challenges: list[str] = []


class DetailedBreakdown(BaseModel):
    """The structured body of the writer's answer."""

    summary: str
    relevant_acts: list[str]
    key_clauses: list[KeyClause]
    caveats: list[str]


class WriterOutput(BaseModel):
    """The writer's full answer, including grounding warnings set by the Supervisor."""

    short_answer: str
    detailed_breakdown: DetailedBreakdown
    warnings: list[str] = []
    analyst_confidence: float = Field(default=1.0, ge=0.0, le=1.0)

    @field_validator("short_answer")
    @classmethod
    def short_answer_max_100_words(cls, v: str) -> str:
        word_count = len(v.split())
        if word_count > 100:
            raise ValueError(f"short_answer must be ≤100 words, got {word_count}")
        return v


class EvaluatorOutput(BaseModel):
    """Quality score and flags from the evaluator.

    pass_ uses a Pydantic alias because 'pass' is a Python reserved word.
    Access it as result.pass_ in code; it serialises as 'pass' in JSON.
    """

    score: float = Field(ge=0.0, le=1.0)
    flags: list[str]
    pass_: bool = Field(alias="pass")

    model_config = {"populate_by_name": True}


class AdvocateOutput(BaseModel):
    """Challenges raised by the devil's advocate."""

    challenges: list[str] = Field(default_factory=list, max_length=5)
    severity: Literal["minor", "major"]


class GroundingOutput(BaseModel):
    """Result of the grounding check — which claims are supported by retrieved text."""

    ungrounded_claims: list[str]
    grounding_passed: bool


class ClarifierOutput(BaseModel):
    """Decision from the clarifier — whether to ask a question or proceed."""

    needs_clarification: bool
    question: Optional[str] = None

    @model_validator(mode="after")
    def question_required_when_clarification_needed(self) -> "ClarifierOutput":
        if self.needs_clarification and self.question is None:
            raise ValueError("question must be provided when needs_clarification is True")
        return self
