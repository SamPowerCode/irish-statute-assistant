from __future__ import annotations

from typing import Literal, Optional
from pydantic import BaseModel, Field, field_validator, model_validator


class ActSection(BaseModel):
    title: str
    url: str
    sections: list[str]


class ResearcherOutput(BaseModel):
    acts: list[ActSection] = Field(min_length=1)


class KeyClause(BaseModel):
    text: str
    act: str
    section: str


class AnalystLLMOutput(BaseModel):
    """Schema fed to the LLM via with_structured_output — no supervisor-side fields."""
    key_clauses: list[KeyClause]
    gaps: list[str]
    confidence: float = Field(ge=0.0, le=1.0)


class AnalystOutput(AnalystLLMOutput):
    """Full analyst context passed through the pipeline.

    advocate_challenges is populated by the Supervisor after the DevilsAdvocateAgent
    runs — it is intentionally absent from AnalystLLMOutput so the LLM never sees it.
    """
    advocate_challenges: list[str] = []


class DetailedBreakdown(BaseModel):
    summary: str
    relevant_acts: list[str]
    key_clauses: list[KeyClause]
    caveats: list[str]


class WriterOutput(BaseModel):
    short_answer: str
    detailed_breakdown: DetailedBreakdown
    warnings: list[str] = []
    analyst_confidence: float = 1.0

    @field_validator("short_answer")
    @classmethod
    def short_answer_max_100_words(cls, v: str) -> str:
        word_count = len(v.split())
        if word_count > 100:
            raise ValueError(f"short_answer must be ≤100 words, got {word_count}")
        return v


class EvaluatorOutput(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    flags: list[str]
    pass_: bool = Field(alias="pass")

    model_config = {"populate_by_name": True}


class AdvocateOutput(BaseModel):
    challenges: list[str] = Field(default_factory=list, max_length=5)
    severity: Literal["minor", "major"]


class GroundingOutput(BaseModel):
    ungrounded_claims: list[str]
    grounding_passed: bool


class ClarifierOutput(BaseModel):
    needs_clarification: bool
    question: Optional[str] = None

    @model_validator(mode="after")
    def question_required_when_clarification_needed(self) -> "ClarifierOutput":
        if self.needs_clarification and self.question is None:
            raise ValueError("question must be provided when needs_clarification is True")
        return self
