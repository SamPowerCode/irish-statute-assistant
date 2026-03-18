# Schemas

All agent inputs and outputs are typed Pydantic v2 models defined in
`src/irish_statute_assistant/models/schemas.py`.

## KeyClause

```python
class KeyClause(BaseModel):
    text: str     # the rule in plain English
    act: str      # full Act name, e.g. "Statute of Limitations Act 1957"
    section: str  # section reference, e.g. "s.11"
```

`KeyClause` enforces structured citations throughout the pipeline. Every key
point in the analyst's output, the writer's breakdown, and the grounding check
must carry an act name and section reference. This makes it possible to verify
claims against source text and to present traceable citations to the user.

## Split-schema pattern: AnalystLLMOutput and AnalystOutput

```python
class AnalystLLMOutput(BaseModel):
    key_clauses: list[KeyClause]
    gaps: list[str]
    confidence: float  # 0.0–1.0

class AnalystOutput(AnalystLLMOutput):
    advocate_challenges: list[str] = []  # populated by Supervisor, not the LLM
```

`AnalystLLMOutput` is what the analyst LLM is bound to via
`with_structured_output(AnalystLLMOutput)`. The LLM never sees `advocate_challenges`
in its JSON schema. After the devil's advocate runs, the Supervisor constructs a
full `AnalystOutput`:

```python
analyst_output = AnalystOutput(**llm_result.model_dump(), advocate_challenges=[])
# ... later, after advocate runs:
analyst_output.advocate_challenges = advocate_result.challenges
```

## EvaluatorOutput and the `pass_` alias

```python
class EvaluatorOutput(BaseModel):
    score: float
    flags: list[str]
    pass_: bool = Field(alias="pass")

    model_config = {"populate_by_name": True}
```

`pass` is a Python reserved word. Pydantic's alias mechanism maps the JSON key
`"pass"` to the Python attribute `pass_`. When constructing in tests:

```python
EvaluatorOutput(score=0.85, flags=[], **{"pass": True})
```

## WriterOutput

```python
class WriterOutput(BaseModel):
    short_answer: str              # ≤100 words, validated
    detailed_breakdown: DetailedBreakdown
    warnings: list[str] = []      # ungrounded claims, set by Supervisor
    analyst_confidence: float = 1.0  # set by Supervisor from AnalystOutput
```

`warnings` and `analyst_confidence` are not populated by the writer LLM — they
are set by the Supervisor after the grounding check and before returning the result.

## AdvocateOutput

```python
class AdvocateOutput(BaseModel):
    challenges: list[str] = Field(default_factory=list, max_length=5)
    severity: Literal["minor", "major"]
```

An empty `challenges` list with `severity="minor"` is valid — it means the
analyst's output was unchallenged.

## GroundingOutput

```python
class GroundingOutput(BaseModel):
    ungrounded_claims: list[str]
    grounding_passed: bool
```

## ClarifierOutput

```python
class ClarifierOutput(BaseModel):
    needs_clarification: bool
    question: str | None = None
```

A `model_validator` enforces that `question` is present when
`needs_clarification=True`.
