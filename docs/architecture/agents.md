# Agents

## ClarifierAgent

Runs first, before any statute retrieval. Reads conversation history from
`ConversationStore.format_for_prompt()` and decides in one LLM call whether
the query is specific enough to proceed. If not, it returns a single focused
question — never asks about jurisdiction (always Irish law).

**Input:** `query: str`, `history: str`
**Output:** `ClarifierOutput(needs_clarification: bool, question: str | None)`

## ResearcherAgent

Searches the vector store for statute sections relevant to the query. Uses the
configured embedding model (`all-MiniLM-L6-v2` by default) to find semantically
similar Act sections. Falls back to a live HTTP fetch from irishstatutebook.ie
if the store is empty, with rate limiting and retry logic.

**Input:** `query: str`
**Output:** `ResearcherOutput(acts: list[ActSection])`

## AnalystAgent

Reads the retrieved statute text and produces structured findings. Returns
`AnalystLLMOutput` (not `AnalystOutput`) — the Supervisor wraps the result into
a full `AnalystOutput` after the devil's advocate runs. This split is intentional:
the `advocate_challenges` field must not appear in the JSON schema the LLM sees.

**Input:** `query: str`, `research: ResearcherOutput`
**Output:** `AnalystLLMOutput(key_clauses: list[KeyClause], gaps: list[str], confidence: float)`

## DevilsAdvocateAgent

Challenges the analyst's findings before the writer proceeds. Two modes:

- **Standard** — finds 1–3 weaknesses: missing exceptions, overriding statutes,
  or claims not supported by the retrieved text
- **Strict** — adversarial; aims for up to 5 challenges. Used when analyst
  confidence is below 0.5, or when re-running after a major-severity result

Challenges are injected into `AnalystOutput.advocate_challenges` and passed to
the writer, which must address them in caveats.

**Input:** `analyst_output`, `query`, `research`, `mode: "standard" | "strict"`
**Output:** `AdvocateOutput(challenges: list[str], severity: "minor" | "major")`

## WriterAgent

Produces the user-facing answer from the analyst's key clauses, devil's advocate
challenges, and any evaluator flags from a previous iteration. The short answer
must be 100 words or fewer. Each key clause is formatted with its act and section
citation.

**Input:** `query`, `analysis: AnalystOutput`, `research`, `evaluator_flags: list[str]`
**Output:** `WriterOutput`

## GroundingCheckerAgent

Cross-references each `KeyClause` in the writer's output against the retrieved
statute text. Any claim that cannot be traced to the source text is added to
`ungrounded_claims`. The Supervisor attaches these to `WriterOutput.warnings`.
When `grounding_passed=False`, the evaluator penalises citation quality.

**Input:** `writer_output: WriterOutput`, `research: ResearcherOutput`
**Output:** `GroundingOutput(ungrounded_claims: list[str], grounding_passed: bool)`

## EvaluatorAgent

Scores the writer's output on four criteria: accuracy, completeness, citation
quality, and plain English. If `grounding_passed=False`, citation quality is
capped at 0.4 in the scoring prompt. Returns `pass_=True` if the overall score
meets `EVALUATOR_PASS_THRESHOLD`. Flags are passed back to the writer on retry.

**Input:** `query`, `output: WriterOutput`, `grounding_passed: bool`
**Output:** `EvaluatorOutput(score: float, flags: list[str], pass_: bool)`

## Supervisor

Orchestrates all agents in the correct order and owns all memory writes. Key
responsibilities:

- Constructs `AnalystOutput` from the analyst's `AnalystLLMOutput` result
- Runs the confidence gate to set `effective_refinements` and `advocate_mode_on_retry`
- Runs `_update_flag_counts` before checking `evaluation.pass_` (so counts are
  accurate for both passing and failing evaluations)
- Calls `_detect_and_save_preferences` on every returned `WriterOutput` (both
  early success and exhausted-refinements paths)
- Detects user preferences from query text and from repeated evaluator signals

## BaseAgent

All agents extend `BaseAgent`, which provides `_invoke_chain(chain, inputs)`.
This method wraps every LLM call with a `TokenUsageCallback` that reads
`usage_metadata` from the response message — compatible with all four supported
providers. The token count is exposed as `agent.last_token_count` and consumed
by the Supervisor into the `QueryContext` budget tracker.
