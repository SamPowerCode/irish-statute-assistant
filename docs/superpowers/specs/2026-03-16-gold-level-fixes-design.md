# Gold-Level Fixes Design

**Date:** 2026-03-16
**Project:** Irish Statute Assistant
**Goal:** Elevate the project from Silver/borderline-Gold to a clean Gold submission by addressing all identified gaps.

---

## Context

A spec compliance review identified the following gaps preventing a clean Gold rating:

| Gap | Spec Requirement |
|-----|-----------------|
| Validation error regeneration absent | Req D — Error Handling |
| `token_budget_per_query` declared but never enforced | Req E — Safety Guardrails |
| `rate_limit_delay` config not wired to `statute_fetcher.py` | Req E — Safety Guardrails |
| `max_retries` config not wired to `statute_fetcher.py` | Req D — Error Handling |
| No typed exception hierarchy | Silver/Gold requirement |
| No adversarial tests | Gold requirement |

---

## Approach: Unified Pipeline Context (QueryContext)

Rather than patching each gap independently, we introduce a thin `QueryContext` dataclass that carries budget state, retry state, and a query ID through the pipeline. All cross-cutting concerns (budget enforcement, retry logic, error types) become explicit, testable, and co-located rather than scattered across files.

---

## Section 1: Typed Exception Hierarchy

**New file:** `src/irish_statute_assistant/exceptions.py`

```
IrishStatuteError (base)
├── TransientError          # retryable: network timeouts, HTTP 5xx
├── ValidationRepairError   # LLM output failed schema validation, retries exhausted
├── BudgetExceededError     # token budget for this query consumed
├── StatuteNotFoundError    # no statutes matched the query
└── FatalError              # unrecoverable — wraps unexpected exceptions
```

**Changes:**
- `researcher.py`: two bare `ValueError` raises become `StatuteNotFoundError`
- `main.py`: bare `except Exception` wraps unknown errors as `FatalError`, with specific handling for each typed exception

---

## Section 2: QueryContext

**New file:** `src/irish_statute_assistant/context.py`

```python
@dataclass
class QueryContext:
    budget: int                    # from config.token_budget_per_query
    query_id: str = field(default_factory=lambda: uuid4().hex[:8])
    tokens_used: int = 0

    def consume(self, tokens: int) -> None:
        """Record token usage; raise BudgetExceededError if limit exceeded."""
        self.tokens_used += tokens
        if self.tokens_used > self.budget:
            raise BudgetExceededError(
                f"Token budget {self.budget} exceeded (used {self.tokens_used})"
            )

    @property
    def remaining(self) -> int:
        return max(0, self.budget - self.tokens_used)

    def summary(self) -> dict:
        return {
            "query_id": self.query_id,
            "tokens_used": self.tokens_used,
            "budget": self.budget,
        }
```

**Integration:**
- `Pipeline.query()` creates one `QueryContext` per query from `config.token_budget_per_query`
- `Supervisor.run()` accepts a `QueryContext` parameter
- After each agent returns, the supervisor calls `context.consume(tokens)` using `response.usage_metadata["total_tokens"]` from LangChain response metadata
- `BudgetExceededError` surfaces to `main.py` with a clear user-facing message

---

## Section 3: Validation Error Retry

**New file:** `src/irish_statute_assistant/retry.py`

```python
def run_with_retry(fn: Callable, max_retries: int) -> Any:
    """Call fn(), retrying on Pydantic ValidationError up to max_retries times."""
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except ValidationError as e:
            last_error = e
            # log attempt and error details
    raise ValidationRepairError(
        f"Agent output failed validation after {max_retries + 1} attempts"
    ) from last_error
```

**Integration:**
- Each agent call in `supervisor.py` is wrapped with `run_with_retry`
- Agent implementations are unchanged — retry logic lives entirely in the supervisor
- `ValidationRepairError` is caught by `main.py` with a specific user-facing message

---

## Section 4: Config Wiring

**`statute_fetcher.py` changes:**
- Remove module-level `RATE_LIMIT_DELAY = 1.0` constant
- Accept `rate_limit_delay: float` and `max_retries: int` as constructor parameters
- Replace hardcoded `stop_after_attempt(3)` with dynamic retry using the `Retrying` context manager from tenacity (since tenacity decorators are applied at class definition time, not instance time)

**`Pipeline.__init__` changes:**
- Pass `config.rate_limit_delay` and `config.max_retries` when constructing `StatuteFetcher`

**Result:** Single source of truth. Changing `config.rate_limit_delay` or `config.max_retries` propagates everywhere automatically.

---

## Section 5: Adversarial Tests

**New file:** `tests/test_adversarial.py`

Five test cases, all using mocks (no live API or network calls):

1. **Prompt injection** — query containing injection text (e.g. `"Ignore all previous instructions"`). Assert the clarifier treats it as a normal legal query without leaking special output.

2. **Budget exhaustion mid-pipeline** — mock agents return high token counts. Assert `BudgetExceededError` is raised after the budget is hit and the pipeline does not proceed further.

3. **Persistent validation failure** — mock LLM always returns malformed output. Assert `ValidationRepairError` is raised after `max_retries + 1` attempts, not a bare Pydantic `ValidationError`.

4. **Empty/nonsense query** — empty string and whitespace-only input. Assert `ClarifierAgent` returns `needs_clarification=True` rather than the pipeline proceeding blindly.

5. **Statute not found** — mock researcher returns nothing. Assert `StatuteNotFoundError` is raised and surfaced cleanly (not a bare `ValueError` or index error).

---

## Files Changed

| File | Change |
|------|--------|
| `src/irish_statute_assistant/exceptions.py` | **New** — typed exception hierarchy |
| `src/irish_statute_assistant/context.py` | **New** — QueryContext dataclass |
| `src/irish_statute_assistant/retry.py` | **New** — run_with_retry helper |
| `src/irish_statute_assistant/agents/supervisor.py` | Accept QueryContext; wrap agent calls with run_with_retry; consume tokens |
| `src/irish_statute_assistant/agents/researcher.py` | Replace ValueError with StatuteNotFoundError |
| `src/irish_statute_assistant/tools/statute_fetcher.py` | Accept rate_limit_delay and max_retries in constructor; use Retrying context manager |
| `src/irish_statute_assistant/pipeline.py` | Create QueryContext per query; pass config values to StatuteFetcher |
| `src/irish_statute_assistant/main.py` | Handle typed exceptions with specific messages |
| `tests/test_adversarial.py` | **New** — 5 adversarial test cases |

---

## What Does Not Change

- All five agent implementations (Clarifier, Researcher, Analyst, Writer, Evaluator)
- All existing schemas in `models/schemas.py`
- All existing tests (they should continue to pass)
- Memory and vector store implementations
- The refinement loop logic in `supervisor.py` (only the agent call sites change)
