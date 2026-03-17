"""QueryContext — carries budget and query state through the pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from uuid import uuid4

from irish_statute_assistant.exceptions import BudgetExceededError


@dataclass
class QueryContext:
    budget: int
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
