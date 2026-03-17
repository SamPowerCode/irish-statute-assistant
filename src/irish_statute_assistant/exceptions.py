"""Typed exception hierarchy for Irish Statute Assistant."""


class IrishStatuteError(Exception):
    """Base exception for all Irish Statute Assistant errors."""


class TransientError(IrishStatuteError):
    """Retryable error: network timeouts, HTTP 5xx responses."""


class ValidationRepairError(IrishStatuteError):
    """LLM output failed schema validation and retries are exhausted."""


class BudgetExceededError(IrishStatuteError):
    """Token budget for this query has been consumed."""


class StatuteNotFoundError(IrishStatuteError):
    """No statutes matched the query."""


class FatalError(IrishStatuteError):
    """Unrecoverable error — wraps unexpected exceptions."""
