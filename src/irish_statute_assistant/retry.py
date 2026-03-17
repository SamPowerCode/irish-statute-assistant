"""Retry helper for agent calls that may fail with validation errors."""
from __future__ import annotations

import json
import logging
from collections.abc import Callable
from typing import Any

from pydantic import ValidationError

from irish_statute_assistant.exceptions import ValidationRepairError

logger = logging.getLogger(__name__)

_RETRYABLE = (ValidationError, ValueError, json.JSONDecodeError)


def run_with_retry(fn: Callable[[], Any], max_retries: int) -> Any:
    """Call fn(), retrying on validation/parse errors up to max_retries times."""
    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except _RETRYABLE as e:
            last_error = e
            logger.warning("Agent attempt %d/%d failed: %s", attempt + 1, max_retries + 1, e)
    raise ValidationRepairError(
        f"Agent output failed validation after {max_retries + 1} attempts"
    ) from last_error
