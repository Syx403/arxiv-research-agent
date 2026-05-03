from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TypeVar

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential_jitter

from src.core.logging import redact_secrets
from src.core.types import RETRY_MAX_ATTEMPTS, RETRY_WAIT_INITIAL_SECONDS, RETRY_WAIT_MAX_SECONDS
from src.llm.errors import LLMRateLimitError, LLMServerError, LLMTimeoutError


F = TypeVar("F", bound=Callable)
logger = logging.getLogger(__name__)


def _before_sleep(retry_state) -> None:
    exc = retry_state.outcome.exception() if retry_state.outcome else None
    logger.warning(
        "Retrying LLM call after %s (attempt %s/%s)",
        redact_secrets(exc) if exc else "unknown error",
        retry_state.attempt_number,
        RETRY_MAX_ATTEMPTS,
    )


def with_retry(func: F | None = None) -> Callable[[F], F] | F:
    def decorator(inner: F) -> F:
        return retry(
            retry=retry_if_exception_type((LLMRateLimitError, LLMTimeoutError, LLMServerError)),
            wait=wait_exponential_jitter(
                initial=RETRY_WAIT_INITIAL_SECONDS,
                max=RETRY_WAIT_MAX_SECONDS,
            ),
            stop=stop_after_attempt(RETRY_MAX_ATTEMPTS),
            before_sleep=_before_sleep,
            reraise=True,
        )(inner)  # type: ignore[return-value]

    if func is not None:
        return decorator(func)
    return decorator
