from __future__ import annotations

import logging
import importlib
from collections.abc import Awaitable, Callable
from typing import TypeVar

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential_jitter

from src.core.logging import redact_secrets
from src.core.types import RETRY_MAX_ATTEMPTS, RETRY_WAIT_INITIAL_SECONDS, RETRY_WAIT_MAX_SECONDS
from src.llm.errors import EmptyProviderResponseError, LLMRateLimitError, LLMServerError, LLMTimeoutError


F = TypeVar("F", bound=Callable)
T = TypeVar("T")
logger = logging.getLogger(__name__)
NETWORK_RETRY_MAX_ATTEMPTS = 3


def _optional_sdk_errors() -> tuple[type[BaseException], ...]:
    errors: list[type[BaseException]] = []
    for module_name, class_names in (
        ("openai", ("APITimeoutError", "APIConnectionError")),
        (
            "cohere",
            (
                "GatewayTimeoutError",
                "ServiceUnavailableError",
                "TooManyRequestsError",
                "InternalServerError",
                "ClientClosedRequestError",
            ),
        ),
    ):
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue
        for class_name in class_names:
            candidate = getattr(module, class_name, None)
            if isinstance(candidate, type) and issubclass(candidate, BaseException):
                errors.append(candidate)
    return tuple(errors)


TRANSIENT_HTTP_ERRORS = (
    httpx.ReadTimeout,
    httpx.ConnectTimeout,
    httpx.RemoteProtocolError,
    httpx.ConnectError,
    EmptyProviderResponseError,
    *_optional_sdk_errors(),
)


def _before_sleep(retry_state) -> None:
    exc = retry_state.outcome.exception() if retry_state.outcome else None
    logger.warning(
        "Retrying LLM call after %s (attempt %s/%s)",
        redact_secrets(exc) if exc else "unknown error",
        retry_state.attempt_number,
        RETRY_MAX_ATTEMPTS,
    )


def _before_network_sleep(retry_state) -> None:
    exc = retry_state.outcome.exception() if retry_state.outcome else None
    logger.warning(
        "Retrying LLM network call after %s (attempt %s/%s)",
        redact_secrets(exc) if exc else "unknown error",
        retry_state.attempt_number,
        NETWORK_RETRY_MAX_ATTEMPTS,
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


async def retry_transient_network(operation: Callable[[], Awaitable[T]]) -> T:
    """Retry raw provider network calls; higher layers still handle empty/JSON repair."""

    @retry(
        retry=retry_if_exception_type(TRANSIENT_HTTP_ERRORS),
        wait=wait_exponential_jitter(
            initial=RETRY_WAIT_INITIAL_SECONDS,
            max=RETRY_WAIT_MAX_SECONDS,
        ),
        stop=stop_after_attempt(NETWORK_RETRY_MAX_ATTEMPTS),
        before_sleep=_before_network_sleep,
        reraise=True,
    )
    async def _run() -> T:
        return await operation()

    return await _run()
