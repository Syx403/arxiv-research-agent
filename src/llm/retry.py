from __future__ import annotations

import asyncio
import contextvars
from contextlib import contextmanager
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import logging
import importlib
from collections.abc import Awaitable, Callable
from typing import TypeVar

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential_jitter

from src.core.logging import redact_secrets
from src.core.types import (
    RETRY_AFTER_CAP_SECONDS,
    RETRY_MAX_ATTEMPTS,
    RETRY_WAIT_INITIAL_SECONDS,
    RETRY_WAIT_MAX_SECONDS,
)
from src.llm.errors import EmptyProviderResponseError, LLMRateLimitError, LLMServerError, LLMTimeoutError
from src.llm import usage


F = TypeVar("F", bound=Callable)
T = TypeVar("T")
logger = logging.getLogger(__name__)
NETWORK_RETRY_MAX_ATTEMPTS = 3
_retry_role: contextvars.ContextVar[str | None] = contextvars.ContextVar("llm_retry_role", default=None)
_single_attempt: contextvars.ContextVar[bool] = contextvars.ContextVar("llm_single_attempt", default=False)


def single_attempt_enabled() -> bool:
    return _single_attempt.get()


@contextmanager
def single_provider_attempt():
    """Let a workflow own its recovery without multiplying hidden paid retries."""
    token = _single_attempt.set(True)
    try:
        yield
    finally:
        _single_attempt.reset(token)


async def _sleep_async(seconds: float) -> None:
    await asyncio.sleep(seconds)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def set_retry_role(role: str):
    return _retry_role.set(role)


def reset_retry_role(token) -> None:
    _retry_role.reset(token)


def _optional_sdk_errors() -> tuple[type[BaseException], ...]:
    errors: list[type[BaseException]] = []
    for module_name, class_names in (
        ("openai", ("APITimeoutError", "APIConnectionError", "RateLimitError")),
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


def _wait_retry_after_or_jitter(retry_state) -> float:
    exc = retry_state.outcome.exception() if retry_state.outcome else None
    if _is_429(exc):
        role = _retry_role.get()
        if role:
            usage.record_provider_429(role)
    retry_after = _retry_after_seconds(exc)
    if retry_after is not None:
        return retry_after
    return float(
        wait_exponential_jitter(
            initial=RETRY_WAIT_INITIAL_SECONDS,
            max=RETRY_WAIT_MAX_SECONDS,
        )(retry_state)
    )


def _is_429(exc: BaseException | None) -> bool:
    if exc is None:
        return False
    status = getattr(exc, "status", None) or getattr(exc, "status_code", None)
    if status == 429:
        return True
    response = getattr(exc, "response", None)
    return getattr(response, "status_code", None) == 429


def _retry_after_seconds(exc: BaseException | None) -> float | None:
    if exc is None or not _is_429(exc):
        return None
    headers = getattr(exc, "headers", None) or {}
    response = getattr(exc, "response", None)
    if not headers and response is not None:
        headers = getattr(response, "headers", {}) or {}
    value = _get_retry_after_header(headers)
    if not value:
        return None
    parsed = _parse_retry_after(value)
    if parsed is None or parsed < 0:
        return None
    return min(parsed, float(RETRY_AFTER_CAP_SECONDS))


def _get_retry_after_header(headers) -> str | None:
    for key, value in dict(headers).items():
        if key.lower() == "retry-after":
            return str(value).strip()
    return None


def _parse_retry_after(value: str) -> float | None:
    if value.isdigit():
        return float(int(value))
    try:
        parsed = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return (parsed - _utcnow()).total_seconds()


def with_retry(func: F | None = None) -> Callable[[F], F] | F:
    def decorator(inner: F) -> F:
        return retry(
            retry=retry_if_exception_type((LLMRateLimitError, LLMTimeoutError, LLMServerError)),
            wait=_wait_retry_after_or_jitter,
            stop=lambda state: single_attempt_enabled() or state.attempt_number >= RETRY_MAX_ATTEMPTS,
            before_sleep=_before_sleep,
            sleep=_sleep_async,
            reraise=True,
        )(inner)  # type: ignore[return-value]

    if func is not None:
        return decorator(func)
    return decorator


async def retry_transient_network(operation: Callable[[], Awaitable[T]]) -> T:
    """Retry raw provider network calls; higher layers still handle empty/JSON repair."""

    if single_attempt_enabled():
        return await operation()

    @retry(
        retry=retry_if_exception_type(TRANSIENT_HTTP_ERRORS),
        wait=_wait_retry_after_or_jitter,
        stop=stop_after_attempt(NETWORK_RETRY_MAX_ATTEMPTS),
        before_sleep=_before_network_sleep,
        sleep=_sleep_async,
        reraise=True,
    )
    async def _run() -> T:
        return await operation()

    return await _run()
