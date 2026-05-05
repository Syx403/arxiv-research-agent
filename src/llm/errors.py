from __future__ import annotations

from src.core.logging import redact_secrets


class LLMError(Exception):
    def __init__(self, message: str = "", *, provider: str, status: int | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.provider = provider
        self.status = status

    def __str__(self) -> str:
        status_part = f" status={self.status}" if self.status is not None else ""
        message = redact_secrets(self.message)
        return f"{self.__class__.__name__}(provider={self.provider}{status_part}): {message}"


class LLMAuthError(LLMError):
    pass


class LLMRateLimitError(LLMError):
    pass


class LLMTimeoutError(LLMError):
    pass


class LLMContextLengthError(LLMError):
    pass


class LLMServerError(LLMError):
    pass


class LLMInvalidRequestError(LLMError):
    pass


class LLMProviderError(LLMError):
    pass


class EmptyProviderResponseError(RuntimeError):
    """Raised when a provider returns a successful but semantically empty response."""


def error_for_status(provider: str, status: int, body: str = "") -> LLMError:
    lowered = body.lower()
    if status in (401, 403):
        return LLMAuthError(body or "authentication failed", provider=provider, status=status)
    if status == 429:
        return LLMRateLimitError(body or "rate limited", provider=provider, status=status)
    if status in (408, 504):
        return LLMTimeoutError(body or "request timed out", provider=provider, status=status)
    if status == 413 or (status == 400 and ("context" in lowered or "token" in lowered)):
        return LLMContextLengthError(body or "context length exceeded", provider=provider, status=status)
    if status == 400:
        return LLMInvalidRequestError(body or "invalid request", provider=provider, status=status)
    if 500 <= status <= 599:
        return LLMServerError(body or "provider server error", provider=provider, status=status)
    return LLMProviderError(body or "provider error", provider=provider, status=status)
