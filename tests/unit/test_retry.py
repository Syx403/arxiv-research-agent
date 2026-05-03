from __future__ import annotations

import pytest

from src.llm.errors import LLMAuthError, LLMRateLimitError
from src.llm.retry import with_retry


def _speed_up_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("src.core.types.RETRY_WAIT_INITIAL_SECONDS", 0.01)
    monkeypatch.setattr("src.core.types.RETRY_WAIT_MAX_SECONDS", 0.05)
    monkeypatch.setattr("src.llm.retry.RETRY_WAIT_INITIAL_SECONDS", 0.01)
    monkeypatch.setattr("src.llm.retry.RETRY_WAIT_MAX_SECONDS", 0.05)


@pytest.mark.asyncio
async def test_retries_rate_limit_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    _speed_up_retry(monkeypatch)
    calls = 0

    @with_retry()
    async def flaky() -> str:
        nonlocal calls
        calls += 1
        if calls < 3:
            raise LLMRateLimitError("retry me", provider="test", status=429)
        return "ok"

    assert await flaky() == "ok"
    assert calls == 3


@pytest.mark.asyncio
async def test_does_not_retry_auth_error() -> None:
    calls = 0

    @with_retry()
    async def fails_auth() -> None:
        nonlocal calls
        calls += 1
        raise LLMAuthError("bad key", provider="test", status=401)

    with pytest.raises(LLMAuthError):
        await fails_auth()
    assert calls == 1


@pytest.mark.asyncio
async def test_retry_log_masks_synthetic_key(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _speed_up_retry(monkeypatch)
    calls = 0
    secret = "Authorization: Bearer abc123def456ghi789"

    @with_retry()
    async def flaky() -> str:
        nonlocal calls
        calls += 1
        if calls < 2:
            raise LLMRateLimitError(secret, provider="test", status=429)
        return "ok"

    assert await flaky() == "ok"
    assert "abc123def456ghi789" not in caplog.text
    assert "[REDACTED]" in caplog.text
