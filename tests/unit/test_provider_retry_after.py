from __future__ import annotations

from datetime import datetime, timedelta, timezone
from email.utils import format_datetime

import pytest

from src.llm import usage
from src.llm.errors import LLMRateLimitError
from src.llm import retry as retry_module


@pytest.fixture(autouse=True)
def reset_usage() -> None:
    usage.reset()


@pytest.mark.asyncio
async def test_retry_after_integer_seconds_is_honored(monkeypatch: pytest.MonkeyPatch) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr(retry_module, "_sleep_async", _fake_sleep(sleeps))
    token = retry_module.set_retry_role("main")
    calls = 0

    async def operation() -> str:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise LLMRateLimitError("rate limited", provider="deepseek", status=429, headers={"Retry-After": "5"})
        return "ok"

    try:
        assert await retry_module.with_retry()(operation)() == "ok"
    finally:
        retry_module.reset_retry_role(token)

    assert sleeps == [5.0]
    assert usage.snapshot()["main"]["provider_429_count"] == 1


@pytest.mark.asyncio
async def test_retry_after_http_date_is_honored(monkeypatch: pytest.MonkeyPatch) -> None:
    sleeps: list[float] = []
    fixed_now = datetime(2026, 5, 5, 12, 0, tzinfo=timezone.utc)
    retry_at = fixed_now + timedelta(seconds=7)
    monkeypatch.setattr(retry_module, "_sleep_async", _fake_sleep(sleeps))
    monkeypatch.setattr(retry_module, "_utcnow", lambda: fixed_now)
    token = retry_module.set_retry_role("main")
    calls = 0

    async def operation() -> str:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise LLMRateLimitError(
                "rate limited",
                provider="deepseek",
                status=429,
                headers={"Retry-After": format_datetime(retry_at, usegmt=True)},
            )
        return "ok"

    try:
        assert await retry_module.with_retry()(operation)() == "ok"
    finally:
        retry_module.reset_retry_role(token)

    assert sleeps == [7.0]


@pytest.mark.asyncio
async def test_retry_after_is_capped(monkeypatch: pytest.MonkeyPatch) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr(retry_module, "_sleep_async", _fake_sleep(sleeps))
    token = retry_module.set_retry_role("main")
    calls = 0

    async def operation() -> str:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise LLMRateLimitError("rate limited", provider="deepseek", status=429, headers={"Retry-After": "9999"})
        return "ok"

    try:
        assert await retry_module.with_retry()(operation)() == "ok"
    finally:
        retry_module.reset_retry_role(token)

    assert sleeps == [60.0]


@pytest.mark.asyncio
async def test_missing_or_garbage_retry_after_uses_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr(retry_module, "_sleep_async", _fake_sleep(sleeps))
    monkeypatch.setattr("src.llm.retry.RETRY_WAIT_INITIAL_SECONDS", 0.01)
    monkeypatch.setattr("src.llm.retry.RETRY_WAIT_MAX_SECONDS", 0.05)
    token = retry_module.set_retry_role("main")
    calls = 0

    async def operation() -> str:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise LLMRateLimitError("rate limited", provider="deepseek", status=429, headers={"Retry-After": "garbage"})
        return "ok"

    try:
        assert await retry_module.with_retry()(operation)() == "ok"
    finally:
        retry_module.reset_retry_role(token)

    assert len(sleeps) == 1
    assert 0.0 <= sleeps[0] <= 0.05


def _fake_sleep(sleeps: list[float]):
    async def _sleep(seconds: float) -> None:
        sleeps.append(seconds)

    return _sleep
