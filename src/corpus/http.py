"""Bounded, cancellable scholarly HTTP requests and a successful-response cache."""
from __future__ import annotations

import asyncio
import hashlib
import json
import time
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from uuid import uuid4

import httpx

from src.core.trace import record_event


class ScholarlyCooldown(httpx.HTTPError):
    """A previous provider failure is still cooling down; no request was sent."""

    def __init__(self, provider: str, retry_after_s: float, http_status: int | None = None):
        super().__init__(f"{provider} temporarily cooling down")
        self.provider, self.retry_after_s, self.http_status = provider, retry_after_s, http_status


class ResponseCache:
    def __init__(self, directory: Path | None = None, ttl_s: float = 86400):
        self.directory = directory or Path(__file__).resolve().parents[2] / "data/cache/scholarly"
        self.ttl_s = ttl_s

    def get(self, key: str, *, max_age_s: float | None = None):
        try:
            row = json.loads(self._path(key).read_text())
            ttl = self.ttl_s if max_age_s is None else min(self.ttl_s, max_age_s)
            if 0 <= time.time() - row["time"] < ttl:
                return row["value"]
        except (OSError, ValueError, KeyError, TypeError):
            pass
        return None

    def put(self, key: str, value) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(f".{uuid4().hex}.tmp")
        try:
            temporary.write_text(json.dumps({"time": time.time(), "value": value}))
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)

    def _path(self, key: str) -> Path:
        return self.directory / (hashlib.sha256(key.encode()).hexdigest() + ".json")


class Pacer:
    def __init__(self, interval_s: float):
        self.interval_s = interval_s
        self.last_request = 0.0
        self.lock = asyncio.Lock()

    async def wait(self):
        async with self.lock:
            remaining = self.interval_s - (time.monotonic() - self.last_request)
            if remaining > 0:
                await asyncio.sleep(remaining)
            self.last_request = time.monotonic()


def retry_after(response: httpx.Response, attempt: int) -> float:
    value = response.headers.get("Retry-After", "")
    try:
        return max(0, float(value))
    except ValueError:
        try:
            return max(0, (parsedate_to_datetime(value) - datetime.now(UTC)).total_seconds())
        except (ValueError, TypeError):
            return 2 ** attempt


async def scholarly_get(client: httpx.AsyncClient, pacer: Pacer, url: str, *,
                        params=None, headers=None, timeout: float | None = None,
                        retry_rate_limits: bool = True) -> httpx.Response:
    # The caller owns the overall deadline. Never retry before Retry-After.
    for attempt in range(2):
        await pacer.wait()
        endpoint = client.base_url.join(url)
        started = time.monotonic()
        try:
            response = await client.get(url, params=params, headers=headers,
                                        **({"timeout": timeout} if timeout is not None else {}))
        except httpx.HTTPError as exc:
            record_event("scholarly_request", host=endpoint.host, path=endpoint.path,
                         attempt=attempt, duration_s=round(time.monotonic() - started, 3),
                         error=type(exc).__name__)
            raise
        record_event("scholarly_request", host=endpoint.host, path=endpoint.path,
                     attempt=attempt, duration_s=round(time.monotonic() - started, 3),
                     http_status=response.status_code)
        retryable = response.status_code in {502, 503, 504} or (response.status_code == 429 and retry_rate_limits)
        if retryable and attempt == 0:
            delay = retry_after(response, attempt)
            if delay <= 5:
                record_event("external_retry", host=endpoint.host, path=endpoint.path,
                             status_code=response.status_code, delay_s=delay)
                await asyncio.sleep(delay)
                continue
        response.raise_for_status()
        return response
    raise AssertionError("unreachable")
