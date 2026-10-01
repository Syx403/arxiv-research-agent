"""Unit tests may use injected transports, never real paid/network services."""
import httpx
import pytest


@pytest.fixture(autouse=True)
def no_live_http(monkeypatch):
    async def forbidden(*args, **kwargs):
        raise AssertionError('Real HTTP is forbidden in unit tests; inject a recorded or mock transport')
    def forbidden_sync(*args, **kwargs):
        raise AssertionError('Real HTTP is forbidden in unit tests')
    monkeypatch.setattr(httpx.AsyncHTTPTransport, 'handle_async_request', forbidden)
    monkeypatch.setattr(httpx.HTTPTransport, 'handle_request', forbidden_sync)
