from __future__ import annotations

import pytest

from src.core.config import get_settings
from src.llm.client import close_llm_clients, get_chat_client


@pytest.mark.asyncio
async def test_close_clients_closes_shared_httpx_client(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-deepseek-key")
    get_settings.cache_clear()
    main = get_chat_client("main")
    fast = get_chat_client("fast")

    assert main.provider_client is fast.provider_client
    assert not main.provider_client.is_closed

    await close_llm_clients()
    assert main.provider_client.is_closed
    get_settings.cache_clear()
