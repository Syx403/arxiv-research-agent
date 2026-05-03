from __future__ import annotations

import os

import pytest

from src.core.types import Message
from src.llm.providers.openrouter import OpenRouterChatClient


pytestmark = pytest.mark.skipif(
    os.getenv("INTEGRATION_TESTS") != "1" or os.getenv("OPENROUTER_SMOKE") != "1",
    reason="OpenRouter smoke is opt-in with INTEGRATION_TESTS=1 and OPENROUTER_SMOKE=1",
)


@pytest.mark.asyncio
async def test_openrouter_smoke_opt_in_only() -> None:
    client = OpenRouterChatClient()
    try:
        response = await client.chat(
            [Message(role="user", content="Hi")],
            model="anthropic/claude-haiku-4.5",
            max_tokens=1,
        )
        assert response.model
    finally:
        await client.aclose()
