from __future__ import annotations

import httpx
import openai
import pytest

from src.llm.providers.openai_embed import OpenAIEmbeddingClient


def _speed_up_network_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("src.llm.retry.RETRY_WAIT_INITIAL_SECONDS", 0.01)
    monkeypatch.setattr("src.llm.retry.RETRY_WAIT_MAX_SECONDS", 0.05)


@pytest.mark.asyncio
async def test_openai_embedding_retries_sdk_connection_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _speed_up_network_retry(monkeypatch)
    fake = _FakeOpenAIClient()
    client = OpenAIEmbeddingClient(api_key="test-key", client=fake)

    vectors = await client.embed(["hello"])

    assert vectors == [[0.01] * 1536]
    assert fake.embeddings.calls == 2


class _FakeOpenAIClient:
    def __init__(self) -> None:
        self.embeddings = _FakeEmbeddings()


class _FakeEmbeddings:
    def __init__(self) -> None:
        self.calls = 0

    async def create(self, **kwargs):
        self.calls += 1
        if self.calls == 1:
            raise openai.APIConnectionError(
                message="connection failed",
                request=httpx.Request("POST", "https://api.openai.test/v1/embeddings"),
            )
        return {"data": [{"embedding": [0.01] * 1536}], "usage": {"prompt_tokens": 1}}
