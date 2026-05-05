from __future__ import annotations

import pytest

from src.llm.errors import EmptyProviderResponseError
from src.llm.providers.openai_embed import OpenAIEmbeddingClient


def _speed_up_network_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("src.llm.retry.RETRY_WAIT_INITIAL_SECONDS", 0.01)
    monkeypatch.setattr("src.llm.retry.RETRY_WAIT_MAX_SECONDS", 0.05)


@pytest.mark.asyncio
async def test_embedding_count_mismatch_raises_empty_provider_response(monkeypatch: pytest.MonkeyPatch) -> None:
    _speed_up_network_retry(monkeypatch)
    fake = _FakeOpenAIClient()
    client = OpenAIEmbeddingClient(api_key="test-key", client=fake)

    with pytest.raises(EmptyProviderResponseError, match="count mismatch"):
        await client.embed(["one", "two"])

    assert fake.embeddings.calls == 3


class _FakeOpenAIClient:
    def __init__(self) -> None:
        self.embeddings = _MismatchedEmbeddings()


class _MismatchedEmbeddings:
    def __init__(self) -> None:
        self.calls = 0

    async def create(self, **kwargs):
        self.calls += 1
        return {"data": [{"embedding": [0.01] * 1536}], "usage": {"prompt_tokens": 1}}
