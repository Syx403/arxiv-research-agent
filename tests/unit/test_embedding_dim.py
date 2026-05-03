from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.llm.errors import LLMProviderError
from src.llm.providers.openai_embed import OpenAIEmbeddingClient


class _FakeEmbeddings:
    async def create(self, **kwargs):
        return SimpleNamespace(data=[SimpleNamespace(embedding=[0.0] * 1024)])


class _FakeOpenAI:
    def __init__(self) -> None:
        self.embeddings = _FakeEmbeddings()
        self.closed = False

    async def close(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_embedding_dim_mismatch_raises_provider_error() -> None:
    client = OpenAIEmbeddingClient(api_key="test-key", client=_FakeOpenAI())
    with pytest.raises(LLMProviderError):
        await client.embed(["hello"])
