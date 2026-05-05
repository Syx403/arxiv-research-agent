from __future__ import annotations

import pytest

from src.llm.errors import EmptyProviderResponseError
from src.llm.providers.cohere_rerank import CohereRerankClient


def _speed_up_network_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("src.llm.retry.RETRY_WAIT_INITIAL_SECONDS", 0.01)
    monkeypatch.setattr("src.llm.retry.RETRY_WAIT_MAX_SECONDS", 0.05)


@pytest.mark.asyncio
async def test_rerank_empty_results_raise_empty_provider_response(monkeypatch: pytest.MonkeyPatch) -> None:
    _speed_up_network_retry(monkeypatch)
    fake = _EmptyRerankClient()
    client = CohereRerankClient(api_key="test-key", client=fake)

    with pytest.raises(EmptyProviderResponseError, match="no results"):
        await client.rerank("query", ["document"], model="rerank-v3.5", top_n=1)

    assert fake.calls == 3


class _EmptyRerankClient:
    def __init__(self) -> None:
        self.calls = 0

    async def rerank(self, **kwargs):
        self.calls += 1
        return {"results": []}
