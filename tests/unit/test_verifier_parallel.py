from __future__ import annotations

import asyncio

import pytest

from src.core.types import ChatResponse, Citation
from src.retrieval.self_rag import verifier


class _CountingChatClient:
    def __init__(self, expected_calls: int) -> None:
        self.expected_calls = expected_calls
        self.calls = 0
        self.active = 0
        self.max_active = 0
        self.all_started = asyncio.Event()

    async def chat(self, *args, **kwargs) -> ChatResponse:
        self.calls += 1
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        if self.calls == self.expected_calls:
            self.all_started.set()
        await self.all_started.wait()
        await asyncio.sleep(0)
        self.active -= 1
        return ChatResponse(
            content='{"supports": true, "rationale": "supported"}',
            model="fake",
            finish_reason="stop",
        )


@pytest.mark.asyncio
async def test_verify_citations_fans_out_with_bounded_concurrency(monkeypatch) -> None:
    client = _CountingChatClient(expected_calls=7)
    monkeypatch.setattr(verifier, "get_chat_client", lambda name: client)
    citations = [Citation(paper_id="p", chunk_id=index, claim_text=f"claim {index}") for index in range(7)]
    evidence_lookup = {index: f"evidence {index}" for index in range(7)}

    report = await asyncio.wait_for(verifier.verify_citations("claim text", citations, evidence_lookup), timeout=1)

    assert report.passed is True
    assert client.calls == 7
    assert client.max_active == 7
