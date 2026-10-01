from __future__ import annotations

import pytest

from src.core.types import ChatResponse
from src.graph.nodes import synthesize
from src.retrieval.index.types import Hit


class _LengthClient:
    def __init__(self) -> None:
        self.calls = 0

    async def chat(self, messages, **kwargs) -> ChatResponse:
        self.calls += 1
        return ChatResponse(
            content=f"Truncated answer attempt {self.calls} [arxiv",
            model="fake-main",
            finish_reason="length",
        )


@pytest.mark.asyncio
async def test_synthesize_degrades_when_compact_retry_still_hits_length(monkeypatch) -> None:
    client = _LengthClient()
    monkeypatch.setattr(synthesize, "get_chat_client", lambda name: client)

    updates = await synthesize.synthesize_node(
        {
            "question": "What is MemGPT?",
            "evidence": [_hit()],
            "messages": [],
        }
    )

    assert client.calls == 2
    assert updates["answer"] == "Truncated answer attempt 2 [arxiv"
    assert updates["citations"] == []
    assert updates["synthesis_format_degraded"] is True
    assert updates["synthesis_finish_reason"] == "length"


@pytest.mark.asyncio
async def test_reasoning_only_truncation_recovers_without_repeating_thinking(monkeypatch):
    from unittest.mock import AsyncMock
    from src.graph.nodes.finalize import finalize_node
    from src.core.types import CitationVerdict, VerificationReport

    client = AsyncMock()
    client.chat.side_effect = [
        ChatResponse(content='', model='test', finish_reason='length'),
        ChatResponse(content='MemGPT uses virtual context management [arxiv:2310.08560#1].',
                     model='test', finish_reason='stop'),
    ]
    monkeypatch.setattr(synthesize, 'get_chat_client', lambda _: client)
    updates = await synthesize.synthesize_node({'question': 'Explain the mechanism', 'evidence': [_hit()]})
    assert 'thinking' not in client.chat.call_args_list[0].kwargs
    assert client.chat.call_args_list[1].kwargs['thinking'] is False
    assert client.chat.await_count == 2
    assert updates['synthesis_format_degraded'] is False
    assert len(updates['citations']) == 1
    # Recovery restores generation, never skips the independent evidence check.
    report = VerificationReport(passed=False, verdicts=[CitationVerdict(
        citation=updates['citations'][0], supports=False, rationale='fixture rejected')])
    final = await finalize_node({**updates, 'question': 'Explain the mechanism', 'verification': report})
    assert 'MemGPT uses' not in final['answer']


def _hit() -> Hit:
    return Hit(
        chunk_id=1,
        paper_id="arxiv:2310.08560",
        section="Intro",
        text="MemGPT uses virtual context management.",
        score=1.0,
    )
