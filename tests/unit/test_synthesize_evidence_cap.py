from __future__ import annotations

import pytest

from src.core.types import (
    SYNTHESIS_EVIDENCE_CHARS_PER_CHUNK,
    SYNTHESIS_EVIDENCE_MAX_CHUNKS,
    ChatResponse,
    RouteDecision,
    SubQResult,
)
from src.graph.nodes import synthesize
from src.retrieval.index.types import Hit


class _CaptureClient:
    def __init__(self) -> None:
        self.messages = []

    async def chat(self, messages, **kwargs) -> ChatResponse:
        self.messages.append(messages)
        return ChatResponse(
            content="The strongest evidence supports the claim [arxiv:top#19].",
            model="fake-main",
            finish_reason="stop",
        )


@pytest.mark.asyncio
async def test_synthesize_caps_and_truncates_prompt_evidence(monkeypatch) -> None:
    client = _CaptureClient()
    monkeypatch.setattr(synthesize, "get_chat_client", lambda name: client)

    await synthesize.synthesize_node(
        {
            "question": "What matters?",
            "evidence": [_hit(index) for index in range(20)],
            "messages": [],
        }
    )

    user_prompt = client.messages[0][1].content
    assert user_prompt.count("Section: Intro") == SYNTHESIS_EVIDENCE_MAX_CHUNKS
    assert "[arxiv:top#19]" in user_prompt
    assert "[arxiv:top#11]" not in user_prompt
    assert "...[truncated from " not in user_prompt
    assert len(synthesize._format_evidence(synthesize._select_synthesis_evidence([_hit(i) for i in range(20)]))) <= (
        SYNTHESIS_EVIDENCE_MAX_CHUNKS * (SYNTHESIS_EVIDENCE_CHARS_PER_CHUNK + 400)
    )


def _hit(index: int) -> Hit:
    return Hit(
        chunk_id=index,
        paper_id="arxiv:top",
        section="Intro",
        text=f"chunk-{index} " + ("x" * (SYNTHESIS_EVIDENCE_CHARS_PER_CHUNK + 200)),
        score=float(index),
    )


def test_comparison_allocates_context_across_queries_instead_of_raw_scores():
    from collections import Counter
    a = [Hit(chunk_id=i, paper_id="A", section="", text=f"A mechanism {i}", score=.4 - i / 100) for i in range(8)]
    b = [Hit(chunk_id=100+i, paper_id="B", section="", text=f"B mechanism {i}", score=.99 - i / 100) for i in range(8)]
    groups = {i: SubQResult(subq_index=i, hits=hits, sufficient=True, route_decision=RouteDecision())
              for i, hits in enumerate([a, b])}
    chosen = synthesize._select_synthesis_evidence(a + b, groups)
    assert Counter(h.paper_id for h in chosen) == {"A": 4, "B": 4}
    assert len({h.chunk_id for h in chosen}) == 8
    assert [h.chunk_id for h in chosen[:2]] == [0, 100]


def test_single_paper_is_not_artificially_padded_with_other_papers():
    hits = [_hit(i) for i in range(10)]
    assert len(synthesize._select_synthesis_evidence(hits)) == 8
    assert {h.paper_id for h in synthesize._select_synthesis_evidence(hits)} == {"arxiv:top"}
