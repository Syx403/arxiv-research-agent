from __future__ import annotations

import pytest

from src.core.types import ChatResponse, EvidenceChunk
from src.graph.nodes.synthesize import _parse_citations
from src.retrieval.self_rag import verifier


class _CaptureClient:
    def __init__(self) -> None:
        self.messages = []

    async def chat(self, messages, **kwargs) -> ChatResponse:
        self.messages.append(messages)
        return ChatResponse(
            content='{"claim_kind":"paper_fact","scope_status":"requested","all_assertions_supported":true,"no_unstated_assumptions":true,"all_citations_contribute":true,"supports": true, "rationale": "substantially supported"}',
            model="fake-main",
            finish_reason="stop",
        )


@pytest.mark.asyncio
async def test_verifier_prompt_defines_substantial_support_with_examples(monkeypatch) -> None:
    client = _CaptureClient()
    monkeypatch.setattr(verifier, "get_chat_client", lambda name: client)

    answer = "ReAct interleaves reasoning [arxiv:2210.03629#1]."
    await verifier.verify_citations(answer, _parse_citations(answer), {
        1: EvidenceChunk(paper_id="arxiv:2210.03629", chunk_id=1, text="ReAct alternates thought and action tokens."),
    })

    prompt = client.messages[0][0].content
    assert "SUBSTANTIALLY SUPPORTED" in prompt
    assert "supports=true (paraphrase)" in prompt
    assert "supports=true (substantial)" in prompt
    assert "supports=false (omitted)" in prompt
