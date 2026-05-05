from __future__ import annotations

import pytest

from src.core.types import ChatResponse, Citation
from src.retrieval.self_rag import verifier


class _CaptureClient:
    def __init__(self) -> None:
        self.messages = []

    async def chat(self, messages, **kwargs) -> ChatResponse:
        self.messages.append(messages)
        return ChatResponse(
            content='{"supports": true, "rationale": "substantially supported"}',
            model="fake-main",
            finish_reason="stop",
        )


@pytest.mark.asyncio
async def test_verifier_prompt_defines_substantial_support_with_examples(monkeypatch) -> None:
    client = _CaptureClient()
    monkeypatch.setattr(verifier, "get_chat_client", lambda name: client)

    await verifier.verify_citations(
        "answer",
        [Citation(paper_id="arxiv:2210.03629", chunk_id=1, claim_text="ReAct interleaves reasoning.")],
        {1: "ReAct alternates thought and action tokens."},
    )

    prompt = client.messages[0][0].content
    assert "SUBSTANTIALLY SUPPORTED" in prompt
    assert "supports=true (paraphrase)" in prompt
    assert "supports=true (substantial)" in prompt
    assert "supports=false (omitted)" in prompt
