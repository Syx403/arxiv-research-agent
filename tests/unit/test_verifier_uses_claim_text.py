from __future__ import annotations

import pytest

from src.core.types import ChatResponse, Citation, Message
from src.retrieval.self_rag import verifier


@pytest.mark.asyncio
async def test_verifier_prompt_uses_sentence_claim_text(monkeypatch) -> None:
    client = _CapturingVerifierClient()
    monkeypatch.setattr(verifier, "get_chat_client", lambda name: client)
    sentence = "ReAct interleaves reasoning and acting [arxiv:2210.03629#123]."
    citation = Citation(
        paper_id="arxiv:2210.03629",
        chunk_id=123,
        claim_text=sentence,
        claim_span=(0, len(sentence)),
    )

    await verifier.verify_citations("answer text", [citation], {123: "ReAct evidence"})

    user_message = client.messages[1].content
    assert user_message is not None
    assert f"Specific claim being verified:\n{sentence}" in user_message
    assert "Specific claim being verified:\n[arxiv:2210.03629#123]" not in user_message


class _CapturingVerifierClient:
    def __init__(self) -> None:
        self.messages: list[Message] = []

    async def chat(self, messages: list[Message], **kwargs) -> ChatResponse:
        self.messages = messages
        return ChatResponse(
            content='{"supports": true, "rationale": "ok"}',
            model="fake",
            finish_reason="stop",
        )
