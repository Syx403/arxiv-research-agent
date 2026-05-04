from __future__ import annotations

import pytest

from src.core.types import ChatResponse, Citation, Message
from src.retrieval.self_rag import verifier


@pytest.mark.asyncio
async def test_verifier_truncates_long_evidence_with_note(monkeypatch) -> None:
    client = _CapturingVerifierClient()
    monkeypatch.setattr(verifier, "get_chat_client", lambda name: client)
    citation = Citation(paper_id="p", chunk_id=7, claim_span=(0, 5))

    report = await verifier.verify_citations("claim text", [citation], {7: "x" * 5000})

    assert report.passed is True
    user_message = client.messages[1].content
    assert user_message is not None
    assert "Evidence chunk 7 [truncated] (truncated to 1200 chars from original 5000 chars):" in user_message
    evidence_body = user_message.rsplit(":\n", maxsplit=1)[1]
    assert evidence_body == "x" * verifier.VERIFIER_EVIDENCE_CHAR_LIMIT


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
