from __future__ import annotations

import pytest

from src.core.types import ChatResponse, Message, EvidenceChunk
from src.graph.nodes.synthesize import _parse_citations
from src.retrieval.self_rag import verifier


@pytest.mark.asyncio
async def test_verifier_preserves_the_entire_supplied_evidence_pack(monkeypatch) -> None:
    client = _CapturingVerifierClient()
    monkeypatch.setattr(verifier, "get_chat_client", lambda name: client)

    answer = "claim text [p#7]."
    report = await verifier.verify_citations(answer, _parse_citations(answer), {
        7: EvidenceChunk(paper_id="p", chunk_id=7, text="x" * 5000),
    })

    assert report.passed is True
    user_message = client.messages[1].content
    assert user_message is not None
    assert "Evidence chunk 7 (exact supplied excerpt):" in user_message
    evidence_body = user_message.rsplit(":\n", maxsplit=1)[1]
    assert evidence_body == "x" * 5000
    assert len(evidence_body) == 5000


class _CapturingVerifierClient:
    def __init__(self) -> None:
        self.messages: list[Message] = []

    async def chat(self, messages: list[Message], **kwargs) -> ChatResponse:
        self.messages = messages
        return ChatResponse(
            content='{"claim_kind":"paper_fact","scope_status":"requested","all_assertions_supported":true,"no_unstated_assumptions":true,"all_citations_contribute":true,"supports": true, "rationale": "ok"}',
            model="fake",
            finish_reason="stop",
        )
