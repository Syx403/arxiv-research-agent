from __future__ import annotations

import pytest

from src.core.types import ChatResponse
from src.retrieval.index.types import Hit
from src.retrieval.self_rag import sufficiency_check


class _FakeChatClient:
    async def chat(self, *args, **kwargs) -> ChatResponse:
        return ChatResponse(
            content='{"sufficient": false, "missing_aspects": ["comparison baseline"]}',
            model="fake",
            finish_reason="stop",
        )


class _BrokenChatClient:
    async def chat(self, *args, **kwargs) -> ChatResponse:
        return ChatResponse(content='{"sufficient": true', model="fake", finish_reason="stop")


@pytest.mark.asyncio
async def test_sufficiency_check_parses_missing_aspects(monkeypatch) -> None:
    monkeypatch.setattr(sufficiency_check, "get_chat_client", lambda name: _FakeChatClient())
    evidence = [Hit(chunk_id=1, paper_id="p", section="S", text="evidence", score=1.0)]

    verdict = await sufficiency_check.is_sufficient("question", evidence)

    assert verdict.sufficient is False
    assert verdict.missing_aspects == ["comparison baseline"]


@pytest.mark.asyncio
async def test_sufficiency_check_falls_back_after_invalid_json(monkeypatch) -> None:
    monkeypatch.setattr(sufficiency_check, "get_chat_client", lambda name: _BrokenChatClient())
    evidence = [Hit(chunk_id=1, paper_id="p", section="S", text="evidence", score=1.0)]

    verdict = await sufficiency_check.is_sufficient("question", evidence)

    assert verdict.sufficient is False
    assert verdict.status == "error"
    assert verdict.missing_aspects == []
    assert verdict.error_code == "sufficiency_judgment_failed"


async def test_missing_requested_aspect_overrides_inconsistent_success(monkeypatch):
    class Client:
        async def chat(self, *args, **kwargs):
            return ChatResponse(content='{"sufficient":true,"missing_aspects":["training dependency"]}',
                                model='fake', finish_reason='stop')
    monkeypatch.setattr(sufficiency_check, 'get_chat_client', lambda _: Client())
    verdict = await sufficiency_check.is_sufficient(
        'Explain the mechanism and its training dependency', [Hit(1, 'p', 'S', 'Mechanism.', 1)])
    assert not verdict.sufficient
    assert verdict.missing_aspects == ['training dependency']
