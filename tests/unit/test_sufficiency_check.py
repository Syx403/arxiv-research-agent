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
        return ChatResponse(content='{"sufficient": true', model="fake", finish_reason="length")


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
    assert verdict.missing_aspects == [
        "PARSE_FAILURE: sufficiency check returned invalid JSON; treating as insufficient."
    ]
