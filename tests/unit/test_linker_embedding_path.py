from __future__ import annotations

from contextlib import asynccontextmanager

import pytest

from src.core.types import ChatResponse, ConceptExtraction
from src.memory.semantic import linker


@pytest.mark.asyncio
async def test_linker_reuses_embedding_match_after_llm_judge(monkeypatch) -> None:
    conn = _EmbeddingFakeConn()
    embed_client = _FakeEmbeddingClient()
    chat_client = _FakeChatClient()

    @asynccontextmanager
    async def fake_acquire_app():
        yield conn

    monkeypatch.setattr(linker.db, "acquire_app", fake_acquire_app)
    monkeypatch.setattr(linker, "get_embedding_client", lambda name: embed_client)
    monkeypatch.setattr(linker, "get_chat_client", lambda name: chat_client)

    result = await linker.link_concept(
        ConceptExtraction(
            display_name="Chain of thought prompting",
            definition="A prompting method that elicits intermediate reasoning steps.",
            aliases=["CoT"],
            evidence_chunk_ids=[11],
        )
    )

    assert result.link_path == "embedding"
    assert result.created is False
    assert result.concept.concept_id == 42
    assert result.concept.evidence_chunk_ids == [5, 11]
    assert embed_client.texts == ["Chain of thought prompting. A prompting method that elicits intermediate reasoning steps."]
    assert chat_client.calls == 1
    assert conn.inserted_aliases == {"chain of thought prompting": 42, "cot": 42}


class _EmbeddingFakeConn:
    def __init__(self) -> None:
        self.inserted_aliases: dict[str, int] = {}

    async def fetchrow(self, query: str, *args):
        if "FROM concept_aliases a" in query:
            return None
        if "FROM concepts\n                WHERE canonical" in query:
            return None
        if "ORDER BY embedding <=>" in query:
            return {
                "concept_id": 42,
                "canonical": "chain-of-thought prompting",
                "display": "Chain-of-thought prompting",
                "definition": "Existing CoT definition",
                "evidence_chunks": [5],
                "similarity": 0.90,
            }
        if "SELECT concept_id FROM concept_aliases WHERE alias" in query:
            return None
        return None

    async def execute(self, query: str, *args):
        if "INSERT INTO concept_aliases" in query:
            self.inserted_aliases[args[0]] = args[1]
            return "INSERT 0 1"
        return "UPDATE 1"


class _FakeEmbeddingClient:
    def __init__(self) -> None:
        self.texts: list[str] = []

    async def embed(self, texts: list[str], *, model: str | None = None) -> list[list[float]]:
        self.texts.extend(texts)
        return [[0.01] * 1536 for _ in texts]


class _FakeChatClient:
    def __init__(self) -> None:
        self.calls = 0

    async def chat(self, *args, **kwargs) -> ChatResponse:
        self.calls += 1
        return ChatResponse(content='{"same": true}', model="fake-main", finish_reason="stop")
