from __future__ import annotations

import json

import pytest

from src.core.types import ChatResponse
from src.memory.semantic import extractor


@pytest.mark.asyncio
async def test_extract_concepts_retries_empty_valid_payload(monkeypatch) -> None:
    client = _EmptyThenConceptClient()
    monkeypatch.setattr(extractor, "get_chat_client", lambda name: client)

    concepts = await extractor.extract_concepts_from_text("ReAct is a method.", source_chunk_ids=[1])

    assert len(concepts) == 1
    assert concepts[0].display_name == "ReAct"
    assert client.calls == 2


class _EmptyThenConceptClient:
    def __init__(self) -> None:
        self.calls = 0

    async def chat(self, *args, **kwargs) -> ChatResponse:
        self.calls += 1
        if self.calls == 1:
            return ChatResponse(content='{"concepts":[]}', model="fake", finish_reason="stop")
        return ChatResponse(
            content=json.dumps(
                {
                    "concepts": [
                        {
                            "display_name": "ReAct",
                            "definition": "A method interleaving reasoning and acting.",
                            "aliases": ["reasoning and acting"],
                            "evidence_chunk_ids": [1],
                            "relations": [],
                        }
                    ]
                }
            ),
            model="fake",
            finish_reason="stop",
        )
