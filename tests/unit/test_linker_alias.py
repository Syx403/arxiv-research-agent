from __future__ import annotations

from contextlib import asynccontextmanager
import json

import pytest

from src.core.types import ConceptExtraction
from src.memory.semantic import linker


@pytest.mark.asyncio
async def test_linker_reuses_concept_by_alias_without_provider_calls(monkeypatch) -> None:
    conn = _AliasFakeConn()

    @asynccontextmanager
    async def fake_acquire_app():
        yield conn

    def fail_provider_call(*args, **kwargs):
        raise AssertionError("alias path must not call providers")

    monkeypatch.setattr(linker.db, "acquire_app", fake_acquire_app)
    monkeypatch.setattr(linker, "get_embedding_client", fail_provider_call)
    monkeypatch.setattr(linker, "get_chat_client", fail_provider_call)

    result = await linker.link_concept(
        ConceptExtraction(
            display_name="react prompting",
            definition="Reasoning and acting with tool calls.",
            aliases=["ReAct"],
            evidence_chunk_ids=[7, 9],
        )
    )

    assert result.link_path == "alias"
    assert result.created is False
    assert result.concept.concept_id == 1
    assert result.concept.evidence_chunk_ids == [3, 7, 9]
    assert conn.updated_evidence == [3, 7, 9]


class _AliasFakeConn:
    def __init__(self) -> None:
        self.aliases = {"react prompting": 1, "react": 1}
        self.updated_evidence: list[int] | None = None

    async def fetchrow(self, query: str, *args):
        if "FROM concept_aliases a" in query:
            alias = args[0]
            if alias in self.aliases:
                return {
                    "concept_id": self.aliases[alias],
                    "canonical": "react prompting",
                    "display": "ReAct prompting",
                    "definition": "Existing definition",
                    "evidence_chunks": [3],
                }
            return None
        if "SELECT concept_id FROM concept_aliases WHERE alias" in query:
            concept_id = self.aliases.get(args[0])
            return {"concept_id": concept_id} if concept_id is not None else None
        return None

    async def execute(self, query: str, *args):
        if "UPDATE concepts" in query:
            self.updated_evidence = json.loads(args[1])
            return "UPDATE 1"
        if "INSERT INTO concept_aliases" in query:
            self.aliases[args[0]] = args[1]
            return "INSERT 0 1"
        return "OK"
