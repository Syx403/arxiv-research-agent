from __future__ import annotations

import pytest

from src.retrieval.query import router


class _FakeAcquire:
    def __init__(self, value: bool) -> None:
        self.conn = _FakeConn(value)

    async def __aenter__(self) -> "_FakeConn":
        return self.conn

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None


class _FakeConn:
    def __init__(self, value: bool) -> None:
        self.value = value
        self.calls: list[tuple[str, str]] = []

    async def fetchval(self, sql: str, subq: str) -> bool:
        self.calls.append((sql, subq))
        return self.value


@pytest.mark.asyncio
async def test_router_sets_concept_graph_on_alias_hit(monkeypatch) -> None:
    acquire = _FakeAcquire(True)
    monkeypatch.setattr(router.db, "acquire_app", lambda: acquire)

    decision = await router.route("How does ReAct use reflection?")

    assert decision.sources == ["local_pgvector"]
    assert decision.use_concept_graph is True
    assert decision.may_use_semantic_scholar_live is True
    assert "concept_aliases" in acquire.conn.calls[0][0]


@pytest.mark.asyncio
async def test_router_leaves_concept_graph_false_on_alias_miss(monkeypatch) -> None:
    monkeypatch.setattr(router.db, "acquire_app", lambda: _FakeAcquire(False))

    decision = await router.route("How does ReAct use reflection?")

    assert decision.use_concept_graph is False
