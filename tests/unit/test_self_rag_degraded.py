from __future__ import annotations

import pytest

from src.graph.nodes import self_rag


@pytest.mark.asyncio
async def test_self_rag_marks_degraded_synthesis_unverified_without_verifier(monkeypatch) -> None:
    calls = 0

    async def fake_verify(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("verifier should not run for degraded synthesis")

    monkeypatch.setattr(self_rag, "verify_citations", fake_verify)

    updates = await self_rag.self_rag_node({"synthesis_format_degraded": True})

    assert calls == 0
    assert updates["verification"].passed is False
    assert updates["verification"].rationale == "synthesis_format_degraded"
