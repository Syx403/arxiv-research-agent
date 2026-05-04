from __future__ import annotations

import pytest

from src.core.types import ReflectionReport
from src.graph.nodes import reflect


@pytest.mark.asyncio
async def test_reflect_node_skip_reflection_returns_empty_report_without_side_effects(monkeypatch) -> None:
    calls = {"append": 0, "reflect": 0}

    async def fail_append(*args, **kwargs):
        calls["append"] += 1
        raise AssertionError("append_message should not be called")

    async def fail_reflect(*args, **kwargs):
        calls["reflect"] += 1
        raise AssertionError("reflect_session should not be called")

    monkeypatch.setattr(reflect.session_store, "append_message", fail_append)
    monkeypatch.setattr(reflect.reflection, "reflect_session", fail_reflect)

    result = await reflect.reflect_node({"skip_reflection": True})

    assert result == {"reflection": ReflectionReport()}
    assert calls == {"append": 0, "reflect": 0}
