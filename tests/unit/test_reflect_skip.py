from __future__ import annotations

import pytest

from src.core.types import ReflectionReport
from src.graph.nodes import reflect


@pytest.mark.asyncio
async def test_reflect_node_skip_reflection_returns_empty_report_without_side_effects(
    monkeypatch,
) -> None:
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


@pytest.mark.asyncio
async def test_failed_answer_is_stored_as_unverified_without_learning(monkeypatch):
    from src.core.types import VerificationReport

    stored = []

    async def append_message(*args, **kwargs):
        stored.append(kwargs["metadata"])

    async def no_reflection(*args):
        pytest.fail("An unverified draft must not feed semantic reflection.")

    monkeypatch.setattr(reflect.session_store, "append_message", append_message)
    monkeypatch.setattr(reflect.reflection, "reflect_session", no_reflection)
    await reflect.reflect_node(
        {
            "thread_id": "test",
            "answer": "Unverified draft",
            "citations": [],
            "verification": VerificationReport(verdicts=[], passed=False),
        }
    )
    assert stored[0]["verification_passed"] is False


@pytest.mark.asyncio
async def test_later_reflection_skips_stored_rejected_answers(monkeypatch):
    from types import SimpleNamespace
    from src.memory import reflection

    async def messages(_):
        return [
            SimpleNamespace(metadata={"verification_passed": False, "citations": [{"chunk_id": 7}]})
        ]

    async def no_chunks(_):
        pytest.fail("Rejected answers must stay out of future reflection.")

    monkeypatch.setattr(reflection, "_fetch_assistant_messages", messages)
    monkeypatch.setattr(reflection, "_fetch_chunks", no_chunks)
    result = await reflection.reflect_session("test")
    assert result.concepts_created == 0
