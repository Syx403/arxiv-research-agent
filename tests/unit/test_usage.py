from __future__ import annotations

import pytest

from src.llm import usage


@pytest.fixture(autouse=True)
def reset_usage() -> None:
    usage.reset()


def test_usage_record_snapshot_and_reset() -> None:
    usage.record("main", prompt=100, completion=25)
    usage.record("main", prompt=50, completion=5)
    usage.record("rerank")

    snapshot = usage.snapshot()

    assert snapshot["main"]["calls"] == 2
    assert snapshot["main"]["prompt_tokens"] == 150
    assert snapshot["main"]["completion_tokens"] == 30
    assert snapshot["rerank"]["calls"] == 1

    usage.reset()

    assert usage.snapshot() == {}


def test_estimated_cost_usd_math() -> None:
    usage.record("main", prompt=1_000_000, completion=1_000_000)
    usage.record("rerank")

    cost = usage.estimated_cost_usd()

    assert cost["main"] == pytest.approx(1.37)
    assert cost["rerank"] == pytest.approx(0.002)
    assert cost["total"] == pytest.approx(1.372)
