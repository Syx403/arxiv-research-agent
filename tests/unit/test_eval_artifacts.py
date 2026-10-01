from __future__ import annotations

import pytest

from src.eval.artifacts import ensure_manifest
from src.eval import runner
from src.eval.datasets.gold_questions import GOLD_QUESTIONS


def test_legacy_or_changed_results_cannot_silently_resume(tmp_path):
    (tmp_path / "per_question.csv").write_text("id\nq-001\n")
    with pytest.raises(ValueError, match="Legacy"):
        ensure_manifest(tmp_path, {"model": "new"})
    fresh = tmp_path / "fresh"
    ensure_manifest(fresh, {"model": "flash-high", "corpus": "v1"})
    ensure_manifest(fresh, {"model": "flash-high", "corpus": "v1"})
    with pytest.raises(ValueError, match="changed"):
        ensure_manifest(fresh, {"model": "flash-low", "corpus": "v1"})


def test_summary_sums_persisted_row_costs_even_with_empty_process_counters(monkeypatch, tmp_path):
    monkeypatch.setattr(runner, "SUMMARY_JSON", tmp_path / "summary.json")
    first = runner._failure_row(GOLD_QUESTIONS[0], RuntimeError("fixture"))
    second = runner._failure_row(GOLD_QUESTIONS[1], RuntimeError("fixture"))
    first.estimated_cost_usd = {"main": .01, "total": .01}
    second.estimated_cost_usd = {"fast": .02, "total": .02}
    first.token_usage = {"main": {"prompt_tokens": 10}}
    restored = runner._row_from_csv_dict({key: str(value) for key, value in runner._row_to_csv_dict(first).items()})
    result = runner._write_summary([restored, second], total_skipped_via_resume=1)
    assert result["estimated_cost_usd"]["total"] == pytest.approx(.03)
    assert result["token_usage"]["main"]["prompt_tokens"] == 10


def test_langsmith_dataset_write_requires_explicit_export(monkeypatch):
    monkeypatch.delenv("ARA_EVAL_LANGSMITH_EXPORT", raising=False)
    def forbidden(*args, **kwargs):
        raise AssertionError("No remote client should be created")
    monkeypatch.setattr(runner, "Client", forbidden)
    assert runner._setup_langsmith_dataset(GOLD_QUESTIONS) is None
