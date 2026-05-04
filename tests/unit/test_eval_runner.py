from __future__ import annotations

from src.eval.datasets.gold_questions import GOLD_QUESTIONS
from src.eval.runner import _failure_row


def test_failure_row_scores_agent_failure_conservatively() -> None:
    row = _failure_row(GOLD_QUESTIONS[0], RuntimeError("boom"))

    assert row.recall_at_5 == 0.0
    assert row.citation_precision == 0.0
    assert row.judge_correctness == 1
    assert row.judge_groundedness == 1
    assert row.judge_completeness == 1
    assert "AGENT_RUN_FAILED" in row.answer
