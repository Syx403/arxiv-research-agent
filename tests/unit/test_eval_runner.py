from __future__ import annotations

import pytest

from src.core.types import JudgeVerdict
from src.eval import runner
from src.eval.datasets.gold_questions import GOLD_QUESTIONS, GoldQuestion
from src.eval.runner import EvalThresholdError, _failure_row


def test_failure_row_scores_agent_failure_conservatively() -> None:
    row = _failure_row(GOLD_QUESTIONS[0], RuntimeError("boom"))

    assert row.recall_at_5 == 0.0
    assert row.citation_precision == 0.0
    assert row.judge_correctness == 1
    assert row.judge_groundedness == 1
    assert row.judge_completeness == 1
    assert row.hard_failure is True
    assert "AGENT_RUN_FAILED" in row.answer


@pytest.mark.asyncio
async def test_run_eval_passes_at_threshold_with_mixed_crashes(monkeypatch, tmp_path) -> None:
    questions = [_gold(index) for index in range(10)]
    summary = await _run_eval_with_fake_agent(monkeypatch, tmp_path, questions, crash_ids={"q-001"})

    assert summary["hard_failure_count"] == 1
    assert summary["hard_failure_rate"] == pytest.approx(0.10)


@pytest.mark.asyncio
async def test_run_eval_raises_when_hard_failure_threshold_exceeded(monkeypatch, tmp_path) -> None:
    questions = [_gold(index) for index in range(5)]

    with pytest.raises(EvalThresholdError, match="hard failure rate 0.20 exceeds threshold 0.10"):
        await _run_eval_with_fake_agent(monkeypatch, tmp_path, questions, crash_ids={"q-001"})


async def _run_eval_with_fake_agent(
    monkeypatch,
    tmp_path,
    questions: list[GoldQuestion],
    *,
    crash_ids: set[str],
) -> dict:
    output_dir = tmp_path / "latest"
    csv_path = output_dir / "per_question.csv"
    summary_path = output_dir / "summary.json"

    async def fake_run(question: str, *, thread_id: str, skip_reflection: bool = False):
        gold_id = thread_id.split("-")[1] + "-" + thread_id.split("-")[2]
        if gold_id in crash_ids:
            raise RuntimeError("boom")
        return {
            "answer": "ok",
            "evidence": [],
            "citations": [],
            "synthesis_format_degraded": False,
        }

    async def fake_judge_answer(question: str, answer: str, gold: GoldQuestion) -> JudgeVerdict:
        return JudgeVerdict(correctness=5, groundedness=5, completeness=5, rationale="ok")

    async def fake_shutdown() -> None:
        return None

    async def fake_cleanup(_thread_id: str) -> None:
        return None

    monkeypatch.setattr(runner, "OUTPUT_DIR", output_dir)
    monkeypatch.setattr(runner, "PER_QUESTION_CSV", csv_path)
    monkeypatch.setattr(runner, "SUMMARY_JSON", summary_path)
    monkeypatch.setattr(runner, "GOLD_QUESTIONS", questions)
    monkeypatch.setattr(runner, "_setup_langsmith_dataset", lambda questions: None)
    monkeypatch.setattr(runner, "run", fake_run)
    monkeypatch.setattr(runner, "judge_answer", fake_judge_answer)
    monkeypatch.setattr(runner, "_cleanup_eval_run", fake_cleanup)
    monkeypatch.setattr(runner, "shutdown", fake_shutdown)

    return await runner.run_eval()


def _gold(index: int) -> GoldQuestion:
    return GoldQuestion(
        id=f"q-{index:03}",
        category="definition",
        question=f"Question {index}?",
        expected_paper_ids=["arxiv:2210.03629"],
        answer_must_mention=["ReAct", "reasoning", "acting"],
    )
