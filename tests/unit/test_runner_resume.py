from __future__ import annotations

import csv

import pytest

from src.eval import runner
from src.eval.datasets.gold_questions import GoldQuestion


def _gold(index: int) -> GoldQuestion:
    return GoldQuestion(
        id=f"q-{index:03}",
        category="definition",
        question=f"Question {index}?",
        expected_paper_ids=["arxiv:2210.03629"],
        answer_must_mention=["ReAct", "reasoning", "acting"],
    )


def _row(gold: GoldQuestion) -> runner.EvalRow:
    return runner.EvalRow(
        id=gold.id,
        question=gold.question,
        expected_paper_ids=gold.expected_paper_ids,
        retrieved_paper_ids=gold.expected_paper_ids,
        cited_paper_ids=gold.expected_paper_ids,
        recall_at_5=1.0,
        mrr=1.0,
        citation_precision=1.0,
        citation_recall=1.0,
        judge_correctness=5,
        judge_groundedness=5,
        judge_completeness=5,
        judge_rationale="ok",
        answer="ok",
    )


@pytest.mark.asyncio
async def test_runner_resumes_from_existing_csv(monkeypatch, tmp_path, capsys) -> None:
    output_dir = tmp_path / "latest"
    output_dir.mkdir()
    csv_path = output_dir / "per_question.csv"
    summary_path = output_dir / "summary.json"
    questions = [_gold(index) for index in range(1, 6)]
    processed: list[str] = []

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=runner.CSV_FIELDNAMES)
        writer.writeheader()
        for gold in questions[:3]:
            writer.writerow(runner._row_to_csv_dict(_row(gold)))

    async def fake_run_one(gold: GoldQuestion, *, langsmith_state):
        processed.append(gold.id)
        return _row(gold)

    async def fake_shutdown() -> None:
        return None

    monkeypatch.setattr(runner, "OUTPUT_DIR", output_dir)
    monkeypatch.setattr(runner, "PER_QUESTION_CSV", csv_path)
    monkeypatch.setattr(runner, "SUMMARY_JSON", summary_path)
    monkeypatch.setattr(runner, "GOLD_QUESTIONS", questions)
    monkeypatch.setattr(runner, "_setup_langsmith_dataset", lambda questions: None)
    monkeypatch.setattr(runner, "_run_one", fake_run_one)
    monkeypatch.setattr(runner, "shutdown", fake_shutdown)

    summary = await runner.run_eval()

    assert processed == ["q-004", "q-005"]
    assert "Skipping 3 already-completed questions" in capsys.readouterr().out
    assert summary["total_questions"] == 5
    assert summary["total_skipped_via_resume"] == 3
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        assert len(list(csv.DictReader(handle))) == 5
