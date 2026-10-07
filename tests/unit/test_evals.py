from typing import Any

import pytest

from evals.graders.retrieval import ndcg_at, recall_at, reciprocal_rank, score
from evals.qasper import questions, select
from evals.stats import mean_ci


def test_retrieval_metrics() -> None:
    ranking = [5, 2, 9, 1]
    assert recall_at(ranking, {2, 1}, 2) == 0.5
    assert reciprocal_rank(ranking, {9}) == pytest.approx(1 / 3)
    assert reciprocal_rank(ranking, {7}) == 0.0
    assert ndcg_at([2, 5], {2}, 10) == 1.0
    assert ndcg_at([5, 2], {2}, 10) == pytest.approx(1 / 1.5849625)


def test_score_takes_the_best_annotator() -> None:
    assert score([3, 4], [{9}, {3}])["mrr"] == 1.0


def test_bootstrap_interval_contains_the_mean() -> None:
    mean, low, high = mean_ci([0.0, 1.0, 1.0, 0.5])
    assert low <= mean <= high
    assert mean == 0.625


def record(qid: str, answers: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "id": "1912.01214",
        "title": "T",
        "abstract": "A.",
        "full_text": {"section_name": ["S"], "paragraphs": [["P one.", "P two."]]},
        "qas": {"question_id": [qid], "question": ["Q?"], "answers": [{"answer": answers}]},
    }


def answer(evidence: list[str], unanswerable: bool = False) -> dict[str, Any]:
    return {"unanswerable": unanswerable, "evidence": evidence}


def test_questions_need_every_answering_annotator_to_cite_paragraphs() -> None:
    [kept] = questions(record("q1", [answer(["P two."]), answer(["P one.", "P two."])]))
    assert kept.references == (frozenset({2}), frozenset({1, 2}))  # index 0 is the abstract
    assert not list(questions(record("q2", [answer(["FLOAT SELECTED: Table 1"])])))
    assert not list(questions(record("q3", [answer([], unanswerable=True)])))


def test_selection_is_reproducible() -> None:
    data = {f"p{i}": record(f"q{i}", [answer(["P one."])]) for i in range(5)}
    assert select(data, seed=7, papers=3, per_paper=1) == select(
        data, seed=7, papers=3, per_paper=1
    )
