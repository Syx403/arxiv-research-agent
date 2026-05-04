from __future__ import annotations

from collections import Counter

from src.eval.datasets.gold_questions import GOLD_QUESTIONS
from src.eval.datasets.seed_papers import SEED_PAPER_IDS


def test_gold_question_expected_papers_are_seed_papers() -> None:
    seed_ids = {f"arxiv:{paper_id}" for paper_id in SEED_PAPER_IDS}

    for question in GOLD_QUESTIONS:
        assert set(question.expected_paper_ids).issubset(seed_ids), question.id


def test_gold_question_distribution_counts() -> None:
    counts = Counter(question.category for question in GOLD_QUESTIONS)

    assert counts == {
        "definition": 10,
        "comparison": 8,
        "multi_hop": 6,
        "survey": 4,
        "application": 2,
    }


def test_gold_question_ids_are_unique() -> None:
    ids = [question.id for question in GOLD_QUESTIONS]

    assert len(ids) == len(set(ids))
    assert len(ids) == 30
