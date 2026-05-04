from __future__ import annotations

import logging

from src.eval.metrics.retrieval_metrics import mrr, recall_at_k


def test_recall_at_k_empty_expected_returns_zero_and_warns(caplog) -> None:
    caplog.set_level(logging.WARNING)

    assert recall_at_k(["p1"], [], 5) == 0.0

    assert "empty expected" in caplog.text


def test_recall_at_k_full_overlap() -> None:
    assert recall_at_k(["p1", "p2"], ["p1", "p2"], 5) == 1.0


def test_recall_at_k_partial_overlap_dedupes_retrieved() -> None:
    assert recall_at_k(["p1", "p1", "p3"], ["p1", "p2"], 5) == 0.5


def test_mrr_first_hit_rank_three() -> None:
    assert mrr(["p9", "p8", "p2"], ["p2"]) == 1 / 3
