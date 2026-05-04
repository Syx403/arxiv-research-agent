from __future__ import annotations

from src.eval.metrics.citation_metrics import citation_precision, citation_recall


def test_citation_precision_empty_cited_returns_zero() -> None:
    assert citation_precision([], ["p1"]) == 0.0


def test_citation_recall_empty_expected_returns_zero() -> None:
    assert citation_recall(["p1"], []) == 0.0


def test_citation_metrics_full_overlap() -> None:
    assert citation_precision(["p1", "p2"], ["p1", "p2"]) == 1.0
    assert citation_recall(["p1", "p2"], ["p1", "p2"]) == 1.0


def test_citation_metrics_partial_overlap() -> None:
    assert citation_precision(["p1", "p3"], ["p1", "p2"]) == 0.5
    assert citation_recall(["p1", "p3"], ["p1", "p2"]) == 0.5
