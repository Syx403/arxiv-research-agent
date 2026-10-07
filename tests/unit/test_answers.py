import pytest

from evals.graders.answers import citation_precision, normalize, score, token_f1
from evals.qasper import UNANSWERABLE, Gold


def test_normalisation_follows_the_official_scorer() -> None:
    assert normalize("The BLEU, score!") == ["bleu", "score"]


def test_token_f1() -> None:
    assert token_f1("BLEU score", "the BLEU") == pytest.approx(2 / 3)
    assert token_f1("ROUGE", "BLEU") == 0.0


def test_citation_precision_takes_the_best_annotator() -> None:
    assert citation_precision([3, 4], [{3}, {3, 4}]) == 1.0


def test_an_abstention_scores_against_unanswerable_gold() -> None:
    golds = [Gold("unanswerable", UNANSWERABLE, frozenset())]
    metrics = score("anything", True, [], golds)
    assert metrics["answer_f1"] == 1.0
    assert metrics["abstain_correct"] == 1.0
    assert "citation_precision" not in metrics


def test_an_answer_scores_against_spans_and_cited_paragraphs() -> None:
    golds = [Gold("extractive", "BLEU, METEOR", frozenset({5}))]
    metrics = score("BLEU and METEOR", False, [5, 6], golds)
    assert metrics["answer_f1"] == pytest.approx(0.8)
    assert metrics["citation_precision"] == 0.5
    assert metrics["abstain_correct"] == 1.0
