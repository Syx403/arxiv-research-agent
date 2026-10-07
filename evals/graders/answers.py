"""Answer metrics for S2 (DESIGN §11.2). Token F1 follows QASPER's official scorer: lower-case,
strip punctuation and articles, compare bags of tokens, take the best annotator."""

import re
import string
from collections import Counter
from collections.abc import Sequence, Set

from evals.qasper import UNANSWERABLE, Gold


def normalize(text: str) -> list[str]:
    text = "".join(ch for ch in text.lower() if ch not in set(string.punctuation))
    return re.sub(r"\b(a|an|the)\b", " ", text).split()


def token_f1(prediction: str, gold: str) -> float:
    predicted, expected = normalize(prediction), normalize(gold)
    common = sum((Counter(predicted) & Counter(expected)).values())
    if not predicted or not expected:
        return float(predicted == expected)
    if common == 0:
        return 0.0
    precision, recall = common / len(predicted), common / len(expected)
    return 2 * precision * recall / (precision + recall)


def citation_precision(cited: Sequence[int], gold: Sequence[Set[int]]) -> float:
    """Share of cited evidence sentences that lie in a gold paragraph, for the best annotator."""
    return max(sum(p in g for p in cited) / len(cited) for g in gold)


def score(
    short: str, abstained: bool, cited: Sequence[int], golds: Sequence[Gold]
) -> dict[str, float]:
    """Per-item metrics; citation precision only when the item is answerable and lines cite."""
    unanswerable = all(g.kind == "unanswerable" for g in golds)
    prediction = UNANSWERABLE if abstained else short
    metrics = {
        "answer_f1": max(token_f1(prediction, g.text) for g in golds),
        "abstained": float(abstained),
        "unanswerable": float(unanswerable),
        "abstain_correct": float(abstained == unanswerable),
    }
    answerable = [g.evidence for g in golds if g.kind != "unanswerable"]
    if answerable and cited:
        metrics["citation_precision"] = citation_precision(cited, answerable)
    return metrics
