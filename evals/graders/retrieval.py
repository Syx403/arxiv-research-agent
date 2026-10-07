"""Retrieval metrics (S1). A ranking lists the source paragraph of each retrieved chunk, in chunk
order, so `k` counts chunks, as the product's top 8 does; a paragraph split into several chunks
counts once, at its first chunk. With several annotators, each metric takes the best reference, as
QASPER's own evidence scoring does."""

import math
from collections.abc import Callable, Sequence, Set


def recall_at(ranking: Sequence[int], gold: Set[int], k: int) -> float:
    return len(gold & set(ranking[:k])) / len(gold)


def reciprocal_rank(ranking: Sequence[int], gold: Set[int]) -> float:
    return next((1 / rank for rank, item in enumerate(ranking, start=1) if item in gold), 0.0)


def ndcg_at(ranking: Sequence[int], gold: Set[int], k: int) -> float:
    """Binary relevance: DCG of the ranking over the DCG of a perfect one."""
    first: dict[int, int] = {}
    for rank, item in enumerate(ranking[:k], 1):
        first.setdefault(item, rank)
    dcg = sum(1 / math.log2(rank + 1) for item, rank in first.items() if item in gold)
    ideal = sum(1 / math.log2(rank + 1) for rank in range(1, min(len(gold), k) + 1))
    return dcg / ideal


def score(ranking: Sequence[int], references: Sequence[Set[int]]) -> dict[str, float]:
    """recall@8 is what select_evidence sees (the top 8 chunks); recall@20 is a wider pool."""

    def best(metric: Callable[[Set[int]], float]) -> float:
        return max(metric(gold) for gold in references)

    return {
        "recall@8": best(lambda gold: recall_at(ranking, gold, 8)),
        "recall@20": best(lambda gold: recall_at(ranking, gold, 20)),
        "mrr": best(lambda gold: reciprocal_rank(ranking, gold)),
        "ndcg@10": best(lambda gold: ndcg_at(ranking, gold, 10)),
    }
