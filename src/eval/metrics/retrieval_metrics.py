from __future__ import annotations

import logging


logger = logging.getLogger(__name__)


def recall_at_k(retrieved_paper_ids: list[str], expected: list[str], k: int) -> float:
    if not expected:
        logger.warning("recall_at_k called with empty expected set")
        return 0.0
    retrieved = _dedupe_preserve_order(retrieved_paper_ids)[:k]
    return len(set(retrieved) & set(expected)) / len(set(expected))


def mrr(retrieved_paper_ids: list[str], expected: list[str]) -> float:
    expected_set = set(expected)
    if not expected_set:
        return 0.0
    for idx, paper_id in enumerate(_dedupe_preserve_order(retrieved_paper_ids), start=1):
        if paper_id in expected_set:
            return 1.0 / idx
    return 0.0


def _dedupe_preserve_order(paper_ids: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for paper_id in paper_ids:
        if paper_id in seen:
            continue
        seen.add(paper_id)
        deduped.append(paper_id)
    return deduped
