from __future__ import annotations

import re

from src.retrieval.index.types import Hit


SHINGLE_SIZE = 5
JACCARD_THRESHOLD = 0.85


def dedupe(hits: list[Hit]) -> list[Hit]:
    unique_by_id = _drop_duplicate_chunk_ids(hits)
    dropped: set[int] = set()
    shingles_by_idx = [_shingles(hit.text) for hit in unique_by_id]
    for idx, hit in enumerate(unique_by_id):
        if idx in dropped:
            continue
        for other_idx in range(idx + 1, len(unique_by_id)):
            if other_idx in dropped:
                continue
            if _jaccard(shingles_by_idx[idx], shingles_by_idx[other_idx]) < JACCARD_THRESHOLD:
                continue
            if unique_by_id[other_idx].score > hit.score:
                dropped.add(idx)
                break
            dropped.add(other_idx)
    return [hit for idx, hit in enumerate(unique_by_id) if idx not in dropped]


def _drop_duplicate_chunk_ids(hits: list[Hit]) -> list[Hit]:
    best_by_chunk_id: dict[int, Hit] = {}
    for hit in hits:
        current = best_by_chunk_id.get(hit.chunk_id)
        if current is None or hit.score > current.score:
            best_by_chunk_id[hit.chunk_id] = hit

    emitted: set[int] = set()
    unique: list[Hit] = []
    for hit in hits:
        if hit.chunk_id in emitted or hit is not best_by_chunk_id[hit.chunk_id]:
            continue
        emitted.add(hit.chunk_id)
        unique.append(hit)
    return unique


def _shingles(text: str) -> set[str]:
    tokens = re.findall(r"\w+", text.lower())
    if len(tokens) < SHINGLE_SIZE:
        return {" ".join(tokens)} if tokens else set()
    return {" ".join(tokens[idx : idx + SHINGLE_SIZE]) for idx in range(len(tokens) - SHINGLE_SIZE + 1)}


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)
