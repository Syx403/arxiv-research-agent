from __future__ import annotations

from src.retrieval.index.types import Hit
from src.retrieval.postprocess.deduplicator import dedupe


def test_dedupe_collapses_duplicate_ids_and_near_duplicate_texts() -> None:
    text = "alpha beta gamma delta epsilon zeta eta theta iota"
    hits = [
        Hit(chunk_id=1, paper_id="p1", section="S", text=text, score=0.5),
        Hit(chunk_id=1, paper_id="p1", section="S", text=text, score=0.4),
        Hit(chunk_id=2, paper_id="p2", section="S", text=text, score=0.8),
        Hit(chunk_id=3, paper_id="p3", section="S", text="completely different evidence text", score=0.1),
    ]

    deduped = dedupe(hits)

    assert [hit.chunk_id for hit in deduped] == [2, 3]
    assert deduped[0].score == 0.8


def test_dedupe_preserves_original_order_of_survivors() -> None:
    near_duplicate = "alpha beta gamma delta epsilon zeta eta theta iota"
    hits = [
        Hit(chunk_id=1, paper_id="p1", section="S", text=near_duplicate, score=0.2),
        Hit(chunk_id=2, paper_id="p2", section="S", text="unrelated evidence text for another chunk", score=0.3),
        Hit(chunk_id=3, paper_id="p3", section="S", text=near_duplicate, score=0.9),
    ]

    deduped = dedupe(hits)

    assert [hit.chunk_id for hit in deduped] == [2, 3]
