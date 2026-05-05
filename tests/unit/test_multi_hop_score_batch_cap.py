from __future__ import annotations

from src.retrieval.multi_hop import citation_walker


def test_multi_hop_score_batch_size_is_capped_to_eight() -> None:
    assert citation_walker.SCORE_BATCH_SIZE == 8


def test_truncate_for_scoring_limits_candidate_text() -> None:
    text = "x" * 500

    truncated = citation_walker._truncate_for_scoring(text)

    assert len(truncated) < len(text)
    assert truncated.endswith("...[truncated]")
    assert truncated.startswith("x" * citation_walker.SCORE_CANDIDATE_TEXT_LIMIT)
