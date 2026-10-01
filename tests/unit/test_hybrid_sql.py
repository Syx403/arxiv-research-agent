from __future__ import annotations

import re

from src.retrieval.index.hybrid_search import HYBRID_SEARCH_SQL


def test_hybrid_sql_contains_required_ctes() -> None:
    assert "WITH vec AS" in HYBRID_SEARCH_SQL
    assert "lex AS" in HYBRID_SEARCH_SQL
    assert "fused AS" in HYBRID_SEARCH_SQL


def test_hybrid_sql_wires_filter_into_both_legs() -> None:
    assert HYBRID_SEARCH_SQL.count("($2::text[] IS NULL OR paper_id = ANY($2))") == 2


def test_hybrid_sql_deduplicates_by_chunk_id() -> None:
    fused = re.search(r"fused AS \((.*?)\)\s*SELECT c\.chunk_id", HYBRID_SEARCH_SQL, re.S)
    assert fused is not None
    assert "UNION ALL" in fused.group(1)
    assert "GROUP BY chunk_id" in fused.group(1)


def test_hybrid_sql_parameter_order_is_stable() -> None:
    assert "embedding <=> $1::vector" in HYBRID_SEARCH_SQL
    assert "paper_id = ANY($2)" in HYBRID_SEARCH_SQL
    assert "LIMIT $3" in HYBRID_SEARCH_SQL
    assert "websearch_to_tsquery('english', $4)" in HYBRID_SEARCH_SQL
    assert "LIMIT $5" in HYBRID_SEARCH_SQL
    assert "$6::float + rnk" in HYBRID_SEARCH_SQL
    assert HYBRID_SEARCH_SQL.rstrip().endswith("LIMIT $7;")
