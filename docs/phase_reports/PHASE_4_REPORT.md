# Phase 4 Report

**Phase title**: Vector Index and Hybrid Retrieval
**Status**: complete
**Commit**: `HEAD (see git log)`
**Owner handoff pending**: none

## Summary
Phase 4 implemented the retrieval index layer over seeded `chunks`: standalone vector search, lexical full-text search, and a single SQL RRF hybrid query. The hybrid query uses the planned `vec`, `lex`, and `fused` CTEs with parameter order fixed as `$1` embedding, `$2` paper filter, `$3` vector top-k, `$4` query text, `$5` lexical top-k, `$6` RRF k, and `$7` final top-k. ADR 0002 documents the RRF decision and the naming distinction between lexical full-text rank and BM25. Integration validation against the clean 30-paper corpus found ReAct chunks in the top five and measured one warmed hybrid SQL query at 9.54 ms.

## Files created
- `src/retrieval/__init__.py` - retrieval package marker.
- `src/retrieval/index/__init__.py` - index package marker.
- `src/retrieval/index/types.py` - frozen `Hit` dataclass shared by retrieval index modules.
- `src/retrieval/index/pgvector_store.py` - vector and lexical search legs.
- `src/retrieval/index/hybrid_search.py` - single-CTE RRF hybrid search.
- `docs/decisions/0002-rrf-fusion.md` - ADR for RRF fusion and lexical naming.
- `tests/unit/test_hybrid_sql.py` - static SQL invariant tests.
- `tests/integration/test_retrieval_basic.py` - live ReAct top-five and latency test.
- `tests/integration/test_retrieval_filter_overlap.py` - live dedup and filter test.

## Files modified
- `docs/STATUS.md` - Phase 4 marked complete.

## Tests run
- Command: `uv run pytest tests/unit/test_hybrid_sql.py -q`
- Outcome: pass; `4 passed in 0.36s`.
- Command: `INTEGRATION_TESTS=1 uv run pytest tests/integration/test_retrieval_basic.py tests/integration/test_retrieval_filter_overlap.py -q -s`
- Outcome: pass; `2 passed in 3.59s`; hybrid SQL latency printed as `9.54 ms`.
- Command: `test -f docs/decisions/0002-rrf-fusion.md && echo "RRF ADR present"`
- Outcome: pass; printed `RRF ADR present`.
- Command: `uv run pytest tests/unit -q`
- Outcome: pass; `48 passed in 0.92s`.
- Command: `uv run ruff check src tests`
- Outcome: pass; `All checks passed!`.

## Validation commands
- Command: `INTEGRATION_TESTS=1 uv run pytest tests/integration/test_retrieval_basic.py tests/integration/test_retrieval_filter_overlap.py -q -s`
- Output (truncated if long): ```
hybrid_search_latency_ms=9.54
top5= [(1501, 'arxiv:2210.03629', 'I NTRODUCTION'), (1496, 'arxiv:2210.03629', 'ABSTRACT'), (1507, 'arxiv:2210.03629', 'Method'), (890, 'arxiv:2308.08155', 'References'), (1502, 'arxiv:2210.03629', 'I NTRODUCTION')]
..
2 passed in 3.59s
```
- Command: `uv run pytest tests/unit/test_hybrid_sql.py -q`
- Output (truncated if long): ```
....                                                                     [100%]
4 passed in 0.36s
```
- Command: `test -f docs/decisions/0002-rrf-fusion.md && echo "RRF ADR present"`
- Output (truncated if long): ```
RRF ADR present
```

## Acceptance checklist
- [x] Hybrid search returns results with no duplicate `chunk_id`s.
- [x] `filter_paper_ids` is honored on both vector and lexical legs.
- [x] Top-five results for `ReAct reasoning and acting interleaved` include ReAct paper chunks.
- [x] Single warmed hybrid query completes under 200 ms locally; observed 9.54 ms.
- [x] ADR `docs/decisions/0002-rrf-fusion.md` exists.

## Deviations from the plan
None.

## Open questions for the planner
None.

## Ready for review
The planner can read these files and run those commands to verify.
