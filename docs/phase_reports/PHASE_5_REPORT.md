# Phase 5 Report

**Phase title**: Agentic RAG Components
**Status**: complete
**Commit**: `HEAD (see git log)`
**Owner handoff pending**: none

## Summary
Phase 5 added the standalone query, postprocess, and self-RAG components that later graph phases will wire together. Shared constants and Pydantic contracts now live in `src/core/types.py`, including sub-question DAGs, route decisions, relevance verdicts, and sufficiency verdicts. All LLM and rerank calls go through `src/llm/client.py` role aliases (`main`, `fast`, `rerank`); no retrieval module imports vendor SDKs directly. Unit tests mock every model path, including decomposition JSON repair, router alias SQL, reranker score overwrites, bounded relevance judging, sufficiency parsing, and citation verification.

## Files created
- `src/retrieval/query/__init__.py` - query package marker.
- `src/retrieval/query/decomposer.py` - sub-question decomposition with trivial-question bypass and one JSON repair retry.
- `src/retrieval/query/rewriter.py` - retrieval rewrite plus heuristic-gated HyDE on `fast`.
- `src/retrieval/query/router.py` - policy router that queries `concept_aliases` and emits flags only.
- `src/retrieval/postprocess/__init__.py` - postprocess package marker.
- `src/retrieval/postprocess/reranker.py` - Cohere rerank adapter through `get_rerank_client("rerank")`.
- `src/retrieval/postprocess/deduplicator.py` - duplicate chunk and 5-shingle near-duplicate filter.
- `src/retrieval/self_rag/__init__.py` - self-RAG package marker.
- `src/retrieval/self_rag/relevance_judge.py` - per-hit relevance judge and semaphore-bounded batch path.
- `src/retrieval/self_rag/sufficiency_check.py` - main-model sufficiency check parser.
- `src/retrieval/self_rag/verifier.py` - citation verifier keyed by required `Citation.chunk_id`.
- `tests/unit/test_decomposer_parsing.py` - decomposition JSON repair and trivial bypass tests.
- `tests/unit/test_rewriter.py` - HyDE skip heuristic test.
- `tests/unit/test_router.py` - concept alias hit/miss tests.
- `tests/unit/test_reranker.py` - rerank order and score overwrite test.
- `tests/unit/test_dedup.py` - duplicate and near-duplicate collapse test.
- `tests/unit/test_relevance_judge.py` - semaphore bound test.
- `tests/unit/test_sufficiency_check.py` - sufficiency JSON parsing test.
- `tests/unit/test_verifier_parser.py` - citation verifier parser and missing-evidence tests.

## Files modified
- `src/core/types.py` - added Phase 5 constants/contracts and set `RERANK_TOP_K=10`.
- `docs/STATUS.md` - Phase 5 marked complete.
- `docs/interview_talking_points.md` - Phase 5 talking points appended.

## Tests run
- Command: `uv run pytest tests/unit -q`
- Outcome: pass; `61 passed in 1.01s`.
- Command: `grep -RIE "from openai|from cohere|import openai|import cohere" src/retrieval || echo "ok"`
- Outcome: pass; printed `ok`.
- Command: `uv run ruff check src tests`
- Outcome: pass; `All checks passed!`.

## Validation commands
- Command: `uv run pytest tests/unit -q`
- Output (truncated if long): ```
.............................................................            [100%]
61 passed in 1.01s
```
- Command: `grep -RIE "from openai|from cohere|import openai|import cohere" src/retrieval || echo "ok"`
- Output (truncated if long): ```
ok
```
- Command: `uv run ruff check src tests`
- Output (truncated if long): ```
All checks passed!
```

## Acceptance checklist
- [x] Each Phase 5 module has a clear public function and at least one unit test.
- [x] All LLM calls go through `src/llm/client.py`; provider grep returns `ok`.
- [x] Router does not perform retrieval; it only queries concept aliases and emits flags.
- [x] Decomposer performs one JSON repair retry on invalid model output.
- [x] Reranker overwrites hit `score` with Cohere relevance scores.
- [x] Citation verifier raises when `evidence_lookup` lacks a required `chunk_id`.

## Deviations from the plan
None.

## Open questions for the planner
None.

## Ready for review
The planner can read these files and run those commands to verify.
