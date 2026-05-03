# Execution Plan Final Review (v1.2)

**Reviewer**: Codex
**Reviewed document**: EXECUTION_PLAN.md v1.2
**Prior reviews**: REVIEW_FEEDBACK.md (v1.0), REVIEW_FEEDBACK_V2.md (v1.1)
**Re-review date**: 2026-05-03
**Verdict**: REWORK_REQUIRED

## Summary
v1.2 resolves all v1.1 Critical and Major findings in the targeted areas. It is still not quite ready because the surgical edits introduced one Major inconsistency in Phase 3 ingestion failure handling and left stale ADR/test-validation references.

## Resolution status of v1.1 round findings

### Critical
- N-C-1: RESOLVED — §7.2 task 1 and Appendix A now require `kwargs={"autocommit": True, "row_factory": dict_row}` for the psycopg pool, and §7.2 task 4 adds a real checkpoint roundtrip test.
- N-C-2: RESOLVED — §6.3 task 1 requires a LangGraph version/security advisory check, explicit Path A/Path B decision, and ADR `0001-model-and-langgraph-selection.md`.

### Major
- N-M-1: RESOLVED — §14.2 task 3 now runs `query.router.route` first and calls `graph_retriever.expand_query_via_graph` only when `route.use_concept_graph` is true; §14.2 task 10 matches this.
- N-M-2: RESOLVED — §6.3 task 2 makes `Citation.chunk_id: int`; §14.2 tasks 5-6 require `[paper_id#chunk_id]`, parse both fields, and build `evidence_lookup` from the evidence list.
- N-M-3: RESOLVED — §7.4, §8.4, §9.4, §11.4, §12.4, §13.4, and §14.4 all include explicit H-3 preconditions.

### Minor
- N-L-1: RESOLVED — §6.5 appends `|| echo "ok"` to the SDK grep command.
- N-L-2: RESOLVED — §4 includes `src/eval/datasets/seed_papers.py`.
- N-L-3: RESOLVED — §6.3 task 11 makes `src/core/types.py` the single constants home; no `constants.py` alternative remains.
- N-L-4: RESOLVED — §6.3 task 1 and Appendix A list actual Cohere IDs: `rerank-v4.0-pro`, `rerank-v4.0-fast`, `rerank-v3.5`.
- N-L-5: RESOLVED — §6.3 task 2 requires `from __future__ import annotations`.
- N-L-6: RESOLVED — §8.2 task 6 collects external artifacts before opening the short DB transaction.

## Stale references / cross-edit consistency

- §6.4 still checks for `docs/decisions/0001-model-selection.md`; it should be `docs/decisions/0001-model-and-langgraph-selection.md` to match §6.3 task 1 and §6.5.
- §7.2 task 4 adds `tests/integration/test_checkpoint_roundtrip.py`, but §7.4 validation runs only `test_db_bootstrap.py`. Add the roundtrip test to §7.4.
- §8.2 task 6 says Phase A errors "just log and exit", but §8.2 task 8 expects an embedding failure to leave `papers.ingestion_status='failed'`. The Phase A failure path should either write a failed row or the test should be revised.
- The historical v1.0 changelog entry still names `0001-model-selection.md`; this is non-executable but stale.

## New issues (if any)

### Critical
None

### Major

#### N3-M-1 Phase A ingestion failure no longer matches retry/test semantics
- **Section**: §8.2 task 6, §8.2 task 8, §8.3
- **Problem**: External artifact collection now happens before the DB transaction, but Phase A errors only "log and exit". The partial-failure test still patches `embed` and expects a `failed` paper row, and idempotency rules still mention `failed -> clean and retry`.
- **Recommendation**: On Phase A failure, write or upsert `papers.ingestion_status='failed'` in a short separate transaction before returning/raising.

### Minor

- N3-L-1 (§6.4): Acceptance criteria references old ADR filename `0001-model-selection.md`.
- N3-L-2 (§7.4): Validation commands omit `tests/integration/test_checkpoint_roundtrip.py`.

## Final verdict justification
The remaining issues are small in number but one is Major because Phase 3's new collect-before-transaction behavior conflicts with its own failure test and retry-state contract. Fix that, update the stale ADR filename, and add the checkpoint roundtrip test to validation; after those edits the plan should be approvable.
