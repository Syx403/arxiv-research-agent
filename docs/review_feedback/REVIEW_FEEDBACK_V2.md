# Execution Plan Re-Review (v1.1)

**Reviewer**: Codex
**Reviewed document**: EXECUTION_PLAN.md v1.1
**Prior review**: REVIEW_FEEDBACK.md (v1.0)
**Re-review date**: 2026-05-03
**Verdict**: REWORK_REQUIRED

## Summary
v1.1 resolves most of the v1.0 structural issues and is much closer to executable. It is not ready yet: the psycopg checkpointer pool is still underspecified in a way known to break `AsyncPostgresSaver`, the pinned LangGraph range has a published checkpoint-deserialization advisory, and the Phase 9 retrieve flow contradicts its own router wiring test.

## Resolution status of prior findings

### Critical issues
- C-1: PARTIALLY_RESOLVED
  - Evidence (v1.1 section): §5.2 task 6 adds `langgraph-checkpoint-postgres` and `psycopg[binary,pool]`; §2.2 and §7.2 task 1 choose split asyncpg/psycopg drivers; Appendix A says not to pass asyncpg to `AsyncPostgresSaver`.
  - Residual (if any): §7.2 task 1 and Appendix A do not require psycopg `autocommit=True` and `row_factory=dict_row`, which LangGraph's Postgres saver documents as required for manual pools.
- C-2: RESOLVED
  - Evidence (v1.1 section): §9.2 task 2 replaces raw weighted SQL with RRF, filters both vector and lexical CTEs, and groups by `chunk_id`.
  - Residual (if any): None.
- C-3: RESOLVED
  - Evidence (v1.1 section): §0 rule 7, §4, and §5.2 tasks 12-15 now put `git init` before `git mv`, exclude `data/` tracking, and create `docs/STATUS.md`.
  - Residual (if any): None for the original issue.
- C-4: RESOLVED
  - Evidence (v1.1 section): §6.3 task 1 removes stale fixed cheap-model IDs and requires Phase 1 model verification plus ADR `0001-model-selection.md`.
  - Residual (if any): None for the original issue.

### Major concerns
- M-1: PARTIALLY_RESOLVED
  - Evidence (v1.1 section): §14.2 task 3 adds router and deduplicator calls; §14.2 task 10 adds `test_router_wired.py`.
  - Residual (if any): §14.2 task 3 calls `graph_retriever.expand_query_via_graph` before `query.router.route`, while §14.2 task 10 expects `use_concept_graph=False` to prevent the graph retriever call.
- M-2: RESOLVED
  - Evidence (v1.1 section): §7.2 task 2 adds `006_session_summaries.sql`; §12.2 task 3 consumes it.
  - Residual (if any): None.
- M-3: RESOLVED
  - Evidence (v1.1 section): §6.3 tasks 6-10 add `aclose()` and `close_llm_clients()`; §16.2 tasks 1 and 3 add UI/FastAPI shutdown cleanup.
  - Residual (if any): None.
- M-4: RESOLVED
  - Evidence (v1.1 section): §7.2 task 1 and Appendix A require `pgvector.asyncpg.register_vector` in the asyncpg pool init callback.
  - Residual (if any): None.
- M-5: RESOLVED
  - Evidence (v1.1 section): §2.4 forbids custom embedding dimensions; §6.3 task 8 validates vector length; §13.2 task 2 defines concept embedding text.
  - Residual (if any): None.
- M-6: RESOLVED
  - Evidence (v1.1 section): §8.2 task 6 adds transaction semantics and `ingestion_status`; §8.2 task 8 adds partial-failure testing.
  - Residual (if any): None for the original issue.
- M-7: RESOLVED
  - Evidence (v1.1 section): §8.3 softens citation coverage to 80%; §9.3 softens retrieval ranking while keeping a hard ReAct-in-top-5 assertion.
  - Residual (if any): None.
- M-8: RESOLVED
  - Evidence (v1.1 section): §6.3 task 13, §9.2 task 3, §10.2 task 9, and §14.2 task 10 add the missing high-risk tests.
  - Residual (if any): None.
- M-9: RESOLVED
  - Evidence (v1.1 section): §2.4 says constants are code defaults only; §5.2 task 5 limits `.env.example` to secrets and infrastructure.
  - Residual (if any): None.
- M-10: PARTIALLY_RESOLVED
  - Evidence (v1.1 section): §6.3 task 2 defines `Message`, `ToolCall`, `ChatResponse`, `Citation`, `VerificationReport`, `CitationVerdict`, and `TokenUsage`.
  - Residual (if any): §14.2 task 5 parses citations with `chunk_id` left `None`, but §10.2 task 8 verifies with `evidence_lookup: dict[int, str]`. The verifier has no deterministic evidence text when only `paper_id` is present.
- M-11: RESOLVED
  - Evidence (v1.1 section): §13.2 task 4 adds a JSONB `evidence_chunks` to `chunks.paper_id` join and a graph retriever integration test.
  - Residual (if any): None.
- M-12: RESOLVED
  - Evidence (v1.1 section): §10 and §11 split query/self-RAG from multi-hop; §12 and §13 split episodic from semantic memory.
  - Residual (if any): None for phase sizing.
- M-13: PARTIALLY_RESOLVED
  - Evidence (v1.1 section): §3.6 adds handoffs H-0 through H-final; §3.7 adds rollback policy.
  - Residual (if any): §3.6 H-3 is not cited in most later phases that run `INTEGRATION_TESTS=1` (§7.4, §8.4, §9.4, §11.4, §12.4, §13.4, §14.4).

### Minor issues
- L-1: RESOLVED
  - Evidence (v1.1 section): Appendix A now points to §5.2 task 6.
- L-2: RESOLVED
  - Evidence (v1.1 section): §3.2 and §5.2 task 13 both use `gh repo create ... --license=mit`.
- L-3: RESOLVED
  - Evidence (v1.1 section): §5.2 task 7 and §16.2 task 5 use `--host 127.0.0.1 --port 8000`, not `-h`.
- L-4: RESOLVED
  - Evidence (v1.1 section): §7.2 task 2 adds `007_triggers.sql` for `concepts.updated_at`.
- L-5: RESOLVED
  - Evidence (v1.1 section): §8.2 task 2 specifies Semantic Scholar fields, rate limits, and retry behavior.
- L-6: RESOLVED
  - Evidence (v1.1 section): §8.2 task 7 uses a Python module, not YAML.
- L-7: RESOLVED
  - Evidence (v1.1 section): §14.2 task 1 uses `Annotated[list[AnyMessage], add_messages]`.
- L-8: RESOLVED
  - Evidence (v1.1 section): §1.4 narrows persistence to "between turns"; §14.2 task 10 adds `test_resume.py`.
- L-9: RESOLVED
  - Evidence (v1.1 section): §16.4 stores PIDs and kills them after UI smoke checks.
- L-10: RESOLVED
  - Evidence (v1.1 section): §15.2 task 5 requires LangSmith dataset upsert.
- L-11: RESOLVED
  - Evidence (v1.1 section): §4 lists `src/core/sql/`.
- L-12: RESOLVED
  - Evidence (v1.1 section): §4 and §5.2 task 12 list/create `docs/STATUS.md`.
- L-13: RESOLVED
  - Evidence (v1.1 section): §6.5 explicitly notes SDK grep excludes tests because mocks may reference SDK types.

### Ambiguities
- A-1: RESOLVED
  - Evidence (v1.1 section): §2.2 and §7.2 task 1 choose asyncpg for app SQL and psycopg for checkpointing.
- A-2: RESOLVED
  - Evidence (v1.1 section): §9.2 task 2 chooses RRF with `RRF_K=60`.
- A-3: PARTIALLY_RESOLVED
  - Evidence (v1.1 section): §10.2 task 3 defines router policy flags only; §14.2 task 3 says retrieve decides downstream.
  - Residual (if any): §14.2 task 3 still invokes `graph_retriever` before consulting the router.
- A-4: RESOLVED
  - Evidence (v1.1 section): §13.2 task 2 embeds `display_name + ". " + definition`.
- A-5: RESOLVED
  - Evidence (v1.1 section): §8.2 task 6 defines transactional ingestion and failed-paper retry behavior.
- A-6: RESOLVED
  - Evidence (v1.1 section): §8.2 task 1 wraps the sync `arxiv` client in `asyncio.to_thread` with a global client.
- A-7: RESOLVED
  - Evidence (v1.1 section): §4 and §5.2 task 3 add `src/__init__.py`; §5.2 task 6 keeps `packages = ["src"]`.
- A-8: PARTIALLY_RESOLVED
  - Evidence (v1.1 section): §3.6 adds owner handoff checkpoints.
  - Residual (if any): Later integration-test phases do not consistently name H-3 where spend/live services are invoked.

### Prior questions
- Q1: answered. Quote the planner's answer (cite v1.1 section): "LangGraph (>=0.2.50,<0.3)" (§2.2) and `langgraph>=0.2.50,<0.3` (§5.2 task 6).
- Q2: answered. Quote the planner's answer (cite v1.1 section): "App DB driver | asyncpg" and "Checkpoint DB driver | psycopg" (§2.2).
- Q3: answered. Quote the planner's answer (cite v1.1 section): "pick the current V4 flagship per their docs" with candidates `deepseek-v4-pro`, `deepseek-v4-flash`, `deepseek-chat` (§6.3 task 1).
- Q4: answered. Quote the planner's answer (cite v1.1 section): "pick the cheapest current Flash-tier model with reliable JSON mode" and "Do not use `google/gemini-2.0-flash-001`" (§6.3 task 1).
- Q5: answered. Quote the planner's answer (cite v1.1 section): "lexical full-text rank" and RRF fusion, "not BM25" (§9.1, §9.2 task 2).
- Q6: answered. Quote the planner's answer (cite v1.1 section): `006_session_summaries.sql` is in Phase 2 (§7.2 task 2).
- Q7: answered. Quote the planner's answer (cite v1.1 section): "Phase 5 - Agentic RAG Components (Query / Rerank / Self-RAG)" and "Phase 6 - Multi-hop Citation Walker" (§10, §11).
- Q8: answered. Quote the planner's answer (cite v1.1 section): "Phase 7 - Episodic Memory and Compression" and "Phase 8 - Semantic Memory and Concept Graph" (§12, §13).
- Q9: answered. Quote the planner's answer (cite v1.1 section): "`data/pdfs/` and `data/eval_outputs/` are not tracked. They are created at runtime" (§4).
- Q10: answered. Quote the planner's answer (cite v1.1 section): "Default: Codex creates a new commit"; amend only with explicit planner approval; revert only when planner says so (§3.7).

## New issues introduced by v1.1

### Critical

#### N-C-1 psycopg checkpoint pool is missing required saver options
- **Section**: §7.2 task 1, §14.2 task 9, Appendix A
- **Problem**: The plan creates `psycopg_pool.AsyncConnectionPool(conninfo, min_size=1, max_size=10, open=False)` but does not require `kwargs={"autocommit": True, "row_factory": dict_row}`. LangGraph's Postgres saver documentation states these are required for manual connections/pools.
- **Impact**: `AsyncPostgresSaver.setup()` may not persist tables, and checkpoint reads can fail with tuple-row indexing errors.
- **Recommendation**: Add `from psycopg.rows import dict_row` and require the pool to be built with `kwargs={"autocommit": True, "row_factory": dict_row}`. Add an integration test that writes and reads a checkpoint, not just verifies tables.

#### N-C-2 Pinned LangGraph range has a published checkpoint deserialization advisory
- **Section**: §2.2, §5.2 task 6, §14.2 task 9, Appendix A
- **Problem**: `langgraph>=0.2.50,<0.3` resolves to a line with a published advisory for unsafe msgpack checkpoint deserialization in persistent checkpointers. The project explicitly persists and resumes checkpoints. PyPI currently lists this advisory on `langgraph` 0.2.76 and says fixed versions are in the 1.x line.
- **Impact**: A compromised checkpoint store can become code execution in the runtime, exposing local secrets. This is defense-in-depth, but it directly touches the plan's persistence feature.
- **Recommendation**: Either move to a fixed LangGraph line and compatible checkpoint package, or document and verify an official strict deserialization mitigation supported by the selected 0.2.x version before execution.

### Major

#### N-M-1 Retrieve node order contradicts router semantics and its own test
- **Section**: §10.2 task 3, §14.2 task 3, §14.2 task 10
- **Problem**: §14.2 task 3 calls `graph_retriever.expand_query_via_graph` before `query.router.route`, but §14.2 task 10 expects `use_concept_graph=False` to prevent graph retriever calls.
- **Recommendation**: Route first, then call graph retriever only when `route.use_concept_graph` is true.

#### N-M-2 Citation parsing cannot support verifier evidence lookup
- **Section**: §6.3 task 2, §10.2 task 8, §14.2 tasks 5-6
- **Problem**: Citations parsed from inline `[paper_id]` leave `chunk_id=None`, but `verify_citations` requires `evidence_lookup: dict[int, str]`.
- **Recommendation**: Require generated citations to include chunk IDs, e.g. `[paper_id#chunk_id]`, or add a deterministic paper_id-to-evidence mapping step before verification.

#### N-M-3 Handoff H-3 is not attached to most live integration phases
- **Section**: §3.6, §7.4, §8.4, §9.4, §11.4, §12.4, §13.4, §14.4
- **Problem**: §3.6 defines H-3 for integration tests and provider spend, but most validation sections invoke `INTEGRATION_TESTS=1` without naming the handoff.
- **Recommendation**: Add a precondition line to each affected phase: "Requires H-3 owner confirmation before running integration commands."

### Minor

- N-L-1 (§6.5): The SDK grep command has no `|| echo "ok"`; a clean result exits 1 and can look like validation failure.
- N-L-2 (§4, §8.2 task 7, Appendix B): `src/eval/datasets/seed_papers.py` is created in Phase 3 but absent from the target directory tree.
- N-L-3 (§4, §6.3 task 11): `src/core/constants.py` is optional, but absent from the target directory tree. Either add it to §4 or require constants to live in `src/core/types.py`.
- N-L-4 (§6.3 task 1, Appendix A): Cohere's current v4 rerank IDs are `rerank-v4.0-pro` and `rerank-v4.0-fast`; the plan's parenthetical `rerank-v4.0` is not the actual model ID.
- N-L-5 (§6.3 task 2): `Message` references `ToolCall` before `ToolCall` is defined. Require `from __future__ import annotations`, quote the type, or define `ToolCall` first.
- N-L-6 (§8.2 task 6): The transaction encloses PDF download, embedding calls, and Semantic Scholar calls. It is correct for rollback but holds a DB connection across long external IO; prefer collecting external artifacts first, then doing one short write transaction.

## Validation of specific high-risk new content

- **RRF SQL** (§9.2 task 2): Syntactically valid for PostgreSQL 16 + pgvector. It deduplicates correctly via `GROUP BY chunk_id`, and parameters match the function signature. Minor hardening: use `ANY($2::text[])` and normalize empty filters to `NULL`.
- **Split-driver pools** (§7.2 task 1, §14.2 task 9): Architecturally valid: asyncpg is for app SQL and psycopg is for checkpointing. Not fully correct yet because the psycopg pool lacks required `autocommit=True` and `row_factory=dict_row`.
- **Concept graph paper-hint join** (§13.2 task 4): The JSONB-array-element join is valid SQL and returns distinct `paper_id`s through `chunks`. Minor hardening: cast `$1::bigint[]` and ensure `evidence_chunks` contains only numeric chunk IDs.
- **Phase split correctness** (§10 vs §11; §12 vs §13): Mostly correct. Query/self-RAG, multi-hop, episodic memory, and semantic memory are now separated with no major duplication. Residuals: `seed_papers.py` is an orphan from §4, and Phase 9 router/graph-retriever ordering is inconsistent.
- **Pydantic data contracts** (§6.3 task 2): Mostly consistent with provider and state usage. Residual: `Citation.chunk_id` may be `None`, but citation verification requires chunk-keyed evidence.
- **Owner Handoff Checkpoints** (§3.6): The checkpoints are technically useful and mostly complete. H-3 should be cited explicitly in every later phase that invokes live integration tests or provider spend.
- **Rollback Policy** (§3.7): Consistent with §3.3's commit/review cycle. The status-file commit-hash amend workflow is awkward but acceptable because §3.7 bounds amend usage.

## Final verdict justification
v1.1 is a meaningful improvement and resolves the majority of the original findings. The remaining blockers are concrete and fixable: correct the psycopg pool construction, address the LangGraph security advisory or mitigation, reorder router/graph-retriever execution, and make citation verification evidence-addressable.

Until those are fixed, execution would still fail in the checkpoint path or pass tests that contradict the intended graph behavior. The verdict remains `REWORK_REQUIRED`.

## What the v1.1 revision does well
- The RRF replacement is a better retrieval plan than raw alpha-weighted scores.
- The phase split is materially more reviewable.
- The model-verification ADR avoids stale provider IDs.
- The ingestion-status and partial-failure test close a real reliability gap.
- The handoff and rollback sections make owner responsibilities explicit.
