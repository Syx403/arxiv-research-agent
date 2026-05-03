# Execution Plan Review Feedback

**Reviewer**: Codex
**Reviewed document**: EXECUTION_PLAN.md v1.0
**Review date**: 2026-05-03
**Verdict**: REWORK_REQUIRED

## Summary
The plan is strong conceptually, but it is not airtight enough for execution without rework. The main blockers are the LangGraph Postgres checkpoint dependency/connection mismatch, incorrect hybrid SQL behavior, Phase 0 ordering/tracking contradictions, and model IDs that will be deprecated or do not mean what the plan says during the expected project window.

## Critical Issues (must resolve before execution)

### C-1 LangGraph Postgres checkpointing cannot work with the declared dependencies
- **Section**: §5.2 task 5, §7.1, §7.2 task 3, §12.2 task 9, Appendix A
- **Problem**: `pyproject.toml` omits `langgraph-checkpoint-postgres`, and `AsyncPostgresSaver` is provided by that separate package. It uses psycopg connections/pools, not an `asyncpg.Pool`; the plan says to wire it to the same asyncpg pool.
- **Impact**: Phase 2 bootstrap and Phase 7 graph compilation will fail or require guessing. `AsyncPostgresSaver.setup()` also needs an awaited setup call with a compatible psycopg connection configuration.
- **Recommendation**: Add `langgraph-checkpoint-postgres` and either explicit `psycopg[binary,pool]` or a documented transitive dependency. Decide whether app DB access remains asyncpg while checkpointing uses psycopg, or whether all DB access moves to psycopg.

### C-2 Hybrid retrieval SQL duplicates rows and does not implement the promised filter
- **Section**: §9.2 tasks 1-2, §9.3
- **Problem**: `unioned AS (SELECT * FROM vec UNION SELECT * FROM bm)` includes the score column, so a chunk found by both vector and lexical search can appear twice with different fifth-column values. The function accepts `filter_paper_ids`, but the SQL snippet has no filter in either CTE. Also, `ts_rank_cd` is PostgreSQL lexical ranking, not BM25.
- **Impact**: Top-k results can contain duplicates, filter acceptance cannot pass as written, and tests can overstate retrieval quality.
- **Recommendation**: Union only stable chunk identifiers, join score CTEs by `chunk_id`, add the `filter_paper_ids` predicate to both CTEs, and either rename BM25 to lexical full-text rank or specify a real BM25 implementation.

### C-3 Phase 0 has ordering and tracking contradictions
- **Section**: §0 rule 7, §5.2 tasks 2, 11, 12, §5.3, Appendix E
- **Problem**: §5.2 task 11 says to use `git mv` after `git init`, but `git init` is task 12. §0 says the plan is moved "after Phase 0 is committed" while also saying the move is part of the Phase 0 commit. Phase 0 must track `.gitkeep` files in empty directories, but `.gitignore` ignores `data/pdfs/` and `data/eval_outputs/`, so `.gitkeep` files under them will not be tracked. Appendix E requires `docs/STATUS.md`, but §4 and §5.2 omit it.
- **Impact**: A strict executor cannot satisfy Phase 0 without reordering or inventing changes.
- **Recommendation**: Move `git init` before `git mv`, clarify the move happens before the Phase 0 commit, add gitignore exceptions for data `.gitkeep` files or remove that tracking requirement, and add `docs/STATUS.md` to §4 and §5.2.

### C-4 Model registry will break or drift during the one-month execution window
- **Section**: §6.2, §6.3 task 3, Appendix A
- **Problem**: `google/gemini-2.0-flash-001` exists on OpenRouter but is marked "Going away June 1, 2026", which lands inside the project budget. DeepSeek official docs no longer support the plan's claim that `deepseek-chat` is V4 Pro; current docs point to explicit V4 IDs such as `deepseek-v4-pro`/`deepseek-v4-flash` and warn about legacy names. `anthropic/claude-haiku-4.5` appears valid on OpenRouter.
- **Impact**: Phase 1 smoke tests and later cheap-model paths can fail or silently use a different model tier than designed.
- **Recommendation**: Update the registry before execution. At Phase 1, verify model IDs through provider model catalogs/docs first, then use optional smoke tests only after the human owner supplies keys.

## Major Concerns (should resolve)

### M-1 Router and deduplicator are built but not wired into the graph
- **Section**: §10.2 tasks 3 and 5, §12.2 task 3
- **Problem**: Phase 5 creates `route()` and `dedupe()`, but the Phase 7 retrieve node does not call either. The source-router behavior described in §2.1 is therefore absent from the executable graph.
- **Recommendation**: Add explicit calls to `query.router.route` and `postprocess.deduplicator.dedupe` in §12.2 task 3, with tests proving the router affects source choices.

### M-2 `session_summaries` belongs in the schema plan, not as a Phase 6 afterthought
- **Section**: §7.2, §11.2 task 3
- **Problem**: Phase 6 says to add `session_summaries` to Phase 2 "if missed"; that is knowingly leaving a required table out of the schema phase.
- **Recommendation**: Add `006_session_summaries.sql` to Phase 2 now, or explicitly define Phase 6 as a migration phase with acceptance criteria for bootstrap order and idempotency.

### M-3 Async client lifecycle is underspecified
- **Section**: §6.3 tasks 4-8, §14.2 task 1
- **Problem**: Singleton `httpx.AsyncClient` and Cohere async clients need deterministic shutdown, but the plan has no `aclose()` API or Chainlit/FastAPI shutdown hook.
- **Recommendation**: Add `close_llm_clients()`/lifespan handling, test client closure, and document ownership of provider clients.

### M-4 pgvector asyncpg type registration is missing
- **Section**: §7.2 task 1, §9.2 tasks 1-2
- **Problem**: Passing Python vectors through asyncpg requires `pgvector.asyncpg.register_vector` on each connection or pool initializer. The plan creates the extension but does not register codecs.
- **Recommendation**: Add registration to `get_pool()` and test inserting/searching a vector through asyncpg.

### M-5 Embedding dimension consistency needs enforcement
- **Section**: §2.4, §6.3 task 6, §7.2 task 2, §11.2 tasks 5-7
- **Problem**: `text-embedding-3-small` defaults to 1536 dimensions, matching `vector(1536)`, but the OpenAI API supports a `dimensions` parameter. Concept embeddings are nullable and the plan does not state how concept definitions are embedded.
- **Recommendation**: For Phase 1, forbid overriding embedding dimensions or validate every returned vector length. For Phase 6, specify how concept embeddings are generated and stored.

### M-6 Ingestion idempotency is unsafe under partial failures
- **Section**: §8.2 task 6, §8.3
- **Problem**: `ingest_paper()` returns early if a paper row exists, but the pipeline can fail after inserting `papers` and before inserting all `chunks` or `citations`.
- **Recommendation**: Use a transaction or add ingestion status columns and repair logic. Add tests for retry after mid-pipeline failure.

### M-7 External-service acceptance criteria are too brittle
- **Section**: §8.3, §9.3, §10.3
- **Problem**: "Each paper has at least 1 citation row", "ReAct chunks rank in top 5", and fixed end-to-end paper traversal depend on arXiv/Semantic Scholar coverage, provider behavior, and scoring calibration.
- **Recommendation**: Convert fragile checks into bounded expectations with fixture-backed tests, and keep live checks as observational integration smoke tests.

### M-8 Test coverage misses the highest-risk logic
- **Section**: §6.3 task 11, §9.2 task 3, §10.2 task 11, §12.2 task 10
- **Problem**: There are no required tests for provider HTTP status mapping, hybrid SQL execution with overlap/filtering, router integration, citation verifier parsing, or graph edge loop budgets.
- **Recommendation**: Add focused unit/integration tests for those paths before relying on the Phase 7 e2e test.

### M-9 `.env.example` is not complete relative to env-overridable config
- **Section**: §2.4, §5.2 task 4, §6.3 task 9
- **Problem**: §2.4 says constants should be overridable by environment variables, but `.env.example` lists only keys, Postgres, LangSmith, and `LOG_LEVEL`.
- **Recommendation**: Add every supported override or change §2.4 to say constants are code defaults only. Also document optional OpenRouter metadata headers if used.

### M-10 Tool-call and citation data contracts are under-specified
- **Section**: §6.3 task 4, §12.2 tasks 5-6
- **Problem**: The plan asks to normalize tool calls and parse citations, but it does not define exact Pydantic models or failure behavior.
- **Recommendation**: Define `ChatResponse`, `ToolCall`, `Citation`, and `VerificationReport` in Phase 1/5 before graph wiring.

### M-11 Concept graph retrieval cannot derive paper hints without an explicit join
- **Section**: §7.2 task 2, §11.2 task 7
- **Problem**: `evidence_chunks` stores chunk IDs, but `GraphExpansion.extra_paper_hints` needs paper IDs. The conversion path is not specified.
- **Recommendation**: Specify a chunk-to-paper join in `graph_retriever.py` and add a test for relation evidence becoming paper filters/biases.

### M-12 Phase 5 and Phase 6 are oversized
- **Section**: §10, §11
- **Problem**: Phase 5 includes decomposition, rewriting, HyDE, routing, rerank, dedupe, three Self-RAG judges, and multi-hop ingestion. Phase 6 includes three memory tiers plus extraction, linking, graph retrieval, reflection, and schema changes.
- **Recommendation**: Split each into two reviewable phases or define a smaller mandatory slice with later enhancements.

### M-13 Operational handoffs and rollback are incomplete
- **Section**: §3.2, §3.3, §5.2 task 12, §7.3, §8.3
- **Problem**: The plan assumes GitHub CLI auth, Docker daemon availability, provider key availability, and database resets. It does not say when the human owner seeds `.env`, nor how to roll back a bad phase commit before the next phase.
- **Recommendation**: Add explicit owner handoffs after Phase 0 and before live integration tests. Add a rollback policy such as `git revert <phase-commit>` unless the planner authorizes amend/reset.

## Minor Issues
- L-1 (§5.2 task 5, Appendix A): Appendix A says dependency versions are in §5.5, but they are actually in §5.2 task 5.
- L-2 (§3.2, §5.2 task 12): §3.2 includes `gh repo create ... --license=mit`; §5.2 omits `--license=mit`. Reconcile the command.
- L-3 (§5.2 task 6, §14.3): `make ui` uses Chainlit `-h`, which is headless. The acceptance criterion should say "serves" at localhost, not "opens" a page.
- L-4 (§7.2 task 2): `concepts.updated_at` is never automatically updated. Add a trigger or require upsert code to set it.
- L-5 (§8.2 task 2): Semantic Scholar `fields` parameters and rate-limit/backoff behavior are not specified.
- L-6 (§8.2 task 7): If the seed list is YAML, `PyYAML` is missing. Prefer the allowed Python module path or add the dependency.
- L-7 (§12.2 task 1): `messages: Annotated[list[Message], add_messages]` should specify LangChain `AnyMessage`/message-like objects, not an undefined local `Message`.
- L-8 (§12.3): "Killing the process mid-run" is only resumable at checkpoint boundaries and has no literal validation command.
- L-9 (§14.4): `make ui &` and `make ui-graph &` validation commands leave background processes running; add cleanup instructions.
- L-10 (§13.2 task 5): LangSmith dataset creation/upsert semantics are unspecified, risking duplicate datasets on repeated `make eval`.
- L-11 (§4, §7.2 task 2): `src/core/sql/` is created in Phase 2 but absent from the target directory tree.
- L-12 (§4, Appendix E): `docs/STATUS.md` is required by Appendix E but absent from the target directory tree.
- L-13 (§6.5): The grep command does not catch vendor SDK imports in tests; that is probably acceptable, but say so explicitly.

## Library / Version Findings
| Name | Pinned | Concern | Recommendation |
|---|---|---|---|
| `langgraph` (§5.2 task 5, Appendix A) | `>=0.2.50,<0.3` | Range exists on PyPI; 0.2.50 exists. Postgres saver is not included in this package. | Keep only if intentionally staying on 0.2.x; add `langgraph-checkpoint-postgres`. |
| `langchain-core` (§5.2 task 5) | `>=0.3,<0.4` | Range exists. No blocker found. | Verify resolver compatibility with the selected LangGraph 0.2.x version. |
| `langsmith` (§5.2 task 5) | `>=0.1.140,<0.2` | Range exists but is old relative to current LangSmith releases. | Keep for compatibility or loosen after checking LangGraph/Chainlit constraints. |
| `anthropic` (§5.2 task 5) | `>=0.40,<1` | Range exists, but plan does not use the SDK directly. | Remove unless needed for future native Anthropic support. |
| `openai` (§5.2 task 5) | `>=1.50,<2` | Range exists. No blocker found. | Keep for embeddings. |
| `cohere` (§5.2 task 5, Appendix A) | `>=5.11,<6` | Range exists; `AsyncClientV2` is plausible, and rerank response uses `relevance_score`, not `score`. | Map Cohere results explicitly to local `score`; test against mocked SDK objects. |
| `httpx` (§5.2 task 5) | `>=0.27,<1` | Range exists. Lifecycle management missing in plan. | Keep and add shutdown handling. |
| `asyncpg` (§5.2 task 5) | `>=0.29,<1` | Range exists. Not usable for `AsyncPostgresSaver`. | Keep for app queries only if paired with psycopg for checkpointing. |
| `pgvector` (§5.2 task 5, §9.2) | `>=0.3,<1` | Range exists; asyncpg vector codec registration is required. | Add `pgvector.asyncpg.register_vector` in pool init. |
| `arxiv` (§5.2 task 5, §8.2) | `>=2.1,<3` | Range exists. Sync library use inside async functions needs care. | Specify `asyncio.to_thread` or direct async HTTP. |
| `pypdf` (§5.2 task 5) | `>=5.0,<6` | Range exists. No blocker found. | Keep. |
| `tiktoken` (§5.2 task 5, §8.2 task 5) | `>=0.8,<1` | Range exists. `cl100k_base` is correct for `text-embedding-3-small` per OpenAI embedding docs. | Keep. |
| `networkx` (§5.2 task 5) | `>=3.3,<4` | Range exists. No blocker found. | Keep. |
| `pydantic` (§5.2 task 5, §13.2 task 1) | `>=2.9,<3` | Range exists. Snippets are compatible with v2. | Keep; use `model_validate_json` for LLM JSON parsing. |
| `pydantic-settings` (§5.2 task 5, §6.3 task 9) | `>=2.6,<3` | Range exists. No blocker found. | Keep. |
| `python-dotenv` (§5.2 task 5) | `>=1.0,<2` | Range exists. No blocker found. | Keep. |
| `tenacity` (§5.2 task 5, §6.3 task 2) | `>=9.0,<10` | Range exists. No blocker found. | Keep; test retry logging masks secrets. |
| `chainlit` (§5.2 task 5, §14) | `>=1.3,<2` | Range exists; project is community-maintained, so API drift should be checked. | Keep but smoke test UI early in Phase 9. |
| `fastapi` (§5.2 task 5, §14.2 task 3) | `>=0.115,<1` | Range exists. No blocker found. | Keep. |
| `uvicorn` (§5.2 task 5, §14.2 task 5) | `>=0.32,<1` | Range exists. No blocker found. | Keep. |
| `pytest` (§5.2 task 5) | `>=8.3,<9` | Range exists. No blocker found. | Keep. |
| `pytest-asyncio` (§5.2 task 5) | `>=0.24,<1` | Range exists, but current stable has crossed 1.x. | Keep if 0.24 behavior is desired; otherwise allow 1.x after checking `asyncio_mode=auto`. |
| `pytest-mock` (§5.2 task 5) | `>=3.14,<4` | Range exists. No blocker found. | Keep. |
| `ruff` (§5.2 task 5) | `>=0.7,<1` | Range exists. No blocker found. | Keep. |
| `mypy` (§5.2 task 5) | `>=1.13,<2` | Range exists. No blocker found. | Keep. |
| `hatchling` (§5.2 task 5 build-system) | build requirement | Range not pinned; acceptable but reproducibility is lower. | Pin a major range if deterministic builds matter. |
| `langgraph-checkpoint-postgres` (§7.2 task 3, Appendix A) | Missing | Required for `from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver` ([PyPI](https://pypi.org/project/langgraph-checkpoint-postgres/)). | Add it. |
| `psycopg` (§7.2 task 3, Appendix A) | Missing/transitive | Required by LangGraph Postgres saver; asyncpg is not a substitute. | Add explicitly or document transitive install. |
| `anthropic/claude-haiku-4.5` (§6.3 task 3) | Model ID | Exists on OpenRouter. | Verify provider availability and tool/JSON support in Phase 1. |
| `google/gemini-2.0-flash-001` (§6.3 task 3) | Model ID | Exists on OpenRouter but is marked going away June 1, 2026. | Replace before execution or set a supported fallback. |
| `deepseek-chat` (§6.2, §6.3 task 3) | Model ID | Legacy/compatibility name; not a reliable V4 Pro identifier. | Use current DeepSeek docs and select `deepseek-v4-pro` or `deepseek-v4-flash`. |
| `text-embedding-3-small` (§2.2, §2.4, §6.3 task 6) | Model ID | Valid OpenAI embedding model; default vector length is 1536 and `cl100k_base` is correct. | Keep; do not pass a different `dimensions` value. |
| `rerank-v3.5` (§2.2, §6.3 task 7, Appendix A) | Model ID | Valid Cohere rerank model, but Cohere now also lists v4.0 rerank models as latest. | Decide whether reproducibility or latest quality matters; verify in Phase 1. |

## Ambiguities Requiring Guessing
For each, propose three concrete options the planner can pick from.

### A-1 Database driver ownership
- **Section**: §7.1, §7.2 task 3, §12.2 task 9
- **Ambiguity**: The plan says "same database" but also implies the same asyncpg pool for LangGraph checkpointing.
- **Options**:
  1. Use asyncpg for app SQL and psycopg only for `AsyncPostgresSaver`.
  2. Switch all database access to psycopg async so one pool type is used.
  3. Use connection strings for checkpointing and keep `src/core/db.py` scoped to app queries only.

### A-2 Hybrid score fusion strategy
- **Section**: §9.2 task 2
- **Ambiguity**: The plan documents raw alpha weighting despite incompatible score ranges.
- **Options**:
  1. Keep raw weighted sum and document it as a baseline.
  2. Use reciprocal rank fusion over vector and lexical ranks.
  3. Normalize each candidate set with min-max or z-score before weighting.

### A-3 Source routing semantics
- **Section**: §10.2 task 3, §12.2 task 3
- **Ambiguity**: The router is supposed to decide live Semantic Scholar use based on local hit count, but local retrieval happens downstream.
- **Options**:
  1. Run an initial local retrieval before routing, then route.
  2. Make router return only policy flags, with retrieve deciding after hit counts.
  3. Remove live Semantic Scholar routing from Phase 5 and leave it to multi-hop ingestion.

### A-4 Concept embedding lifecycle
- **Section**: §11.2 tasks 4-7
- **Ambiguity**: The plan uses concept vector search but never says what text is embedded or when.
- **Options**:
  1. Embed `display_name + definition` during extraction.
  2. Embed only the canonical name first, then refresh with definition after upsert.
  3. Skip concept vector search until enough definitions exist; rely on aliases first.

### A-5 Ingestion partial failure behavior
- **Section**: §8.2 task 6
- **Ambiguity**: Idempotency behavior after a partial insert is not defined.
- **Options**:
  1. Wrap paper/chunk/citation writes in one transaction.
  2. Add `ingestion_status` and resume incomplete papers.
  3. On retry, delete incomplete chunks/citations for that paper and rebuild.

### A-6 arXiv async and rate limiting
- **Section**: §8.2 tasks 1 and 7
- **Ambiguity**: The `arxiv` package is sync, while the public functions are async and seed ingestion runs concurrently.
- **Options**:
  1. Use one global `arxiv.Client` with its built-in delay and call it in a thread.
  2. Use direct httpx calls with an async global rate limiter.
  3. Serialize all arXiv metadata/PDF requests and parallelize only PDF parsing/embedding.

### A-7 Package/import layout
- **Section**: §4, §5.2 task 5, §6.5
- **Ambiguity**: The plan imports `src.llm...` but the tree does not include `src/__init__.py`, and Hatch is configured with `packages = ["src"]`.
- **Options**:
  1. Add `src/__init__.py` and intentionally make `src` the import package.
  2. Rename to a real package such as `arxiv_research_agent` under `src/`.
  3. Keep repo-root execution only and remove packaging expectations.

### A-8 Human-owner handoff timing
- **Section**: §3.1, §5.2 task 4, §6.4, §7.3
- **Ambiguity**: The plan does not say exactly when the owner creates `.env`, opens Docker, or authenticates `gh`.
- **Options**:
  1. Add a post-Phase-0 owner checklist before Phase 1 starts.
  2. Let Codex stop with a questions file whenever a handoff is missing.
  3. Mark all live-provider checks optional until the owner explicitly announces `.env` readiness.

## Questions for the Planner
Numbered, each answerable in 1-3 sentences.
1. Should the project stay on `langgraph<0.3`, or should the plan update to the current LangGraph 1.x line before execution?
2. Do you want asyncpg for application SQL plus psycopg for checkpointing, or one database driver everywhere?
3. Which DeepSeek model should `deepseek-v4-pro` map to in the registry: current `deepseek-v4-pro`, current `deepseek-v4-flash`, or legacy `deepseek-chat`?
4. What model should replace `google/gemini-2.0-flash-001` before its June 1, 2026 OpenRouter retirement?
5. Should lexical retrieval be called BM25, or should the implementation change to actual BM25/rank fusion?
6. Should `session_summaries` be a Phase 2 schema table or a Phase 6 migration?
7. Should Phase 5 be split into retrieval query/rerank/self-RAG first and multi-hop second?
8. Should Phase 6 be split into episodic memory first and semantic graph/reflection second?
9. Should `data/pdfs/` and `data/eval_outputs/` be tracked with `.gitkeep` or created lazily at runtime?
10. What is the expected rollback action if a phase commit passes locally but the planner rejects it after review?

## What the Plan Does Well
- Strong phase-gate discipline: one phase, one report, one commit, no push.
- Secret-handling rules are explicit and mostly enforceable.
- The architecture focuses on the two stated learning goals: agentic RAG and tiered memory.
- The SQL schema is broadly reasonable for PostgreSQL 16 + pgvector once the retrieval/query issues are fixed.
- The tokenizer and embedding dimension choices for `text-embedding-3-small` are technically correct.
- The skipped-by-default integration tests are the right pattern for live provider calls.
