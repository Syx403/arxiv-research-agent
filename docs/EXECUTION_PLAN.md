# Execution Plan: arxiv-research-agent

> **Document version**: 1.5
> **Owner**: Project planner (Claude)
> **Executor**: Codex (GPT-5.5)
> **Reviewer**: Project planner (Claude), per phase
> **Audience**: Codex executor and the human owner. This document is Codex's only source of context. Read it from start to finish before doing anything.

## Changelog v1.4 → v1.5

This revision answers a Phase 0 task 13 blocker raised by Codex (`docs/phase_reports/PHASE_0_QUESTIONS.md`).

The `gh` CLI rejects `--source` together with `--license` (error: `the --source option is not supported with --clone, --template, --license, or --gitignore`). Since `--source=.` is the correct mode for our flow (we never push remote-initialized content), `--license=mit` is dropped from the canonical command. The local `LICENSE` file remains the only license artifact, which is consistent with the original intent (the plan already said the local file is the source of truth).

- §3.2 and §5.2 task 13: command is now `gh repo create arxiv-research-agent --private --source=. --remote=origin` (no `--license` flag).
- §3.2 surrounding paragraph: removed the now-obsolete instruction to overwrite a remote-generated `LICENSE` (none is generated under `--source` mode).
- §5.2 surrounding paragraph: clarifies that the remote is empty on GitHub by design.

## Changelog v1.3 → v1.4

This revision answers a Phase 0 blocker raised by Codex: the project root already contains three review-feedback artifacts (`REVIEW_FEEDBACK.md`, `REVIEW_FEEDBACK_V2.md`, `REVIEW_FEEDBACK_V3.md`) generated during plan iteration. The plan now formally preserves them as project history.

- §4 directory tree: adds `docs/review_feedback/` containing the three review files.
- §5.2 task 14 (Phase 0): extends the move step to also `git mv` the three review files from the repo root into `docs/review_feedback/`, alongside the existing `EXECUTION_PLAN.md` move. The move is part of the Phase 0 commit.
- §5.3 acceptance: adds a check that `docs/review_feedback/` contains the three files and the repo root no longer does.

## Changelog v1.2 → v1.3

This revision addresses Codex's third-round review (`REVIEW_FEEDBACK_V3.md`). Three small consistency fixes from v1.2 surgical edits.

- **N3-M-1**: §8.2 task 6 now requires writing `papers.ingestion_status='failed'` in a short separate transaction on Phase A errors (not only Phase B), reconciling with the §8.2 task 8 partial-failure test and the §8.3 idempotency contract.
- **N3-L-1**: §6.4 acceptance criterion updated to reference the renamed ADR `0001-model-and-langgraph-selection.md`.
- **N3-L-2**: §7.4 validation commands now include `tests/integration/test_checkpoint_roundtrip.py` (added in v1.2 §7.2 task 4 but missed in §7.4).

## Changelog v1.1 → v1.2

This revision addresses Codex's second-round review (`REVIEW_FEEDBACK_V2.md`). The v1.1 changes still stand; v1.2 fixes the residuals and the new issues introduced by v1.1.

- **N-C-1**: psycopg connection pool now requires `kwargs={"autocommit": True, "row_factory": dict_row}` per LangGraph PostgresSaver documentation. Phase 2 also adds an integration test that performs a real checkpoint write and read, not just table existence.
- **N-C-2**: Phase 1 model-verification task is extended to also verify the current LangGraph stable line and any open security advisory (notably the historical msgpack checkpoint deserialization advisory on the 0.2.x line). The decision (stay on `<0.3` with documented mitigation, or upgrade to 1.x) is recorded in `docs/decisions/0001-model-and-langgraph-selection.md` (the prior `0001-model-selection.md` is renamed to absorb this).
- **N-M-1 / residual A-3**: Phase 9 retrieve node reorders steps so that `query.router.route` runs **first**, and `graph_retriever.expand_query_via_graph` is invoked only when `route.use_concept_graph == True`. The `test_router_wired.py` integration test is now coherent.
- **N-M-2 / residual M-10**: Citations in generated answers must use the format `[paper_id#chunk_id]`. The synthesizer prompt requires both fields. The citation regex extracts both. `Citation.chunk_id` is now `int` (not Optional). The verifier's `evidence_lookup` is keyed by the same `chunk_id` and is constructed deterministically from the synthesizer's evidence list.
- **N-M-3 / residual M-13 / residual A-8**: Every integration-test-bearing phase (Phase 2, 3, 4, 6, 7, 8, 9) now begins its validation section with an explicit "Requires Owner Handoff H-3" line.
- **N-L-1**: §6.5 grep command appends `|| echo "ok"` so a clean run does not exit non-zero.
- **N-L-2**: `src/eval/datasets/seed_papers.py` is now in the directory tree (§4) and explicitly listed in Phase 3.
- **N-L-3**: All code constants live in `src/core/types.py`. The "or `src/core/constants.py`" alternative is removed.
- **N-L-4**: Cohere rerank candidate IDs corrected to actual model identifiers: `rerank-v3.5`, `rerank-v4.0-pro`, `rerank-v4.0-fast`.
- **N-L-5**: `src/core/types.py` opens with `from __future__ import annotations` so the `Message` → `ToolCall` forward reference resolves.
- **N-L-6**: Phase 3 ingestion is refactored. External I/O (PDF download, embedding API calls, Semantic Scholar lookups) happens **before** any DB transaction is opened. The transaction is short and only performs the row inserts. On error during external I/O, no transaction was held.

## Changelog v1.0 → v1.1

This revision incorporates the findings of Codex's pre-execution review (saved at the time as `REVIEW_FEEDBACK.md`). Highlights:

- **C-1**: Added `langgraph-checkpoint-postgres` and `psycopg[binary,pool]`. Decided on a split-driver architecture: asyncpg for application SQL, psycopg only for LangGraph checkpointing.
- **C-2**: Replaced alpha-weighted hybrid SQL with **Reciprocal Rank Fusion (RRF, k=60)**. Renamed "BM25" to "lexical full-text rank" everywhere because we use `ts_rank_cd`, not real BM25. Added `filter_paper_ids` predicate to both CTEs and de-duplicated by `chunk_id`.
- **C-3**: Reordered Phase 0 tasks: `git init` before `git mv`. `data/pdfs/` and `data/eval_outputs/` are now created lazily at runtime — they have no `.gitkeep`. Added `docs/STATUS.md` to the directory tree and to Phase 0 tasks.
- **C-4**: Removed pinned non-Anthropic model IDs from the document body. Phase 1 has a new mandatory **model verification task** that fetches each provider's current model catalog and writes the chosen IDs into `src/llm/registry.py` and `docs/decisions/0001-model-selection.md`.
- **M-1**: Phase 9 retrieve node now explicitly invokes `query.router.route` and `postprocess.deduplicator.dedupe`, with tests.
- **M-2**: `session_summaries` table moved into Phase 2 schema (`006_session_summaries.sql`).
- **M-3**: Provider clients now expose `aclose()`. Chainlit and FastAPI bind a lifespan that closes them. Tests cover client closure.
- **M-4**: `get_pool()` registers the pgvector codec via `pgvector.asyncpg.register_vector` on every new connection.
- **M-5**: Embedding client validates returned vector length matches `EMBEDDING_DIM`. Concept embeddings are generated from `display_name + ". " + definition` at extraction time.
- **M-6**: Ingestion writes `papers`, `chunks`, and `citations` for one paper inside a single transaction.
- **M-7**: Acceptance criteria that depend on external services are softened to "should typically" with a clear pass margin. The "≥ 1 citation per paper" check becomes "≥ 80% of papers have at least 1 citation row".
- **M-8**: Added required tests for HTTP status mapping, hybrid SQL with overlap and filter, router integration, citation parser, and edge budget.
- **M-9**: Section 2.4 now states constants are code defaults; only secrets and infrastructure config are environment-overridable.
- **M-10**: `ChatResponse`, `ToolCall`, `Citation`, `VerificationReport` Pydantic models defined in Phase 1.
- **M-11**: `graph_retriever.py` explicitly joins `chunks` to derive paper IDs from chunk evidence.
- **M-12**: Split former Phase 5 into Phase 5 (query/rerank/self-RAG) + Phase 6 (multi-hop), and former Phase 6 into Phase 7 (episodic) + Phase 8 (semantic). Total phases now 13 (Phase 0 through Phase 12).
- **M-13**: New Section 3.6 (Owner Handoff Checkpoints) and Section 3.7 (Rollback Policy).
- **L-1 through L-13**: All resolved inline.
- **A-1 through A-8**: Resolved per the planner's choices and woven into the affected phases.

---

## 0. Reading Instructions for Codex

You (Codex) are the sole executor of this project. You have **no prior context** about why we are building this or what decisions led to the architecture below. Everything you need is in this document.

Hard rules:

1. **Do not deviate from this plan without asking the human owner first.** If something seems wrong, missing, or under-specified, stop and ask. Do not improvise architecture.
2. **Execute one phase at a time.** Each phase ends with a commit (no push) and a phase report. Wait for the planner's review approval before starting the next phase.
3. **Never push to GitHub.** Only `git commit`. The human owner controls `git push` manually.
4. **Never read, cat, or print the contents of `.env`, `.env.local`, or any file containing secrets.** Never write code that logs API keys, tokens, headers containing `Authorization`, or full request payloads. Mask secrets in any logs.
5. **Never paste API keys into code as literals.** All keys come from environment variables loaded by `python-dotenv` from `.env` (which is gitignored). Code only references variable names, never values.
6. **Track your progress.** Use a checklist for each phase's tasks. Mark them complete as you go.
7. **Move this document.** As part of the Phase 0 commit, this file (`EXECUTION_PLAN.md`) is moved from the repository root into `docs/EXECUTION_PLAN.md` using `git mv`. The move is included in the Phase 0 commit.

What "review approval" looks like in practice:

- After committing a phase, write a phase report at `docs/phase_reports/PHASE_<N>_REPORT.md` (template in Appendix C).
- The human owner will start a new conversation with the planner. The planner reads your report, reads the affected files, and runs validation commands.
- The planner will either approve or request changes. Apply the changes, recommit (per the rollback policy in Section 3.7), regenerate the report, and wait for re-review.

---

## 1. Project Overview

### 1.1 What we are building

A research-oriented conversational agent that helps the user deeply explore the academic literature on LLM-based agents. The user asks research questions in natural language; the agent answers with grounded citations to specific papers, follows multi-hop citation chains across the corpus when useful, accumulates a per-session research notebook (concept graph), and persists memory across sessions.

The seed corpus is a curated set of ~30 foundational papers on LLM agents (ReAct, Reflexion, MemGPT, Toolformer, MetaGPT, Self-RAG, CRAG, etc.). The corpus expands lazily on demand: when retrieval requires a paper not in the local index, the agent fetches it from arXiv, indexes it, and continues.

### 1.2 Why these design choices

This project exists to give the human owner deep, hands-on knowledge of agent architecture for engineering interviews. Two subsystems are the focus and must be implemented with depth:

1. **Agentic RAG** — query decomposition, query rewriting (HyDE), source routing, hybrid retrieval (vector + lexical, fused via RRF), reranking, self-RAG (relevance + sufficiency + citation grounding verification), and bounded multi-hop traversal across the citation graph.
2. **Tiered memory** — working memory (per-task scratchpad), episodic memory (conversation history with summarization-based compression), semantic memory (a small concept knowledge graph that is *queried by retrieval*, not just visualized), and end-of-session reflection that promotes information from episodic to semantic.

Other subsystems (corpus ingestion, vector index, LLM provider adapter, evaluation, UI) exist to support these two. They should be solid but not over-engineered.

### 1.3 Out of scope

- Training or fine-tuning any model.
- Multi-tenant access control. Single user assumed.
- A production-grade deployment story. Local development on macOS only.
- Multi-modal input (images, audio).
- Non-English papers.
- Building our own vector index, embedding model, or reranker. We use external APIs and `pgvector`.

### 1.4 Success criteria for the project

By the end of Phase 12:

1. The agent answers research questions about LLM agents with grounded citations on a 30-50 question gold evaluation set, with measurable retrieval recall, citation precision, and answer correctness scores logged to LangSmith.
2. The agent demonstrates multi-hop reasoning on at least one canonical example (e.g., "Compare ReAct's reasoning step with Reflexion's self-critique" must traverse from one paper to the other via citation).
3. The concept graph contains 50+ nodes after a few demo sessions and is queried by the retrieval layer.
4. The Chainlit UI shows live agent reasoning steps and lets the user click any citation to see the source paper.
5. Persistence works: a conversation interrupted between turns can resume in a new process via the LangGraph PostgresSaver checkpoint.
6. All key decisions are documented in `docs/decisions/` (ADR style).
7. `docs/interview_talking_points.md` lists 15+ specific technical points the human owner can discuss in an interview, each tied to a file in this repository.

---

## 2. Architecture Overview

### 2.1 High-level data flow

```
User question (Chainlit UI)
  -> LangGraph entry node
  -> Query decomposer (sub-questions)
  -> For each sub-question:
       -> Query rewriter (optionally HyDE)
       -> Source router (returns policy flags only)
       -> Retrieve node (decides live SS use based on local hit count)
       -> Hybrid retrieval (vector + lexical, fused via RRF)
       -> Deduplicator
       -> Cohere reranker (top-5)
       -> Self-RAG relevance judge (per chunk)
       -> If insufficient: rewrite query and retry (bounded)
       -> If multi-hop justified: walk citations 1-2 hops
  -> Sufficiency check across all sub-question results
  -> Synthesizer node generates grounded answer with inline citations
  -> Citation verifier confirms each cited claim is supported
  -> Reflection node updates semantic memory (concept graph)
  -> LangGraph checkpoint persists state to Postgres
  -> Chainlit streams steps to user
```

### 2.2 Tech stack

| Layer | Choice | Why |
|---|---|---|
| Orchestration | LangGraph (>=0.2.50,<0.3) | Explicit state graph, checkpointing, time travel |
| LLM (chat/tool) | Anthropic Claude Haiku 4.5 (main brain), plus a current Flash-tier model and a current DeepSeek V4 flagship for cheap routing | Multi-model for cost-performance trade-off |
| Chat routing | OpenRouter for Haiku and the chosen Flash-tier; native DeepSeek API for the chosen DeepSeek model | OpenRouter does not cover all needs |
| Embedding | OpenAI `text-embedding-3-small` (1536 dim), direct OpenAI API | OpenRouter has weak embedding coverage |
| Reranker | Cohere rerank, direct Cohere API; specific version selected in Phase 1 | OpenRouter has no rerank |
| Database | PostgreSQL 16 + `pgvector` extension, single Docker service | Single source of truth for vectors, metadata, concept graph, and LangGraph checkpoints |
| App DB driver | `asyncpg` | Fast async access for retrieval, ingestion, memory |
| Checkpoint DB driver | `psycopg` (via `langgraph-checkpoint-postgres`) | LangGraph's PostgresSaver requires psycopg, not asyncpg |
| Persistence | LangGraph `AsyncPostgresSaver` from `langgraph.checkpoint.postgres.aio` | Same Postgres database as everything else, but via psycopg |
| Eval | LangSmith (traces + datasets + LLM-as-judge) | Native LangGraph integration |
| UI | Chainlit (chat + step trace) + a small FastAPI route serving a static `react-flow` page for the concept graph | Chainlit gives reasoning trace visualization for free |
| Testing | pytest + pytest-asyncio | Standard |
| Retry | tenacity | Standard |
| Config | pydantic-settings + python-dotenv | Type-safe env loading |

### 2.3 Module responsibility table

| Module | Responsibility | Talked-about-in-interview point |
|---|---|---|
| `src/llm/` | Provider adapter: unified chat/embedding/rerank clients with retry, error normalization, multi-provider routing, deterministic shutdown | "I built a thin model gateway so business code never knows which provider serves a model" |
| `src/corpus/` | arXiv/Semantic Scholar/GitHub clients, PDF loading, section-aware chunking, transactional ingestion (seed batch and lazy on-demand) | "Lazy ingestion: when retrieval misses, the agent fetches and indexes a new paper mid-flight, transactionally" |
| `src/retrieval/index/` | pgvector store, RRF-fused hybrid (vector + lexical) search SQL | "Hybrid retrieval is RRF over a vector candidate set and a `ts_rank_cd` candidate set — score-unit-free fusion" |
| `src/retrieval/query/` | Query decomposition, rewriting, HyDE, source routing (policy flags only) | "Decomposition turns one user question into a small dependency graph of sub-questions; the router emits flags, not decisions" |
| `src/retrieval/postprocess/` | Cohere rerank, deduplicator (exact and near-duplicate text removal) | |
| `src/retrieval/self_rag/` | Per-chunk relevance judge, sufficiency check across all retrieved evidence, post-generation citation verifier | "Self-RAG: the agent grades its own retrieved chunks before generating, and re-retrieves if insufficient" |
| `src/retrieval/multi_hop/` | Citation walker with depth and budget controls; frontier queue | "Bounded multi-hop: I cap depth at 2 and use an LLM-judged frontier to avoid combinatorial explosion" |
| `src/memory/working.py` | Per-task scratchpad inside graph state | |
| `src/memory/episodic/` | Session message history with rolling summary compression cached to `session_summaries`; backed by LangGraph checkpoint | "Compression strategy: keep last K turns verbatim, summarize older turns recursively, cache summary keyed by upper message id" |
| `src/memory/semantic/` | Concept graph extractor, entity linker (alias), graph store (Postgres), graph retriever — concept graph is *queried by retrieval* | "The concept graph isn't decoration — when the agent retrieves, it expands the query along graph neighbors" |
| `src/memory/reflection.py` | End-of-session distillation: extract concepts and relations from episodic into semantic | |
| `src/graph/` | LangGraph state schema, nodes, edges, builder, AsyncPostgresSaver wiring | |
| `src/eval/` | Gold dataset, retrieval/citation/judge metrics, LangSmith runner | "I built my own eval harness with retrieval recall, citation precision, and a G-Eval style judge" |
| `src/ui/` | Chainlit app, citation cards, embedded react-flow graph view, lifespan-managed client cleanup | |
| `src/core/` | Config (pydantic-settings), DB connection pool with vector codec registration, common types, masked logging | |

### 2.4 Configuration constants

These constants are referenced throughout the document. Codify them in `src/core/config.py` as **code defaults** (typed module-level constants or class fields). They are **not** environment-overridable. Only secrets and infrastructure parameters (DB host/port/credentials, log level, LangSmith project name) come from `.env`.

| Name | Default | Notes |
|---|---|---|
| `EMBEDDING_DIM` | 1536 | `text-embedding-3-small` default; do not pass `dimensions` |
| `CHUNK_TOKEN_SIZE` | 800 | section-aware splitting target |
| `CHUNK_TOKEN_OVERLAP` | 100 | |
| `RETRIEVAL_TOP_K_VECTOR` | 30 | vector candidates before fusion |
| `RETRIEVAL_TOP_K_LEXICAL` | 30 | lexical candidates before fusion |
| `RRF_K` | 60 | RRF smoothing constant |
| `RETRIEVAL_TOP_K_AFTER_FUSION` | 30 | candidates fed to rerank |
| `RERANK_TOP_K` | 5 | after Cohere rerank |
| `MULTI_HOP_MAX_DEPTH` | 2 | from a paper, max hops to follow |
| `MULTI_HOP_FRONTIER_LIMIT` | 8 | per hop, max papers to expand |
| `MAX_SUB_QUESTIONS` | 5 | decomposer cap |
| `SELF_RAG_MAX_RETRIES` | 2 | rewrite-and-retry cap on insufficient retrieval |
| `EPISODIC_VERBATIM_TURNS` | 6 | keep this many recent turns uncompressed |
| `LLM_REQUEST_TIMEOUT_SECONDS` | 60 | per LLM call |
| `RETRY_MAX_ATTEMPTS` | 4 | shared retry policy |

---

## 3. Operational Discipline (Critical)

### 3.1 Secret handling

- Secrets live in `.env` at the repo root. `.env` is in `.gitignore`. Never commit `.env`.
- A template `.env.example` lives in the repo root and lists every required environment variable name with an empty value or a placeholder. It is committed.
- Code only references variable names via `pydantic-settings`. No code path may read a key file directly other than through the settings loader.
- You (Codex) must never:
  - Run `cat .env`, `head .env`, `tail .env`, `less .env`, or any command that prints the file contents.
  - Run `printenv`, `env`, `set`, or any command that dumps environment variables.
  - Read the `.env` file via the file-read tool.
  - Log API keys, request headers containing `Authorization`, or full request bodies that may contain keys.
  - Paste any real API key into source code, even temporarily.
- All HTTP clients must mask or omit `Authorization` and `x-api-key` headers from any logs.
- Configure `httpx` to disable automatic request/response logging. Configure `tenacity` retry log extras to redact sensitive fields.

### 3.2 Git workflow

- The repo is initialized in Phase 0.
- Each phase ends with exactly one `git commit -m "phase N: <short description>"` (extended message body in HEREDOC, see template in Appendix C).
- **Never run `git push`.** The human owner pushes manually.
- The remote is set in Phase 0 to a private GitHub repo named `arxiv-research-agent` under the user's account.
- Use exactly: `gh repo create arxiv-research-agent --private --source=. --remote=origin`. Do **not** pass `--license`: the `gh` CLI rejects it together with `--source`. The local `LICENSE` file (created in §5.2 task 9) is the only license artifact; the remote is intentionally empty on GitHub until the owner pushes.
- Never run `gh repo create ... --push`.
- Branch: `main` only for now. No feature branches.

### 3.3 Phase commit / review cycle

For each phase:

1. Read the phase section in this document end to end.
2. Make a checklist of every task and acceptance criterion.
3. Implement each task. Run tests as you go.
4. When all acceptance criteria pass, write the phase report at `docs/phase_reports/PHASE_<N>_REPORT.md` using the template in Appendix C.
5. Update `docs/STATUS.md` with the new phase status, commit hash placeholder (filled after commit), and date.
6. `git add` the new and modified files. **Do not** `git add` `.env` or anything in `.gitignore`.
7. `git commit` with the message template in Appendix C.
8. Replace the commit hash placeholder in `docs/STATUS.md` with the actual hash, then `git commit --amend` only if no review has begun, otherwise create a follow-up commit per Section 3.7.
9. Stop. Do not start the next phase. Do not push.
10. Wait for the human owner to bring the planner in for review.

### 3.4 If you are blocked or unsure

- Stop work.
- Write your question into `docs/phase_reports/PHASE_<N>_QUESTIONS.md` with full context and three to five concrete options if applicable.
- Notify the human owner. Do not guess.

### 3.5 Logging hygiene

- Use Python `logging` module configured in `src/core/logging.py`.
- Default level `INFO`. `DEBUG` for development.
- Mask secrets: implement a logging filter that scrubs `Authorization`, `x-api-key`, `api_key`, and any value matching common API key patterns.
- Never `print()` in library code. `print` is allowed only in CLI entry scripts and tests.
- httpx `event_hooks` may be used for timing but must not log request/response bodies or headers.

### 3.6 Owner handoff checkpoints

The plan assumes certain actions are performed by the human owner, not by Codex. Each handoff is explicit and named.

**Handoff H-0 — before Phase 0 starts** (already done by the time you read this):
- Project folder exists at `/Users/richsion/Desktop/arxiv-research-agent/`.
- This document exists in that folder.
- Owner has `gh` CLI installed and authenticated against their GitHub account.
- Owner has `docker` installed and the daemon running.
- Owner has `uv` installed.

**Handoff H-1 — after Phase 0 commit, before Phase 1 starts**:
- Owner copies `.env.example` to `.env`.
- Owner fills in real values for all required keys: `OPENROUTER_API_KEY`, `DEEPSEEK_API_KEY`, `OPENAI_API_KEY`, `COHERE_API_KEY`, and optionally `LANGSMITH_API_KEY`, `SEMANTIC_SCHOLAR_API_KEY`, `GITHUB_TOKEN`.
- Owner confirms readiness in the next planner conversation. **Codex must not start Phase 1 until this confirmation is recorded in `docs/STATUS.md`.**

**Handoff H-2 — after Phase 2 commit, before Phase 3 starts**:
- Owner runs `make db-up` and confirms the container is healthy.
- The same handoff confirms Phase 3 may run live ingestion (which costs money via embeddings).

**Handoff H-3 — before any phase that runs integration tests**:
- Owner exports `INTEGRATION_TESTS=1` for the test session.
- Owner is informed of expected provider spend.

**Handoff H-final — Phase 12 complete**:
- Owner reviews final eval numbers and decides whether to push to GitHub.

Codex prompts the owner with a clearly named handoff line (e.g., "Awaiting H-1 from owner") in the phase report whenever a handoff is pending.

### 3.7 Rollback policy

If the planner rejects a phase commit during review:

- **Default**: Codex creates a new commit that fixes the issues and references the original commit. Use a commit message of the form `phase N (fix): <short description>` and reference the rejected commit hash in the body.
- **`git commit --amend`**: allowed only if (a) the planner explicitly approves amending in writing, and (b) no other commits have been added on top.
- **`git revert`**: used to undo a phase entirely. Allowed when the planner says "revert phase N". Codex creates a `git revert <hash>` commit, then proceeds with a fresh re-implementation in the next commit.
- **`git reset --hard`**: never. The reflog is shared with the owner; do not destroy commits.
- **Force push**: not applicable; we never push.

---

## 4. Target Directory Structure

This is the final shape. Phase 0 creates the empty structure (folders with `.gitkeep` placeholders where appropriate). Subsequent phases populate it.

`data/pdfs/` and `data/eval_outputs/` are **not** tracked. They are created at runtime by ingestion and eval scripts. They have no `.gitkeep`.

```
arxiv-research-agent/
├── src/
│   ├── __init__.py
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── client.py
│   │   ├── retry.py
│   │   ├── errors.py
│   │   ├── registry.py
│   │   └── providers/
│   │       ├── __init__.py
│   │       ├── openrouter.py
│   │       ├── deepseek.py
│   │       ├── openai_embed.py
│   │       └── cohere_rerank.py
│   ├── corpus/
│   │   ├── __init__.py
│   │   ├── arxiv_client.py
│   │   ├── semantic_scholar.py
│   │   ├── github_client.py
│   │   ├── pdf_loader.py
│   │   ├── chunking.py
│   │   └── ingestion.py
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── index/
│   │   │   ├── __init__.py
│   │   │   ├── pgvector_store.py
│   │   │   └── hybrid_search.py
│   │   ├── query/
│   │   │   ├── __init__.py
│   │   │   ├── decomposer.py
│   │   │   ├── rewriter.py
│   │   │   └── router.py
│   │   ├── postprocess/
│   │   │   ├── __init__.py
│   │   │   ├── reranker.py
│   │   │   └── deduplicator.py
│   │   ├── self_rag/
│   │   │   ├── __init__.py
│   │   │   ├── relevance_judge.py
│   │   │   ├── sufficiency_check.py
│   │   │   └── verifier.py
│   │   └── multi_hop/
│   │       ├── __init__.py
│   │       ├── citation_walker.py
│   │       └── frontier.py
│   ├── memory/
│   │   ├── __init__.py
│   │   ├── working.py
│   │   ├── reflection.py
│   │   ├── episodic/
│   │   │   ├── __init__.py
│   │   │   ├── session_store.py
│   │   │   └── compressor.py
│   │   └── semantic/
│   │       ├── __init__.py
│   │       ├── extractor.py
│   │       ├── linker.py
│   │       ├── graph_store.py
│   │       └── graph_retriever.py
│   ├── graph/
│   │   ├── __init__.py
│   │   ├── state.py
│   │   ├── edges.py
│   │   ├── builder.py
│   │   └── nodes/
│   │       ├── __init__.py
│   │       ├── decompose.py
│   │       ├── retrieve.py
│   │       ├── self_rag.py
│   │       ├── multi_hop.py
│   │       ├── synthesize.py
│   │       └── reflect.py
│   ├── eval/
│   │   ├── __init__.py
│   │   ├── runner.py
│   │   ├── datasets/
│   │   │   ├── __init__.py
│   │   │   ├── seed_papers.py
│   │   │   └── gold_questions.py
│   │   └── metrics/
│   │       ├── __init__.py
│   │       ├── retrieval_metrics.py
│   │       ├── citation_metrics.py
│   │       └── llm_judge.py
│   ├── ui/
│   │   ├── __init__.py
│   │   ├── app.py
│   │   ├── elements.py
│   │   └── graph_view/
│   │       ├── __init__.py
│   │       ├── server.py
│   │       └── static/
│   │           └── index.html
│   └── core/
│       ├── __init__.py
│       ├── config.py
│       ├── logging.py
│       ├── types.py
│       ├── db.py
│       └── sql/
│           ├── 001_papers.sql
│           ├── 002_chunks.sql
│           ├── 003_citations.sql
│           ├── 004_concepts.sql
│           ├── 005_sessions.sql
│           ├── 006_session_summaries.sql
│           └── 007_triggers.sql
├── scripts/
│   ├── bootstrap_db.py
│   ├── ingest_seed_corpus.py
│   └── eval_run.py
├── tests/
│   ├── __init__.py
│   ├── unit/
│   │   └── __init__.py
│   └── integration/
│       └── __init__.py
├── docker/
│   └── docker-compose.yml
├── docs/
│   ├── EXECUTION_PLAN.md      (moved here in Phase 0)
│   ├── STATUS.md              (created in Phase 0)
│   ├── architecture.md        (Phase 12)
│   ├── interview_talking_points.md (started Phase 0, grown each phase)
│   ├── decisions/             (ADRs, written as decisions are made)
│   ├── phase_reports/         (one report per phase)
│   └── review_feedback/       (pre-execution review artifacts, moved in Phase 0)
│       ├── REVIEW_FEEDBACK.md      (v1.0 review)
│       ├── REVIEW_FEEDBACK_V2.md   (v1.1 review)
│       └── REVIEW_FEEDBACK_V3.md   (v1.2 review)
├── data/                      (folder exists in tree only at runtime)
├── .env.example
├── .gitignore
├── pyproject.toml
├── Makefile
├── LICENSE                    (MIT)
└── README.md
```

---

## 5. Phase 0 — Bootstrap

### 5.1 Goal

Create the project skeleton: directory layout, dependency manifest, environment template, Postgres+pgvector docker-compose, license, README, git repository, GitHub remote (without pushing), and move this execution plan into `docs/`.

### 5.2 Tasks (executed in this order)

1. **Verify working directory.** You should be inside `/Users/richsion/Desktop/arxiv-research-agent/`. If not, stop and ask.

2. **Create the directory tree** as specified in Section 4. Add a `.gitkeep` file in any source/test directory that would otherwise be empty so git tracks it. Do not add `.gitkeep` under `data/` paths because those are gitignored.

3. **Create `src/__init__.py`** (empty) so `src` is an importable package.

4. **Create `.gitignore`** at the repo root. Contents:

```
# Secrets
.env
.env.local
.env.*.local

# Python
__pycache__/
*.py[cod]
*$py.class
*.egg-info/
.pytest_cache/
.mypy_cache/
.ruff_cache/
.coverage
htmlcov/

# Virtual envs
.venv/
venv/
env/

# Data (created at runtime; not tracked)
data/

# OS
.DS_Store
Thumbs.db

# Editor
.idea/
.vscode/

# Build
dist/
build/
*.egg
```

5. **Create `.env.example`** at the repo root. Lists secrets and infrastructure config only (per Section 2.4, code constants are not env-overridable):

```
# LLM providers (required for Phase 1+)
OPENROUTER_API_KEY=
DEEPSEEK_API_KEY=
OPENAI_API_KEY=
COHERE_API_KEY=

# LangSmith (optional but recommended)
LANGSMITH_API_KEY=
LANGSMITH_PROJECT=arxiv-research-agent
LANGSMITH_TRACING=true

# Postgres (defaults match docker-compose.yml)
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=arxiv_agent
POSTGRES_USER=arxiv_agent
POSTGRES_PASSWORD=arxiv_agent_dev

# Optional API keys
SEMANTIC_SCHOLAR_API_KEY=
GITHUB_TOKEN=

# Optional OpenRouter app metadata (recommended in their docs)
OPENROUTER_HTTP_REFERER=
OPENROUTER_X_TITLE=arxiv-research-agent

# App
LOG_LEVEL=INFO
```

6. **Create `pyproject.toml`** with `uv` style. Pin major versions; allow minor updates. The `anthropic` SDK is intentionally **not** included because we route Claude through OpenRouter:

```toml
[project]
name = "arxiv-research-agent"
version = "0.1.0"
description = "An agent that helps deeply explore the LLM agents literature"
requires-python = ">=3.11"
license = "MIT"
dependencies = [
  # Orchestration
  "langgraph>=0.2.50,<0.3",
  "langgraph-checkpoint-postgres>=2.0,<3",
  "langchain-core>=0.3,<0.4",
  "langsmith>=0.1.140,<0.2",
  # LLM SDKs
  "openai>=1.50,<2",
  "cohere>=5.11,<6",
  # HTTP
  "httpx>=0.27,<1",
  # DB
  "asyncpg>=0.29,<1",
  "pgvector>=0.3,<1",
  "psycopg[binary,pool]>=3.2,<4",
  # Corpus
  "arxiv>=2.1,<3",
  "pypdf>=5.0,<6",
  "tiktoken>=0.8,<1",
  # Memory graph
  "networkx>=3.3,<4",
  # Config / runtime
  "pydantic>=2.9,<3",
  "pydantic-settings>=2.6,<3",
  "python-dotenv>=1.0,<2",
  "tenacity>=9.0,<10",
  # UI
  "chainlit>=1.3,<2",
  "fastapi>=0.115,<1",
  "uvicorn>=0.32,<1",
]

[dependency-groups]
dev = [
  "pytest>=8.3,<9",
  "pytest-asyncio>=0.24,<1",
  "pytest-mock>=3.14,<4",
  "ruff>=0.7,<1",
  "mypy>=1.13,<2",
]

[build-system]
requires = ["hatchling>=1.25,<2"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src"]

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]

[tool.ruff]
line-length = 100
target-version = "py311"
```

7. **Create `Makefile`** with common targets:

```makefile
.PHONY: install fmt lint test ingest eval ui ui-graph db-up db-down db-reset

install:
	uv sync

fmt:
	uv run ruff format src tests scripts

lint:
	uv run ruff check src tests scripts

test:
	uv run pytest -q

db-up:
	docker compose -f docker/docker-compose.yml up -d

db-down:
	docker compose -f docker/docker-compose.yml down

db-reset:
	docker compose -f docker/docker-compose.yml down -v
	docker compose -f docker/docker-compose.yml up -d
	sleep 3
	uv run python scripts/bootstrap_db.py

ingest:
	uv run python scripts/ingest_seed_corpus.py

eval:
	uv run python scripts/eval_run.py

ui:
	uv run chainlit run src/ui/app.py --host 127.0.0.1 --port 8000

ui-graph:
	uv run uvicorn src.ui.graph_view.server:app --host 127.0.0.1 --port 9000
```

8. **Create `docker/docker-compose.yml`** for Postgres with pgvector:

```yaml
services:
  postgres:
    image: pgvector/pgvector:pg16
    container_name: arxiv_agent_pg
    environment:
      POSTGRES_DB: ${POSTGRES_DB:-arxiv_agent}
      POSTGRES_USER: ${POSTGRES_USER:-arxiv_agent}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD:-arxiv_agent_dev}
    ports:
      - "5432:5432"
    volumes:
      - pgdata:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${POSTGRES_USER:-arxiv_agent}"]
      interval: 5s
      timeout: 3s
      retries: 10

volumes:
  pgdata:
```

9. **Create `LICENSE`** with the standard MIT license text. Copyright holder placeholder: `<owner>` (the human owner fills it in). Year: current year (2026).

10. **Create `README.md`** at the repo root, brief. Sections: project name, one-paragraph description (no Bridge5 mention), prerequisites (Python 3.11+, Docker, `uv`, OpenRouter/OpenAI/Cohere/DeepSeek API keys), quickstart (clone, copy `.env.example` to `.env` and fill, `make db-up`, `make install`, `make ingest`, `make ui`), directory layout (paste the tree from Section 4), and a "Status: in development" note.

11. **Create `docs/interview_talking_points.md`** with a header and an empty list. Each phase will append entries.

12. **Create `docs/STATUS.md`** with the table from Appendix E (all phases marked "not started").

13. **Initialize git, create the GitHub repo, set the remote, but do not push.**

```
git init
git branch -M main
gh repo create arxiv-research-agent --private --source=. --remote=origin
git remote -v
```

`--license=mit` is intentionally not passed because the `gh` CLI disallows `--source` together with `--license`. The local `LICENSE` file (created in task 9) is the only license artifact. The remote on GitHub is empty by design — no commits, no LICENSE — until the owner pushes.

If `gh repo create` complains the repo already exists, stop and ask the human owner.

14. **Move this execution plan and the pre-execution review artifacts** from the repo root into `docs/`:

```
git mv EXECUTION_PLAN.md docs/EXECUTION_PLAN.md
git mv REVIEW_FEEDBACK.md docs/review_feedback/REVIEW_FEEDBACK.md
git mv REVIEW_FEEDBACK_V2.md docs/review_feedback/REVIEW_FEEDBACK_V2.md
git mv REVIEW_FEEDBACK_V3.md docs/review_feedback/REVIEW_FEEDBACK_V3.md
```

The three `REVIEW_FEEDBACK*.md` files exist in the repo root because they were produced during plan iteration before Phase 0 began. They are preserved as project history under `docs/review_feedback/` and committed alongside the rest of Phase 0.

15. **Stage and commit** everything. **Do not push.** Use the commit message template in Appendix C.

### 5.3 Acceptance criteria

- [ ] Directory tree matches Section 4 exactly. Use `find . -type d -not -path './.git*'` to verify.
- [ ] Every directory listed in the tree contains either at least one source file or a `.gitkeep` file. Excluded: `data/`.
- [ ] `.env` does not exist at this point (only `.env.example` is committed).
- [ ] `.env` is in `.gitignore`.
- [ ] `git status` after the commit is clean.
- [ ] `git log --oneline` shows exactly one commit.
- [ ] `git remote -v` shows the `origin` pointing at `arxiv-research-agent` on GitHub.
- [ ] `gh repo view --json visibility -q .visibility` returns `PRIVATE`.
- [ ] `EXECUTION_PLAN.md` is at `docs/EXECUTION_PLAN.md`, not in the repo root.
- [ ] The three `REVIEW_FEEDBACK*.md` files are at `docs/review_feedback/` and **not** in the repo root.
- [ ] `docs/phase_reports/PHASE_0_REPORT.md` exists and follows the template.
- [ ] `docs/STATUS.md` exists with the initial table; the row for Phase 0 is marked "complete".
- [ ] `src/__init__.py` exists and is empty.
- [ ] `data/` directory is **not** tracked (no entries from `data/` in `git ls-files`).

### 5.4 Validation commands the planner will run during review

```
find . -type d -not -path './.git*' | sort
find . -type f -not -path './.git/*' | sort | head -100
cat .gitignore
cat .env.example
cat pyproject.toml
cat docker/docker-compose.yml
cat Makefile
cat README.md
cat LICENSE | head -3
cat docs/STATUS.md
git log --oneline
git status
git remote -v
gh repo view --json visibility,name,description -q '.'
test -f docs/EXECUTION_PLAN.md && echo "execution plan moved" || echo "MISSING"
test -f .env && echo "FAIL: .env should not exist" || echo "ok no .env"
git ls-files | grep -E "(\.env$|\.env\.local$)" && echo "FAIL: env file tracked" || echo "ok no tracked env"
git ls-files | grep -E "^data/" && echo "FAIL: data tracked" || echo "ok no data tracked"
```

### 5.5 Commit message

```
phase 0: project bootstrap

- Empty directory skeleton with .gitkeep placeholders (excluding data/)
- Dependency manifest (pyproject.toml) with langgraph-checkpoint-postgres + psycopg
- Postgres+pgvector docker-compose
- .env.example template, .gitignore, MIT LICENSE
- README skeleton, interview_talking_points.md initialized
- docs/STATUS.md with initial phase table
- Execution plan moved to docs/EXECUTION_PLAN.md
- GitHub repo created (private), remote set, not pushed
```

---

## 6. Phase 1 — Provider Adapter Layer

### 6.1 Goal

Build the LLM/embedding/rerank adapter layer. Business code in later phases will call `get_chat_client(model_name)`, `get_embedding_client()`, and `get_rerank_client()` and never touch a vendor SDK directly. This phase also verifies and pins the actual model IDs at the four providers.

### 6.2 Background

We use four providers:

- **OpenRouter** for chat models routed through it: Claude Haiku 4.5 and a current Flash-tier model (Google or other). Model IDs are OpenRouter-style (`provider/model`).
- **DeepSeek** native API for the current DeepSeek V4 flagship. Their endpoint is `https://api.deepseek.com/v1`.
- **OpenAI** direct for `text-embedding-3-small`.
- **Cohere** direct for the current rerank model.

### 6.3 Tasks

1. **Model and LangGraph verification (do this first).**
   - Open Codex's web search or fetch tool. Pull the current model lists:
     - OpenRouter `GET https://openrouter.ai/api/v1/models` (no auth required for the list).
     - DeepSeek docs: `https://api-docs.deepseek.com/`.
     - Cohere rerank docs: `https://docs.cohere.com/reference/rerank-2`.
     - OpenAI embedding docs: `https://platform.openai.com/docs/guides/embeddings`.
   - For each provider, choose:
     - **OpenRouter chat (main)**: `anthropic/claude-haiku-4.5` (this is the planner's anchor; verify it exists; if not, ask the owner).
     - **OpenRouter chat (cheap routing)**: pick the cheapest current Flash-tier model with reliable JSON mode. Candidates to check in order: `google/gemini-2.5-flash`, `google/gemini-flash-1.5`, then any current `mistral` Flash-tier. **Do not** use `google/gemini-2.0-flash-001` (retiring 2026-06-01).
     - **DeepSeek**: pick the current V4 flagship per their docs. Candidates: `deepseek-v4-pro`, `deepseek-v4-flash`, `deepseek-chat`. Pick the one their current docs identify as the latest production-quality model; document the mapping.
     - **Cohere rerank**: pick the latest stable rerank model. Candidate IDs in order of preference: `rerank-v4.0-pro`, `rerank-v4.0-fast`, `rerank-v3.5`. Reproducibility matters more than newness — choose one and pin.
     - **OpenAI embedding**: `text-embedding-3-small`, dimension 1536.
   - **LangGraph version and security check** (resolves N-C-2):
     - Check PyPI and GitHub advisories for the `langgraph` and `langgraph-checkpoint-postgres` packages. Specifically look for the historical msgpack checkpoint deserialization advisory affecting the 0.2.x line.
     - Decide between two paths:
       - **Path A — stay on `langgraph>=0.2.50,<0.3`**: only acceptable if the latest 0.2.x release contains the deserialization fix or a documented strict-mode opt-in. Document the mitigation: our checkpoint store is local Postgres only and never accepts checkpoints from untrusted sources.
       - **Path B — upgrade to `langgraph>=1.0,<2`**: acceptable if 1.x is GA and `langgraph-checkpoint-postgres` has a compatible release. Update `pyproject.toml` accordingly. Verify the `AsyncPostgresSaver` import path and `add_messages` reducer still apply (they do across versions, but confirm).
     - Pick one path. **Do not pick a path that has neither the fix nor a compatible 1.x line; ask the human owner instead.**
   - Write `docs/decisions/0001-model-and-langgraph-selection.md` recording each model choice, the LangGraph path chosen, the date checked, and the URLs of the supporting documentation and advisory.
   - Update `src/llm/registry.py` (created later in this phase) with the chosen IDs. If Path B was selected, also update `pyproject.toml`.

2. **`src/core/types.py`** — Define shared Pydantic models and code constants. The first line of the file must be `from __future__ import annotations` so forward references (e.g., `Message` referring to `ToolCall` defined later) resolve cleanly.

   Models:
   - `class ToolCall(BaseModel)`: `id: str`, `name: str`, `arguments: dict`.
   - `class TokenUsage(BaseModel)`: `prompt_tokens: int`, `completion_tokens: int`, `total_tokens: int`.
   - `class Message(BaseModel)`: `role: Literal["system","user","assistant","tool"]`, `content: str | None`, `tool_calls: list[ToolCall] | None`, `tool_call_id: str | None`, `name: str | None`.
   - `class ChatResponse(BaseModel)`: `content: str | None`, `tool_calls: list[ToolCall]`, `usage: TokenUsage | None`, `model: str`, `finish_reason: str`.
   - `class Citation(BaseModel)`: `paper_id: str`, **`chunk_id: int`** (required, not Optional — produced by the synthesizer in the form `[paper_id#chunk_id]` and parsed back), `quote: str | None`, `claim_span: tuple[int,int] | None`.
   - `class VerificationReport(BaseModel)`: `verdicts: list[CitationVerdict]`, `passed: bool`.
   - `class CitationVerdict(BaseModel)`: `citation: Citation`, `supports: bool`, `rationale: str`.

   Code constants (per Section 2.4) are also declared here as module-level uppercase constants (`EMBEDDING_DIM = 1536`, etc.). They are not environment-overridable.

3. **`src/llm/errors.py`** — Define a hierarchy:
   - `LLMError` (base)
   - `LLMAuthError` (401/403)
   - `LLMRateLimitError` (429)
   - `LLMTimeoutError`
   - `LLMContextLengthError` (token cap exceeded)
   - `LLMServerError` (5xx)
   - `LLMInvalidRequestError` (400)
   - `LLMProviderError` (provider-specific generic)
   - Each error class accepts a `*, provider: str, status: int | None = None` constructor; `__str__` masks any header-like fields if accidentally passed.

4. **`src/llm/retry.py`** — A tenacity-based decorator `@with_retry()`:
   - Retry on `LLMRateLimitError`, `LLMTimeoutError`, `LLMServerError`.
   - Do not retry on `LLMAuthError`, `LLMContextLengthError`, `LLMInvalidRequestError`.
   - Exponential backoff with jitter, max attempts from `RETRY_MAX_ATTEMPTS` config.
   - Log each retry at `WARNING`, masking sensitive fields. Use `tenacity.before_sleep_log` with a custom callable that scrubs the error message.

5. **`src/llm/registry.py`** — A `ModelRegistry` mapping logical names to provider routes. Populate from the choices in Task 1:

```python
@dataclass(frozen=True)
class ProviderRoute:
    provider: Literal["openrouter", "deepseek_native", "openai_native", "cohere_native"]
    vendor_model: str
    capability: Literal["chat", "embedding", "rerank"]

REGISTRY: dict[str, ProviderRoute] = {
    "haiku-4.5":      ProviderRoute("openrouter",      "<verified-id>", "chat"),
    "deepseek-v4":    ProviderRoute("deepseek_native", "<verified-id>", "chat"),
    "flash-cheap":    ProviderRoute("openrouter",      "<verified-id>", "chat"),
    "embed-small":    ProviderRoute("openai_native",   "text-embedding-3-small", "embedding"),
    "rerank":         ProviderRoute("cohere_native",   "<verified-id>", "rerank"),
}
```

   Use logical aliases (`flash-cheap`, `deepseek-v4`) so the rest of the code does not encode vendor model IDs. The Phase 1 owner-handoff allows live verification but must not commit secrets.

6. **`src/llm/providers/openrouter.py`** — `OpenRouterChatClient`:
   - Built on `httpx.AsyncClient` (not the official OpenAI SDK).
   - `async def chat(messages: list[Message], *, model: str, tools: list | None = None, temperature: float = 0.0, max_tokens: int | None = None, response_format: dict | None = None) -> ChatResponse`
   - `async def aclose()` — closes the underlying httpx client.
   - Set `Authorization: Bearer ${OPENROUTER_API_KEY}` plus optional `HTTP-Referer` and `X-Title` headers from settings.
   - Map HTTP status codes to our error classes via a single `_raise_for_status` helper.

7. **`src/llm/providers/deepseek.py`** — `DeepSeekChatClient`. Same interface and `aclose()`. Endpoint: `https://api.deepseek.com/v1/chat/completions`.

8. **`src/llm/providers/openai_embed.py`** — `OpenAIEmbeddingClient`:
   - `async def embed(texts: list[str], *, model: str = "text-embedding-3-small") -> list[list[float]]`.
   - **Validate every returned vector length equals `EMBEDDING_DIM` (1536).** Raise `LLMProviderError` otherwise.
   - Do **not** pass a custom `dimensions` parameter.
   - Batch up to 256 inputs per request; auto-chunk longer batches.
   - `aclose()`.

9. **`src/llm/providers/cohere_rerank.py`** — `CohereRerankClient`:
   - Use Cohere `AsyncClientV2`.
   - `async def rerank(query: str, documents: list[str], *, model: str, top_n: int) -> list[RerankedDoc]`.
   - `RerankedDoc(BaseModel)`: `index: int`, `score: float` (Cohere SDK calls this `relevance_score`; map explicitly to local `score`).
   - Filter empty strings before sending; raise `LLMInvalidRequestError` if all are empty.
   - `aclose()`.

10. **`src/llm/client.py`** — Public API:
    - `def get_chat_client(model_name: str) -> ChatClient` (returns the right provider instance, looked up via registry).
    - `def get_embedding_client(model_name: str = "embed-small") -> EmbeddingClient`.
    - `def get_rerank_client(model_name: str = "rerank") -> RerankClient`.
    - Each is a process singleton constructed lazily.
    - `async def close_llm_clients()` — calls `aclose()` on every cached client. Must be called by the UI lifespan and by integration tests in a finalizer.

11. **`src/core/config.py`** — `pydantic-settings` `Settings` class. Fields:
    - All provider API keys (each `Optional[SecretStr]`).
    - LangSmith fields.
    - Postgres fields.
    - Optional OpenRouter app metadata.
    - Code constants from Section 2.4 are **not** here; they live as module-level constants in `src/core/types.py` (single canonical home, no separate `constants.py`).
    - Loaded from `.env` automatically.
    - When a provider client is constructed but its key is missing, raise a clear `LLMAuthError` with a message naming the env var.

12. **`src/core/logging.py`** — Setup `logging.basicConfig` based on `LOG_LEVEL`. Implement a `SecretMaskingFilter` that scrubs any value matching common API key patterns (`sk-`, `sk-ant-`, length ≥ 32 alphanumeric strings preceded by `key`/`token`/`Authorization`/`Bearer`). Apply the filter to the root logger.

13. **Tests**:
    - `tests/unit/test_retry.py` — Mock raises `LLMRateLimitError` twice then succeeds; verify retry count. Verify no retry on `LLMAuthError`. Verify the retry log line does not contain a synthetic API key string fed via the mock.
    - `tests/unit/test_registry.py` — Round-trip lookups; unknown model raises `KeyError`.
    - `tests/unit/test_logging_filter.py` — Verify the masking filter scrubs `Authorization: Bearer abc123def456...` patterns.
    - `tests/unit/test_status_mapping.py` — For each provider client, mock httpx to return status codes 400/401/403/429/500/503 and verify the right error class is raised.
    - `tests/unit/test_embedding_dim.py` — Mock OpenAI to return a 1024-dim vector; verify `LLMProviderError` is raised.
    - `tests/unit/test_close_clients.py` — Build and close clients; verify httpx `is_closed` is true after `aclose()`.
    - `tests/integration/test_provider_smoke.py` — A skipped-by-default test (skip if `INTEGRATION_TESTS=1` env not set) that does a one-token call against each provider to verify auth and basic round trip.

### 6.4 Acceptance criteria

- [ ] `from src.llm.client import get_chat_client; client = get_chat_client("haiku-4.5")` works in a Python REPL after `make install` and a populated `.env`.
- [ ] `uv run pytest tests/unit -q` passes with all unit tests green.
- [ ] No code path imports `openai` or `cohere` outside `src/llm/providers/` (a comment in the grep validation acknowledges that test files may reference SDK types via mocks).
- [ ] Logging masks API keys: a manual check by running a provider call and inspecting logs shows no full key visible.
- [ ] `src/core/config.py` raises a clear error if a required key is absent when its provider is invoked.
- [ ] `docs/decisions/0001-model-and-langgraph-selection.md` exists with each chosen model ID, the LangGraph version path (Path A or Path B), and source URLs / advisory references.
- [ ] Integration smoke test passes when `INTEGRATION_TESTS=1` and all keys are present (this is an optional check the human owner runs; it requires Handoff H-1 confirmation in `docs/STATUS.md`).

### 6.5 Validation commands

```
uv run pytest tests/unit -q
# Note: the grep below excludes tests/ because mocks reference SDK types intentionally.
# Append `|| echo "ok"` so a clean run (no leaks) does not exit non-zero from grep.
grep -RIE "from openai|from cohere|import openai|import cohere" src --exclude-dir=llm || echo "ok"
uv run python -c "from src.llm.client import get_chat_client, get_embedding_client, get_rerank_client; print('imports ok')"
test -f docs/decisions/0001-model-and-langgraph-selection.md && echo "model+langgraph ADR present"
```

### 6.6 Commit message

```
phase 1: provider adapter layer

- Unified ChatClient/EmbeddingClient/RerankClient interfaces
- OpenRouter, DeepSeek native, OpenAI embed, Cohere rerank providers
- Tenacity-based retry with provider-aware error normalization
- Pydantic-settings config and secret-masking logging filter
- Shared Pydantic data contracts (Message, ToolCall, ChatResponse, Citation, VerificationReport)
- Async client lifecycle with close_llm_clients()
- Embedding dimension validation
- ADR 0001 records verified model selections
- Unit tests for retry, registry, masking filter, status mapping, dim validation, lifecycle
- Optional integration smoke test (skipped by default)
```

### 6.7 Interview talking points to add this phase

Append to `docs/interview_talking_points.md`:
- Provider abstraction: business code never touches vendor SDKs; one swap point for adding/removing providers.
- Retry policy: classify errors as retryable vs deterministic; exponential backoff with jitter; bounded attempts.
- Secret masking: filter at the logging layer means even unintentional `logger.debug(headers)` cannot leak.
- Embedding dimension validation: catches silent provider drift before it pollutes the index.
- Client lifecycle: `aclose()` + lifespan hook ensures no leaked TCP connections.

---

## 7. Phase 2 — Database Schema and Connection

### 7.1 Goal

Bring up Postgres+pgvector, create the schema for everything we will store (papers, chunks with vectors, citations, concept graph, sessions, messages, session summaries, triggers), and provide both an asyncpg pool (for app code) and a psycopg pool (for LangGraph's checkpointer). Wire LangGraph's `AsyncPostgresSaver` to the same database via psycopg.

### 7.2 Tasks

1. **`src/core/db.py`** — connection pool factory:
   - `async def get_asyncpg_pool() -> asyncpg.Pool` (singleton, async-initialized). On first connect, runs `CREATE EXTENSION IF NOT EXISTS vector;` and `CREATE EXTENSION IF NOT EXISTS pg_trgm;`. Sets up the asyncpg vector codec via `pgvector.asyncpg.register_vector` in the pool's `init` callback.
   - `async def get_psycopg_pool() -> psycopg_pool.AsyncConnectionPool` (singleton). Used **only** by the LangGraph checkpointer. Construction must include the kwargs LangGraph's `AsyncPostgresSaver` documents as required for manual pools (resolves N-C-1):

     ```python
     from psycopg.rows import dict_row
     from psycopg_pool import AsyncConnectionPool

     pool = AsyncConnectionPool(
         conninfo,
         min_size=1,
         max_size=10,
         open=False,
         kwargs={"autocommit": True, "row_factory": dict_row},
     )
     await pool.open()
     ```

     Without `autocommit=True` and `row_factory=dict_row`, `AsyncPostgresSaver.setup()` may not persist tables and reads can fail with tuple-row indexing errors.
   - Context managers: `async with db.acquire_app() as conn: ...` (asyncpg) and `async with db.acquire_checkpoint() as conn: ...` (psycopg).
   - `async def close_pools()` for clean shutdown.

2. **Schema files in `src/core/sql/`**. Each file is idempotent (`CREATE TABLE IF NOT EXISTS`, `CREATE INDEX IF NOT EXISTS`). The `bootstrap_db.py` script runs them in lexical order.

   - `001_papers.sql`:

```sql
CREATE TABLE IF NOT EXISTS papers (
  paper_id      TEXT PRIMARY KEY,        -- e.g. arxiv:2210.03629
  source        TEXT NOT NULL,           -- 'arxiv' | 'semantic_scholar' | 'github'
  title         TEXT NOT NULL,
  authors       JSONB NOT NULL DEFAULT '[]',
  abstract      TEXT,
  published_at  DATE,
  url           TEXT,
  pdf_path      TEXT,                    -- local cache path
  ingestion_status TEXT NOT NULL DEFAULT 'complete', -- 'complete' | 'in_progress' | 'failed'
  raw_metadata  JSONB NOT NULL DEFAULT '{}',
  ingested_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS papers_source_idx ON papers(source);
CREATE INDEX IF NOT EXISTS papers_published_idx ON papers(published_at DESC);
```

   - `002_chunks.sql`:

```sql
CREATE TABLE IF NOT EXISTS chunks (
  chunk_id     BIGSERIAL PRIMARY KEY,
  paper_id     TEXT NOT NULL REFERENCES papers(paper_id) ON DELETE CASCADE,
  section      TEXT,
  ord          INT NOT NULL,
  text         TEXT NOT NULL,
  token_count  INT NOT NULL,
  embedding    vector(1536) NOT NULL,
  tsv          tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED,
  created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  UNIQUE (paper_id, ord)
);
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw ON chunks USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);
CREATE INDEX IF NOT EXISTS chunks_tsv_idx ON chunks USING gin (tsv);
CREATE INDEX IF NOT EXISTS chunks_paper_idx ON chunks(paper_id);
```

   - `003_citations.sql`:

```sql
CREATE TABLE IF NOT EXISTS citations (
  citing_paper_id TEXT NOT NULL REFERENCES papers(paper_id) ON DELETE CASCADE,
  cited_paper_id  TEXT NOT NULL,         -- not FK; cited paper may not be in our DB yet
  cited_title     TEXT,
  evidence_chunk  BIGINT REFERENCES chunks(chunk_id) ON DELETE SET NULL,
  PRIMARY KEY (citing_paper_id, cited_paper_id)
);
CREATE INDEX IF NOT EXISTS citations_cited_idx ON citations(cited_paper_id);
```

   - `004_concepts.sql`:

```sql
CREATE TABLE IF NOT EXISTS concepts (
  concept_id   BIGSERIAL PRIMARY KEY,
  canonical    TEXT NOT NULL UNIQUE,
  display      TEXT NOT NULL,
  definition   TEXT,
  embedding    vector(1536),
  evidence_chunks JSONB NOT NULL DEFAULT '[]',
  created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS concept_aliases (
  alias        TEXT PRIMARY KEY,
  concept_id   BIGINT NOT NULL REFERENCES concepts(concept_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS concept_aliases_concept_idx ON concept_aliases(concept_id);

CREATE TABLE IF NOT EXISTS concept_relations (
  src_id       BIGINT NOT NULL REFERENCES concepts(concept_id) ON DELETE CASCADE,
  dst_id       BIGINT NOT NULL REFERENCES concepts(concept_id) ON DELETE CASCADE,
  rel_type     TEXT NOT NULL,            -- 'extends' | 'compares' | 'uses' | 'cites' | 'is-a'
  evidence_chunks JSONB NOT NULL DEFAULT '[]',
  weight       REAL NOT NULL DEFAULT 1.0,
  created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  PRIMARY KEY (src_id, dst_id, rel_type)
);
CREATE INDEX IF NOT EXISTS concept_relations_dst_idx ON concept_relations(dst_id);
```

   - `005_sessions.sql`:

```sql
CREATE TABLE IF NOT EXISTS sessions (
  session_id   TEXT PRIMARY KEY,
  title        TEXT,
  created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  last_active  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS session_messages (
  message_id   BIGSERIAL PRIMARY KEY,
  session_id   TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
  role         TEXT NOT NULL,
  content      TEXT NOT NULL,
  metadata     JSONB NOT NULL DEFAULT '{}',
  created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS session_messages_session_idx ON session_messages(session_id, created_at);
```

   - `006_session_summaries.sql`:

```sql
CREATE TABLE IF NOT EXISTS session_summaries (
  session_id          TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
  up_to_message_id    BIGINT NOT NULL,
  summary             TEXT NOT NULL,
  created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  PRIMARY KEY (session_id, up_to_message_id)
);
```

   - `007_triggers.sql`:

```sql
CREATE OR REPLACE FUNCTION set_updated_at_now()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = NOW();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS concepts_set_updated_at ON concepts;
CREATE TRIGGER concepts_set_updated_at
  BEFORE UPDATE ON concepts
  FOR EACH ROW EXECUTE FUNCTION set_updated_at_now();
```

3. **`scripts/bootstrap_db.py`**:
   - Run all SQL files in `src/core/sql/` in lexical order via the asyncpg pool.
   - Then call `AsyncPostgresSaver.setup()` using the psycopg pool to create LangGraph's checkpoint tables.
   - Idempotent: re-running should be a no-op.

4. **Tests**:
   - `tests/integration/test_db_bootstrap.py` (skipped without `INTEGRATION_TESTS=1`):
     - Bootstrap on the running Docker Postgres.
     - Verify each table exists.
     - Verify the vector and pg_trgm extensions are installed.
     - Verify HNSW and GIN indexes exist.
     - Verify LangGraph checkpoint tables (`checkpoints`, `checkpoint_blobs`, `checkpoint_writes`, `checkpoint_migrations`) exist.
     - Verify the `concepts.updated_at` trigger fires (insert, update, observe `updated_at` advanced).
     - Verify a vector can round-trip through asyncpg (insert a 1536-dim vector, select it back, compare).
   - `tests/integration/test_checkpoint_roundtrip.py` (skipped without `INTEGRATION_TESTS=1`) — **resolves N-C-1's residual concern**: build an `AsyncPostgresSaver` from the psycopg pool, call `aput` with a fabricated checkpoint payload for a synthetic `thread_id`, then `aget_tuple` and assert the round-tripped checkpoint payload matches what was written. This is a stronger guarantee than verifying tables exist.

### 7.3 Acceptance criteria

- [ ] `make db-up` starts Postgres healthy.
- [ ] `make db-reset` recreates the volume and runs bootstrap successfully.
- [ ] `psql` shows all expected tables and indexes.
- [ ] LangGraph's `checkpoints`, `checkpoint_blobs`, `checkpoint_writes`, `checkpoint_migrations` tables exist after bootstrap.
- [ ] Integration test `test_db_bootstrap.py` passes.
- [ ] Re-running `make db-reset` produces no errors.

### 7.4 Validation commands

> **Requires Owner Handoff H-2 and H-3 confirmation in `docs/STATUS.md` before running these commands** (Docker daemon up, `.env` populated for any provider calls).

```
make db-up
sleep 3
uv run python scripts/bootstrap_db.py
docker exec arxiv_agent_pg psql -U arxiv_agent -d arxiv_agent -c '\dt'
docker exec arxiv_agent_pg psql -U arxiv_agent -d arxiv_agent -c "SELECT extname FROM pg_extension;"
docker exec arxiv_agent_pg psql -U arxiv_agent -d arxiv_agent -c "\d chunks"
docker exec arxiv_agent_pg psql -U arxiv_agent -d arxiv_agent -c "\d concepts"
INTEGRATION_TESTS=1 uv run pytest tests/integration/test_db_bootstrap.py tests/integration/test_checkpoint_roundtrip.py -q
```

### 7.5 Commit message

```
phase 2: database schema and connection

- asyncpg pool with vector codec registration on connection init
- psycopg pool dedicated to LangGraph checkpointing
- SQL schema (numbered 001-007): papers (with ingestion_status), chunks (vector+tsvector+HNSW+GIN), citations, concepts/aliases/relations, sessions/messages, session_summaries, updated_at trigger
- bootstrap_db.py runs SQL files and AsyncPostgresSaver.setup()
- Integration test verifies tables, extensions, indexes, trigger, vector round-trip
```

---

## 8. Phase 3 — Corpus Clients and Ingestion

### 8.1 Goal

Fetch papers from arXiv and Semantic Scholar, parse PDFs, chunk them, embed, and write to the database transactionally. Provide both a batch seed-corpus path and a lazy on-demand ingestion path used by retrieval in later phases.

### 8.2 Tasks

1. **`src/corpus/arxiv_client.py`**:
   - `class ArxivClient` wrapping `arxiv.Client(page_size=10, delay_seconds=3)` (the package handles rate limiting internally).
   - `async def fetch_metadata(arxiv_id: str) -> PaperMetadata` — uses `asyncio.to_thread` to call the sync arxiv package.
   - `async def download_pdf(arxiv_id: str, dest_dir: Path) -> Path` — uses `asyncio.to_thread`.
   - One global `arxiv.Client` instance per process to honor the 3-second pacing.

2. **`src/corpus/semantic_scholar.py`**:
   - `async def fetch_paper(s2_id_or_arxiv: str) -> S2Paper` — direct httpx call to `https://api.semanticscholar.org/graph/v1/paper/<id>` with `fields=externalIds,title,abstract,authors,year,referenceCount,citationCount`.
   - `async def fetch_references(paper_id: str, limit: int = 50) -> list[S2Reference]` — `GET /paper/<id>/references?limit=<limit>&fields=externalIds,title,abstract`.
   - `async def fetch_citations(paper_id: str, limit: int = 50) -> list[S2Citation]` — analogous.
   - Add `x-api-key` header from settings if `SEMANTIC_SCHOLAR_API_KEY` is set.
   - Implement an async semaphore-based rate limiter: 1 req/sec without key, 10 req/sec with key. On 429, exponentially back off and retry up to 5 times (delegated to tenacity).
   - `aclose()` on the httpx client.

3. **`src/corpus/github_client.py`** (minimal):
   - `async def fetch_readme(repo_full_name: str) -> str | None`. Used optionally in later phases.

4. **`src/corpus/pdf_loader.py`**:
   - `def load_pdf(path: Path) -> list[Section]` — `pypdf.PdfReader`, page-by-page text extraction. Detect section headers via heuristics (lines that are short, ALL CAPS, or match common section names). Return a list of `(section_name, text)` tuples in document order.

5. **`src/corpus/chunking.py`**:
   - `def chunk_sections(sections: list[Section], target_tokens: int = CHUNK_TOKEN_SIZE, overlap: int = CHUNK_TOKEN_OVERLAP) -> list[Chunk]` — uses `tiktoken.get_encoding("cl100k_base")`. Section-aware: prefer chunk boundaries at sentence ends; never cross sections.

6. **`src/corpus/ingestion.py`**:
   - `async def ingest_paper(arxiv_id: str, *, allow_redownload: bool = False) -> str` returns `paper_id`.
   - **Collect-then-short-transaction behavior** (resolves M-6 / A-5 / N-L-6):

     **Phase A — collect external artifacts (no DB transaction held):**
     1. Idempotency check: open a short read-only connection, query `papers` for the `paper_id`. If status is `complete`, return immediately. If `in_progress` or absent, proceed. If `failed`, plan a clean retry.
     2. `arxiv_client.fetch_metadata(arxiv_id)` — sync wrapped in `asyncio.to_thread`.
     3. `arxiv_client.download_pdf(...)`.
     4. `pdf_loader.load_pdf(path)` then `chunking.chunk_sections(...)` — pure Python, no I/O.
     5. `embedding_client.embed([chunk.text for chunk in chunks])` — batched up to 256.
     6. `semantic_scholar.fetch_references(...)` — best effort; on failure log and proceed with empty references.

     **Phase B — short DB transaction:**
     1. Open one asyncpg transaction.
     2. If retrying a previously failed paper, delete its rows from `chunks` and `citations` first.
     3. Upsert `papers` row with `ingestion_status='in_progress'`.
     4. Bulk insert into `chunks` (use `executemany` or COPY; the embedding column already has its codec registered).
     5. Bulk insert into `citations`.
     6. Update `papers.ingestion_status='complete'`.
     7. Commit.

     On any error during Phase A: no transaction was held during the long external I/O; before raising, open a short separate transaction to upsert `papers` with `ingestion_status='failed'` (so retries can identify the paper as failed and the §8.2 task 8 partial-failure test sees the expected state). On any error during Phase B: the transaction rolls back; then in a separate new transaction upsert `papers.ingestion_status='failed'`. In both cases, no `chunks` or `citations` rows are committed.

     This refactor avoids holding a DB connection across the long external I/O of Phase A while preserving atomicity of the actual writes.
   - Idempotency rules: `complete` → return early; `in_progress` (orphan from a crashed run) → treat as `failed`, retry; `failed` → clean and retry.

7. **`scripts/ingest_seed_corpus.py`**:
   - **Use a Python module for the seed list, not YAML** (resolves L-6). The list lives at `src/eval/datasets/seed_papers.py` (Codex creates this file in Phase 3) as `SEED_PAPER_IDS: list[str] = [...]` populated from Appendix B of this document.
   - Concurrency cap of 4 simultaneous papers; arXiv pacing is enforced by the global client.
   - Print a summary at the end: papers ingested, papers skipped, papers failed, chunks inserted, citations inserted.

8. **Tests**:
   - `tests/unit/test_chunking.py` — synthetic sections, verify chunk sizes and section boundaries.
   - `tests/unit/test_pdf_loader.py` — Use a tiny fixture PDF generated at test time (or include a small fixture) to verify section extraction.
   - `tests/integration/test_ingest_one.py` (skipped by default) — Ingest a single small arXiv paper end-to-end. Verify `papers`, `chunks`, `citations` rows created.
   - `tests/integration/test_ingest_partial_failure.py` (skipped by default) — Patch `OpenAIEmbeddingClient.embed` to raise on the second batch. Run `ingest_paper`. Verify the transaction rolled back: `papers.ingestion_status` is `failed`, no `chunks` rows exist for that paper.

### 8.3 Acceptance criteria

- [ ] `make ingest` ingests the full seed list without crashing. Rate limits handled.
- [ ] Database afterward contains `>= 25` rows in `papers` (some IDs may 404; tolerate up to 5 missing) and `>= 800` chunks.
- [ ] **At least 80% of ingested papers have at least one citation row** (resolves M-7; SS occasionally lacks reference data for some IDs).
- [ ] `tests/unit/test_chunking.py` passes.
- [ ] Re-running `make ingest` is a no-op for already-complete papers and retries failed papers.
- [ ] Partial-failure integration test passes.

### 8.4 Validation commands

> **Requires Owner Handoff H-3 confirmation in `docs/STATUS.md` before running these commands** (provider keys populated; `make ingest` will spend on embedding API calls).

```
make db-reset
make ingest
docker exec arxiv_agent_pg psql -U arxiv_agent -d arxiv_agent -c "SELECT count(*) FROM papers WHERE ingestion_status='complete';"
docker exec arxiv_agent_pg psql -U arxiv_agent -d arxiv_agent -c "SELECT count(*) FROM chunks;"
docker exec arxiv_agent_pg psql -U arxiv_agent -d arxiv_agent -c "SELECT count(*) FROM citations;"
docker exec arxiv_agent_pg psql -U arxiv_agent -d arxiv_agent -c "SELECT 100.0 * count(DISTINCT citing_paper_id) / NULLIF((SELECT count(*) FROM papers WHERE ingestion_status='complete'),0) AS pct FROM citations;"
uv run pytest tests/unit/test_chunking.py tests/unit/test_pdf_loader.py -q
INTEGRATION_TESTS=1 uv run pytest tests/integration/test_ingest_partial_failure.py -q
```

### 8.5 Commit message

```
phase 3: corpus clients and transactional ingestion

- arXiv client (sync arxiv package via asyncio.to_thread), Semantic Scholar (async httpx), GitHub minimal client
- PDF loader with section heuristics
- Section-aware chunking with tiktoken cl100k_base
- Transactional ingestion (atomic per paper); ingestion_status tracking
- Seed corpus script using a Python module, concurrency-capped
- Unit tests for chunking and PDF; integration test for partial-failure rollback
```

---

## 9. Phase 4 — Vector Index and Hybrid Retrieval

### 9.1 Goal

Implement the retrieval layer over `chunks`: cosine vector search, lexical full-text search (via `ts_rank_cd`), and a hybrid scoring SQL using **Reciprocal Rank Fusion** combining the two. No agentic logic yet.

### 9.2 Tasks

1. **`src/retrieval/index/pgvector_store.py`**:
   - `async def search_vector(query_embedding: list[float], top_k: int, *, filter_paper_ids: list[str] | None = None) -> list[Hit]`
   - `async def search_lexical(query_text: str, top_k: int, *, filter_paper_ids: list[str] | None = None) -> list[Hit]`
   - `Hit = (chunk_id, paper_id, section, text, score)`

2. **`src/retrieval/index/hybrid_search.py`** — RRF-based fusion (resolves C-2 and A-2):
   - `async def hybrid_search(query_text: str, query_embedding: list[float], *, top_k_vector: int = 30, top_k_lexical: int = 30, rrf_k: int = 60, top_k_final: int = 30, filter_paper_ids: list[str] | None = None) -> list[Hit]`
   - Single SQL using CTEs:

```sql
WITH vec AS (
  SELECT chunk_id,
         ROW_NUMBER() OVER (ORDER BY embedding <=> $1::vector) AS rnk
  FROM chunks
  WHERE ($2::text[] IS NULL OR paper_id = ANY($2))
  ORDER BY embedding <=> $1::vector
  LIMIT $3
),
lex AS (
  SELECT chunk_id,
         ROW_NUMBER() OVER (ORDER BY ts_rank_cd(tsv, plainto_tsquery('english', $4)) DESC) AS rnk
  FROM chunks
  WHERE tsv @@ plainto_tsquery('english', $4)
    AND ($2::text[] IS NULL OR paper_id = ANY($2))
  ORDER BY ts_rank_cd(tsv, plainto_tsquery('english', $4)) DESC
  LIMIT $5
),
fused AS (
  SELECT chunk_id, SUM(1.0 / ($6::float + rnk)) AS score
  FROM (
    SELECT chunk_id, rnk FROM vec
    UNION ALL
    SELECT chunk_id, rnk FROM lex
  ) AS combined
  GROUP BY chunk_id
)
SELECT c.chunk_id, c.paper_id, c.section, c.text, f.score
FROM fused f
JOIN chunks c ON c.chunk_id = f.chunk_id
ORDER BY f.score DESC
LIMIT $7;
```

   - Parameters: `$1` query embedding, `$2` filter_paper_ids array (NULL when no filter), `$3` top_k_vector, `$4` query text, `$5` top_k_lexical, `$6` rrf_k, `$7` top_k_final.
   - Document the fusion choice in `docs/decisions/0002-rrf-fusion.md`: why RRF (rank-based, score-unit-free), the choice of `k=60` (TREC-OAR convention), and the implication that we still call the lexical leg "lexical full-text rank" rather than "BM25" since `ts_rank_cd` is not BM25.

3. **Tests**:
   - `tests/unit/test_hybrid_sql.py` — verify the SQL string is constructed with correct parameter placeholders, `filter_paper_ids` is wired into both CTEs, and the fused CTE deduplicates by `chunk_id`.
   - `tests/integration/test_retrieval_basic.py` (skipped by default) — after seed ingest, query "ReAct reasoning and acting interleaved" and assert ReAct paper chunks **should typically appear** in the top 5 (failure surfaces as a soft warning the planner reviews). Hard assertion: top-5 contains at least one chunk from the ReAct paper.
   - `tests/integration/test_retrieval_filter_overlap.py` (skipped by default) — verify (a) results contain no duplicate `chunk_id`s, and (b) `filter_paper_ids` constrains both legs (vector and lexical).

### 9.3 Acceptance criteria

- [ ] Hybrid search returns results with no duplicate `chunk_id`s.
- [ ] `filter_paper_ids` is honored on both legs.
- [ ] Top-5 results for "ReAct reasoning and acting interleaved" should typically include the ReAct paper; hard assertion: at least one of top-5 belongs to the ReAct paper.
- [ ] Latency: a single hybrid query against the seed corpus completes in under 200 ms locally on the Docker DB.
- [ ] ADR `0002-rrf-fusion.md` exists.

### 9.4 Validation commands

> **Requires Owner Handoff H-3 confirmation in `docs/STATUS.md` before running integration commands.**

```
INTEGRATION_TESTS=1 uv run pytest tests/integration/test_retrieval_basic.py tests/integration/test_retrieval_filter_overlap.py -q -s
uv run pytest tests/unit/test_hybrid_sql.py -q
test -f docs/decisions/0002-rrf-fusion.md && echo "RRF ADR present"
```

### 9.5 Commit message

```
phase 4: vector index and RRF hybrid retrieval

- pgvector_store with vector and lexical (ts_rank_cd) search legs
- Hybrid scoring via Reciprocal Rank Fusion (k=60), unit-free
- filter_paper_ids honored on both legs; deduplicated by chunk_id
- ADR 0002 documents fusion choice and naming (lexical, not BM25)
- Integration tests for retrieval basics, dedup, and filter
```

---

## 10. Phase 5 — Agentic RAG Components (Query / Rerank / Self-RAG)

### 10.1 Goal

Build the per-sub-question retrieval-side agentic components as standalone modules with clear interfaces and tests. Multi-hop is its own phase (Phase 6); graph wiring is in Phase 9.

### 10.2 Tasks

1. **`src/retrieval/query/decomposer.py`**:
   - `async def decompose(question: str, *, max_subq: int = MAX_SUB_QUESTIONS) -> Decomposition`
   - `Decomposition` carries a list of `SubQuestion(text, depends_on: list[int])`. Dependencies form a DAG.
   - Use `haiku-4.5` with a prompt requiring JSON output validated by `model_validate_json`.
   - For trivial questions, return a single sub-question equal to the input.
   - On JSON parse failure: one auto-retry asking the model to fix the JSON; then raise.

2. **`src/retrieval/query/rewriter.py`**:
   - `async def rewrite_for_retrieval(subq: str) -> str` — tightens the query for retrieval.
   - `async def hyde(subq: str) -> str` — generates a hypothetical answer for HyDE-style retrieval. Use `flash-cheap`.
   - A heuristic decides which to use: skip HyDE if the sub-question is shorter than 8 tokens or matches a keyword pattern.

3. **`src/retrieval/query/router.py`**:
   - `async def route(subq: str) -> RouteDecision`
   - `RouteDecision = {"sources": ["local_pgvector"], "use_concept_graph": bool, "may_use_semantic_scholar_live": bool}`
   - The router returns **policy flags only** (resolves A-3). The retrieve node decides downstream whether to actually trigger live Semantic Scholar based on local hit count (e.g., fewer than 3 high-confidence chunks after retrieval+rerank).
   - Set `use_concept_graph = True` if any concept alias from the database matches a substring of the sub-question (do a quick query against `concept_aliases`).

4. **`src/retrieval/postprocess/reranker.py`**:
   - `async def rerank(query: str, hits: list[Hit], top_n: int = RERANK_TOP_K) -> list[Hit]`
   - Calls Cohere rerank with `documents=[hit.text for hit in hits]`. Return hits in new order with `score` overwritten by the Cohere `relevance_score` (mapped explicitly).

5. **`src/retrieval/postprocess/deduplicator.py`**:
   - `def dedupe(hits: list[Hit]) -> list[Hit]` — removes duplicate `chunk_id` and near-duplicate text via a fast 5-shingle Jaccard threshold (e.g., 0.85).

6. **`src/retrieval/self_rag/relevance_judge.py`**:
   - `async def judge_relevance(subq: str, hit: Hit) -> RelevanceVerdict`
   - `RelevanceVerdict = {"relevant": bool, "rationale": str}`. Use `flash-cheap` for batched judgement.
   - `async def judge_batch(subq: str, hits: list[Hit]) -> list[RelevanceVerdict]` — concurrency-bounded with `asyncio.Semaphore(8)`.

7. **`src/retrieval/self_rag/sufficiency_check.py`**:
   - `async def is_sufficient(question: str, evidence: list[Hit]) -> SufficiencyVerdict`
   - `SufficiencyVerdict = {"sufficient": bool, "missing_aspects": list[str]}`. Use `haiku-4.5`.

8. **`src/retrieval/self_rag/verifier.py`**:
   - `async def verify_citations(answer: str, citations: list[Citation], evidence_lookup: dict[int, str]) -> VerificationReport`
   - For each citation, ask `haiku-4.5` whether the cited chunk supports the specific claim_span. Return per-citation verdicts.

9. **Tests**:
   - `tests/unit/test_decomposer_parsing.py` — Mock LLM client; verify Pydantic parsing handles malformed JSON via the one-retry path.
   - `tests/unit/test_dedup.py` — Near-duplicate texts collapsed to one.
   - `tests/unit/test_router.py` — Concept alias substring detection sets `use_concept_graph` true; missing alias sets it false.
   - `tests/unit/test_verifier_parser.py` — Synthetic LLM verdict JSON parsed into `VerificationReport`.

### 10.3 Acceptance criteria

- [ ] Each module has a clear public function and at least one unit test.
- [ ] All LLM calls go through `src/llm/client.py`; no module here imports a vendor SDK directly.
- [ ] Router does not perform retrieval; it only emits flags.

### 10.4 Validation commands

```
uv run pytest tests/unit -q
grep -RIE "from openai|from cohere|import openai|import cohere" src/retrieval || echo "ok"
```

### 10.5 Commit message

```
phase 5: agentic RAG components (query, rerank, self-RAG)

- Query decomposition with sub-question DAG and JSON repair retry
- Query rewrite + heuristic-gated HyDE
- Source router emitting policy flags only (no retrieval inside router)
- Cohere reranker with explicit relevance_score mapping
- Near-duplicate filter (5-shingle Jaccard)
- Self-RAG: per-chunk relevance judge (batched), sufficiency check, citation verifier
- Unit tests for parsing, dedup, router flags, verifier parsing
```

### 10.6 Interview talking points to add

- Decomposition produces a DAG, not a flat list.
- HyDE is gated by a heuristic, not always-on.
- Self-RAG's relevance judge is batched on a cheap model; sufficiency uses the main brain — cost-tiered.
- Router stays pure: emits flags, never side-effects.
- JSON repair retry: one shot at "fix your JSON" before raising.

---

## 11. Phase 6 — Multi-hop Citation Walker

### 11.1 Goal

Implement the bounded multi-hop citation traversal subsystem. The walker is invoked when a sub-question's local retrieval is insufficient and budget remains. It can lazily ingest newly discovered papers via Phase 3's ingestion pipeline.

### 11.2 Tasks

1. **`src/retrieval/multi_hop/frontier.py`**:
   - `class FrontierItem(BaseModel)`: `paper_id: str`, `score: float`, `depth: int`.
   - `class Frontier`: priority queue keyed by score (descending).
   - Methods: `add(item)`, `pop() -> FrontierItem | None`, `mark_visited(paper_id)`, `__len__`.
   - Internally use `heapq` with a `(-score, monotonic_counter, item)` tuple to break score ties.
   - Dedup: items with `paper_id` already visited or already enqueued are dropped.

2. **`src/retrieval/multi_hop/citation_walker.py`**:
   - `async def walk_citations(seed_paper_ids: list[str], question: str, *, max_depth: int = MULTI_HOP_MAX_DEPTH, frontier_limit: int = MULTI_HOP_FRONTIER_LIMIT) -> WalkResult`
   - `WalkResult = {"visited_paper_ids": list[str], "newly_ingested_paper_ids": list[str]}`.
   - Algorithm:
     1. Mark each seed as visited (depth 0) and enqueue its references.
     2. Score each candidate via a quick LLM judge ("does this paper title+abstract relate to <question>?", returns 0-10), using `flash-cheap` for cost control. Batch.
     3. Keep top-`frontier_limit` per hop.
     4. For each kept candidate, lazy-ingest it via `ingestion.ingest_paper` if not already in the DB (this writes to `papers` and `chunks` so subsequent retrieval finds the new chunks).
     5. If `depth < max_depth`, fetch the new paper's references and enqueue them with depth+1.
   - Honor a global wall-clock budget (default: 60 seconds for the entire walk).

3. **Tests**:
   - `tests/unit/test_frontier.py` — Priority semantics, dedup, score-tie ordering deterministic.
   - `tests/integration/test_multi_hop_walk.py` (skipped by default) — Seed with the ReAct paper and the question "How is reflection added on top of ReAct in Reflexion?". Expect Reflexion paper to be visited at depth ≥ 1.

### 11.3 Acceptance criteria

- [ ] `Frontier` deterministically orders ties by insertion order.
- [ ] Walker respects `max_depth` and `frontier_limit`.
- [ ] Lazy ingestion path is exercised: at least one newly fetched paper is added to the DB during the integration test.
- [ ] Wall-clock budget terminates the walk if exceeded; the partial result is returned.

### 11.4 Validation commands

> **Requires Owner Handoff H-3 confirmation in `docs/STATUS.md` before running integration commands.**

```
uv run pytest tests/unit/test_frontier.py -q
INTEGRATION_TESTS=1 uv run pytest tests/integration/test_multi_hop_walk.py -q -s
```

### 11.5 Commit message

```
phase 6: bounded multi-hop citation walker

- Priority frontier with deterministic tie-breaking and dedup
- LLM-judged scoring of citation candidates (cheap-model batch)
- Lazy ingestion of newly discovered papers via Phase 3 pipeline
- Depth- and wall-clock-bounded traversal
- Unit + integration tests
```

### 11.6 Interview talking points to add

- Bounded multi-hop with LLM-judged frontier prevents combinatorial explosion of citation expansion.
- Lazy ingestion means the corpus grows during a session — the agent extends its own memory.
- Wall-clock budget is a separate axis from depth/frontier limits — three independent safety knobs.

---

## 12. Phase 7 — Episodic Memory and Compression

### 12.1 Goal

Implement working memory (typed dict for graph state) and episodic memory (session message history with rolling-summary compression cached in `session_summaries`).

### 12.2 Tasks

1. **`src/memory/working.py`** — Types only:
   - `WorkingMemory(TypedDict)` to be embedded in the LangGraph state in Phase 9. Fields: `current_question`, `decomposition`, `subq_results`, `evidence`, `tool_calls`, etc.

2. **`src/memory/episodic/session_store.py`**:
   - `async def get_or_create_session(session_id: str, title: str | None = None) -> Session`.
   - `async def append_message(session_id: str, role: str, content: str, metadata: dict | None = None) -> int` — returns `message_id`.
   - `async def get_recent(session_id: str, limit: int) -> list[Message]`.

3. **`src/memory/episodic/compressor.py`**:
   - `async def compressed_history(session_id: str, *, verbatim_turns: int = EPISODIC_VERBATIM_TURNS) -> list[Message]`
   - Strategy:
     1. Fetch recent N turns; keep them verbatim.
     2. For older messages, look up `session_summaries` for the latest cached row where `up_to_message_id <= cutoff_message_id`. If hit, prepend that summary as a system message.
     3. If older messages exist past the cached `up_to_message_id`, summarize the new prefix incrementally using `flash-cheap` and upsert into `session_summaries` keyed by `(session_id, new_up_to_message_id)`.
   - Never re-summarize the same prefix twice.

4. **Tests**:
   - `tests/unit/test_compressor_cache.py` — Two consecutive calls with no new messages must not re-call the LLM (use a counting mock).
   - `tests/integration/test_compressor_persist.py` (skipped by default) — Insert 30 messages; verify cached summary covers messages ≤ id N-K, verbatim covers id > N-K.

### 12.3 Acceptance criteria

- [ ] `compressed_history` never re-summarizes the same prefix twice.
- [ ] Cache hits are observable via the counting mock test.
- [ ] Persistent summaries land in `session_summaries`.

### 12.4 Validation commands

> **Requires Owner Handoff H-3 confirmation in `docs/STATUS.md` before running integration commands.**

```
uv run pytest tests/unit/test_compressor_cache.py -q
INTEGRATION_TESTS=1 uv run pytest tests/integration/test_compressor_persist.py -q
docker exec arxiv_agent_pg psql -U arxiv_agent -d arxiv_agent -c "SELECT count(*) FROM session_summaries;"
```

### 12.5 Commit message

```
phase 7: episodic memory and rolling-summary compression

- WorkingMemory typed dict for upcoming graph state
- session_store CRUD for session_messages
- Rolling-summary compressor with session_summaries cache keyed by upper message id
- Cache-hit counting test guarantees no redundant summarization
```

### 12.6 Interview talking points to add

- Compression policy: verbatim recent + recursive summary of older; cache keyed by upper message id keeps cost bounded.
- The compressor is a pure consumer of `session_messages` and `session_summaries` — no global state, fully testable.

---

## 13. Phase 8 — Semantic Memory and Concept Graph

### 13.1 Goal

Build the semantic memory tier: a concept graph with extractor, three-tier entity linker, graph store, and the **Input Mode** graph retriever that biases retrieval. End the phase with end-of-session reflection that promotes information from episodic to semantic.

### 13.2 Tasks

1. **`src/memory/semantic/extractor.py`**:
   - `async def extract_concepts_from_text(text: str, *, source_chunk_ids: list[int]) -> list[ConceptExtraction]`.
   - LLM-driven extraction with strict JSON: each concept has `display_name`, `definition`, `aliases`, `evidence_chunk_ids`, optional `relations: list[{target_display, type, evidence_chunk_ids}]`.
   - Use `haiku-4.5`. Be conservative; instruct the model to extract only concepts that have clear definitions in the text.

2. **`src/memory/semantic/linker.py`**:
   - `async def link_concept(extraction: ConceptExtraction) -> ConceptLinkResult`.
   - Steps (three-tier fallback):
     1. Normalize `display_name` (lowercase, strip, collapse whitespace) → candidate canonical.
     2. Look up `concept_aliases` for an exact match.
     3. If miss: look up `concepts.canonical` for an exact match.
     4. If miss: vector search on `concepts.embedding` (cosine ≥ 0.85), then LLM judge confirms same concept (`haiku-4.5`).
     5. If still miss: insert a new concept; insert canonical and any provided aliases.
   - Concept embeddings: at insert time, embed the string `display_name + ". " + definition` via `embed-small` (resolves M-5 / A-4). Store the vector in `concepts.embedding`.

3. **`src/memory/semantic/graph_store.py`**:
   - `async def upsert_concept(...)`, `async def upsert_relation(src_concept_id, dst_concept_id, rel_type, evidence_chunks)`.
   - `async def get_concept(*, by_id=None, by_canonical=None, by_alias=None)`.
   - `async def get_neighbors(concept_id: int, rel_types: list[str] | None = None, limit: int = 20) -> list[Neighbor]`.
   - `Neighbor = (concept, rel_type, weight, evidence_chunks)`.

4. **`src/memory/semantic/graph_retriever.py`** — Input Mode:
   - `async def expand_query_via_graph(question: str) -> GraphExpansion`.
   - Steps:
     1. Detect concept mentions: extract candidate phrases via a quick LLM call (`flash-cheap`); for each candidate, look up `concept_aliases`/`concepts.canonical`/embedding-search. Collect resolved `concept_ids`.
     2. For each resolved concept, fetch neighbors of types `extends`, `compares`, `is-a`, `uses` up to a small limit.
     3. **Derive paper hints by joining `evidence_chunks` JSONB array elements to `chunks.paper_id`**, then to distinct paper IDs (resolves M-11). SQL pattern:

```sql
WITH ec AS (
  SELECT (jsonb_array_elements_text(evidence_chunks))::bigint AS chunk_id
  FROM concept_relations
  WHERE src_id = ANY($1) OR dst_id = ANY($1)
)
SELECT DISTINCT c.paper_id
FROM ec
JOIN chunks c ON c.chunk_id = ec.chunk_id;
```

     4. Return `GraphExpansion = {seed_concept_ids, neighbor_concept_ids, extra_paper_hints: [paper_id, ...]}`.
   - This object is consumed by the retrieval node in Phase 9: hits whose `paper_id` is in `extra_paper_hints` receive a small score boost (or are biased via filter when sufficient).

5. **`src/memory/reflection.py`**:
   - `async def reflect_session(session_id: str) -> ReflectionReport`.
   - At end of session, take the assistant's final answers and their cited chunks, run them through `extractor.py`, then `linker.py`, then `graph_store.upsert_relation`.
   - Idempotent: re-running reflection on the same session must not create duplicate concepts or duplicate aliases.

6. **Tests**:
   - `tests/unit/test_linker_alias.py` — Linker correctly merges "ReAct" and "react prompting" via alias path.
   - `tests/unit/test_linker_embedding_path.py` — Mock embedding similarity ≥ 0.85 + mock LLM "same concept": linker reuses the existing concept.
   - `tests/integration/test_graph_retriever.py` (skipped by default) — Pre-seed the concept graph with two concepts and one relation backed by chunks of two papers; verify `expand_query_via_graph` returns those two paper IDs in `extra_paper_hints`.
   - `tests/integration/test_reflection_loop.py` (skipped by default) — Simulate a small session with two assistant answers; reflection inserts at least 5 concepts and at least 1 relation; second run inserts zero new rows.

### 13.3 Acceptance criteria

- [ ] Concept embeddings are stored for every newly created concept.
- [ ] `expand_query_via_graph` returns non-empty `extra_paper_hints` after reflection runs over the seed corpus.
- [ ] Reflection is idempotent.

### 13.4 Validation commands

> **Requires Owner Handoff H-3 confirmation in `docs/STATUS.md` before running integration commands.**

```
uv run pytest tests/unit/test_linker_alias.py tests/unit/test_linker_embedding_path.py -q
INTEGRATION_TESTS=1 uv run pytest tests/integration/test_graph_retriever.py tests/integration/test_reflection_loop.py -q
docker exec arxiv_agent_pg psql -U arxiv_agent -d arxiv_agent -c "SELECT count(*) FROM concepts;"
docker exec arxiv_agent_pg psql -U arxiv_agent -d arxiv_agent -c "SELECT count(*) FROM concept_relations;"
```

### 13.5 Commit message

```
phase 8: semantic memory and concept graph

- Concept extractor with strict JSON schema
- Three-tier linker: alias / canonical / embedding+LLM-judge merge
- Concept embeddings on display_name + definition
- Graph store with upsert and neighbor queries
- Input Mode graph retriever; chunk -> paper_id derivation via JSONB join
- Reflection promoting episodic to semantic; idempotent
- Unit + integration tests
```

### 13.6 Interview talking points to add

- Three-tier entity linking is the practical fallback for noisy LLM extractions.
- Concept graph biases retrieval (Input Mode); the graph isn't decoration.
- Reflection idempotency: each session's promotion is safe to replay.

---

## 14. Phase 9 — LangGraph Orchestration

### 14.1 Goal

Wire all components from Phases 1, 4, 5, 6, 7, and 8 into a LangGraph state graph with `AsyncPostgresSaver` checkpointer (psycopg-backed). End-to-end: a question goes in, a grounded answer comes out, state persists.

### 14.2 Tasks

1. **`src/graph/state.py`**:
   - Define `AgentState` as a `TypedDict` with explicit reducers:
     - `messages: Annotated[list[AnyMessage], add_messages]` (use `langchain_core.messages.AnyMessage`).
     - `question: str`
     - `decomposition: Decomposition | None`
     - `subq_results: Annotated[dict[int, SubQResult], merge_dicts]`
     - `evidence: list[Hit]`
     - `graph_expansion: GraphExpansion | None`
     - `multi_hop_visited: list[str]`
     - `multi_hop_calls: int`
     - `answer: str | None`
     - `citations: list[Citation]`
     - `verification: VerificationReport | None`
     - `reflection: ReflectionReport | None`
     - `tool_iters: int`
   - Implement `merge_dicts` reducer.

2. **`src/graph/nodes/decompose.py`** — calls `query.decomposer.decompose`, sets state.

3. **`src/graph/nodes/retrieve.py`** — for each sub-question (resolves M-1 / N-M-1; router runs first, graph_retriever is conditional):
   1. Call `query.router.route` to obtain policy flags (`use_concept_graph`, `may_use_semantic_scholar_live`).
   2. **If** `route.use_concept_graph` is True: call `graph_retriever.expand_query_via_graph` to obtain `extra_paper_hints`. Otherwise skip and use empty hints.
   3. Call `query.rewriter.rewrite_for_retrieval` (and HyDE conditionally).
   4. Call `index.hybrid_search.hybrid_search` (with paper hint biasing applied as a post-filter ranking boost).
   5. Call `postprocess.deduplicator.dedupe`.
   6. Call `postprocess.reranker.rerank`.
   7. Call `self_rag.relevance_judge.judge_batch`.
   8. If `is_sufficient` is False and `tool_iters < SELF_RAG_MAX_RETRIES`: rewrite query and retry from step 4.
   9. If still insufficient and `multi_hop_calls < 1` and `route.may_use_semantic_scholar_live` is True: emit a routing decision to the multi-hop node.

   The router-first ordering makes `test_router_wired.py` coherent: when the router sets `use_concept_graph=False`, the graph retriever is not invoked.

4. **`src/graph/nodes/multi_hop.py`** — calls `multi_hop.citation_walker.walk_citations`, then re-enters `retrieve` for the same sub-question.

5. **`src/graph/nodes/synthesize.py`** — assembles final answer (resolves N-M-2):
   - Constructs a prompt that lists each kept evidence chunk in the form `[chunk_id={cid} paper_id={pid} section={section}]\n{text}` so the model sees both identifiers.
   - Instructs `haiku-4.5` to write a grounded answer where every cited claim ends with the inline citation token `[paper_id#chunk_id]` (both fields required, separated by `#`). Forbid `[paper_id]`-only citations.
   - Parses citations with the regex `\[(?P<pid>[^\]#\s]+)#(?P<cid>\d+)\]`. Constructs `Citation(paper_id=pid, chunk_id=int(cid), ...)` for each match. If the model produces malformed citations, retry once with a "fix the citation format" message, then raise.
   - Builds an `evidence_lookup: dict[int, str]` from the synthesizer's evidence list (chunk_id → chunk text). This lookup is passed into the verifier in step 6.

6. **`src/graph/nodes/self_rag.py`** — runs `verify_citations(answer, citations, evidence_lookup)` on the answer (resolves N-M-2). The verifier looks up evidence text by `chunk_id` from the lookup deterministically, no guessing. If any citation fails verification, removes the citation and asks the model either to find an alternative (one retry) or drop the claim.

7. **`src/graph/nodes/reflect.py`** — calls `memory.reflection.reflect_session`.

8. **`src/graph/edges.py`** — conditional routing functions:
   - `route_after_retrieve`: returns `synthesize` if sufficient, `multi_hop` if budget remains, `synthesize` (best effort) otherwise.
   - `route_after_verify`: returns `synthesize` if any failed citation needs replacement (and retry budget remains), else `reflect`.
   - Each edge function reads explicit budget counters (`tool_iters`, `multi_hop_calls`) to bound loops.

9. **`src/graph/builder.py`**:
   - Compiles the graph and binds `AsyncPostgresSaver` from the psycopg pool.
   - Exposes `async def run(question: str, *, thread_id: str) -> AgentState` for tests and CLI.
   - Exposes `async def shutdown()` that closes pools and LLM clients.

10. **Tests**:
    - `tests/unit/test_edge_budgets.py` — Build a fake state with `tool_iters` at the max and `multi_hop_calls` at the max; verify `route_after_retrieve` returns `synthesize` regardless of sufficiency.
    - `tests/integration/test_router_wired.py` (skipped by default) — Patch `query.router.route` to set `use_concept_graph=False`; verify the retrieve node does not call `graph_retriever`. Then patch with `True`; verify it does. This proves M-1 wiring.
    - `tests/integration/test_graph_e2e.py` (skipped by default) — End-to-end on "What is ReAct?". Asserts: at least one citation in the answer; the citation's `paper_id` exists in `papers`; verification passes; reflection inserts at least one concept.
    - `tests/integration/test_resume.py` (skipped by default) — Run one turn with `thread_id=t-resume`; assert checkpoint row exists; spawn a new process (via `subprocess`) calling `run` again with the same `thread_id`; verify it resumes (no re-decomposition).

### 14.3 Acceptance criteria

- [ ] `uv run python -c "import asyncio; from src.graph.builder import run, shutdown; ..."` produces a grounded answer with at least one citation, then cleanly shuts down.
- [ ] Resuming with the same `thread_id` reads the checkpoint (resume test passes).
- [ ] LangSmith traces appear under the configured project (manual check during owner handoff H-3).
- [ ] No edge cycles exceed their declared budgets (`tool_iters`, `multi_hop_calls`).

### 14.4 Validation commands

> **Requires Owner Handoff H-3 confirmation in `docs/STATUS.md` before running integration commands.**

```
uv run pytest tests/unit/test_edge_budgets.py -q
INTEGRATION_TESTS=1 uv run pytest tests/integration/test_router_wired.py tests/integration/test_graph_e2e.py tests/integration/test_resume.py -q -s
```

### 14.5 Commit message

```
phase 9: LangGraph orchestration

- AgentState typed dict with explicit reducers (add_messages, merge_dicts)
- Nodes: decompose, retrieve, multi_hop, synthesize, self_rag, reflect
- Retrieve node now wires router and deduplicator (M-1)
- Conditional edges with explicit budget counters
- AsyncPostgresSaver checkpointer wired to psycopg pool
- run()/shutdown() entry points
- Tests for edge budgets, router wiring, end-to-end, resume
```

### 14.6 Interview talking points to add

- LangGraph state uses explicit reducers — `add_messages`, `merge_dicts` — not implicit override semantics.
- Retrieval is one node; multi-hop is a separate node entered only when sufficiency fails and budget remains.
- Persistence via `AsyncPostgresSaver` (psycopg-backed) enables checkpointed resume across processes.

---

## 15. Phase 10 — Evaluation Harness

### 15.1 Goal

Create a 30-50 question gold dataset, three families of metrics, and a LangSmith-integrated runner. Establish baseline numbers for the human owner to iterate against.

### 15.2 Tasks

1. **`src/eval/datasets/gold_questions.py`** — A list of `GoldQuestion` records:

```python
class GoldQuestion(BaseModel):
    id: str
    question: str
    expected_paper_ids: list[str]
    answer_must_mention: list[str]
    answer_must_not_mention: list[str] = []
    notes: str = ""
```

   - Hand-author 30 questions spanning: definitions, comparisons, multi-hop, survey-style, application-level. Cover the full seed corpus.

2. **`src/eval/metrics/retrieval_metrics.py`**:
   - `def recall_at_k(retrieved_paper_ids: list[str], expected: list[str], k: int) -> float`.
   - `def mrr(retrieved_paper_ids: list[str], expected: list[str]) -> float`.

3. **`src/eval/metrics/citation_metrics.py`**:
   - `def citation_precision(cited_paper_ids: list[str], expected: list[str]) -> float`.
   - `def citation_recall(cited_paper_ids: list[str], expected: list[str]) -> float`.

4. **`src/eval/metrics/llm_judge.py`**:
   - `async def judge_answer(question: str, answer: str, gold: GoldQuestion) -> JudgeVerdict`.
   - G-Eval-style: `haiku-4.5` scores 1-5 on `correctness`, `groundedness`, `completeness`. Output JSON, parsed strictly.

5. **`src/eval/runner.py`** and **`scripts/eval_run.py`**:
   - For each gold question, run the agent (`graph.builder.run` with a fresh thread_id).
   - Use LangSmith `Client.upsert_dataset` (or equivalent) so reruns do not duplicate datasets (resolves L-10).
   - Log run metadata.
   - Compute aggregate metrics: average recall@5, citation precision/recall, judge scores; per-question CSV in `data/eval_outputs/`.

6. **Tests**:
   - `tests/unit/test_retrieval_metrics.py` — boundary cases for recall and MRR.
   - `tests/unit/test_citation_metrics.py` — empty cited list; full overlap; partial overlap.

### 15.3 Acceptance criteria

- [ ] `make eval` runs all gold questions, prints aggregate metrics, writes CSV to `data/eval_outputs/`.
- [ ] LangSmith dashboard shows the runs grouped under `LANGSMITH_PROJECT`.
- [ ] Re-running `make eval` updates the same dataset rather than creating a duplicate.
- [ ] Baseline numbers documented in `docs/decisions/0003-eval-baseline.md`.

### 15.4 Validation commands

```
make eval
ls data/eval_outputs/
test -f docs/decisions/0003-eval-baseline.md && echo "baseline ADR present"
```

### 15.5 Commit message

```
phase 10: evaluation harness

- 30-question hand-authored gold dataset
- Retrieval, citation, and LLM-judge metrics
- LangSmith dataset upsert (no duplicates on rerun)
- Eval runner with CSV outputs
- Baseline metrics in ADR 0003
```

---

## 16. Phase 11 — User Interface

### 16.1 Goal

A Chainlit chat UI that streams agent reasoning steps and shows citations as clickable cards, plus a separate FastAPI page for the live concept graph.

### 16.2 Tasks

1. **`src/ui/app.py`** — Chainlit application:
   - `@cl.on_chat_start`: create a `session_id`, store in `cl.user_session`.
   - `@cl.on_message`: invoke `graph.builder.run` via `astream_events`. For each event, render with `cl.Step`. Step labels match graph node names.
   - `@cl.on_chat_end`: call `close_llm_clients()` and `close_pools()` for clean shutdown (resolves M-3).
   - At the end of each turn, render the final answer with citations using `cl.Message` and inline citation elements.

2. **`src/ui/elements.py`**:
   - `def citation_card(paper_id, section, snippet, url)` — returns a `cl.Text` with paper title, section, snippet, and link.

3. **`src/ui/graph_view/server.py`**:
   - A FastAPI app with one route `GET /graph/data?session_id=...` returning JSON `{nodes: [...], edges: [...]}`.
   - Use FastAPI's `lifespan` context manager to call `close_llm_clients()` and `close_pools()` on shutdown.

4. **`src/ui/graph_view/static/index.html`**:
   - Single HTML file using `react-flow` via CDN. Fetches `/graph/data` and renders. Polls every 5 seconds.

5. **Run scripts**:
   - `Makefile` `ui` target runs Chainlit on port 8000 (resolves L-3 — uses `--host 127.0.0.1 --port 8000`, not `-h`).
   - `Makefile` `ui-graph` target runs uvicorn on port 9000.

6. **Tests**:
   - No automated UI tests. Manual smoke test in acceptance.

### 16.3 Acceptance criteria

- [ ] `make ui` serves Chainlit at `http://127.0.0.1:8000`.
- [ ] Asking "What is ReAct?" streams visible reasoning steps named `decompose`, `retrieve`, `synthesize`, `verify`, `reflect`.
- [ ] The final answer shows at least one clickable citation that links to the arXiv URL.
- [ ] `make ui-graph` serves the concept graph at `http://127.0.0.1:9000`; the page updates after a session that produces new concepts.
- [ ] Stopping the UI cleanly closes all LLM clients and DB pools (verified by absence of `unclosed connector` warnings in the log).

### 16.4 Validation commands

```
# These commands launch background processes. Stop them with `kill %1; kill %2` after verifying.
make ui &
UI_PID=$!
sleep 5
curl -sS -o /dev/null -w "%{http_code}\n" http://127.0.0.1:8000
make ui-graph &
GR_PID=$!
sleep 3
curl -sS http://127.0.0.1:9000/graph/data | head -c 200
kill $UI_PID $GR_PID
```

### 16.5 Commit message

```
phase 11: chainlit UI and concept graph view

- Chainlit app streaming agent steps via astream_events
- Citation cards with arXiv links
- FastAPI route serving live concept graph data, with lifespan cleanup
- Static react-flow page (CDN, no build step)
- Manual smoke test acceptance
```

---

## 17. Phase 12 — Polish and Documentation

### 17.1 Goal

Performance pass, documentation, and a curated list of interview talking points.

### 17.2 Tasks

1. **Performance pass**: run the eval suite, identify the slowest two stages, and apply at least two improvements — examples: parallelize sub-question retrieval, cache frequent embedding lookups, batch relevance judgments more aggressively. Document each in an ADR.
2. **`docs/architecture.md`**: a single architecture document with a mermaid diagram, state schema, and a numbered tour of a single request.
3. **`docs/decisions/`**: ensure every non-trivial decision in Phases 1-11 has an ADR. Required entries:
   - 0001 Model and LangGraph selection (Phase 1)
   - 0002 RRF fusion (Phase 4)
   - 0003 Eval baseline (Phase 10)
   - 0004 Why pgvector over Chroma
   - 0005 Why split asyncpg + psycopg drivers
   - 0006 Why OpenRouter for chat but direct providers for embedding/rerank
   - 0007 Episodic compression strategy
   - 0008 Concept linking with three-tier fallback
   - 0009 Multi-hop budget and frontier ordering
   - 0010-0012 Performance pass decisions
4. **`docs/interview_talking_points.md`**: 15+ entries spanning all phases. Each entry: one-line claim, file reference, 3-sentence elaboration.
5. **`README.md`** polish: small architecture diagram, links to ADRs and talking points, an example session.
6. **Final eval run**: rerun the eval suite, save final numbers to `data/eval_outputs/final.csv` and `docs/decisions/0013-final-numbers.md`.

### 17.3 Acceptance criteria

- [ ] All ADRs listed above exist.
- [ ] `docs/interview_talking_points.md` has at least 15 entries.
- [ ] Final eval numbers committed.
- [ ] `README.md` polished and accurate.

### 17.4 Commit message

```
phase 12: polish, performance pass, and documentation

- ADRs for all major decisions and performance changes
- Architecture document with state schema and request tour
- Curated interview talking points (15+ entries)
- Final eval numbers
- README polish
```

---

## Appendix A — Pinned Dependency Versions and Library Notes

The exact versions in Section 5.2 task 6 are the planner's recommendations. If a version is yanked or has a known critical bug at execution time, pick the nearest stable version and document the swap in `docs/decisions/`.

Library-specific notes for Codex:

- **LangGraph**: We rely on `from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver`. Its `setup()` is async and creates tables on first run. The `AsyncPostgresSaver` accepts a psycopg `AsyncConnectionPool`; **do not** pass an asyncpg pool. Phase 1 verifies the LangGraph version line and any open security advisory; the chosen line is recorded in ADR `0001-model-and-langgraph-selection.md`.
- **psycopg pool**: must be constructed with `kwargs={"autocommit": True, "row_factory": dict_row}` (`from psycopg.rows import dict_row`). Without these, `AsyncPostgresSaver.setup()` may not persist tables and reads can fail. Example:

  ```python
  from psycopg.rows import dict_row
  from psycopg_pool import AsyncConnectionPool
  pool = AsyncConnectionPool(conninfo, min_size=1, max_size=10, open=False,
                             kwargs={"autocommit": True, "row_factory": dict_row})
  await pool.open()
  ```

  Call `await pool.close()` on shutdown.
- **OpenRouter**: HTTP base is `https://openrouter.ai/api/v1`. Headers: `Authorization: Bearer $OPENROUTER_API_KEY`, optional `HTTP-Referer` and `X-Title`. Body is OpenAI-compatible.
- **DeepSeek**: HTTP base is `https://api.deepseek.com/v1`. OpenAI-compatible.
- **Cohere**: `cohere.AsyncClientV2`. Current rerank model IDs are typically `rerank-v4.0-pro`, `rerank-v4.0-fast`, and `rerank-v3.5`. Phase 1 picks one and pins it. Rerank response uses `relevance_score`; map explicitly to local `score`.
- **arXiv**: 3-second rate limit between requests. Sync API; wrap in `asyncio.to_thread`.
- **Semantic Scholar**: Public Graph API at `https://api.semanticscholar.org/graph/v1`. Without a key, ~1 req/sec. With a key, ~10 req/sec.
- **Chainlit**: 1.3.x. Use `astream_events` for streaming.
- **pgvector + asyncpg**: Call `pgvector.asyncpg.register_vector(connection)` in pool init.

---

## Appendix B — Seed Corpus arXiv IDs

Codex must use this list verbatim in `src/eval/datasets/seed_papers.py` as `SEED_PAPER_IDS`. If any ID 404s at fetch time, log and skip — do not silently substitute.

Foundational LLM agents:
- 2210.03629 (ReAct)
- 2303.11366 (Reflexion)
- 2305.10601 (Tree of Thoughts)
- 2308.09687 (Graph of Thoughts)
- 2310.04406 (LATS)

Memory and persistence:
- 2310.08560 (MemGPT)
- 2502.12110 (A-MEM)
- 2304.03442 (Generative Agents)
- 2405.14831 (HippoRAG)

Tool use:
- 2302.04761 (Toolformer)
- 2305.15334 (Gorilla)
- 2307.16789 (ToolLLM)

RAG core and agentic RAG:
- 2005.11401 (RAG, Lewis et al.)
- 2310.11511 (Self-RAG)
- 2401.15884 (CRAG)
- 2212.10496 (HyDE)
- 2112.01488 (ColBERT v2)
- 2312.10997 (Survey: RAG)
- 2501.09136 (Survey: Agentic RAG)

Multi-agent and SOP frameworks:
- 2308.00352 (MetaGPT)
- 2308.08155 (AutoGen)
- 2303.17760 (CAMEL)

Reasoning and CoT:
- 2201.11903 (Chain of Thought)
- 2203.11171 (Self-Consistency)
- 2305.04091 (Plan-and-Solve)

Evaluation and benchmarks:
- 2308.03688 (AgentBench)
- 2311.12983 (GAIA)
- 2310.06770 (SWE-bench)

Surveys:
- 2308.11432 (LLM-based autonomous agents)
- 2304.08354 (Tool learning with foundation models)

Total: 30 papers.

---

## Appendix C — Templates

### C.1 Phase report template (`docs/phase_reports/PHASE_<N>_REPORT.md`)

```markdown
# Phase <N> Report

**Phase title**: <copy from execution plan>
**Status**: complete | blocked
**Commit**: `<hash>`
**Owner handoff pending**: <H-X if any, else "none">

## Summary
A 3-5 sentence description of what was built and any deviations from the plan.

## Files created
- `path/to/file.py` — <one-line role>

## Files modified
- `path/to/file.py` — <one-line description>

## Tests run
- Command: `<exact command>`
- Outcome: `<pass/fail with summary>`

## Validation commands
- Command: `<from acceptance section>`
- Output (truncated if long): ```...```

## Acceptance checklist
- [x] criterion 1
- [x] criterion 2

## Deviations from the plan
None — or — <list with rationale>

## Open questions for the planner
None — or — <list>

## Ready for review
The planner can read these files and run those commands to verify.
```

### C.2 Commit message template (HEREDOC form)

```
git commit -m "$(cat <<'EOF'
phase <N>: <short title>

- bullet 1
- bullet 2
- bullet 3
EOF
)"
```

### C.3 Question template (`docs/phase_reports/PHASE_<N>_QUESTIONS.md`)

```markdown
# Phase <N> Questions

## Question 1: <one-line title>

**Context**: <what you were doing>
**Problem**: <what's blocking>
**Options considered**:
1. <option A> — pros: ... cons: ...
2. <option B> — pros: ... cons: ...
3. <option C> — pros: ... cons: ...
**Recommendation**: <your best guess>
```

---

## Appendix D — Common Errors and Resolutions

- **`asyncpg` connection refused**: Postgres container not up. Run `make db-up` and wait for the healthcheck.
- **`pgvector` extension not found**: Container is plain Postgres. Verify image is `pgvector/pgvector:pg16`.
- **OpenRouter 402 / 429**: Rate limit or insufficient credits. Surface to the user; do not auto-downgrade silently.
- **LangGraph checkpoint errors on first run**: `AsyncPostgresSaver.setup()` was not called. Add it to `bootstrap_db.py`.
- **Cohere rerank empty results**: Documents list contained empty strings. Filter empty strings before sending.
- **Chainlit missing dependencies**: ensure `chainlit` is in `pyproject.toml` dependencies.
- **`unclosed connector` warnings on shutdown**: client `aclose()` not awaited. Verify lifespan or `close_llm_clients()` is called.
- **asyncpg `cannot adapt type 'list'`** when inserting embeddings: the pgvector codec is not registered. Verify `pgvector.asyncpg.register_vector` is in the pool's `init` callback.

---

## Appendix E — Project Status Board

Phase 0 creates `docs/STATUS.md` with this initial content:

```markdown
# Project Status

| Phase | Title | Status | Commit | Date | Owner handoff |
|-------|-------|--------|--------|------|---------------|
| 0 | Bootstrap | not started | - | - | - |
| 1 | Provider Adapter Layer | not started | - | - | - |
| 2 | Database Schema and Connection | not started | - | - | - |
| 3 | Corpus Clients and Ingestion | not started | - | - | - |
| 4 | Vector Index and Hybrid Retrieval | not started | - | - | - |
| 5 | Agentic RAG Components | not started | - | - | - |
| 6 | Multi-hop Citation Walker | not started | - | - | - |
| 7 | Episodic Memory and Compression | not started | - | - | - |
| 8 | Semantic Memory and Concept Graph | not started | - | - | - |
| 9 | LangGraph Orchestration | not started | - | - | - |
| 10 | Evaluation Harness | not started | - | - | - |
| 11 | User Interface | not started | - | - | - |
| 12 | Polish and Documentation | not started | - | - | - |
```

After each phase's commit, Codex updates the matching row, including the actual commit hash and owner-handoff status if a handoff is pending.

---

## End of Execution Plan
