# ARA v2 — Decision log

Each entry: context, decision, alternatives, consequences. New decisions are appended; a changed
decision gets a new entry that names the one it replaces.

## D1 — Rebuild on branch `v2` instead of refactoring v1 (2026-10-07, Ewan)
- Context: v1 works but its core is hard to explain: a 156-line central router, a flat state with
  60+ fields, very long verifier functions and dense defensive code. LangGraph was used as a star
  around one routing function, so its value was not visible.
- Decision: rebuild from scratch on `v2`; v1 stays at tag `baseline-2026-10-07` for reference.
- Consequence: v1 evaluation numbers do not carry over; v2 earns its own.

## D2 — LangGraph owns orchestration and state only (2026-10-07)
- Decision: use checkpointer threads, `interrupt`, `Send`, subgraphs, node retry/timeout/error
  handlers, the Store and streaming, because the product needs each of them (DESIGN §4).
  Retrieval, metrics, the LLM gateway and the eval runner stay plain Python.
- Alternatives: plain asyncio + a durable-execution engine (Temporal, DBOS; Pydantic AI integrates
  both); OpenAI Agents SDK. Rejected because interrupts, fan-out, checkpoints and memory would be
  rebuilt by hand, which is what v1 partly did.
- Consequence: the workflow is declarative and can be drawn from the compiled graph in the UI.

## D3 — Two model families with fixed roles (2026-10-07) [proposal; confirmed as D10]
- Decision: `gpt-6-luna` for machine-consumed structured decisions (strict schemas, cheapest
  tokens); `deepseek-flash` for the tool loop and user-facing prose. The verifier differs from the
  writer. Judge rule: an evaluation judge never shares a family with the component that produced or
  selected the graded artifact (DESIGN §6.2).
- Alternatives: one model everywhere (simpler, but no independent checking and no strict schemas
  on DeepSeek); Luna everywhere with DeepSeek only as judge (cheapest; reconsider if measurements
  favour it).
- Consequence: two providers in one gateway; the verifier choice is settled by suite S5.

## D4 — Cache-aware prompt layout (2026-10-07) [proposal; replaced by D11]
- Decision: every prompt is static → shared → item, with explicit breakpoints (OpenAI), stable
  serialisation, prewarm before fan-out, append-only conversations, fixed tools/schema/effort per
  stage. Cache-hit rate is measured per stage.
- Consequence: verification of N claims pays for the evidence pack about once; this is visible in
  the UI timeline and the eval report.

## D5 — PostgreSQL as the single store (2026-10-07)
- Decision: one Postgres with pgvector + BM25 (ParadeDB, verified at M0, see D12) for chunks, plus
  LangGraph checkpoints and Store, the cost ledger, and eval results.
- Alternatives: a separate vector database or search engine — unnecessary at this scale.
- Consequence: native full-text search (no IDF) remains available as an ablation against BM25.

## D6 — Small, tiered evaluation with a cost cap (2026-10-07, Ewan)
- Decision: suites S1–S7 (DESIGN §11); one full round ≤ US$1; dry run by default; judges
  calibrated on Ewan's labels; paired comparisons with bootstrap CIs; public data where it fits
  (QASPER, PaSa) plus small labeled sets.
- Consequence: results are directional; the report always states n and CIs.

## D7 — No mocks; small live tests; review before scale (2026-10-07, Ewan)
- Decision: unit tests cover deterministic code against real Postgres; LLM behaviour is checked
  with small live calls under the approval threshold. No fake LLMs or recorded cassettes.
  Large-scale LLM testing waits for Ewan's architecture review.

## D8 — arXiv only, English first (2026-10-07, Ewan)
- Decision: no Semantic Scholar. arXiv's limit (one request per three seconds, one connection)
  means parallelism goes into LLM steps (screening, reading, verification), not search calls.

## D9 — UI for interviews (2026-10-07, Ewan)
- Decision: FastAPI + SSE + React. Page priority: workflow (live graph from the compiled
  LangGraph), evidence, evaluation, memory (optional).

## D10 — Stage allocation confirmed, with DeepSeek effort and judge output fixed (2026-10-07, Ewan)
- Replaces the D3 proposal. Context: M0 verification showed DeepSeek has no schema-enforced output
  (`json_object` only; strict schemas only through beta tool calls) and only three thinking
  efforts (low / high / max), and that V4.1 caching favours sequential append-only calls.
- Decision: the DESIGN §6.2 table as proposed, plus: researcher effort low, synthesize / repair /
  relevance judge effort high; the DeepSeek relevance judge returns `json_object`, validated by
  Pydantic, and a parse failure is a counted `judge_error` (no repair). Every stage has a fixed
  `max_output_tokens`, because the ledger reserves the maximum and DeepSeek's thinking default is 64K.
- Alternatives: leave efforts to M3 measurement; Luna everywhere with DeepSeek only as judge.
- Consequence: fan-out stages are all on Luna, sequential stages on DeepSeek, which matches both
  providers' caching. Efforts are re-measured in M3.

## D11 — Cache design corrected after verification (2026-10-07, Ewan)
- Replaces the D4 proposal. Context: OpenAI's default implicit mode writes each request's tail at
  1.25× the input price; top-level `instructions` cannot hold a breakpoint; prewarm is a separate
  request; DeepSeek V4.1 stores a shared prefix only after two requests share it.
- Decision: OpenAI calls always use explicit mode, with breakpoints on the parts the stage table
  lists (static + shared for fan-out stages, + item for understand); static instructions go into a
  developer message; prewarm only when a fan-out has ≥ 2 calls and the shared prefix is ≥ 1,024
  tokens, with the same model, schema and effort as the calls that follow; no DeepSeek warm-up.
- Alternatives: implicit mode with extra breakpoints (simpler, pays the 1.25× write on every tail).
- Consequence: fan-out items are billed at the plain input rate; cache-hit rate per stage stays
  measurable from the ledger.

## D12 — ParadeDB 0.26 on PostgreSQL 18, pinned (2026-10-07)
- Context: D5 chose ParadeDB pending verification. The 0.26.0 image bundles PostgreSQL 18.6,
  pgvector 0.8.6 and pg_search 0.26.0; BM25, pgvector (exact and HNSW) and native FTS all ran.
- Decision: pin `paradedb/paradedb:0.26.0-pg18`. Use `USING paradedb` without `key_field`, the
  `|||` operator and `pdb.score(id)`. The v2 database runs in its own container on
  127.0.0.1:5434 with its own volume; v1's container (port 5433) is left untouched.
- Consequence: upgrading ParadeDB is a deliberate change (syntax moved between 0.25 and 0.26).

## D13 — M0 sequencing (2026-10-07)
- Decision: migrations add tables in the milestone that first uses them (M0: `llm_calls`);
  `gateway.tool_loop` arrives with the researcher in M3 and fault hooks in M5, as DESIGN §14
  assigns them. v1 code and v1 documents are removed from this branch (Ewan approved); both stay
  at tag `baseline-2026-10-07`.
- Consequence: no empty placeholder modules; every module on the branch is exercised by a test.

## D14 — Prompt version = file name + content hash (2026-10-07)
- Context: DESIGN §6.4 requires a version tag per prompt in traces; hand-written tags go stale.
- Decision: the version is `<name>@<first 8 hex of sha256(text)>`, computed when the file is loaded.
  The static part of a `Prompt` is therefore one `Instructions` object (a prompt file) rather than a
  list of blocks, as first sketched in DESIGN §6.3; the output schema travels in `text.format`.
- Consequence: a trace or ledger row always identifies the exact prompt text, and a prompt cannot
  be built from an inline string; renaming a prompt is the only way to change its name.
