# ARA v2 — Design

Status: approved (2026-10-07). The model allocation and cache design (§6.2–6.3) were confirmed by
Ewan on 2026-10-07 after the M0 verification of external facts (§16). A future section marked
**[proposal]** needs Ewan's confirmation before it is coded.

ARA is an arXiv research agent: it clarifies a research need, finds papers on arXiv, reads the
chosen papers, and answers with sentence-level citations that were checked against the source.
v2 is a rebuild. v1 (tag `baseline-2026-10-07`) is kept for reference only; v1 code is not modified.

---

## 1. Goals, non-goals, constraints

Goals
- Show, through a working personal project, how a mature agent system is built: LangGraph
  orchestration, RAG, an LLM layer, memory and a database — and how it is evaluated.
- Every design choice must be explainable in an interview, with its trade-offs.
- Evaluation is a first-class module: "how do you know it works" must have a measured answer.

Non-goals
- Not production, no real users, no claims of production experience.
- No paper-level experiments: small, honest samples with confidence intervals.
- No Semantic Scholar or other new data providers. arXiv is the only paper source.

Constraints
- Budget: whole v2 (development + evaluation) ≤ US$10. One full evaluation round ≤ US$1
  (tentative; any change is reported to Ewan first).
- API keys: reuse the existing `.env` (DeepSeek, OpenAI, Cohere trial, LangSmith).
  `gpt-6-luna` on the existing OpenAI key is approved.
- Billable batches need approval (see `CLAUDE.md`). No large-scale LLM testing before the
  architecture review gate (§14).
- Code: elegant and concise. Guards only at trust boundaries. No mock frameworks or fake LLMs.
- Product language: English first (inputs in Chinese are accepted).

---

## 2. Product scope and user flows

| Flow | Example | Path |
|---|---|---|
| Clarify | "Find the best paper to cut my agent's cost" | understand → clarify ⏸ → understand |
| Discover | "Recent papers on KV-cache eviction for long-context inference" | understand → discover → respond |
| Discover + read | "Find 2 papers on tool-call scheduling and compare their mechanisms" | understand → discover → choose_papers (⏸ if ambiguous) → read → answer → respond |
| Read named/selected papers | "Read 2210.03629 and explain how it interleaves reasoning and actions" / "read the second one" | understand → read → answer → respond |
| Ask the library | "Which papers did we read about MoE routing, and what did they conclude?" | understand → answer (library scope) → respond |
| Remember / forget | "I only use hosted APIs, no fine-tuning" / "forget that" | any path → remember |

Hard limits per turn: ≤ 5 papers listed, ≤ 3 papers read, ≤ 8 arXiv tool calls, one evidence
requery, one answer repair.

---

## 3. Architecture overview

```
Browser UI (React)  ──SSE──  FastAPI  ──  LangGraph app (AsyncPostgresSaver + AsyncPostgresStore)
                                              │
                    ┌─────────────────────────┼──────────────────────────┐
                 LLM gateway              RAG services                arXiv client
          (gpt-6-luna, deepseek-flash,  (ingest, chunk, embed,      (rate-limited 1 req / 3 s,
           budget ledger, usage,         hybrid search, rerank)      single connection)
           cache-aware prompts)
                    └──────────── PostgreSQL (pgvector + BM25 + checkpoints + store + ledger + eval) ───┘
Tracing: LangSmith (project `ara-v2`).   Evaluation: `evals/` package, results stored in Postgres.
```

Layering rule: LangGraph owns orchestration and state only. Retrieval, scoring and metrics are
plain functions. The LLM gateway is a thin in-house layer over the official `openai` SDK
(DeepSeek is OpenAI-compatible). The evaluation runner is plain Python, not a graph.

Package layout. A module is created in the milestone that first needs it (M0: `settings.py`,
`db/`, `llm/`; M1: `tokens.py`, `rag/`, `arxiv/`, `evals/`; M2: `graph/` read, answer, qa, state;
`rag/retrieve.py`; `evals/prompts/`):

```
ara/
  settings.py   keys from .env, database URL, budget caps, tracing
  tokens.py     cl100k token counting (chunk sizes, cost estimates)
  graph/        app.py (top-level graph), discover.py, read.py, answer.py, routing.py, state.py
  rag/          sources.py (arxiv html/pdf, qasper), chunking.py, embed.py, ingest.py, search.py,
                retrieve.py (stemmed BM25 + dense → RRF → rerank top 30 → top 8 passages)
  llm/          gateway.py, prompt.py, prompts/ (text files), stages.py, pricing.py, ledger.py
  memory/       store.py, extract.py
  db/           migrations/*.sql, migrate.py, pool.py
  arxiv/        client.py (metadata, HTML/PDF fetch; search API in M3)
  api/          server.py, events.py
ui/             React + Vite + TypeScript
evals/          datasets/, suites/, graders/, judges/, prompts/ (data-generation prompts),
                qasper.py, stats.py, runner.py, report.py, cli.py (`ara eval`)
tests/          unit (deterministic code, real Postgres) and live (opt-in, small)
docs/v2/        DESIGN.md, DECISIONS.md, PROGRESS.md
```

---

## 4. LangGraph design

LangGraph 1.2 (1.2.14 checked at M0; API names below were read from the installed package). Each
primitive below is used because the product needs it; nothing is added for decoration.

| Need | Primitive |
|---|---|
| Multi-turn conversation per thread, resumable turns | `AsyncPostgresSaver`, `thread_id` = conversation |
| Ask the user mid-turn (clarification, paper choice) | `interrupt()` + `Command(resume=...)` |
| Parallel per-candidate screening, per-paper reading, per-claim verification | `Send` fan-out + list reducers |
| Phase isolation and per-phase evaluation | subgraphs with their own state schemas |
| Bounded loops (tool calls, requery, repair) | conditional edges + counters + `recursion_limit` |
| Transient failures and slow calls | `add_node(retry_policy=RetryPolicy(...), timeout=TimeoutPolicy(...), error_handler=...)` |
| Cross-session memory | `AsyncPostgresStore` with a pgvector index |
| Live progress in the UI | `astream(..., subgraphs=True, version="v2")` (typed `StreamPart`s) + custom events |
| Per-run configuration and dependencies | `context_schema` / `Runtime` (no globals) |
| Debugging and component replay | `get_state_history`, forking from a checkpoint |

### 4.1 Top-level graph

```
START → load_context → understand ─┬─► clarify ⏸ ──────────────────────────────► understand
                                   ├─► discover ⟦sub⟧ → choose_papers (⏸ if ambiguous) ─┬─► read ⟦sub⟧ → answer ⟦sub⟧ → respond
                                   │                                                    └─► respond
                                   ├─► read ⟦sub⟧ → answer ⟦sub⟧ → respond
                                   ├─► answer ⟦sub⟧ (library scope) → respond
                                   └─► respond
respond → remember → END
```

- `load_context`: resets turn-scoped fields (papers, evidence, answer) so nothing leaks from the
  previous turn; loads the user profile from the Store.
- `understand` (LLM): returns either a `ResearchRequest` or a clarification question.
  Hard constraints must be literal quotes from human messages (v1 rule, enforced by code).
- `clarify`: calls `interrupt(question)`. On resume the node re-runs from its start, so it holds
  no side effects before the interrupt.
- `choose_papers`: deterministic (top direct matches up to the limit); interrupts only when the
  user asked to read but the choice is ambiguous.
- `respond`: renders the final message; `remember` writes memory (§7).

### 4.2 Subgraphs

discover — `researcher ⇄ arxiv_tools` (tool loop, ≤ 8 calls) → `prerank` (embedding similarity of
abstracts to the need; deterministic) → `prewarm` → `screen` × batches of 8 (`Send`) → `rank`.

read — `ingest` × paper (`Send`; idempotent, cached by paper version + pipeline version) →
`gather` × (paper, question) (`Send`: hybrid search → rerank → `select_evidence`) → `collect`
→ at most one requery for reported missing aspects.

answer — `synthesize` → `prewarm` → `verify` × claim (`Send`) → `assemble` (keep only sentences
whose every citation passed) → at most one `repair` → `verify` again → `finalize`.

Answer form (M2, D19): the first line is `Answer: <direct answer> [E…]`, followed by explanation
lines, each ending with its citations. Each cited line is one claim and is verified on its own;
uncited lines are dropped. If the direct answer fails verification the answer becomes the
abstention "Not stated in the provided papers."; with no evidence, `finalize` abstains without
calling a model. `select_evidence` returns sentence labels (`S3`), parsed by regex so a label with
the sentence appended still counts.

### 4.3 State

Graph states are `TypedDict`s; domain objects are Pydantic models.

```python
class ConversationState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    request: ResearchRequest | None
    papers: list[PaperCard]          # turn-scoped
    selected: list[str]              # paper version ids, turn-scoped
    evidence: list[Evidence]         # turn-scoped
    answer: Answer | None            # turn-scoped
    status: Literal["running", "needs_input", "complete", "partial", "failed"]
    stop_reason: str
```

Subgraph states add only what the phase needs (e.g. `DiscoveryState.tool_calls`,
`AnswerState.claims`, `AnswerState.verdicts: Annotated[list[Verdict], operator.add]`).
Cost and usage are not kept in graph state; they live in the ledger (§6.5).

Routing functions are small pure functions of state (unit-tested without any LLM):

```python
def after_understand(s) -> Literal["clarify", "discover", "read", "answer", "respond"]:
    r = s["request"]
    return "clarify" if r.clarification else ROUTE_BY_INTENT[r.intent]
```

### 4.4 Reliability

- One retry layer: node `RetryPolicy` for transient network errors (SDK retries off).
- Per-node timeouts with `add_node(timeout=...)` (async nodes only): ingest 60 s, LLM nodes 45 s,
  arXiv tool 20 s. A timeout raises `NodeTimeoutError`, which the default `RetryPolicy` retries.
- Node error handlers (`add_node(error_handler=...)`; the handler receives a `NodeError` and
  returns a `Command`) turn failures into explicit partial results (`status="partial"`,
  `stop_reason`), never into an exception the user sees.
- Budget: the gateway raises `BudgetExceeded` before sending; the error handler finishes the turn
  with what is already verified.
- Durability: checkpoints after every step; completed `Send` branches are not re-run on resume.

---

## 5. RAG design

Sources
- arXiv: HTML (`arxiv.org/html/<id>v<n>`) first, PDF fallback. Keep the section tree, paragraph
  ids and sentence offsets. Math keeps the LaTeX alttext (v1 `html_loader`).
- QASPER: the dataset's own full text (sections → paragraphs), used by evaluation so retrieval
  is measured against exactly aligned gold paragraphs.

Chunking
- Paragraph-aligned chunks inside one section, target ≤ 400 tokens (cl100k); long paragraphs split
  at sentence boundaries; no chunk crosses a section.
- Each chunk stores its heading path ("Title › Section › Subsection"). The heading path is
  prepended for embedding and BM25 only (a cheap contextual header); display uses raw text.
- LLM-generated chunk context (contextual retrieval) is an optional ablation, not the default.

Indexing
- `text-embedding-3-small` (1536 d). Embeddings are content-addressed (hash of model + text), so
  re-ingesting unchanged text costs nothing.
- `PIPELINE_VERSION` constant (parser + chunker + embedding model). Chunks of other versions are
  never mixed into a search.

Search (scoped to the selected paper versions, or to the user's library)
- BM25 over `chunks.search_text` with ParadeDB `pg_search` 0.26: one ParadeDB index per table
  holding `id`, `document_id` (so the scope filter is pushed down) and the text tokenised twice,
  plain words and English stemming (alias `search_text_stemmed`, D15); queried with `|||` (any
  query token) and ordered by `pdb.score(id)`. The product uses the stemmed field (D15). Native
  full-text search (`ts_rank_cd`, query terms OR-ed) stays as an ablation arm (it has no IDF).
- Dense: exact cosine scan when scoped to ≤ 3 papers (a MATERIALIZED scope keeps the planner off
  any vector index); HNSW for library-wide search (pgvector ≥ 0.8 iterative scans for filtered
  queries), added with library questions (M4).
- Reciprocal rank fusion, k = 60, top 50 from each list → Cohere rerank (`rerank-v4.0-pro`,
  trial key) top 30 → top 8.
- `select_evidence` (LLM) returns sentence ids from those chunks plus missing aspects; this merges
  v1's separate relevance and sufficiency calls into one structured call.

---

## 6. LLM layer

### 6.1 Models (prices checked 2026-10-07, see §17)

| Model | Input | Cached input | Cache write | Output | Notes |
|---|---:|---:|---:|---:|---|
| `gpt-6-luna` (OpenAI) | $0.10 | $0.01 | $0.125 | $0.50 | strict structured outputs; effort none…max; Batch API −50% |
| `deepseek-flash` (V4.1 Flash) | $0.30 peak / $0.15 off | $0.006 / $0.003 | — | $1.20 / $0.60 | thinking mode (effort low / high / max); JSON output is `json_object` only (no schema); schema-enforced output only through strict tool calls (beta endpoint) |

DeepSeek peak = 01:00–04:00 and 06:00–10:00 UTC on weekdays, i.e. 09:00–12:00 and 14:00–18:00
Singapore time. Paid evaluation rounds should run off-peak.

### 6.2 Stage allocation (confirmed 2026-10-07, D10)

Principles
1. Machine-consumed decisions go to `gpt-6-luna` with strict JSON schemas: no repair code.
2. Open-ended generation (tool loop, user-facing prose) goes to `deepseek-flash`.
3. The checker differs from the writer: answers are written by DeepSeek and verified by Luna.
4. Judge rule: an evaluation judge is never from the same family as the component that produced or
   selected the graded artifact.
5. Calls that share a large prefix use the same model (caches are per model).

| Stage | Model | Effort | Output | Fan-out |
|---|---|---|---|---|
| understand (intent, constraints, clarification) | gpt-6-luna | low | strict schema | 1 |
| researcher (arXiv tool loop) | deepseek-flash | thinking, low | tool calls | ≤ 8 calls |
| screen candidates | gpt-6-luna | medium | strict schema | batches of 8 |
| select_evidence | gpt-6-luna | low | strict schema | per (paper, question) |
| synthesize / repair | deepseek-flash | thinking, high | prose with `[E#]` citations | 1 (+1) |
| verify | gpt-6-luna | medium | strict schema | per claim, after prewarm |
| remember | gpt-6-luna | low | strict schema | 1 |
| judge: answers (written by DeepSeek) | gpt-6-luna, Batch API | medium | strict schema | offline |
| judge: paper relevance (selected by Luna) | deepseek-flash, off-peak | thinking, high | `json_object` + Pydantic validation | offline |

- DeepSeek accepts only low / high / max (medium maps to high). The researcher uses low because each
  of its up to 8 steps thinks before acting; high would lengthen every turn (to be measured in M3).
- The paper-relevance judge has no schema enforcement on DeepSeek. Its output is validated with
  Pydantic; a parse failure is recorded as `judge_error` and counted in the report, never repaired.
  Strict tool calls (beta) remain the fallback if the error rate is material (decided in M3).
- Every stage has a fixed `max_output_tokens` (reasoning included): DeepSeek's default in thinking
  mode is 64K, and the ledger reserves the maximum before sending (§6.5).

Known trade-off: the answer judge and the in-product verifier are both Luna, so their blind spots
may correlate. Mitigation: answer quality is graded against QASPER gold answers and gold evidence
(reference-based), and the judge is calibrated on Ewan's labels (§11.4). The verifier model itself
is chosen by measurement on the verifier suite (S5 compares Luna and DeepSeek, about $0.05).
  Measured in M2 (D20): both at F1 1.00 on held-out; Luna kept.

### 6.3 Cache-aware prompts (confirmed 2026-10-07, D11)

Every prompt is three ordered parts:

```python
@dataclass(frozen=True)
class Prompt:
    instructions: Instructions  # static: one versioned prompt file (the schema goes in text.format)
    shared: tuple[Block, ...]   # context shared by a group of calls: evidence pack, request, transcript
    item: tuple[Block, ...]     # the part that differs per call
```

- The renderer puts static → shared → item, one message per block, serialises JSON with sorted
  keys and stable ordering, and never puts timestamps or run ids before the item part. Static
  instructions go into a developer message, because the top-level `instructions` field cannot
  carry a cache breakpoint.
- OpenAI calls always use `prompt_cache_options.mode = "explicit"`. The stage table lists which
  parts end with a breakpoint (`"prompt_cache_breakpoint": {"mode": "explicit"}` on the last
  `input_text` block of that part): static and shared for fan-out stages (screen, select_evidence,
  verify); static, shared and item for understand, whose new message is part of the next turn's
  prefix. Reason: the default implicit mode writes every request's tail to the cache at 1.25× the
  input price, which is waste for fan-out items that are used once.
- Fan-outs warm the shared prefix first with a separate OpenAI request carrying
  `prompt_cache_options.prewarm = true` (no output; written tokens billed at the cache-write rate).
  It uses exactly the model, schema (`text.format`) and effort of the calls that follow, otherwise
  the prefix differs. Prewarm only when the fan-out has ≥ 2 calls and the shared prefix is
  ≥ 1,024 tokens (`worth_prewarming`): one write at 1.25× plus N reads at 0.1× is cheaper than
  paying for the prefix N times (at 1×, or 1.25× when every concurrent call writes it) from N = 2.
- DeepSeek has no warm-up step. Under V4.1 caching a request creates cache units at the end of its
  input and of its output, and a later request hits only by fully matching a unit; a common prefix
  is stored on its own only after two requests share it (long inputs also get fixed-interval
  units). Sending one item first would therefore not warm the others. DeepSeek stages are
  sequential and append-only (researcher loop, synthesize → repair), which hit naturally; the
  offline DeepSeek judge accepts misses (off-peak input is $0.15 per 1M).
- Conversations stay append-only. Compaction happens only at a turn boundary, by appending a
  summary note, never by rewriting earlier messages.
- Each stage keeps a fixed tool list, schema and reasoning effort; changing any of them breaks
  the cached prefix.
- OpenAI cache lifetime is 30 minutes after the last write or reuse (`ttl: "30m"`, the only value).
  `prompt_cache_key` is not used: from GPT-5.6 on, routing is automatic and the key only separates
  cache accounting.

| Stage | Shared prefix that should hit |
|---|---|
| understand | instructions + user profile + earlier conversation |
| researcher | tool definitions + instructions + growing transcript |
| screen | instructions + the research request (all batches) |
| verify | instructions + the answer's evidence pack (all claims) |

The gateway records `usage.input_tokens_details.cached_tokens` / `cache_write_tokens` (OpenAI)
and `usage.prompt_cache_hit_tokens` / `prompt_cache_miss_tokens` (DeepSeek). The UI and the eval
report show cache-hit rate per stage. OpenAI caches only prefixes of ≥ 1,024 tokens; short prompts
are not padded.

### 6.4 Gateway

- `gateway.structured(stage, prompt, Schema, scope=...)`, `gateway.text(stage, prompt, scope=...)`
  and `gateway.prewarm(stage, prompt, Schema, scope=...)`. The stage table decides model, effort,
  output cap and cache breakpoints; each call site always passes the same schema (prewarm needs it
  because `text.format` is part of the cached prefix); `scope` names the evaluation run and the
  turn whose budget caps apply. `worth_prewarming(prompt, Schema, calls)` applies the §6.3 rule.
  `gateway.tool_loop(...)` serves the researcher (added in M3). `gateway.embed(texts)` and
  `gateway.rerank(query, documents, top_n)` (Cohere v2 REST, calls spaced 6 s for the trial's
  10 per minute) go through the same metering, so embeddings and reranks are in the ledger too.
- Luna: Responses API with strict `json_schema` from the Pydantic model.
  DeepSeek: Chat Completions; `structured` uses JSON mode (`response_format: json_object`) with
  the schema stated in a system message after the static prefix, and the reply validated by
  Pydantic (a failure raises `InvalidOutput`); tool arguments are validated by Pydantic. With tools in thinking mode
  DeepSeek requires every earlier `reasoning_content` to be sent back (HTTP 400 otherwise), which
  also keeps its cached prefix intact.
- SDK retries are off (`max_retries=0`); the node `RetryPolicy` is the only retry layer.
- Prompts are files under `ara/llm/prompts/`. The version tag recorded in traces and in the ledger
  is the file name plus a short hash of its content, so it cannot go stale.

### 6.5 Budget ledger

- Postgres table `llm_calls`. Before a call: reserve an upper bound (input estimate + max output);
  after: settle with reported usage; unknown outcomes keep the reservation (v1 rule).
- Caps: global $10, per eval round (`--max-usd`), per turn (default $0.05). A transaction-scoped
  advisory lock serialises reservations across processes.
- Prices live in `ara/llm/pricing.py` with the date they were checked. DeepSeek bills this account
  in CNY; the ledger uses its USD price list, so DeepSeek costs are close estimates. Peak hours are
  applied to every weekday (Chinese public holidays are off-peak in reality), which errs high.

---

## 7. Memory

| Kind | Where | Written by | Read by |
|---|---|---|---|
| Conversation | checkpoints (thread state) | graph | every turn (append-only) |
| Profile facts (interests, constraints, preferences) | Store `("users", uid, "profile")` | `remember` | `understand`, `synthesize` (shared prompt part) |
| Episodes (one summary per research turn) | Store `("users", uid, "episodes")`, semantic index | `remember` | `understand` when the user refers to earlier work |
| Library (papers read or saved) | SQL `library_items` | `read`, UI | library questions, UI |

- A profile fact must quote the user literally and keep its source turn (v1 rule). A newer fact
  with the same key supersedes the old one; "forget" deletes it.
- Structured data (library) lives in tables, not in vector memory.

---

## 8. Database

PostgreSQL 18.6 with pgvector 0.8.6 and pg_search 0.26.0: image `paradedb/paradedb:0.26.0-pg18`,
pinned (verified at M0). One database (`ara`; unit tests use `ara_test` on the same server):

| Table | Purpose / key points |
|---|---|
| `papers` | arXiv id, version, metadata; `source` ∈ {arxiv, qasper} |
| `documents` | one parsed edition per (paper version, `PIPELINE_VERSION`); unique; written with its chunks in one transaction, so a row means complete (D16) |
| `chunks` | `document_id`, `ord`, `heading_path`, `text`, `search_text`, sentence offsets, `embedding vector(1536)`, `tsv` (generated) |
| `embedding_cache` | content hash → vector |
| `library_items` | user, paper version, status, note |
| `llm_calls` | ledger + usage + cache tokens + latency + stage + turn/run ids |
| `eval_runs`, `eval_results`, `labels` | evaluation outputs and Ewan's labels |
| LangGraph tables | checkpoints, store (created by `setup()`) |

Indexes: HNSW on `chunks.embedding` (M4); the ParadeDB (BM25) index on `chunks(id, document_id,
search_text ×2 tokenizers)`; GIN on `chunks.tsv`; btree on `chunks(document_id)`, `llm_calls(turn_id)`,
`llm_calls(run_id)`, `eval_results(run_id)`.
Ingestion takes a per-paper advisory lock and writes a document in one transaction.
Migrations: numbered SQL files and a ~30-line runner. Each table arrives with the milestone that
first uses it (M0: `llm_calls`; M1: corpus tables, `embedding_cache`, `eval_runs`, `eval_results`).

---

## 9. API and UI

Delivery form: a local web app. One command (`make start`) starts the Docker database, applies
migrations, builds the UI if needed and runs FastAPI; then open **http://127.0.0.1:8000** in a
browser. It binds to 127.0.0.1 only, single user, no login (same form as v1). The UI does not
depend on LangSmith or LangGraph Studio and works with tracing turned off; Studio is only an
optional developer tool. Requirements: Docker, uv (Python), Node (only to build the UI; v26.8.2
is installed). During development `make ui-dev` runs the Vite dev server with hot reload and
proxies API calls to FastAPI.

UI purpose: show the work in interviews. Priority: (1) workflow, (2) evidence, (4) evals,
(3) memory (optional).

| Priority | Page | Content |
|---|---|---|
| 1 | Workflow + chat | Chat; the live graph rendered from the compiled LangGraph (`get_graph(xray=True)` → Mermaid) with the active node highlighted; a step timeline (node, model, latency, tokens, cache-hit %, cost); interrupt cards (clarification, paper choice) |
| 2 | Evidence | Answer with citation chips → source viewer with the cited sentences highlighted and the verifier's verdict; paper cards with fit and constraint checks |
| 4 | Evaluation | Rounds, metrics with CIs per suite, per-item table, cost, LangSmith trace links |
| 3 | Memory | Profile facts with their quotes, library, forget |

The diagram is generated from the compiled graph, so it always matches the code.

API (FastAPI + SSE): `POST /threads`, `POST /threads/{id}/messages` (SSE run stream),
`POST /threads/{id}/resume`, `GET /threads/{id}`, `GET /graph`, `GET /papers/{id}/document`,
`GET /evals`, `GET /evals/{run}`, `GET /memory`, `DELETE /memory/{key}`.
Frontend: React + Vite + TypeScript + mermaid.js, served by FastAPI. A demo mode uses
pre-ingested papers so a live demo turn stays short.

---

## 10. Observability

- LangSmith tracing for graph runs and gateway calls (project `ara-v2`); metadata on every LLM
  span: stage, model, prompt version, cached / written / uncached tokens, cost.
- Free plan: 5k traces per month, 14-day retention. Evaluation results are therefore stored
  locally (Postgres, plus a JSONL export and a markdown report per run in `data/eval_reports/`;
  reports cited by a decision are copied to `docs/v2/eval/`); LangSmith is for inspection and
  experiment comparison.

---

## 11. Evaluation

### 11.1 Principles
- Small, tiered suites; grade outcomes rather than paths; code graders first.
- LLM judges only where needed, calibrated against Ewan's labels; judge rule in §6.2.
- Ablations are paired (same items); report n, mean and a 95% bootstrap CI. At n = 30 and a
  70% pass rate, the CI is about ±16 points, so results are directional, not accuracy claims.
- Tune on dev items, report on held-out items. Failures from real use become new items.

### 11.2 Suites (one full round ≈ $0.7 estimated, hard cap $1)

| Suite | Data | n | Metrics | Graders | Est. cost |
|---|---|---|---|---|---:|
| S1 retrieval | QASPER validation, dataset full text | 10 papers × 3 questions | evidence recall@k, MRR, nDCG@10 for BM25 / dense / RRF / RRF+rerank / native FTS | code | ≈ $0 (≈ 30 rerank calls) |
| S2 reading QA | the 30 S1 questions + 5 unanswerable ones drawn (seed 20261008) from other validation papers, since the S1 papers have none (D19) | 30, plus 5 × 3 trials | answer F1 (extractive, yes/no), judged equivalence to gold (free-form), abstention P/R, citation precision vs gold evidence, latency, $ | code + Luna judge | ≈ $0.29 |
| S2-baselines | same items | 30 each | closed-book, whole paper in context, naive RAG | same | ≈ $0.05 |
| S3 discovery | PaSa: AutoScholarQuery (dev 15), RealScholarQuery (test 15) | 15 per round | candidate-pool recall, precision@5 (gold lower bound + adjudicated), hit@5, constraint violations | code + DeepSeek judge + Ewan | ≈ $0.10 |
| S4 understand/clarify | v1 UI questions + edge cases, labeled by Ewan | 50 | intent accuracy, false-clarify, missed-clarify | code | ≈ $0.02 |
| S5 verifier | QASPER evidence (one sentence per S1 item); 30 DeepSeek paraphrases, 30 perturbed: number 8 and negation 7 by code, entity 8 and over-generalisation 7 by DeepSeek; reviewed by Ewan; 58 after review (D19, D20) | 58 | P/R/F1 on "unsupported", recall per kind; Luna vs DeepSeek | code | $0.0093 measured (D20) |
| S6 multi-turn + memory | scripted scenarios | 6 × ~3 turns | assertion pass rate (reference resolution, constraint retention, update, forget, abstain) | code | ≈ $0.13 |
| S7 robustness | fault hooks + one prompt-injection document | 6 + 3 turns | graceful-degradation rate, injection success (must be 0) | code | ≈ $0.02 |

Efficiency is reported for every suite: requests, tokens, cache-hit rate, $/task, latency p50/p95.
Costs are estimates from v1 unit costs and §6.1 prices; they are re-measured after the M2 smoke
run and reported to Ewan.

### 11.3 Discovery metrics
The product lists ≤ 5 papers but PaSa gold sets average 15.8 (RealScholarQuery), so recall@20 does
not apply. Search quality = gold recall in the candidate pool. Selection quality = precision of
the final list. Gold-based precision is a lower bound (gold is pooled, not exhaustive); non-gold
papers in the final list are adjudicated by the judge, and the judge is checked against Ewan's
labels. Search uses the query's `published_time` as the date cutoff.

### 11.4 Judges and labels
- Ewan labels about 40 answer-equivalence items and 30 paper-relevance items. Cohen's κ is
  reported; κ ≥ 0.6 is the working bar. The judge prompt is tuned on these items only, then frozen
  (version recorded) before round E1.
- Answer judging uses the OpenAI Batch API (−50%); results arrive asynchronously.

### 11.5 Runner
- `ara eval plan <suites>`: dry run; prints item counts, expected requests and estimated cost.
- `ara eval run <suites> --execute --max-usd 1.0`: runs through LangSmith `aevaluate`, writes
  `eval_runs` / `eval_results`, stops at the cap and reports what was not run.
- `ara eval report <run>`: markdown report (held-out and dev items separately, each with paired
  differences between arms, D18) + JSONL export + the UI evaluation page.
- Datasets: manifests (item ids, seed, source hash) are committed under `evals/datasets/`;
  raw data stays in git-ignored `data/datasets/` (PaSa is gated, CC BY-NC-SA 4.0; QASPER is
  CC BY 4.0).

---

## 12. Testing (no mocks)

- Unit tests cover deterministic code against a real Postgres: chunking, search SQL, RRF,
  routing functions, prompt rendering (byte-stable prefixes), ledger, metrics.
- Live tests (`-m live`, opt-in) call the real models. Each is small; batches above the approval
  threshold need Ewan's approval.
- No fake LLM classes, no recorded cassettes. Failure paths are exercised through small
  environment-driven fault hooks in the gateway and arXiv client.

---

## 13. Reuse from v1 (reference only, tag `baseline-2026-10-07`)

| Idea | v1 location |
|---|---|
| Reserve-before-send ledger, unknown outcomes keep the reservation | `/Users/richsion/Desktop/arxiv-research-agent/src/llm/budget.py` |
| Per-paper advisory lock, transactional ingestion | `/Users/richsion/Desktop/arxiv-research-agent/src/corpus/ingestion.py` |
| HTML parsing that keeps math alttext | `/Users/richsion/Desktop/arxiv-research-agent/src/corpus/html_loader.py` |
| Sentence spans; deliver only verified sentences | `/Users/richsion/Desktop/arxiv-research-agent/src/retrieval/evidence.py`, `/Users/richsion/Desktop/arxiv-research-agent/src/graph/delivery.py` |
| Constraints must be literal user quotes; ids must come from input/history | `/Users/richsion/Desktop/arxiv-research-agent/src/retrieval/paper_search.py`, `/Users/richsion/Desktop/arxiv-research-agent/src/graph/conversation.py` |
| Deterministic search recovery (relax phrase, widen scope, next page) | `/Users/richsion/Desktop/arxiv-research-agent/src/graph/controller.py` |
| Dry-run-by-default eval runner with explicit spend cap | `/Users/richsion/Desktop/arxiv-research-agent/scripts/eval_agent.py` |

Not reused: the central router function, the flat 60-field state, inline prompts with case-specific
patches, the JSON-file ledger, hand-written asyncio fan-out inside nodes, layered retries,
Chinese strings inside logic.

---

## 14. Milestones

Development may call real models in small amounts (threshold in `CLAUDE.md`). Large-scale LLM
testing starts only after the architecture review.

| Milestone | Scope | Done when |
|---|---|---|
| M0 Foundation | v2 skeleton (v1 code and documents removed from this branch), Docker DB, migrations, settings, gateway (both providers, prompt rendering, usage, ledger), LangSmith, CI (ruff, mypy, pytest) | unit tests pass; one live call per provider shows cache tokens on the repeated call |
| M1 RAG + S1 | sources (arXiv, QASPER), chunking, embeddings, BM25 + dense + RRF + rerank; eval runner + datasets + S1 | S1 runs end to end on 2 papers (live, approved) |
| M2 Read + answer | read and answer subgraphs, prewarmed verification, finalize; S2, S5 | two live questions answered with verified citations |
| M3 Understand + discover | understand, clarify interrupt, researcher loop, screen, choose_papers; S3, S4 | a live discover → read turn and a clarify turn |
| M4 Memory | Store namespaces, remember, library, forget; S6 | a scripted cross-session scenario passes live |
| M5 Reliability + UI | retry/timeout/error handlers, fault hooks, S7; UI pages 1, 2, 4 (3 optional); README | `make start` → http://127.0.0.1:8000 shows pages 1, 2, 4 and a demo turn works |
| **Gate** | **Ewan reviews the architecture** | approved |
| E1–E6 | judge calibration, then baseline, comparison, iteration and final rounds (≤ $1 each) | results reported with CIs |
| Interview | facts / stories / Q&A in the interview hub | — |

---

## 15. Budget plan (US$10 total)

| Item | Cap |
|---|---:|
| Development smoke runs (M0–M5) | 1.0 |
| Data preparation and judge calibration | 0.5 |
| Evaluation rounds E1–E6 | 6.0 |
| Reserve | 2.5 |

---

## 16. Verified at M0 (2026-10-07)

1. ParadeDB: `paradedb/paradedb:0.26.0-pg18` (arm64 and amd64) bundles PostgreSQL 18.6,
   pgvector 0.8.6 and pg_search 0.26.0. BM25 (`USING paradedb`, `|||`, `pdb.score`), exact and
   HNSW vector search (`hnsw.iterative_scan`) and native `ts_rank_cd` were run in a throwaway
   container. `key_field` is deprecated and ignored since 0.26.0.
2. `gpt-6-luna` is listed for the existing OpenAI key (`GET /v1/models/gpt-6-luna`). Parameter
   names, from SDK 3.26 types and the caching guide: `prompt_cache_options.{mode, prewarm, ttl}`,
   `prompt_cache_breakpoint: {"mode": "explicit"}` on `input_text` blocks (up to 4 written per
   request), usage `input_tokens_details.{cached_tokens, cache_write_tokens}`. The guide says
   "GPT-5.6 and later" without naming Luna; the M0 live check confirmed it: a prewarm wrote 2,690
   tokens and the following call read 2,690 of its 2,710 input tokens from the cache. A breakpoint
   before the 1,024-token minimum (the short static part) is accepted without error.
3. DeepSeek: `deepseek-flash` is V4.1-Flash (released 2026-09-10), listed for the key. Thinking:
   `extra_body={"thinking": {"type": "enabled"}}` (default on) and `reasoning_effort` low / high /
   max. `response_format` supports only `json_object`; strict tool calls need
   `https://api.deepseek.com/beta`. M0 live check: an appended second turn read 2,560 of its 2,795
   input tokens from the cache. 2,560 is a multiple of 256 below the first turn's 2,754-token input,
   so the hit most likely came from a fixed-interval cache unit (inferred, not documented).
4. LangGraph 1.2.14: `add_node(timeout=float | TimeoutPolicy(run_timeout, idle_timeout))`
   raising `NodeTimeoutError`; `add_node(error_handler=...)` with `NodeError(node, error)`;
   `astream(..., version="v2")` yielding `StreamPart`; `RunControl` / `GraphDrained` for shutdown.
5. Cohere: the trial key is valid and lists `rerank-v4.0-pro`. Responses carry
   `x-endpoint-monthly-call-limit: 1000` but no remaining count; the remaining monthly calls are
   visible only on the Cohere dashboard (to be checked by Ewan before M1).
6. QASPER: the Hugging Face repo holds only a loading script (`qasper.py`, v0.3). The data is read
   from the auto-converted Parquet (`refs/convert/parquet`, validation = 281 papers, 4.7 MB) with
   pyarrow. Fields: `id` (arXiv id), `full_text.{section_name, paragraphs}`,
   `qas.answers[].answer[].{unanswerable, extractive_spans, yes_no, free_form_answer, evidence,
   highlighted_evidence}`. The original AllenAI S3 archive is still online as a fallback.

---

## 17. External facts used (checked 2026-10-07)

- gpt-6-luna: $0.10 input, $0.01 cached, $0.125 cache write, $0.50 output per 1M tokens;
  1.05M context; reasoning effort none/low/medium/high/xhigh/max; Responses, Chat Completions and
  Batch. https://developers.openai.com/api/docs/models/gpt-6-luna
- OpenAI prompt caching (≥ 1,024 tokens; explicit breakpoints and prewarm for GPT-5.6+; cache
  writes 1.25×, reads 0.1×; tools, schema and reasoning effort are part of the prefix).
  https://developers.openai.com/api/docs/guides/prompt-caching
- DeepSeek pricing, models `deepseek-flash` and `deepseek-v4-pro`, peak hours.
  https://api-docs.deepseek.com/quick_start/pricing
- DeepSeek context caching (default on, best effort, built within seconds; V4.1 cache units must
  be matched in full). https://api-docs.deepseek.com/guides/kv_cache
- DeepSeek V4.1-Flash release, 2026-09-10 (`deepseek-v4-pro` now routes to V4.1-Flash).
  https://api-docs.deepseek.com/news/news260910
- DeepSeek thinking mode, JSON output, strict tool calls (beta).
  https://api-docs.deepseek.com/guides/thinking_mode, https://api-docs.deepseek.com/guides/json_mode,
  https://api-docs.deepseek.com/guides/tool_calls
- ParadeDB 0.26.0 (2026-10-03; `key_field` requirement removed; `USING paradedb`).
  https://github.com/paradedb/paradedb/releases/tag/v0.26.0,
  https://www.paradedb.com/docs/reference/indexing/create-index
- LangGraph 1.1 (typed streaming, 2026-03-10) and 1.2 (node timeouts, error handlers, graceful
  shutdown, 2026-05-12). https://docs.langchain.com/oss/python/releases/changelog.md
- arXiv API: at most one request every three seconds, one connection.
  https://info.arxiv.org/help/api/tou.html
- Cohere rerank trial: 10 calls per minute; trial keys 1,000 calls per month.
  https://docs.cohere.com/docs/rate-limits
- Cohere rerank v2 (checked 2026-10-08): `POST /v2/rerank` with `model`, `query`, `documents`
  (strings), optional `top_n`; results carry `index` and `relevance_score`.
  https://docs.cohere.com/reference/rerank
- text-embedding-3-small (checked 2026-10-08): $0.02 per 1M input tokens; one request takes at most
  2,048 inputs, 8,192 tokens each and 300,000 tokens in total.
  https://developers.openai.com/api/docs/models/text-embedding-3-small,
  https://developers.openai.com/api/docs/api-reference/embeddings/create
- LangSmith Developer plan: 5k traces per month, 14-day retention.
  https://www.zenml.io/blog/langsmith-pricing
- QASPER (CC BY 4.0; arXiv ids; paragraph and sentence evidence).
  https://huggingface.co/datasets/allenai/qasper
- PaSa datasets (gated, CC BY-NC-SA 4.0; `answer_arxiv_id`, `published_time`).
  https://huggingface.co/datasets/CarlanLark/pasa-dataset
- Anthropic, "Demystifying evals for AI agents" (2026-01-09).
  https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents
