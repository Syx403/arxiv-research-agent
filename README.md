# ARA — Agentic Research Assistant

ARA is a conversational arXiv research agent. It discovers relevant papers from live arXiv searches, explains the matches using abstracts, and retrieves original evidence for deeper questions. One LangGraph controller admits actions, observes progress, and stops on evidence, time or cost limits.

## Start

Python 3.11+, Docker Desktop (running), `uv`, and Make are required. Run commands from the repository root. The default local database is exposed on port **5433**, and the browser on **8000**.

On a fresh clone, create `.env` from `.env.example` without overwriting an existing configuration:

```sh
test -f .env || cp .env.example .env
```

Set `DEEPSEEK_API_KEY` for discovery and analysis, plus `OPENAI_API_KEY` and a **Cohere Trial** `COHERE_API_KEY` for full-text retrieval. The application can also load the named ARA DeepSeek credential from the macOS Keychain; `uv run python -m scripts.configure_deepseek_key` provides the local secure setup dialog. OpenRouter and LangSmith are not required for the default workflow. Credentials stay in `.env`/Keychain, not in Git or the browser.

Start the complete local demo:

```sh
make start
```

This installs the locked dependencies, waits for PostgreSQL readiness, applies idempotent schema migrations, and starts the browser server at **http://127.0.0.1:8000**. Keep the terminal running; Ctrl+C stops the web server. `make db-down` stops the database while retaining its volume. To use another browser port: `make start PORT=8001`.

The individual steps remain available:

```sh
make install
make db-up
make db-init
make ui
```

Open http://127.0.0.1:8000. No seed-corpus ingestion is needed. The schema migration preserves existing paper records, chunks and checkpoints. Do not use `db-reset` to apply migrations: it deletes the Docker volume.

## Interview demo

Use **新的研究**, set Main and Fast to **Low** for the initial walkthrough, then try:

1. `找到 Agentless（arXiv:2407.01489v1），只基于摘要，用两句话解释研究目标。`
2. `阅读这篇论文的原文，解释它如何逐级定位需要修改的代码，再生成补丁；只讲方法，不报实验分数。`
3. Expand the sources and **查看本轮过程** to show the resolved paper/version, retrieval, verification, stop reason, and cost.

The first turn demonstrates discovery; the second demonstrates contextual reference resolution and full-text RAG. Previously indexed originals are reused. New papers can require downloads and embedding, and model/network latency varies. A completed search uses abstracts; it does not claim to have read the whole paper.

The Chinese operator guide, including restart and failure checks, is in [docs/INTERVIEW_DEMO.md](docs/INTERVIEW_DEMO.md).

## Command-line entry

Live CLI questions use the **same AgentRuntime, checkpoints, and cumulative cost ledger as the browser**:

```sh
uv run python -m src.cli "Find Agentless and briefly explain its research goal using only the abstract."
# Use the UUID printed by the previous command to continue that conversation:
uv run python -m src.cli "Read this paper and explain its localization process." --session <session-uuid>
```

Both roles default to Low in the CLI; `--main-effort high --fast-effort low` changes them. `--json` emits structured output. Results and traces are saved in `data/cli/`. Exit codes: 0 for complete or a clarification question, 2 for partial/incomplete, 1 for a runtime/configuration error.

For a **hand-authored offline output example**, requiring no database, network, or credentials:

```sh
make demo
# Or machine-readable output:
uv run python -m src.cli --example --json
```

The example is visibly labelled; it does not claim a live search or model verification. See [the fixture explanation](docs/evidence-example.md).

## Research flow

```mermaid
flowchart TD
    U[Question and bounded conversation history] --> P[Plan: intent, literal constraints, queries, depth]
    P --> C[Deterministic research controller]
    C -->|Material ambiguity| N[Ask for clarification]
    C -->|New discovery| A[Live arXiv API; official HTML fallback]
    A --> C
    C -->|New candidates| J[Abstract relevance and coverage assessment]
    J --> C
    C -->|Recover search| A
    C -->|Brief paper discovery| L[Up to five papers and abstract explanations]
    C -->|Selected source or deep comparison| R[Acquire versioned originals; reuse material/index]
    R --> C
    C --> T[Scoped hybrid retrieval and evidence selection]
    T --> C
    C -->|Allowed related evidence needed| E[Propose bounded related queries]
    E --> C
    C --> S[Generate with bound evidence]
    S --> V[Check citations and semantic support]
    V --> C
    C -->|Verified, partial or stopped| F[Deliver supported answer and gaps]
```

- Every **new discovery** makes an arXiv HTTP request. There is no local/auto/online mode or user toggle. If the API is unavailable, the adapter can query official arXiv HTML, with separately recorded provenance. A cached old search is never presented as a fresh search.
- Model responsibilities: interpret the request, propose short queries, assess topical relevance and uncertain constraints, select evidence and explain findings. Hard constraints and soft preferences must quote human messages. Earlier assistant recommendations do not become user requirements.
- Program responsibilities: validate identities/schema, select admissible actions, track new candidates, enforce bounds and deliver honest partial results. Empty title searches can broaden to title/abstract; empty conjunctions can relax; crowded pages can switch ordering or paginate. An unchanged pool does not trigger another assessment. Within a turn, unchanged sources already excluded as namesakes, incidental uses or known violations are not rejudged; changed source content invalidates that exclusion. Low-recall compound phrases can relax one modifier before relevance is assessed again.
- An explicitly chosen paper bypasses recommendation screening. A request for implementation, experimental details, or rigorous comparison can trigger original-text reading without the words “read PDF”. A simple discovery request uses abstracts. More than three selected originals triggers a batching clarification.
- Primary papers and directly related mechanism papers are eligible. Incidental usage and mathematical namesakes are excluded unless those are the user's subject. Unknown constraint compliance is recorded as uncertainty. Latest searches prioritize confirmed **first publication**, not update time. No padding to five papers or popularity claims.
- Reading retains RAG: query rewrite → embedding → scoped vector/full-text recall → fusion → reranking → source-sentence selection → sufficiency check. Originals prefer version-pinned arXiv HTML with mathematical alttext preserved; unavailable/invalid HTML falls back to PDF, with provenance recorded. Retrieval is restricted to acquired paper versions. Synthesis and verification use the same source excerpts. An unavailable judge is not a negative relevance verdict.
- Explicit result quantities are enforced after relevance/freshness selection, without narrowing the candidate pool. Focused reading keeps the user's resolved question as the acceptance scope; planner suggestions cannot add unrequested training/inference comparisons. Comprehensive overviews can use a bounded outline.
- One related-paper expansion is allowed for broad/comparative requests when evidence is insufficient, within the same total reading cap. A focused single-paper task stays within its source unless the user requests related work.

## Storage and caching

Postgres stores checkpoints and source material. It is not the boundary of the discoverable literature.

| Material | Identity and reuse |
|---|---|
| Search responses | Stored for diagnostics/replay; new discovery still performs HTTP |
| Exact-version metadata/original | arXiv ID plus `vN`; HTML/PDF format and material hash recorded; unversioned identity lookups refresh metadata |
| Indexed full text | Versioned paper ID plus parser/chunker and embedding model/configuration fingerprint |
| Embeddings | Provider, model, dimension and exact text hash; only missing texts call the API |
| Plan/assessment | Request/history, current date, source contents, prompts/schema and actual model/reasoning settings |

Changing the paper version creates a separate material record. Changing the processing fingerprint creates new chunk IDs; old IDs remain available to historical citations. Retrieval uses the active index for each selected version. Concurrent acquisitions share a PostgreSQL advisory lock. Failed/cancelled work cannot replace an already complete index with a failed one.

## Limits and costs

Defaults: 8 search actions, 60 candidates, 3 semantic assessments, at most 5 result cards, and at most 3 read papers. Search acquisition/assessment shares a 90-second window; a whole deep-reading turn has a 300-second deadline and retains its $0.15 request-admission cost limit. Source checks reserve three seconds for checkpoint/delivery and preserve completed siblings when another check times out. Late semantic retries are not admitted. Individual indexing is capped at 75 seconds and 160 chunks. Original-evidence retrieval has bounded local recovery and answer verification permits at most two rewrites. These are execution bounds, not completeness guarantees.

The UI and CLI use the existing **cumulative US$2.50** ledger at `data/eval_outputs/round1-budget.json`, including all previous requests, with a 3,000-request guard. Each search has an estimated US$0.05 admission ceiling, raised to US$0.15 when original reading is required; the cumulative ceiling always wins. Reservations happen before transmission, and concurrent calls/retries share the ledger. Unknown charges keep their reservations. Estimates are not provider invoices. Cohere is admitted under the configured Trial policy. Do not delete the ledger to bypass limits; inspect remaining allowance before a live demo.

Opening/reloading the page, creating a conversation and offline tests do not call a model. Main (Pro) and fast reasoning can each be changed in **思考设置**; both use Flash. Each turn freezes its settings. More thinking can cost more and need not improve an answer.

## Inspect and test

**查看本轮过程** shows resolved intent, query actions, candidates, decisions, exact-version acquisitions, retrieval and citation checks. Local JSONL traces also preserve visible structured model output and parse errors; hidden reasoning and credentials are not recorded. Incremental UI polling updates affected regions without rebuilding the conversation.

```sh
# Unit tests block live HTTP, including accidental paid calls.
uv run pytest tests/unit -q
# Real database, synthetic acquisition/embedding adapters; no paid APIs.
INTEGRATION_TESTS=1 uv run pytest tests/integration/test_db_bootstrap.py tests/integration/test_ingestion_concurrency.py tests/integration/test_ingest_partial_failure.py tests/integration/test_paper_search_resume.py tests/integration/test_resume.py -q
uv run ruff check src tests scripts
```

The old fixed-corpus gold questions and metrics under `src/eval` are historical diagnostic assets, not an evaluation of this architecture. Their automatic CLI entry was removed. Current research evaluations include a **20-scenario / 26-turn** suite split into calibration and acceptance, plus targeted runtime/structured-output experiments. Passing behavioral checks or a few live examples is not a claim of general research accuracy.

For a Chinese walkthrough, see [RUNTIME_WALKTHROUGH.md](docs/RUNTIME_WALKTHROUGH.md). Historical design/phase reports document earlier versions and do not override this runtime description. Legacy database tables and historical result files are retained; unused concept graph, reflection, citation-walk, HyDE, publisher-specific acquisition and local-corpus runtime branches have been removed.

The research suites and their actual execution results are described in [RESEARCH_EVALUATION.md](docs/RESEARCH_EVALUATION.md); the shared runtime and independent Python evaluation entry are described in [AGENT_RUNTIME.md](docs/AGENT_RUNTIME.md). Mechanical checks and semantic source review remain separate.
