# ARA (arXiv Research Agent) — v2

Personal project by Ewan (Syx403). Purpose: demonstrate, with honest evidence, how a mature
agent system is built and evaluated (LangGraph, RAG, LLM layer, memory, database). It is not
production and must never be described as production.

Read first: `docs/v2/DESIGN.md` (spec), `docs/v2/DECISIONS.md` (why), `docs/v2/PROGRESS.md`
(where we are). v1 is kept only at tag `baseline-2026-10-07`; read it with
`git show baseline-2026-10-07:<path>` or a separate worktree, never revive it wholesale.

## Hard rules

1. Billable APIs (LLM, embedding, rerank): a run of ≤ 3 requests and ≤ US$0.02 may be done and
   reported afterwards. Anything larger needs Ewan's approval first: state the request count and
   estimated cost. Report actual spend after every billable run and log it in `PROGRESS.md`.
2. Budget: whole v2 ≤ US$10 (development + evaluation); one full evaluation round ≤ US$1. If
   either needs to change, tell Ewan before acting.
3. No large-scale LLM testing until Ewan has reviewed the architecture (DESIGN §14 gate).
4. No mocks: no fake LLM classes, no recorded cassettes. Unit tests cover deterministic code
   against real Postgres; LLM behaviour is checked with small live tests.
5. Code style: elegant and concise. Validate at trust boundaries (LLM output, external APIs, user
   input) only; no scattered defensive checks; small functions; prompts live in files.
6. Never silently change something agreed in `DESIGN.md` / `DECISIONS.md`. If a deviation is
   needed, say so at that moment, record a new decision, and repeat it in the session summary.
7. Never fabricate when context is long: re-read the docs or ask.
8. Git: work on `v2`. Do not merge into `main` or push to `main` unless Ewan asks.
9. Keys come from the existing `.env`; never print or commit secrets. Do not add new providers
   (no Semantic Scholar).
10. Datasets: raw data stays in git-ignored `data/datasets/`; PaSa is gated (CC BY-NC-SA 4.0) and
    must not be committed or redistributed.

## Working with Ewan

- Talk to Ewan in Chinese; repository code, docs and UI are in English.
- Cite code as absolute paths, e.g. `/Users/richsion/Desktop/arxiv-research-agent/ara/graph/app.py:12`.
- Flow per milestone: confirm scope → implement → small live check (rule 1) → update
  `PROGRESS.md` → summarise what changed, what was spent and any deviations.
- Ewan's known gaps (databases, hand-written code fluency): explain DB and LangGraph choices in
  plain terms; keep the code readable enough for him to defend line by line.

## Commands

- `make install` — `uv sync --locked` (Python 3.13).
- `make db-up` / `make db-down` — v2 database container `ara-v2-db` (ParadeDB 0.26, PostgreSQL 18) on
  127.0.0.1:5434 with databases `ara` (app, ledger) and `ara_test` (unit tests). v1's container on
  5433 is separate; leave it alone.
- `make migrate` — apply `ara/db/migrations/*.sql` to `ara`; unit tests migrate `ara_test` themselves.
- `make check` — ruff format check, ruff lint, mypy (strict), unit tests. Same steps run in CI
  (`.github/workflows/ci.yml`, on pushes to `v2`).
- `make test-live` — billable (rule 1): real model calls, recorded in `llm_calls` and traced in
  LangSmith project `ara-v2`. Each live test module sets a run cap equal to its approval.
- Spend so far (`charge_usd` is what counts against the caps; it includes open reservations):
  `docker exec ara-v2-db psql -U ara -d ara -c "SELECT run_id, count(*), sum(cost_usd) AS spent, sum(charge_usd) AS charged FROM llm_calls GROUP BY 1 ORDER BY min(created_at)"`.
- Run a single live test module with `uv run pytest -m live <path>`; `make test-live` runs all of
  them, and each run needs its own approval when it exceeds the rule-1 threshold.
- Evaluation: `uv run ara eval prepare` (QASPER into `data/datasets/qasper/`, S1 manifest), `uv run
  ara eval plan s1 [--papers N]` (free dry run: requests and cost), `uv run ara eval run s1 --papers N
  --execute --max-usd X` (billable, needs approval), `uv run ara eval report <run_id>` (also written
  to `data/eval_reports/`).
- API server and UI: added in M5.
