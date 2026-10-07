# ARA v2 — Progress

Update this file at the end of every working session: what was done, what was spent, what is next.

## Milestones

| Milestone | Status | Notes |
|---|---|---|
| Design | done (2026-10-07) | `DESIGN.md`, `DECISIONS.md`, root `CLAUDE.md` |
| Data: PaSa query sets | done (2026-10-07) | `data/datasets/pasa/` (git-ignored, checksums in its README) |
| M0 Foundation | done (2026-10-07) | §16 verified; §6.2–6.3 confirmed (D10, D11); 28 unit tests on real Postgres; live check passed (cache hits on both providers) |
| M1 RAG + S1 | not started | |
| M2 Read + answer | not started | |
| M3 Understand + discover | not started | |
| M4 Memory | not started | |
| M5 Reliability + UI | not started | |
| Architecture review (Ewan) | — | gate before any large-scale LLM testing |
| E1–E6 evaluation rounds | — | ≤ US$1 each |

## Spend (US$10 cap)

| Date | Session | Purpose | Requests | US$ | Approved by |
|---|---|---|---:|---:|---|
| 2026-10-07 | design | none (no billable calls) | 0 | 0.00 | — |
| 2026-10-07 | M0 verification | free metadata endpoints only (model lists, key checks, balance) | 0 | 0.00 | — |
| 2026-10-07 | M0 live check | Luna prewarm + call; DeepSeek turn + appended turn (`m0-live-20261007T125043`) | 4 | 0.0009 | Ewan (≤ 4 requests, ≤ US$0.005) |
| | | **Total so far** | 4 | **0.0009** | |

## Session log

- 2026-10-07 — Design session (interview-hub conversation). Created branch `v2`, wrote the design
  documents, downloaded the PaSa query sets. Next: start M0 in a new session rooted at this repo.
- 2026-10-07 — M0 session, part 1. Verified the six §16 items (DESIGN §16 has the results).
  Ewan confirmed §6.2 with two refinements (D10) and §6.3 with three corrections (D11); also
  recorded D12 (ParadeDB 0.26 / PostgreSQL 18, v2 DB on 127.0.0.1:5434), D13 (M0 sequencing) and
  D14 (prompt version = name + content hash). The `.env` DeepSeek key was invalid; Ewan entered a
  new one through a native masked dialog and it was written to `.env` unseen (`/models` → 200).
  DeepSeek balance is CNY 11.62 (about US$1.6), likely short for E1–E6. Approved for M0: the live
  check (4 requests, ≤ US$0.005), pushing `v2` to GitHub for CI, removing v1 documents. Blocked:
  the auto-mode classifier refused the v1 removal command, so Ewan runs it himself; M0 code
  continues after that. Open: Ewan checks Cohere's remaining monthly calls on the dashboard.
- 2026-10-07 — M0 session, part 2. Ewan ran the v1 removal (303 files; `data/eval_outputs` only
  untracked). Built the foundation: `ara/settings.py`; `ara/db` (numbered SQL migrations with a
  30-line runner, `0001_ledger` = extensions + `llm_calls` with a generated `charge_usd` column,
  async pool); `ara/llm` (Prompt / Instructions / Block with byte-stable OpenAI and DeepSeek
  renderers and explicit breakpoints, the D10 stage table, prices with DeepSeek peak hours,
  reserve → settle ledger under an advisory lock with total / turn / run caps, and the gateway:
  `structured` + `prewarm` on Luna via the Responses API, `text` on DeepSeek thinking mode, every
  call metered and traced in LangSmith `ara-v2`). Docker DB `ara-v2-db` on 127.0.0.1:5434,
  Makefile, CI workflow, v2 README stub. Live check (4 requests, US$0.0009 of the approved
  US$0.005): Luna prewarm wrote 2,690 tokens and the call read 2,690 / 2,710 from cache; the
  DeepSeek follow-up read 2,560 / 2,795; all four runs visible in LangSmith with the ledger's
  token counts and costs. Deviation recorded: `Prompt.static` became one `Instructions` object
  (D14). CI: the first run failed because setup-uv has no moving `v10` tag (pinned v10.2.0);
  the second run passed (ruff, mypy, 28 unit tests against the ParadeDB service).
  Next: M1 (RAG + S1), after Ewan's review of M0 and the Cohere quota check.
