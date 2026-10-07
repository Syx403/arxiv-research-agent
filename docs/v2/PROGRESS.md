# ARA v2 — Progress

Update this file at the end of every working session: what was done, what was spent, what is next.

## Milestones

| Milestone | Status | Notes |
|---|---|---|
| Design | done (2026-10-07) | `DESIGN.md`, `DECISIONS.md`, root `CLAUDE.md` |
| Data: PaSa query sets | done (2026-10-07) | `data/datasets/pasa/` (git-ignored, checksums in its README) |
| M0 Foundation | done (2026-10-07), reviewed | §16 verified; §6.2–6.3 confirmed (D10, D11); 29 unit tests on real Postgres; live checks passed (cache hits on both providers; a rejected request is released) |
| M1 RAG + S1 | done (2026-10-08) | sources (arXiv HTML/PDF, QASPER), chunking, cached embeddings, ingestion, BM25 ×2 / FTS / dense / RRF / rerank, eval runner + S1; S1 live on 2 papers (6 items) |
| M2 Read + answer | in progress (2026-10-08) | read + answer subgraphs, S2 suite; live check: 2 questions answered with verified citations; S5 data generated, awaiting Ewan's review; S5 suite not built |
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
| 2026-10-07 | M0 review | one request rejected by OpenAI (HTTP 400), released at $0 (`m0-reject-20261007T131416`) | 1 | 0.00 | rule-1 threshold, reported |
| 2026-10-08 | M1 live check | S1 on 2 papers × 3 questions (`s1-20261007T161240`): 3 embedding requests, 6 Cohere reranks | 9 | 0.0002 | Ewan (≈ 9 requests, ≤ US$0.002) |
| 2026-10-08 | S1 full round | 10 papers × 3 questions (`s1-20261007T162246`): 9 embedding requests, 30 reranks | 39 | 0.0009 | Ewan |
| 2026-10-08 | M2 live check | S2 on 2 questions: first attempt (`s2-20261007T165545`, every label discarded) and the fixed run (`s2-20261007T165742`) | 16 | 0.0021 | Ewan (≈ 30 requests, ≤ US$0.03) |
| 2026-10-08 | S5 data | DeepSeek paraphrases + perturbations, 6 batches (`s5-data-20261007T165950` stopped by the cap after 5; `s5-data-20261007T170445` the last) | 6 | 0.0113 | Ewan (≈ 15 requests, ≤ US$0.01; cap then raised) |
| | | **Total so far** | 75 | **0.0154** | |

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
- 2026-10-07 — M0 strict review (严格审核). Re-read every file against DESIGN, DECISIONS and
  CLAUDE.md. Fixed: D14 overstated that a prompt cannot be built from an inline string (only
  the version is guaranteed); D11's prewarm rule was missing from code (`worth_prewarming`,
  tested); DESIGN §6.4 showed the wrong `prewarm` signature and no `scope`; an unknown-outcome
  call got `settled_at` while still reserved (now only settled and released calls have it);
  DESIGN §5 had fixed an English stemmer that was never agreed (now an M1 choice); the
  `PDB_TUNE=false` choice was undisclosed (now in D12); D12–D14 lacked alternatives; the §6.3
  cost wording was loose; CI runner pinned to ubuntu-24.04; `.gitignore` now covers `.env.*`;
  unused `pricing.CHECKED` removed; a parameter shadowing `breakpoint` renamed; ledger tests
  check stored usage; the CLAUDE.md spend query also shows `charge_usd`. Added a live check that
  a rejected request is released (1 request, HTTP 400, US$0).
- 2026-10-08 — M1. Ewan approved `PDB_TUNE=false`, the S1 selection (seeded draw), measuring both
  BM25 tokenizers, and the live check. Built `ara/rag` (sources: arXiv LaTeXML HTML with math as
  LaTeX, PDF fallback, QASPER; one chunk per paragraph ≤ 400 tokens with sentence offsets;
  content-addressed embedding cache; idempotent ingestion under a per-paper advisory lock; BM25
  plain/stemmed, OR-ed native FTS, exact scoped dense, RRF), `ara/arxiv/client.py` (one connection,
  3 s spacing, HTML then PDF), gateway `embed` and `rerank` (metered), migrations 0002–0003, and
  `evals` (QASPER download + sha256, S1 manifest committed, retrieval graders, bootstrap CIs,
  LangSmith `aevaluate` runner, markdown report, `ara eval prepare|plan|run|report`). ReAct
  (2210.03629v3) parsed from real HTML: 65 paragraphs, 66 chunks, headings and math intact. S1 live
  (6 items, US$0.0002): rrf_rerank best (MRR 0.69, nDCG@10 0.77), stemmed BM25 ≥ plain on every
  metric (D15, provisional). Cohere rerank p95 latency 23 s (corrected in the M1 review: one cold call
  of 27.8 s among 6; the full round's p50 / p95 are 0.5 s / 1.9 s). Deviations recorded in
  D16 (no `documents.status`, HNSW deferred to M4). 43 unit tests. Open: Cohere monthly quota still
  unchecked; next is M2 (read + answer).
- 2026-10-08 — S1 full round (`s1-20261007T162246`, 30 items, 0 failed, US$0.00086). rrf_rerank
  recall@8 1.00, MRR 0.71 [0.60, 0.82], nDCG@10 0.79 [0.70, 0.87]; dense MRR 0.61; stemmed BM25
  0.55; plain BM25 0.51. Paired: stemming never loses top-8 recall but is neutral on ordering;
  rerank is the only clear gain; RRF ties dense. D15 confirmed as D17. Report and paired table in
  `docs/v2/eval/`. Cohere: Ewan reports the trial quota unused this month (1,000 calls); 36 used.
- 2026-10-08 — M1 strict review (严格审核), then fixes approved by Ewan. Fixed: the S1 report
  pooled dev and test and had no paired comparison (now split, paired bootstrap Δ with items
  better/worse; D17 restated as D18: the tokenizer choice holds on dev and held-out, rerank's
  ordering gain holds on dev only, dense is the strongest single arm on held-out); the request
  timeout promised in the M0 review (120 s on all provider clients); rerank spacing skipped after a
  failed call; the LangSmith dataset could keep stale gold (now named by an examples hash); the
  dry-run plan matched papers by arXiv id across sources; BM25 had no tie-break; results were
  written only after the whole run (now per item); the PDF fallback made one paragraph per paper
  (now line-based paragraphs, checked on 2210.03629v3); recall@k counted paragraphs, not chunks;
  the 23 s rerank latency claim; DESIGN §3, §10, §11.5, §17 (embedding price and limits, Cohere v2
  rerank, checked 2026-10-08); JSONL export added. 47 unit tests. No billable calls (the arXiv
  PDF fetch is free). The Cohere quota is resolved (Ewan: unused this month before M1).
- 2026-10-08 — M2, part 1. Ewan chose S2 = S1's 30 + 5 seeded unanswerable questions, the answer
  form (direct answer + cited lines; F1 on the direct answer), code-only grading until E1, and S5
  perturbations by code + DeepSeek with his review (D19). Built `ara/rag/retrieve.py`, `ara/graph`
  (read: ingest × paper → gather × document → collect → ≤ 1 requery; answer: synthesize → prewarm
  if worth it → verify × claim → assemble → ≤ 1 repair → finalize; `qa.answer_question`), the
  DeepSeek JSON path in the gateway, `evals/graders/answers.py`, `evals/suites/s2.py`,
  `evals/suites/s5_data.py`, `ara eval run s2 --limit` and `ara eval perturb`. Live check
  (`s2-20261007T165742`, 2 questions): both answered, every delivered line verified, citation
  precision 0.83, answer F1 0.25 — low because of the metric (one answer covered both annotators'
  answers in one line; the other quoted the paper's 4,528 employees where the gold says 26,972
  sentences), which is what the E1 judge is for. No prewarm (the pack was under 1,024 tokens) and
  no repair needed. S5 data: 60 claims in `evals/datasets/s5_claims.json`; the generation overran
  its first approval (D19). Noted for Ewan's review: two S1 questions share one source sentence
  (the 4,528 employees), so one claim pair is duplicated; code negations are sometimes awkward
  ("has not 100 training queries"). 62 unit tests. Next: Ewan reviews the S5 claims; then the S5
  suite (Luna vs DeepSeek verifier), which needs its own approval.
