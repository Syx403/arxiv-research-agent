# ARA v2 — Progress

Update this file at the end of every working session: what was done, what was spent, what is next.

## Milestones

| Milestone | Status | Notes |
|---|---|---|
| Design | done (2026-10-07) | `DESIGN.md`, `DECISIONS.md`, root `CLAUDE.md` |
| Data: PaSa query sets | done (2026-10-07) | `data/datasets/pasa/` (git-ignored, checksums in its README) |
| M0 Foundation | done (2026-10-07), reviewed | §16 verified; §6.2–6.3 confirmed (D10, D11); 29 unit tests on real Postgres; live checks passed (cache hits on both providers; a rejected request is released) |
| M1 RAG + S1 | done (2026-10-08) | sources (arXiv HTML/PDF, QASPER), chunking, cached embeddings, ingestion, BM25 ×2 / FTS / dense / RRF / rerank, eval runner + S1; S1 live on 2 papers (6 items) |
| M2 Read + answer | done (2026-10-08), reviewed | read + answer subgraphs, S2 suite; live check: 2 questions answered with verified citations; S5 data reviewed (58 claims) and S5 run: Luna and DeepSeek both F1 1.00 held-out, Luna stays (D20); review fixes and a live check of requery, prewarm and repair (D21) |
| M3 Understand + discover | done (2026-10-08), reviewed; S3/S4 first rounds run (D25) | top-level graph with Postgres checkpointer, understand + clarify interrupt, researcher tool loop, prerank, screen, choose_papers; live: a clarify turn and a discover → read turn passed (D22); review fixes (D23); priorities, titles, recency, today, English S4 of 62 items (D24); S4 labels await Ewan's review |
| M4 Memory | done (2026-10-09); S6 reviewed and run; review fixes (D29) and library redesign (D30) built, awaiting the S6 round | profile + research records in the Store, library + HNSW, `memory` intent; live: 2 S6 scenarios (D27); named papers read directly (D29); arXiv first with a library pre-check, no paper-choice question, per-paper evidence with one rerank (D30) |
| M5 Reliability + UI | done (2026-10-09): M5a (D34) S7 graceful 6/6, injection 0/3, S6 19/19 without an error; M5b (D35) `make start` → http://127.0.0.1:8000, four pages, a demo turn checked live | |
| Architecture review (Ewan) | in progress: review done, 16 fixes approved and built (D39); live check and Ewan's pass pending | gate before any large-scale LLM testing |
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
| 2026-10-08 | S5 smoke | one claim, both verifier arms (`s5-smoke-20261007T173949`) | 2 | 0.0002 | rule-1 threshold, reported |
| 2026-10-08 | S5 round | 58 claims × Luna / DeepSeek (`s5-20261007T174137`) | 116 | 0.0093 | Ewan (116 requests, cap US$0.10) |
| 2026-10-08 | M2 review check | requery, prewarm + cached verify, off-question verify, repair (`m2-review-20261007T180521`, `…180557` a mistaken rerun, `…180728`) | 23 | 0.0025 | Ewan (≈ 10 requests, < US$0.005; exceeded in count by the rerun) |
| 2026-10-08 | M3 live check | clarify turn; discover → read turn, a crashed first run (embedding bug) and its rerun (`m3-live-20261007T183501`, `…183610`, `…183711`) | 49 | 0.0171 | Ewan (≈ 40 requests, ≤ US$0.05; count exceeded by the rerun, 5 of 49 are free reranks) |
| 2026-10-08 | M3 review check | understand with the shown-papers block on two S4 items (`m3-review-20261008T051722`) | 2 | 0.0003 | rule-1 threshold, reported |
| 2026-10-08 | S3 calibration | 2 dev + 2 held-out queries, off-peak (`s3-20261008T052442`): 11 researcher steps, 4 embedding requests, 10 screen batches | 25 | 0.0098 | Ewan (calibration first; cap US$0.05) |
| 2026-10-08 | D24 live check | understand on 5 S4 items (`d24-live-…T085334`, `…T085356`); latest-papers turn and read-then-follow-up turn, off-peak (`…T103156`) | 39 | 0.0119 | Ewan (≈ 40 requests, ≤ US$0.02) |
| 2026-10-08 | S4 round | 62 understand requests (`s4-20261008T085452`) | 62 | 0.0084 | Ewan (cap US$0.05) |
| 2026-10-08 | S3 round | 30 PaSa queries, off-peak (`s3-20261008T103432`) | 189 | 0.0893 | Ewan (cap US$0.20) |
| 2026-10-09 | M4 live check | S6 scenarios s6-carry and s6-library (`s6-20261008T162055`) | 42 | 0.0176 | Ewan (≈ 40 requests, ≤ US$0.03; 2 over in count) |
| 2026-10-09 | S6 round | 6 scenarios, 14 turns, off-peak (`s6-20261008T164126`) | 108 | 0.0280 | Ewan (cap US$0.08) |
| 2026-10-09 | S6 rerun | s6-episode after the D27 prompt fix and D28 (`s6-20261008T170341`) | 21 | 0.0026 | Ewan |
| 2026-10-09 | S6 round, first attempt | 13 scenarios; 9 stopped by arXiv HTTP 429 / timeouts (`s6-20261009T040017`) | 88 | 0.0128 | Ewan (cap raised to US$0.15) |
| 2026-10-09 | S4 re-baseline | 62 understand requests on the D30 code (`s4-20261009T040617`) | 62 | 0.0082 | Ewan (cap US$0.05) |
| 2026-10-09 | S6 round | 13 scenarios, off-peak; 3 stopped by arXiv timeouts (`s6-20261009T044915`) | 239 | 0.0450 | Ewan (cap US$0.15) |
| 2026-10-09 | S3 re-baseline | 30 PaSa queries, off-peak (`s3-20261009T045853`) | 187 | 0.0907 | Ewan (cap US$0.20) |
| 2026-10-09 | S4 re-baseline | 62 understand requests on the D32 prompt (`s4-20261009T112903`) | 62 | 0.0080 | Ewan (cap US$0.05) |
| 2026-10-09 | S6 round | 17 scenarios, 33 turns, off-peak (`s6-20261009T113005`) | 394 | 0.0950 | Ewan (cap US$0.15) |
| 2026-10-09 | S4 re-baseline | 62 understand requests on the D33 prompt (`s4-20261009T121555`) | 62 | 0.0084 | Ewan (cap US$0.05) |
| 2026-10-09 | S6 round | 19 scenarios, 37 turns, off-peak (`s6-20261009T121705`) | 489 | 0.1265 | Ewan (S6 cap raised to US$0.20) |
| 2026-10-09 | S7 round | 6 fault turns, 3 injection questions (`s7-20261009T141936`) | 44 | 0.0085 | Ewan (cap US$0.05) |
| 2026-10-09 | S6 rerun after M5a | 19 scenarios, off-peak (`s6-20261009T142221`) | 500 | 0.1251 | Ewan (cap US$0.20) |
| 2026-10-09 | M5b live check | one demo turn in the app (`ui:` turn ids, no run id) | 8 | 0.0019 | Ewan (≤ US$0.01) |
| 2026-10-10 | Chat UI live check | three turns in two conversations (`ui:` turn ids) | 33 | 0.0064 | Ewan (≤ US$0.01) |
| 2026-10-10 | Ewan's own app turns | the "find" and "I" conversations, investigated in D38 | 25 | 0.0032 | Ewan (his own use) |
| 2026-10-10 | D37/D38 live check | read, switch away and back, two stops, a follow-up, a reload mid-turn, an arXiv 429 turn (`ui:` turn ids) | 42 | 0.0114 | Ewan (≤ US$0.03) |
| | | **Total so far** (from the ledger; it also counts free embedding and rerank rows not itemised above) | 3063 | **0.7618** | |

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
- 2026-10-07 — M0 strict review. Re-read every file against DESIGN, DECISIONS and
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
- 2026-10-08 — M1 strict review, then fixes approved by Ewan. Fixed: the S1 report
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
- 2026-10-08 — M2, part 2. Ewan's S5 review: one duplicated pair removed, four code negations
  reworded, the rest accepted (58 claims). Built the S5 suite (`evals/suites/s5.py`: the product's
  verify prompt on each claim with its one evidence sentence, Luna vs DeepSeek; report adds P/R/F1
  on "unsupported" and recall per kind), `ara eval run s5`, and a guard so `perturb` cannot
  overwrite the review. Smoke (2 requests, $0.0002), then the round (116 requests, $0.0093, 0
  failed): held-out both models F1 1.00; dev DeepSeek 1.00, Luna 0.96 (one false alarm on an
  ambiguous paraphrase). Luna stays the verifier; the suite is at its ceiling (D20). 65 unit
  tests. M2 remaining: Ewan's review of M2; the full S2 round belongs to E1.
- 2026-10-08 — M2 strict review, then fixes approved by Ewan (D21). Found: the requery
  told synthesize that aspects other searches had covered were missing; the verifier never saw the
  question, so a direct answer was checked as a bare phrase; a failed direct answer still
  delivered its explanation lines (against D19); repair calls carried only the synthesize version;
  S2 lost first-draft rejections; S2 had no trials and a stale cost estimate; `run s2` / `run s5`
  did not compare the estimate with `--max-usd`; a stored paper was fetched again; label columns
  appeared in paired differences; S5's DeepSeek bias was unstated. Fixed all of them (D21).
  New live check `tests/live/test_answer_paths.py`: prewarm wrote 2,234 tokens and verify read
  them from the cache, the requery fired, an off-question phrase was rejected, one repair removed
  a rejected line. Observed, not changed: Luna rejects "no" for a yes/no question whose evidence
  implies but does not state it (E1 measures this). Spend: 23 requests, US$0.0025; a mistaken
  rerun of the live module took it past the approved ≈ 10 requests (still under US$0.005).
  73 unit tests. Next: M3 (understand + discover).
- 2026-10-08 — M3. Ewan chose code-only S3 grading until E1, S4 from v1's 18 questions plus 32
  drafted items with his review, PaSa traces without a LangSmith dataset, and the live-check
  budget (D22). Built `ara/graph/app.py` (load_context → understand → clarify ⏸ / discover /
  resolve → read → answer → respond; `AsyncPostgresSaver`; `send` for a turn or a resume),
  `ara/graph/discover.py` (researcher ⇄ arxiv_tools on DeepSeek, prerank by embeddings, Luna
  screening in batches of 8, rank), `gateway.tool_step`, tool turns in `Block`, arXiv `search` and
  `lookup`, prompts `understand`, `researcher`, `screen`; `evals/pasa.py`, S3 (manifest of ids,
  seed 20261009), S4 (50 items, labels proposed, `reviewed: false`), `graders/discovery.py`,
  `ara eval run s3|s4`. Fixed a latent M1 bug: cached embeddings were not numpy arrays. Live:
  the clarify turn and the discover → read turn passed (49 requests, US$0.0171; over the
  approved count because of the crashed run). 93 unit tests. Next: Ewan reviews the S4 labels;
  then the S3 (≈ US$0.24) and S4 (≈ US$0.025) rounds, which need approval.
- 2026-10-08 — M3 strict review, then fixes approved by Ewan (D23). Found and fixed:
  understand's prompt promised the shown papers but never received them (now a data block before
  the latest message); ids were checked by pattern only (now they must appear in the conversation);
  S4 graded constraints by count (now by quote; v1-c03x label added); violation quotes compared
  exactly (now folded); screen judgements with an unexpected id dropped papers silently (now matched
  by bare id, unjudged papers counted in S3); read-by-id references never matched stored papers and
  a paper named and pointed at was read twice (now stored ids, one per paper, bare ids pinned to the
  latest version). Recorded: S3 stays at 30 queries per round (Ewan), screening never caches, D22's
  1,124 / 1,192 tokens, researcher effort not measured in M3 (proposal: an S3 arm in the E rounds).
  Item 11 (a named off-topic paper screened out) moved to E1. Live check: 2 requests, US$0.00034;
  ref-mixed showed the model choosing `discover_read` with both references filled (left for S4).
  97 unit tests. Next: Ewan reviews the S4 labels; S3 (suggested: `--limit 2` first to calibrate the
  unit cost) and S4 rounds need approval.
- 2026-10-08 — S3 calibration (`s3-20261008T052442`, 4 queries, 0 failed, US$0.0098): US$0.0012-0.0031
  per query (screening 63% of it), 30-37 s each, so a 30-query round is ≈ US$0.07-0.09 and ≈ 18
  minutes, well under the D22 estimate of US$0.24. The researcher used 2-4 steps per query (about four
  searches each, so its 8-search budget) and gathered 5-56 candidates; pool recall 0.28 held-out, 0.33 dev (n = 2 each, directional).
- 2026-10-08 — D24, after Ewan's S4 label review. Built: `priorities`, `titles` and `prefer_recent`
  in the request (understand prompt rewritten, quotes checked in code), `Context.today` in the
  understand item and as the search end (S3 pins it to the PaSa date), `search_arxiv(newest_first)`
  and date order within a relevance grade, priorities in the synthesize item, papers just read
  become `shown`; all project data English; S4 rebuilt as 62 fully labelled English items graded
  per field. No billable calls in this step. 100 unit tests. Next: Ewan reviews the S4 draft; a
  live check of the changed turns (≈ 40 requests, ≤ US$0.02) needs approval; then S4 and S3.
- 2026-10-08 — Ewan accepted every S4 label; first S4 and S3 rounds (D25). S4 held-out: intent
  0.94, every field right 0.69, priorities the weakest field, no missed clarification. S3 held-out:
  pool recall 0.44, hit@5 0.87; search recall is the bottleneck (the researcher uses its 8
  searches, about four per step; an earlier line here wrongly said it stopped early). Live check:
  the latest-papers turn passed; the read-then-follow-up turn failed at its first step (ids taken as find-then-read). No change was tuned on held-out results: three
  candidates (inclusive date limits, tighter priorities, an ids-means-read rule) go to the E rounds.
  Fixed a D24 slip in the understand prompt (`paper_ids` from the latest message). S3's dry-run unit
  cost lowered to US$0.005 from the calibration. Spend this session: 290 requests, US$0.1096; v2 total
  582 requests, US$0.1643. Next: M4 (memory), after Ewan's go-ahead.
- 2026-10-09 — M4 (D27). Ewan approved the plan: a `memory` intent, the profile into understand
  only, research records written by code, HNSW kept and measured. Built `ara/memory` (Store,
  remember with its trust boundary, library), migration 0004, the `remember` and `library` nodes,
  S6 (six scenarios) and `ara eval run s6`. HNSW: top-10 overlap 0.977 with the exact scan, but the
  planner keeps the exact scan at 686 chunks. Live check: s6-carry passed; s6-library failed one
  check (a library question was clarified), fixed by a routing rule; checkpoint types registered.
  109 unit tests. Next: Ewan reviews the S6 scenarios; the full S6 round (≈ US$0.05) needs approval.
- 2026-10-09 — S6 reviewed by Ewan (library misses now offer an arXiv search) and run: 5 of 6
  scenarios passed, US$0.0280. s6-episode failed: the understand prompt did not let records' ids be
  used (fixed, not rerun), and choose_papers asked about a single listed paper (open for Ewan).
- 2026-10-09 — D28: papers named by id or title are locked (code title match, else the screen's
  `named`; always shortlisted, listed first, read without asking; a name not found is reported).
  s6-episode rerun passed (understand used the research record's id). 113 unit tests. Next: M5.
- 2026-10-09 — M4 strict review, then fixes approved by Ewan (D29). Review found: an incomplete
  checkpoint allowlist (subgraph types loaded back as dicts), the library route not built as the
  approved plan said (an unrecorded deviation), library replies and records listing uncited papers,
  empty quotes passing both trust boundaries, `remember` calling Luna for profile constraints, a
  false "could not find on arXiv" on library questions, no source turn on facts, S6 tuned on a
  held-out scenario, and stale M3 baselines. Ewan added a rule: papers the user names (id or title)
  are read directly, and the reply says when a question does not fit the paper or a named paper
  breaks a constraint. Built: `resolve` finds titles by code (library, then an arXiv title search),
  discovery only for titles it cannot find; `conflicts` node (screen prompt on unscreened named
  papers); premise correction in the synthesize and verify prompts; per-paper "does not seem to
  discuss this" notes quoting the stored abstract; library questions fuse HNSW with research
  records and search their papers together; quotes non-empty; facts keep thread, turn and day;
  records split read and listed; `ara eval hnsw` (free: top-10 overlap 1.00 on 879 chunks and on a
  300-chunk library; the planner still picks the exact scan); S6 s6-episode to dev plus five draft
  scenarios; four S4 labels `discover_read` → `read`. 124 unit tests. No billable calls (one unit
  test reached OpenAI with a placeholder key before its embedding was cached: HTTP 401, nothing
  billed). Next: Ewan reviews the S6 drafts and the S4 label changes; then the approved runs: S6
  (≈ US$0.07, cap US$0.10), S4 re-baseline (≈ US$0.01, cap US$0.05), S3 re-baseline (≈ US$0.09,
  cap US$0.20), off-peak.
- 2026-10-09 — D30, Ewan's redesign of the library route after D29. Built: `library` narrowed to
  explicit references to our history; discovery adds the user's close papers to its candidates and
  marks "read before"; migration 0005 `papers.published` (ingestion writes it; `ara db backfill`
  dated the 5 stored arXiv rows with one free lookup); `choose_papers` never asks (interrupt
  removed); history questions screen library papers with their nearest passages, list every
  relevant one, read up to three, and search arXiv once when they do not answer; every reading
  path searches each paper on its own with one shared rerank and keeps 8 passages per paper;
  citations listed per sentence. S6: s6-library-many reworded and rechecked, two drafts added.
  126 unit tests. No billable calls. S6 dry run: 13 scenarios, 25 turns, US$0.104 at the flat
  US$0.008 per scenario, over the US$0.10 cap; measured scenarios cost US$0.0004–0.0116 (mean
  about US$0.005), so the expected cost is about US$0.08. Next: Ewan reviews the S6 drafts and
  decides the cap or the unit cost; then S6, S4 and S3 rounds off-peak.
- 2026-10-09 — D31 (Ewan): the fallback's opening is the model's own "Context:" line, unverified.
  First S6 attempt hit arXiv HTTP 429 (the host was rate-limited for about 45 minutes; free probes
  until it answered) and exposed a D30 gap: a history question whose paper id came from a research
  record took the named path and never fell back to arXiv (fixed: any unanswered library question
  falls back, names dropped). Re-baselines on the D30/D31 code: S4 held-out intent 1.00, all fields
  0.72, false clarify 1/32; S3 held-out pool recall 0.58, hit@5 0.93, gold precision@5 ≥ 0.51,
  dev pool recall 0.58, hit@5 0.53. Pool recall comes before screening, which is all D29-D30
  changed in discovery, so the moves against M3 (held-out +0.14, dev -0.12) are the researcher's
  run-to-run variance, not an effect. S6 (`s6-20261009T044915`): 7 of 10 completed scenarios
  passed (dev 4/4, held-out 3/6); 3 stopped by arXiv timeouts (carry, forget, refer). Failures:
  s6-library-many (understand took both ids from research records, so the named path ran and
  nothing was listed), s6-premise (the model abstained instead of correcting the premise),
  s6-conflict (understand did not apply the remembered "never fine-tune" to a named read, so no
  conflict check ran). s6-library-fallback passed (opening: "We previously read the ReAct paper,
  which did not cover the number of tokens consumed by each task, so this evidence comes from a
  new arXiv search."), but its answer abstained: the question still asked what "the papers we
  read" said. Proposals await Ewan; nothing tuned on these held-out results.
- 2026-10-09 — D32 fixes (Ewan approved a-d): library questions name papers only by ids the user
  wrote; `need` states the topic, not our history; remembered facts apply to named reads; a shared
  title ("Gorilla:") resolves to the paper closest to the need (found while drafting, before any
  run). Four scenarios moved to dev; four held-out drafts await Ewan. 129 unit tests. No billable
  calls. Next: Ewan reviews the drafts; S6 (dry run US$0.136, cap US$0.15) and S4 (≈ US$0.008).
- 2026-10-09 — D32 rounds (arXiv answering again; off-peak). S4 (`s4-20261009T112903`): held-out
  intent 1.00, all fields 0.62 [0.47, 0.78] (0.72 the run before; 3 items better, 6 worse, all in
  known error kinds: titles copied from the shown list, prefer_recent with a date limit, the
  inclusive end date), dev 0.80; read as run-to-run variance, not tuned. S6
  (`s6-20261009T113005`, 17 scenarios, 0 errors): held-out 6/8 passed (checks 0.93), dev 7/9.
  New held-out passed: s6-history-list, s6-premise-parallel ("No; LLMCompiler executes functions
  in parallel …", verified), s6-conflict-train (Gorilla resolved to 2305.15334 among two "Gorilla:"
  titles; conflict note shown). Failed: s6-history-fallback (held-out: the library screen judged
  LLMCompiler not relevant to "prompt injection in parallel function calling", so the reply was
  "not discussed" although we had read it); s6-library (held-out: "And what did the papers we read
  say about diffusion models …" became `discover_read`, a regression from D32(b)'s topic-only
  need); s6-library-fallback (dev: fallback and opening worked, but the check wants the id and the
  model's opening names the title); s6-premise (dev: still abstains). Proposals await Ewan.
- 2026-10-09 — D33 (Ewan: option A): `history` says which papers a library question means; the
  library screen judges against it; held-out S6 frozen until the E rounds. S4
  (`s4-20261009T121555`): held-out intent 1.00, all fields 0.69, history 1.00; dev intent 0.95
  (f-limits: a follow-up on the paper just read became `library`). S6 (`s6-20261009T121705`):
  held-out 6/8 (checks 0.96), dev 8 of 10 completed plus 1 error. Held-out, reported and not
  acted on: s6-history-gap and s6-history-back passed; s6-history-list fell back to arXiv although
  LLMCompiler answers it (the answer from our papers abstained, and the model's opening then said
  they "did not cover this question"); s6-conflict-train showed no conflict note this time (the
  screen did not flag Gorilla's fine-tuning against "never train models myself"; it did in the
  previous run). Dev: s6-history-fallback did everything right but its check wanted "LLMCompiler"
  where the opening writes the full title (check fixed to the title); s6-library stopped with
  `InvalidOutput('synthesize: empty answer')` (DeepSeek returned no text; error handling is M5);
  s6-premise still abstains. Unit tests 130.
- 2026-10-09 — M5a (D34; Ewan approved the plan and the split into M5a / M5b). Checked LangGraph
  1.2.14's retry and error-handler behaviour on scratch graphs first (429 not retried by default,
  OpenAI 400 retried, handlers need `Command(goto)`, none for `Send` tasks). Built
  `ara/graph/reliability.py` (transient, RETRY, timeouts, `degrade`), `ara/faults.py`
  (ARA_FAULTS), handlers on every failing node, a "Note:" line and `status="partial"`, arXiv
  Retry-After spacing, migration 0006 (`synthetic` source), S7 suite and data (6 fault turns, 3
  injection questions, drafted for Ewan's review). 141 unit tests, 11 of them on degraded paths
  with injected faults (no billable calls). Next: Ewan reviews S7; S7 (dry run US$0.036, expected
  ≈ US$0.02) and an S6 rerun to confirm no external error stops it (≈ US$0.13) need approval.
- 2026-10-09 — M5a rounds (Ewan approved). S7 (`s7-20261009T141936`, US$0.0085): graceful 6/6
  (arXiv down or refusing once, a paper not fetched, empty answers, a verifier that never answers,
  memory not saved: each ended in a reply with its note), injection success 0/3 (the "93% vs 81%"
  answer, an abstention on the link question, "2,000" not "42"). S6 rerun (`s6-20261009T142221`,
  US$0.1251): 19 scenarios, 0 errors, so M5a's done criterion holds; held-out 7/8 (checks 0.99),
  dev 8/11. Reported, not acted on: s6-conflict-train (held-out) again shows no conflict note (the
  constraint was there and the screen ran, but it did not flag Gorilla's fine-tuning against "never
  train models myself"); dev s6-conflict the same with Toolformer; dev s6-library's diffusion
  question read ReAct and fell back to arXiv instead of "not discussed" (the library screen let
  ReAct through); dev s6-premise still abstains. These go to the E rounds (dev).
- 2026-10-09 — M5b (D35; Ewan approved: subgraphs as separate diagrams, pages 1, 2, 4 and a simple
  memory page, an Anthropic-like visual language without its branding). Built the FastAPI server
  (SSE turns, graph, documents, evaluations, memory), the React UI, `make start`, a UI CI job.
  Found on LangGraph 1.2.14: xray cannot draw subgraphs a node invokes; streaming with
  `subgraphs=True` re-raises an already handled error after the run finished (worked around).
  Live: one demo turn in the browser, 8 calls, US$0.0019. 145 unit tests (4 for the API). Next:
  the architecture review (DESIGN §14 gate), then the E rounds.
- 2026-10-10 — D36: the app rebuilt as a research agent's chat (Ewan): sidebar of conversations
  (grouped by day, rename, delete), a fold under each reply leading to a side panel with that
  turn's workflow, evidence and papers; migration 0007 records conversations and turns. Live check
  in the browser: three turns, two conversations, US$0.0064; fixed recorded graphs drawn without
  colours. 146 unit tests. Next: the architecture review (DESIGN §14 gate).
- 2026-10-10 — D37 and D38 (Ewan's review of the chat app; read-only investigation first, then
  all fixes approved). Answers explain in paragraphs (4–12 cited sentences). Turns run as server
  tasks: switching away and back or reloading rejoins them; Stop (or Esc) undoes the turn and
  returns its message to the composer. Enter no longer sends while an input method composes.
  Failures are said in words, streamed live, and an arXiv that refuses ends the search after two
  rounds (Retry-After capped at 20 s, a deviation from D34). New memory page (long lists, search,
  a paper drawer with the conversations that read it) and evaluation page (protocol, a card per
  suite, metrics explained, items). Live: 42 calls, US$0.0114. 152 unit tests. Next: the
  architecture review (DESIGN §14 gate).
- 2026-10-10 — Architecture review (read-only) and D39: Ewan approved all 16 fixes (one finding,
  an ingest race, was wrong and dropped). Shared rate-limit queues no longer time out nodes;
  every turn ends in done, stopped or error, failures are undone, startup repairs half-done turns;
  library and memory are written only when a turn commits; titles must be the user's words;
  evaluation rounds record code and prompt versions and the page marks stale rounds; understand
  sees a moving window of history and the most relevant facts; replies are recorded in parts;
  running spend totals; `ara db prune`. 164 unit tests; no billable calls yet. Next: a live check
  (needs approval), then the E rounds.
