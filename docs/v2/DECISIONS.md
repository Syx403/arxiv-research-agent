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
  127.0.0.1:5434 with its own volume; v1's container (port 5433) is left untouched. ParadeDB's
  memory auto-tuning is off (`PDB_TUNE=false`), so the dev database, CI and later measurements all
  run on stock PostgreSQL settings instead of values sized from each host's RAM.
- Alternatives: the pgvector image with native full-text search only (no IDF, the §16 fallback);
  ParadeDB 0.25 with the older `bm25` / `key_field` syntax.
- Consequence: upgrading ParadeDB is a deliberate change (syntax moved between 0.25 and 0.26).

## D13 — M0 sequencing (2026-10-07)
- Context: M0's done-criteria need only the ledger, the gateway and their tests.
- Decision: migrations add tables in the milestone that first uses them (M0: `llm_calls`);
  `gateway.tool_loop` arrives with the researcher in M3 and fault hooks in M5, as DESIGN §14
  assigns them. v1 code and v1 documents are removed from this branch (Ewan approved); both stay
  at tag `baseline-2026-10-07`.
- Alternatives: create every DESIGN §8 table in 0001 (one migration, but tables with no code yet).
- Consequence: no empty placeholder modules; every module on the branch is exercised by a test.

## D14 — Prompt version = file name + content hash (2026-10-07)
- Context: DESIGN §6.4 requires a version tag per prompt in traces; hand-written tags go stale.
- Decision: the version is `<name>@<first 8 hex of sha256(text)>`, computed when the file is loaded.
  The static part of a `Prompt` is therefore one `Instructions` object (a prompt file) rather than a
  list of blocks, as first sketched in DESIGN §6.3; the output schema travels in `text.format`.
- Alternatives: a hand-written version tag in each file (goes stale when someone forgets it).
- Consequence: every call carries a version that identifies its exact instruction text (in traces
  and in the ledger). Product prompts are loaded from files with `Instructions.load`; the type does
  not forbid an inline `Instructions(name, text)`, which the unit tests use.

## D15 — BM25 tokenizer: English stemming, measured against plain words (2026-10-08, Ewan)
- Context: DESIGN §5 left the tokenizer to M1. ParadeDB can tokenise one field twice, so both
  variants live in one index and S1 compares them at no cost.
- Decision: S1 keeps both arms (`bm25`, `bm25_stemmed`); the product and the RRF arm use the
  stemmed field. In the M1 check (6 items) stemming was equal or better on every metric
  (recall@8 1.00 vs 0.83, nDCG@10 0.64 vs 0.57), but the intervals overlap, so the choice is
  re-checked on the full S1 round.
- Alternatives: plain words only (exact terms; misses "eviction" for "evicting").
- Consequence: one index carries both fields; switching is a query change, not a re-index.

## D16 — M1 implementation choices (2026-10-08)
- One chunk per paragraph (a long paragraph split at sentences, an overlong sentence by words, all
  ≤ 400 cl100k tokens). Gold evidence in QASPER is per paragraph, so metrics are exact; the
  heading path gives each chunk its context. Alternative: merging short paragraphs (more context,
  but paragraph-level grading would credit unretrieved neighbours).
- `documents` has no `status` column (DESIGN §8 listed one): a document and its chunks are written
  in one transaction, so a row exists only when complete. A failure status comes back if M2 needs
  to remember unreadable papers.
- The HNSW index is deferred to M4: scoped search is an exact scan, and nothing searches the whole
  library yet.
- Embeddings and Cohere reranks go through the gateway and the ledger (stages `embed`, `rerank`);
  Cohere is called over REST with a Pydantic-validated response instead of its SDK. The trial is
  free, so reranks cost $0 in the ledger but are counted against the 1,000-calls-a-month quota.
- S1 items: 10 QASPER validation papers × 3 questions drawn with seed 20261007 from questions whose
  every answering annotator cites text paragraphs; the first 4 papers are dev, the rest test. Each
  metric takes the best annotator (as QASPER's evidence score does). The manifest is committed; the
  Parquet file stays in `data/` and is checked by sha256.
- The runner uploads the S1 items once to a LangSmith dataset (`ara-s1-<manifest hash>`; QASPER is
  CC BY 4.0) because `aevaluate` reads examples from a dataset. Query embeddings are batched before
  the per-item runs, so each item costs one rerank call and no embedding request.

## D17 — D15 confirmed by the full S1 round (2026-10-08, Ewan approved the run)
- Context: D15 chose stemmed BM25 on 6 items. The full S1 round (`s1-20261007T162246`, 30 items,
  US$0.00086) gives paired differences, recorded in `docs/v2/eval/s1-20261007T162246.md`.
- Decision: keep English stemming. Against plain words it never lost top-8 recall (+0.05,
  CI [+0.00, +0.13], 2 better / 0 worse) and was neutral on ordering (MRR and nDCG@10 CIs include
  0). The top 8 is what reranking and evidence selection receive, so recall is the deciding metric.
- Also measured: rerank is the one step with a clear gain over RRF (recall@8 +0.18, CI [+0.07,
  +0.33]; nDCG@10 +0.11, CI [+0.005, +0.23]); RRF ties dense alone.
- Consequence: the M2 read pipeline uses stemmed BM25 + dense → RRF → rerank as designed.

## D18 — M1 review: S1 reported by split and paired; D17 restated; review fixes (2026-10-08)
- Context: the M1 review found that the S1 report pooled dev and test items (§11.1: tune on dev,
  report on held-out) and compared arms through separate intervals rather than paired differences
  (D6); D17's figures pool all 30 items. recall@k also counted de-duplicated paragraphs rather than
  the top k chunks the product passes on.
- Decision: `ara eval report` prints held-out (test, 18 items) and dev (12) separately, each with
  paired differences (bootstrap of per-item Δ, items better/worse) for the suite's pairs. Ranks
  count chunks; a paragraph split into several chunks counts once, at its first chunk. In the full
  round no paragraph was split (463 chunks, 463 paragraphs), so its stored metrics are unchanged.
- Restated from the regenerated report (`docs/v2/eval/s1-20261007T162246.md`):
  - Tokenizer, chosen on dev: stemmed − plain recall@8 +0.083 [+0.000, +0.250], 1 better / 0
    worse; MRR −0.014 (2/4); nDCG@10 +0.018 (3/3). Held-out agrees: recall@8 +0.028 (1/0), MRR
    +0.077 [−0.027, +0.208] (5/5). Stemming stays (it never lost top-8 recall in either split), but
    the evidence is two items and no effect on ordering is shown.
  - Rerank: on dev a clear gain over RRF (MRR +0.225 [+0.081, +0.397], nDCG@10 +0.193 [+0.062,
    +0.337]); on held-out only top-8 recall improved (+0.139 [+0.000, +0.306], 3/0, reaching 1.00),
    with no ordering gain (MRR −0.004 [−0.178, +0.189], nDCG@10 +0.060 [−0.090, +0.229]). D17's
    ordering gain holds on dev only.
  - Held-out first stage: dense alone is the strongest arm (MRR 0.69, recall@8 0.86); RRF ties it
    and beats stemmed BM25 (MRR +0.132 [+0.038, +0.248], 9/0).
- Other review fixes: every provider client gets a 120 s request timeout instead of the SDK's
  600 s (a backstop for callers outside the graph; long enough for 8K DeepSeek thinking tokens;
  node timeouts stay as §4.4 in M5). The S1 LangSmith dataset is named by a hash of its examples,
  gold references included, instead of the manifest hash (D16), so a change in gold derivation
  cannot grade against stale references. The PDF fallback regroups pypdf's lines into paragraphs
  (a sentence-final line well short of full width ends one; running headers, page numbers and
  undecodable text dropped); before, a whole PDF became one paragraph (checked on 2210.03629v3:
  1 → 449 paragraphs).
- Alternatives: keep pooled figures (larger n, but choices and reported numbers on the same items).
- Consequence: the M2 pipeline is unchanged (stemmed BM25 + dense → RRF → rerank → top 8), on the
  strength of top-8 recall; whether rerank's ordering gain is real is re-measured in S2.

## D19 — M2 choices: S2 items, answer form, S5 data (2026-10-08, Ewan)
- S2 items: the 30 S1 questions plus 5 questions every annotator marked unanswerable, drawn with
  seed 20261008 from other QASPER validation papers (the S1 papers have none). Manifest
  `evals/datasets/s2_qasper.json`; 2 of the 5 are dev. Deviation from DESIGN §11.2 ("same 30
  questions, ≈ 5 unanswerable"), approved by Ewan.
- Answer form: one `Answer:` line with the direct answer, then cited explanation lines. Each cited
  line is a claim and is verified alone; uncited lines are dropped; an unverified direct answer
  turns the whole answer into the abstention. Answer F1 is QASPER's token F1 on the direct answer
  only.
- Grading in M2 is by code only. The Luna equivalence judge for free-form answers moves to E1,
  where it is calibrated against Ewan's labels (§11.4).
- `select_evidence` returns labels; the first live attempt returned "S8: <sentence>" and every
  label was discarded, so both questions abstained. The prompt now asks for labels only and the
  parser extracts `S\d+` from each entry.
- S5 data: one evidence sentence per S1 item (with a number preferred, then the longest).
  Supported claims are DeepSeek paraphrases; unsupported ones change one thing — number (8) and
  negation (7) by code, entity (8) and over-generalisation (7) by DeepSeek (`s5_perturb`, high
  effort, 5 sentences per request). Every item carries `reviewed: false` until Ewan reviews it.
  Generation saves each batch to `data/s5_rewrites.json` and skips batches already saved.
- S5 cost: approved as ≈ 15 requests, ≤ $0.01. Thinking tokens grew from 394 to 5,250 per request,
  and the run cap stopped the sixth (last) request after 5 requests, $0.0087. The 5 replies were
  recovered from the LangSmith traces, not re-bought; Ewan raised the cap and the last request
  ($0.0026) completed the set. Total: 6 requests, $0.0113 — over the first approval by $0.0013.
- Alternatives: S2 with only the 30 answerable items (no abstention signal); all perturbations by
  LLM (cheaper code paths unused, less control over what changed).

## D20 — S5 round: Luna stays the verifier; S5 as built is at its ceiling (2026-10-08, Ewan approved the run)
- Data: Ewan's review of the S5 claims removed one duplicated pair (two S1 questions share the
  same source sentence) and reworded four code negations by hand ("has not 100" → "does not have
  100", "can not" → "cannot", "have not special" → "do not have special"); the rest was accepted.
  58 claims (29 pairs), all marked reviewed; the file records the review. `ara eval perturb`
  now refuses to overwrite the reviewed file.
- Run `s5-20261007T174137` (116 requests, $0.0093, 0 failed, 0 invalid DeepSeek replies; report
  in `docs/v2/eval/`): held-out (36 claims) both models P = R = F1 = 1.00 on "unsupported",
  every kind caught; dev (22) DeepSeek 1.00, Luna F1 0.96 with one false alarm. That false alarm
  is a label ambiguity, not a clear miss: the source says the system "create[s] … features,
  generate[s] document vectors and feed[s] them" to the SVM, the paraphrase reads "them" as
  features and vectors, and Luna objected that only the vectors are said to be fed.
- Decision: the product keeps Luna for `verify` (D10). The suite cannot separate the models
  (paired Δ 0/0 on held-out), so the choice rests on the other grounds: checker ≠ writer (D10
  principle 3, the answers are written by DeepSeek), and Luna was cheaper here ($0.000062 vs
  $0.000099 per claim, DeepSeek off-peak; DeepSeek peak doubles it). DeepSeek was faster (p50
  0.84 s vs 1.60 s).
- Limitation: one evidence sentence and one obvious change per claim make S5 an easy test. It
  shows the verify prompt catches the four perturbation kinds; it does not measure how often the
  verifier errs on real answer lines against multi-sentence packs. That is measured through S2
  (lines dropped by verification, judged in E1). A harder S5 (several sentences per pack with
  distractors, subtler changes) is a proposal for the E rounds, not built.
- Cost estimate corrected: the plan assumed ≈ $0.05 per S5 round (§11.2); measured $0.0093.

## D21 — M2 review fixes (2026-10-08, Ewan approved the fixes and the live check)
- Context: the M2 strict review found that the requery reported aspects as missing that other
  searches had covered; that the verifier never saw the question, so a direct answer was checked as
  a bare phrase (live: "4528 employees" passed for "What is the size of the real-life dataset?",
  gold 26,972 sentences); that a failed direct answer still delivered its explanation lines, against
  D19; that repair calls carried only the synthesize prompt's version (D14); that S2 recorded no
  verifier rejections from the first draft; and smaller gaps listed below.
- Decisions:
  - Requery per document: each document is searched again, once, for up to 3 aspects
    (`MAX_REQUERIES`) its own first-round selection reported missing, so at most 9 extra gathers
    for 3 papers. `synthesize` is told only the aspects no search covered: no first-round document
    chose sentences without listing the aspect, and the aspect's own search chose nothing. The
    requery's own `missing` lists are ignored, because they judge the whole question from one
    aspect's passages. Alternative: intersect the first-round lists (cheaper, but blind to an aspect
    one paper covers and another lacks).
  - The direct answer (line 0) is verified with the question before the claim; explanation lines
    stay question-free, so the evidence pack remains the shared prefix and S5 is unchanged. The
    verify prompt gains one paragraph for this (new version `verify@88327669`; the S5 round of D20
    used the earlier version).
  - A direct answer that fails verification or cites nothing withholds the whole answer (D19 as
    written; Ewan chose this over keeping verified lines). A model abstention still keeps its one
    verified line on what the evidence covers, as the synthesize prompt allows; a cited abstention
    is recognised as an abstention.
  - `Prompt.follow_up`: a second prompt file sent as the item part; the version becomes
    `synthesize@…+repair@…` (D14 extended).
  - `Answer` records `checked` (distinct lines verified over both drafts) and `rejected` (lines the
    verifier rejected, over both drafts); S2 reports `rejection_rate` beside `verified_share`, which
    is what D20 relies on to measure the verifier on real answer lines.
  - S2 runs each unanswerable item 3 times (DESIGN §11.2 "5 × 3 trials", not built in M2): 45 runs.
    Per-item cost re-measured: $0.00075 on the happy path; the dry-run estimate is $0.003 per item,
    $0.136 for the round (was $0.29). `run s2` and `run s5` refuse when the estimate exceeds
    `--max-usd`, as S1 does.
  - The read graph does not fetch a reference that is already a stored paper id. The report leaves
    label columns (`unanswerable`, `unsupported`) out of paired differences.
- Live check (`tests/live/test_answer_paths.py`): prewarm wrote 2,234 tokens and both verify calls
  read 2,234 from the cache; the requery fired (1 and then 2 aspects); a true but off-question
  phrase ("five independent decoders" for "How many images are in each sequence?") was rejected;
  one repair removed the rejected line and its version tag carried `+repair@`.
- Observed, not changed: with the question, Luna rejected "no" for "Do the decoder LSTMs all have
  the same weights?" (QASPER gold "No"), because the evidence says "five independent decoders" and
  not that their weights differ. The verifier is strict about yes/no answers that rest on an
  inference; whether that is wanted is measured in E1 (S2 has one yes/no item) before the verify
  prompt is loosened.
- Limitation added to D20: DeepSeek wrote all 29 S5 paraphrases and 15 of the perturbations and is
  also one of the two arms, so S5 may favour it slightly; the suite was at its ceiling anyway.
- Spend: the check was approved at about 10 requests under US$0.005. A mistaken rerun of the whole
  module (a command meant only to show the failures) doubled it: 23 requests (5 of them free
  reranks), US$0.0025 in all. The run cap was raised from US$0.005 to US$0.01 for the last two
  requests, because one repair call reserves about US$0.0052 before it settles (it cost US$0.0006).

## D22 — M3 choices and implementation (2026-10-08, Ewan chose the evaluation options)
- Ewan's choices: S3 is graded by code in M3 (pool recall, shortlist recall, gold precision@5 as a
  lower bound, hit@5); the DeepSeek relevance judge, its calibration on Ewan's 30 labels, the
  adjudication of non-gold papers and constraint violations move to E1. S4 = v1's 18 distinct
  questions plus 32 drafted edge cases, labels proposed by Claude and reviewed by Ewan before a
  round (`reviewed` per item; `run s4 --execute` refuses unreviewed data). PaSa (CC BY-NC-SA,
  gated): traces may reach the private LangSmith project, but S3 creates no LangSmith dataset or
  experiment; results stay in Postgres, the committed manifest and reports name queries by id.
- `gateway.tool_step` replaces the planned `gateway.tool_loop` (DESIGN §6.4): one DeepSeek step
  returning the assistant turn (content, reasoning, tool calls); the loop is the graph's
  `researcher ⇄ arxiv_tools`, which owns the budget (8 tool calls; calls past it get a "not run"
  tool message, and the loop goes to prerank without asking the researcher again). `Block` gained
  `reasoning`, `calls`, `call_id` and the role `tool`; OpenAI rendering refuses tool turns.
- The arXiv client gained `search` (arXiv query syntax; the request's date window is appended by
  code, from 1991-01-01 because arXiv rejects earlier starts) and `lookup` (many ids in one
  request); a rejected query raises `SearchError`, which the tool node returns to the model.
- Top-level graph additions not drawn in DESIGN §4.1: `resolve` (read by reference), `shown`
  kept across turns, at most 2 clarifications per turn, a read request without a resolvable paper
  becomes find-then-read, and the deterministic `choose_papers` rule (count, else 1-3 direct
  matches, else ask). No memory until M4: library questions get a notice; no `remember` node.
- Understand's trust boundary in code: constraints must quote a human message (whitespace and
  case folded), ids must match the arXiv pattern, positions must point at a shown paper, dates
  must parse, a count must be positive. The conversation is the shared prompt part and the
  latest message the item, so each turn extends the previous turn's cached prefix (measured:
  the resumed understand call read 1,124 of 1,192 input tokens from the cache; corrected in D23).
- Fixed on the way: cached embeddings came back as pgvector `Vector` objects, not numpy arrays
  (latent since M1, where cached vectors only went back into SQL); `embed` now converts them.
- Live check (`tests/live/test_turns.py`, approved ≈ 40 requests, ≤ US$0.05): 49 requests, of
  which 5 free reranks; US$0.0171. The count exceeded the estimate because the first discover →
  read run crashed on the embedding bug after 4 calls and was rerun. Clarify turn: a question,
  then on resume discovery with both constraints kept verbatim (8 requests, US$0.0058). Discover →
  read: "find 2 papers on scheduling LLM tool calls in parallel and compare" found LLMCompiler
  (2312.04511) and LLM-Tool Compiler (2405.17438), read both and answered with every line
  verified (verify read 21,744 of 24,907 input tokens from the prewarmed cache).
- Costs re-estimated: S3 ≈ US$0.008 per query (≈ US$0.24 a round, DESIGN had US$0.10; screening
  on Luna medium is the largest part, ≈ US$0.001 per batch); S4 ≈ US$0.025 a round.
- Alternatives: the researcher on Luna (cheaper tool calls, but D10 puts open-ended loops on
  DeepSeek); no prerank (screen the whole pool: up to ≈ 100 abstracts, about 4× the screening
  cost).

## D23 — M3 review fixes (2026-10-08, Ewan approved fixes 1–10; S3 at 30 queries; item 11 to E1)
- Context: the M3 strict review found that understand's prompt promised the papers shown last but
  the code never sent them; that ids were checked by pattern only, so an id the model recalled
  (not one the user wrote) would skip discovery (DESIGN §13 keeps v1's "ids come from the
  conversation"); that S4 graded constraints by count only; that DESIGN's S3 size moved from "15 per
  round" to 30 without a record; that the screen cache claim in §6.3 does not hold; a wrong number
  in D22; that D10's "efforts re-measured in M3" was neither done nor deferred; that violation
  quotes were compared exactly; that screen judgements with an unexpected id silently dropped a
  paper; and that read-by-id references never matched stored papers.
- Decisions:
  - `understand` gets the shown papers as a data block (n, id with version, title, date) placed
    in the item part before the latest message, only when there are any. Cost: after a turn that
    showed papers, the next turn's cache hit ends before the previous latest message (one message,
    tens of tokens), because the block is not repeated. Alternative: put the block in the shared
    part (it would break the prefix every time the list changes). The prompt file is unchanged.
  - An id survives `checked` only if its bare form appears in the conversation text (user or
    assistant messages, so an id listed by ARA earlier counts). A read request whose ids all fail
    becomes find-then-read, as before.
  - S4 labels constraints as quotes; a prediction passes with the same number of constraints, each
    containing or contained in its own labelled quote (whitespace and case folded). v1-c03x now
    carries the 不优化模型权重 constraint, as v1-c03 does. Labels still await Ewan's review.
  - S3 stays at 30 queries per round (15 dev + 15 held-out), Ewan's choice; part of the cost rise
    from DESIGN's US$0.10 to ≈ US$0.24 is this doubling, not only screening (D22 attributed it to
    screening alone).
  - Recorded, not changed: screening never caches (its shared prefix is a few hundred tokens, below
    the 1,024 minimum); the M3 live screen calls read 0 cached tokens.
  - D22 corrected: the resumed understand call read 1,124 of 1,192 input tokens (1,134 was another
    run's call).
  - Researcher effort: not measured in M3. Claude's proposal, pending Ewan: a paired S3 arm with
    researcher effort high on the 15 dev queries in the E rounds (≈ US$0.12 or more, needs approval).
    Until then the product keeps low (D10).
  - `rank` compares violation quotes folded (`state.plain`, shared with understand's check);
    `screen` keeps judgements by bare arXiv id ("2401.00001v2", "arXiv:2401.00001" match); the
    subgraph reports `unjudged` shortlisted papers and S3 records their count per query.
  - `resolve` turns named ids and shown positions into stored ids (`arxiv:<id>[v<n>]`), one per
    paper, a versioned one winning over a bare one; `ingest_paper` pins a bare id to its latest
    version (one arXiv metadata request) before checking for a stored document, so a stored paper
    is not fetched again (D21) on this path either.
- Moved to E1 (Ewan): a paper the user names in a find-then-read request can be filtered out by
  screening when it is off-topic for the need (v1-c03 names GQA among tool-scheduling papers).
- Live check (2 Luna requests, `m3-review-20261008T051722`, US$0.00034): with the block,
  "最后一篇用了哪些数据集？" resolved to `listed: [3]`. On ref-mixed ("Compare the first one with
  2305.18323") the model returned `discover_read` with both references filled; code does not
  override it, so S4 will count it as an intent error. Not changed: it is the kind of model
  behaviour S4 measures, and with item 11 it belongs to E1.
