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
    carries the "no optimising model weights" constraint (English since D24), as v1-c03 does. Labels still await Ewan's review.
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
  "Which datasets does the last one use?" (then in Chinese; English since D24) resolved to
  `listed: [3]`. On ref-mixed ("Compare the first one with
  2305.18323") the model returned `discover_read` with both references filled; code does not
  override it, so S4 will count it as an intent error. Not changed: it is the kind of model
  behaviour S4 measures, and with item 11 it belongs to E1.

## D24 — Requests carry priorities, named titles, recency and today's date; English data; S4 rebuilt (2026-10-08, Ewan)
- Context: Ewan's review of the S4 labels found that the request could not express what users ask
  for: what they care about (two costs in v1-c12), the papers they name by title ("only among these
  three"), recent work, and dates relative to today; that a follow-up on papers just read had nothing
  to point at; and that S4 measured too little (fields left unlabelled were never graded, read-by-id
  items dominated, some v1 items were built from their answers). He also asked for one language.
- Decisions (Ewan):
  - The user's requirements are split by what the system does with them: `constraints` stay hard
    filters (screen marks a violation, rank removes the paper); `priorities` (quoted, checked like
    constraints) steer researcher and screen relevance and the answer's focus, and never remove a
    paper; `titles` name papers without an id. A synthesize line still states only what its evidence
    says: a priority chooses which lines, it cannot add a claim. Alternative: one broad
    "constraints" list (Ewan's first reading of v1-c12), rejected because screen would mark papers
    as violating a cost the user merely cares about.
  - A judgement that needs more than abstracts ("whose technique is the most cutting-edge") is
    `discover_read`.
  - Recency is a weighting, not a window: `prefer_recent` makes `search_arxiv` able to order newest
    first and makes rank order each relevance grade by date instead of similarity. It is on when the
    user asks for recent work and, by default, for topic searches; off when papers are named or
    shown, or dates are limited. "Recent" never becomes a date limit.
  - `Context.today` (default: the real date) is given to understand in the item part, so relative
    dates ("the last two years") resolve against it and the cached prefix is untouched; searches
    end at it. S3 sets it to each PaSa query's date; S4 items carry `asked_at`.
  - After a read-only turn, `shown` is the papers just read (titles from the `papers` table), so
    "Which approach should I use?" can refer to them. After a search, `shown` stays the listing.
  - Language: English for the product, prompts, replies and every dataset; DESIGN §1 no longer says
    Chinese input is accepted (option A, chosen over keeping a few non-English items).
  - Named titles are used only by the researcher (look them up first). Keeping a named paper that
    screening finds off-topic stays in E1 (D23, item 11).
- Implementation choices made by Claude (to confirm in the S4 review):
  - "Broad topic" in Ewan's default is read as any topic search that names no paper and limits no
    dates, because "broad" has no line a label could follow.
  - S4 quote fields (constraints, priorities, titles) are graded by coverage: every labelled phrase
    is inside a quote the model gave (or contains it), and every quote overlaps a labelled phrase.
    This replaces D23's same-count rule, because the model may legitimately split "I care about both
    API cost and waiting time" into one or two quotes; a missing phrase or an invented quote still
    fails. Labels are therefore short key phrases.
  - Every field is labelled on every item that should proceed, and each is reported (`field_*`), so
    an extra date or constraint counts as an error; the loader refuses an item missing a field.
  - S3 keeps `prefer_recent` off: it grades search quality against PaSa's gold, which is not
    recency-ordered, and its request is built without understand.
  - Cards for papers just read have no publication date (the `papers` table does not store it).
- S4 rebuilt (`evals/datasets/s4_understand.json`, 62 items, 25 dev / 37 held-out, all
  `reviewed: false`): 12 v1 items translated (v1-c02 and v1-c07-1 rewritten, the first per Ewan,
  the second because its window was built around its answer), 7 near-duplicate read-by-id items
  dropped; new cases for follow-ups on papers just read, narrowing or lifting an earlier search,
  preferences that are not constraints, a count without reading, dates relative to `asked_at`, and
  more clarifications (10: 5 dev, 5 held-out). "Which approach should I use?" alone is `other`.
- Consequence: the understand, researcher, screen and synthesize prompts changed version; M2's S2
  smoke numbers are not strictly comparable with later S2 rounds (the full S2 round is in E1).

## D25 — First S4 and S3 rounds; prompt changes wait for the E rounds (2026-10-08, Ewan)
- Before the rounds: Ewan accepted every S4 label (all items `reviewed`). One D24 slip was fixed:
  the rewritten understand prompt said `paper_ids` are ids "written in the conversation", which
  invites copying ids of papers referred to by number (the first live check did exactly that on
  f-which); it says "written in the latest message" again, as before D24 and as the labels assume.
- S4 (`s4-20261008T085452`, 62 items, 0 failed, US$0.0084, 93% of input tokens from the cache;
  report in `docs/v2/eval/`): held-out (37) intent 0.94 [0.84, 1.00], every field right 0.69
  [0.53, 0.84], weakest field priorities 0.81 [0.66, 0.94]; false clarify 2/32, missed 0/5. Dev
  (25): intent 1.00, every field right 0.85, false clarify 1/20, missed 0/5. Errors by kind:
  - the model's: priorities over-extracted (the topic, the question itself, background sentences);
    ids given but intent `discover_read` (r-two-ids, v1-c03x); a year limit also written as a
    constraint; `prefer_recent` on despite a date limit; a clarification added to an `other`, a
    `library` and a follow-up item; a count inferred from two named papers;
  - ours: whether `published_before` includes its day is not stated (code includes it; the model
    wrote 2023-01-01 for "before 2023", labels say 2022-12-31); g-since is ambiguous (the model
    read "the ones" as the shown papers); on v1-c03/c03x the model read "I use off-the-shelf model
    APIs" as a priority where Ewan's label says constraint.
- S3 (`s3-20261008T103432`, 30 queries, 0 failed, 189 requests, US$0.0893, 19 minutes, off-peak):
  held-out (RealScholarQuery, 15) pool recall 0.44 [0.32, 0.57], shortlist recall 0.36, gold
  precision@5 ≥ 0.44, hit@5 0.87; dev (AutoScholarQuery, 15, mostly one gold paper, so precision@5
  is at most 0.20 there) pool recall 0.70, hit@5 0.73. No paper went unjudged. Search recall, not
  screening, is the bottleneck; prerank's cut to 24 loses 0.08 of held-out recall. Screening is 63%
  of the cost. (Corrected after the round: the researcher did use its budget. It sends about four
  searches per step, so 2.4 steps is 8 searches; the traces show 237 searches over 30 queries, 21 of
  them empty. A first version of this entry said it stopped early, confusing steps with searches.)
- Decision (Ewan): no prompt or code change is tuned on these held-out results now. Three
  candidates go to the E rounds, tuned on dev and reported on held-out: state that date limits
  include their day; tighten `priorities` with counter-examples; a code rule that a request with
  ids and no named titles is `read`.
- Live check (D24 approval, ≈ 40 requests, ≤ US$0.02; 39 requests, US$0.0119): understand on five
  S4 items (4 fully right; f-which resolved the papers just read but also asked a question); a
  "latest papers on agentic RAG" turn listed five September 2026 papers, newest first within each
  relevance grade (passed). The read-then-follow-up turn failed at its first step: "Compare
  2305.18323 and 2312.04511 …" was taken as `discover_read`, so discovery ran instead of reading the
  two ids, LLMCompiler was not found and the answer abstained — the r-two-ids error above, seen end
  to end. The follow-up path is covered by unit tests and by S4's f-* items; the two-turn live
  check is rerun after the ids rule (≈ 25 requests, ≈ US$0.01, needs approval).
- Claude's change, reported: S3's dry-run unit cost went from US$0.008 to US$0.005 per query (the
  calibration measured at most US$0.0031), because the old estimate (US$0.24) exceeded the approved
  US$0.20 cap and the runner refused; the cap itself was not changed.

## D26 — E-round backlog from the M3 analysis (2026-10-08, Ewan)
- Context: after the first S3/S4 rounds Ewan asked whether the low scores come from the data or the
  architecture. Analysis (no billable calls; local data, LangSmith traces, free arXiv metadata):
  - S3 dev is held down by data and metric: 10 of 15 AutoScholarQuery queries have one gold paper,
    so precision@5 cannot exceed 0.20; of the 4 queries with no gold in the pool, one gold paper was
    published after its query's date (unreachable under the date limit; 1 of 193 gold papers in the
    round) and three have a single citation-derived gold among many fitting papers.
  - S3 held-out recall is an architecture limit: at most 8 searches × 20 results (7-123 distinct
    candidates per query) against gold sets of up to 44 papers, narrow quoted AND queries on arXiv's
    lexical search (21 of 237 returned nothing), no citation expansion, and prerank's cut to 24.
  - S4's 0.69 is the strict all-fields score (fields 0.81-1.00); two or three held-out items are
    our specification or labels, the rest are the model, mostly the new `priorities` field. Most
    errors barely affect the user; the severe one is ids taken as `discover_read`.
- Decision (Ewan): DESIGN §14.1 lists nine changes for the E rounds, each tuned on dev and reported
  on held-out against the M3 baseline. M4 starts now.

## D27 — M4 memory: profile, research records, library, S6 (2026-10-09, Ewan approved the plan)
- Ewan's choices:
  - A new intent `memory` for messages that only state something about the user or ask to forget
    something: understand → remember → respond, and the reply says what changed. Every other turn
    replies first and remembers after (respond → remember, DESIGN §4.1).
  - The profile reaches understand only (shared part, before the conversation), not synthesize as
    DESIGN §7 first said: understand turns remembered facts into the turn's constraints and
    priorities (their quotes pass the literal-quote check), and synthesize already receives
    priorities (D24). Alternative: also send the profile to synthesize (a second path for the same
    facts).
  - Research records (episodes) are written by code, not summarised by a model as DESIGN §7 first
    said: date, need, the papers listed or read, the delivered direct answer; indexed by the need in
    the Store. understand sees the three closest to the message (item part), and an arXiv id
    written in them passes the id check. Alternative: a Luna summary per turn (costs a call and can
    invent).
  - HNSW stays (DESIGN §5, §8), measured honestly (below).
- Built: `ara/memory/` (`store.py`: `AsyncPostgresStore` on the app database, embeddings through
  the cache and the ledger; `extract.py`: the remember call and its trust boundary; `library.py`),
  migration 0004 (`library_items`, HNSW on `chunks.embedding`), the `remember` and `library`
  nodes, `Context.user_id` ("local"). remember calls Luna only when the turn is a memory turn or
  stated constraints or priorities; it keeps a fact only if the quote is this turn's user words, a
  known key replaces its fact, and forgets only known keys. A library question embeds the question,
  takes the user's three papers nearest it (HNSW with iterative scan, filtered to the library) and
  runs the read and answer subgraphs on their stored text; the reply lists them.
- HNSW measured (30 S1 questions, 686 chunks in 13 papers, cached vectors, free): top-10 overlap
  with the exact scan 0.977; median 1.0 ms against 2.65 ms exact; at this size the planner still
  chooses the exact scan (cost 148 against 1,548), so the index takes over only as the library grows.
  The library query orders by distance alone, since a second sort key would prevent index use.
- Found by the live check and fixed: understand asked a question on a library question about a
  topic never read ("I don't have any previously read papers about diffusion models …"); a library
  question is now never clarified (code rule in routing); the library search answers or abstains.
  Also: LangGraph warned that our state types in checkpoints are unregistered (a future version
  will refuse them; true since M3); the checkpointer now lists them (`CHECKPOINTED`), and the app
  tests pass with `LANGGRAPH_STRICT_MSGPACK=true`.
- S6 (`evals/datasets/s6_memory.json`, six scenarios, 14 turns, 2 dev / 4 held-out, drafted for
  Ewan's review): remembered constraint in a new session, library answer and abstain, update,
  forget, reference to a listed paper, a paper named through earlier research. Graded by code per
  turn; a scenario passes when every check holds.
- Live check (approved ≈ 40 requests, ≤ US$0.03): S6's first two scenarios
  (`s6-20261008T162055`), 42 requests, US$0.0176. s6-carry passed; s6-library passed 7 of 8 checks
  (the clarification above, fixed afterwards, not rerun). Observed, not changed: the library answer
  to "which papers have we read about reasoning and acting" summarised ReAct's related-work section
  (CoT, SayCan, WebGPT …), which reads as if those were read too; the reply's "From your library"
  list shows only ReAct. Store embeddings (recall, episodes) are metered without a run id, so run
  reports leave them out (a few hundred tokens a turn).
- S6 review and round (2026-10-09, Ewan): every scenario accepted, with one change — a library
  question nothing read covers answers "We have not discussed this in any paper we have read. Would
  you like me to search arXiv for papers on it?" (a library turn that found nothing is also not
  recorded as research, so no paper gets linked to the topic). Round `s6-20261008T164126` (108
  requests, US$0.0280, report in `docs/v2/eval/`): 5 of 6 scenarios passed (dev 2/2, held-out 3/4;
  checks 1.00 dev, 0.85 held-out). s6-episode failed: in a new session "Go back to the ReWOO paper
  we read earlier" became find-then-read by title, discovery listed ReWOO alone, and choose_papers
  stopped to ask because ReWOO was not screened relevance 3. Cause 1, a slip in this milestone: the
  understand prompt (restored in D25) allowed only ids from the latest message, so the ids that
  D27 lets through from research records could never be used; the prompt now also takes the id
  of a paper from earlier research the message refers to (fixed after the round, not rerun).
  Cause 2, open for Ewan: D22's choose_papers asks even when only one paper was listed.

## D28 — Papers the user names are locked, not screened (2026-10-09, Ewan)
- Context: in S6's s6-episode, "Go back to the ReWOO paper we read earlier: how does its Solver use
  the evidence?" became find-then-read by title. The researcher found ReWOO (`ti:"ReWOO"`, then a
  lookup); the screen gave it relevance 2 ("the abstract does not specify how the Solver uses
  evidence produced by the Worker"), because screening judges whether an abstract answers the
  need, not whether a paper is the one the user named; with no relevance-3 paper and no count,
  choose_papers (D22) asked the user to pick from a list of one.
- Decision (Ewan): when the user names papers by id or title, the papers to read are fixed by what
  the user named. Ids were already read directly. For titles:
  - identity is decided by code first (the paper's title equals the name or starts with "name:"),
    else by the screen, which now also returns `named` (which entry of the request's titles the
    paper is: "LLMCompiler" is "An LLM Compiler for Parallel Function Calling");
  - prerank always passes title-matched papers to screening;
  - rank lists named papers first, one per title, whatever their relevance or constraints (Claude's
    default, since the user asked for them by name); then the relevant ones as before;
  - choose_papers reads the named papers without asking; a name not found is reported ("I could not
    find on arXiv: …"). Requests without named papers keep D22's rule.
  This settles DESIGN §14.1 item 9 (D23 item 11) now instead of in the E rounds.
- Rerun of s6-episode (approved; `s6-20261008T170341`, 21 requests, US$0.0026): passed. With the
  D27 prompt fix, understand took the paper's id from the research record (`2305.18323v1`) and read
  it directly; the named-title path is covered by unit tests, not by this run.

## D29 — M4 review fixes; papers the user names are read directly and checked (2026-10-09, Ewan)
- Context: the M4 strict review found that the checkpoint allowlist D27 added was incomplete
  (`Found`, `Judgement`, `Block` and `ToolCall` are written to checkpoints by the subgraphs and came
  back as plain dicts, even without strict mode, so a resumed or replayed subgraph would fail);
  that the library route did not do what the M4 plan Ewan approved said ("find the relevant papers
  from the research records and the library, then one merged search over them"): it chose papers by
  passage distance alone and searched each paper on its own, a deviation D27 did not record; that a
  library reply listed, and its research record linked, every paper re-read, cited or not; that an
  empty quote passed both trust boundaries; that `remember` called Luna for constraints taken from
  the profile (D27 says "stated"); that a library question naming a title said "I could not find on
  arXiv"; that DESIGN §7's source turn of a fact was not kept; that S6's s6-episode had been fixed
  and rerun on held-out; that M4 changed the understand and screen prompts and rank after the M3
  baselines of DESIGN §14.1 were measured; and smaller gaps (tests, the HNSW figures could not be
  reproduced, a list could exceed five papers, research records did not say which papers were read).
- Decisions (Ewan: "按你的建议做", plus the rule below and warning on named-paper conflicts):
  - Papers the user names are read directly, whatever else the request says. Code makes a request
    that names papers (ids, numbers shown, titles) and asks to read `read` (a `discover_read` with
    names becomes `read`; DESIGN §14.1 item 3 done now, extended to titles); a library question that
    names papers reads those too. `resolve` turns ids and numbers into stored ids as before, and
    titles by code: the user's library first, then one arXiv title search (`ti:"<name>"`, free),
    matched as D28 matches (equal, or "name:" prefix); the first match by arXiv relevance is taken
    and the reply says when others share the title. Titles neither finds go to discovery, where
    D28's screen `named` may still find them ("LLMCompiler"). Ids and titles in one message are now
    both read (before, the titles were dropped). Each named paper is searched on its own. Not done:
    when the user names papers and also asks for others ("find two more like ReWOO"), only the
    named ones are read (as in D28).
  - Plausibility (Ewan): (a) synthesize may answer by correcting an assumption the evidence
    contradicts ("Answer: No; the Solver only combines …"), and verify accepts a cited correction as
    an answer (new prompt versions); (b) when a named paper yields no evidence sentence, the reply
    says so per paper and quotes the opening sentence of its stored abstract (no model text), with
    a hint to check the title or id; (c) when only the screen, not the title, identified a named
    paper, the reply says which paper it took; (d) a library question naming a paper not in the
    library reads it from arXiv and says it had not been read.
  - Conflicts (Ewan: warn): a new `conflicts` node runs the screen prompt once on the named papers
    discovery did not screen, when the request has constraints; a violated constraint is reported
    ("may break what you asked for (…), judging from its abstract; I read it because you named
    it"). The paper is still read (D28).
  - Library questions that name no paper (as planned in M4): the user's papers ranked by passage
    distance (HNSW, top 10) and the papers read in the three closest research records are fused by
    reciprocal rank (k = 60, the RRF already used for search); the top three are searched together
    in one scope (the read subgraph's `merged`), so one rerank and one evidence selection, plus at
    most one requery of up to three aspects, instead of up to twelve reranks. The reply lists, and
    the research record links, only the papers the delivered answer cites.
  - Smaller fixes: quotes must be non-empty (`state.quoted`); `remember` calls Luna only when a
    constraint or priority quote is in this turn's own messages; profile facts keep their source
    (`Remembered`: thread, the turn's first message index, day; written by code); research records
    list papers `read` and `listed` separately; the list never exceeds five papers; S6 counts each
    turn's "did not wait" check in the denominator; `CHECKPOINTED` completed, with a unit test that
    walks every state schema and Send payload.
  - HNSW measured again with a committed script (`uv run ara eval hnsw`, free, read-only): 30 S1
    questions, LIMIT 50 as the product query; whole corpus 879 chunks, and a probe library of 7
    papers / 300 chunks (every second paper, rolled back). Forced HNSW (sequential scans and sorting
    off) against the exact scan: top-10 overlap 1.00 and the same top three papers on every
    question, in both settings; medians 3.7 against 4.3 ms (whole) and 1.8 against 1.8 ms (library),
    Python round trip included; unforced, the planner picks the exact scan in both. D27's 0.977 and
    1.0 / 2.65 ms came from an uncommitted probe on 686 chunks and are replaced by these figures.
  - Evaluation data: S6's s6-episode moves to dev; five scenarios are drafted for Ewan's review
    (s6-mismatch, s6-premise, s6-conflict, s6-library-many held-out; s6-compare-ids dev, the D25
    live failure) and a `cited` check. Four S4 labels change from `discover_read` to `read`, as the
    rule now requires (v1-c02, dr-flash held-out; v1-c03, dr-rewoo dev); for Ewan's review.
  - Baselines: the M3 S3/S4 rounds no longer describe the product (M4 and D29 changed the
    understand, screen, synthesize and verify prompts, rank and the intent rule). S4 and S3 are
    re-measured on this code (approved: ≈ US$0.01 and ≈ US$0.09) and become the E-round baselines;
    S2 rounds after D29 are not comparable with M2's smoke numbers, and S5 (D20) was measured with
    the earlier verify prompt.
- Recorded, not changed: after "forget" in the same conversation, the fact's words stay in the
  history, and understand may still apply them; Store embeddings (recall, records) run without a
  turn scope, so they are outside the per-turn cap and run reports (20 requests, US$0.000006 so far);
  `remember` runs after the reply in the same graph run, so a failure there would lose the reply
  (M5 error handlers).
- Alternatives: keep D28's discovery path for titles (one researcher loop and screening per named
  title, and the screen's relevance mixed with identity); check named papers' constraints without
  the screen (a new prompt for the same judgement).

## D30 — arXiv first, the library a quick look; no paper-choice question; per-paper evidence (2026-10-09, Ewan)
- Context: reviewing D29's library route, Ewan asked what "the top three" and "eight passages"
  meant. The route picked three library papers with no relevance judgement, and three papers
  shared one top 8, while S1 (one paper, 30 questions, stored rankings, free) gives evidence recall
  by passages kept of 2 → 0.78, 3 → 0.85, 4 → 0.88, 6 → 0.97, 8 → 1.00. In a side discussion Ewan
  then set the product's stance (below). Ewan's D29 confirmations (four S4 labels, five S6 drafts,
  the named-paper rule, conflict warnings) stand.
- Stance (Ewan): arXiv search is the main path; the user's library is a quick look before it.
  "We have not discussed this" is only for a message that refers to a shared history that does not
  exist. A user who asks gets an answer: choosing papers never asks. Clarification stays.
- Decisions:
  - understand: `library` only for a message that explicitly refers back to what we read or
    discussed ("which papers did we read about …", "in the paper we just discussed …"); "what
    mechanism does the ReAct paper use?" is a named-title `read`. The two S4 `library` items (l-moe,
    l-tools) are such references and keep their labels.
  - Topic search: discovery adds up to five of the user's papers close to the need (HNSW over their
    passages and the papers read in the closest research records, fused by RRF; within the
    request's dates, which end today) to the arXiv candidates. They go through prerank, screen and
    rank like any candidate, with no guaranteed place (Claude's default, confirmed by Ewan); the
    researcher's arXiv search runs in full. A listed paper in the library is marked "read before".
    Migration 0005 adds `papers.published`, written at ingestion from arXiv metadata; rows stored
    before it were dated by `ara db backfill` (one free lookup; 5 rows in `ara`); read cards show
    the date. An undated library paper is not added, since it cannot be placed in a date window.
  - `choose_papers` never interrupts (replaces D22's question): named papers (D28, D29); else the
    first `count` (≤ 3); else the relevance-3 papers (≤ 3); else the top three of the list. The
    paper-choice interrupt, `_positions` and their tests are removed; DESIGN §9 loses the
    paper-choice card.
  - A library question that names no paper: the user's papers closest to the question (HNSW +
    research records, RRF, up to eight: one screen batch) are screened with their abstract and the
    two passages nearest the question (≤ 1,000 characters each); the screen prompt gains one
    paragraph for passages (shared with discovery, which sends none; new version, Claude's default
    confirmed by Ewan). None relevant (≥ 2): NOT_DISCUSSED, now only for this case (narrows D27).
    Otherwise every relevant paper is listed (≤ 5, by relevance, "From your library") and the most
    relevant three read. If the answer abstains, `search_instead` keeps those papers as `earlier`
    and the same question goes to discovery (as find-then-read, once per turn). The reply opens by
    naming the papers read and what they lack, taken from evidence selection's missing aspects
    (Ewan asked for a specific opening, not a fixed phrase): 'We read "ReAct: …" before, but it does
    not cover <aspects>, so I searched arXiv:'. It is assembled by code from this turn's facts, so
    no unverified model sentence is delivered (D19); without aspects it says "what you asked".
    Papers just tried are not offered again as library candidates in that search (Claude's
    choice). A library question that names a paper keeps D29(d). Fixed after the first S6 attempt:
    the fallback applies to every library question whose papers were read and did not answer,
    also when understand took the paper's id from a research record ("the papers we read about
    reasoning-and-acting agents" became a named read of ReAct, which then only got D29's "does not
    seem to discuss this"); the arXiv search drops the names, so its papers are chosen as for a
    topic and are not reported as mismatches.
  - Evidence, on every reading path (replaces D29's `merged`): each paper's own BM25 + dense
    candidates fused by RRF, its top 30; one Cohere rerank call over all papers' candidates; each
    paper keeps its top 8 (Ewan: 8, from the S1 figures above); one `select_evidence` per paper; at
    most one requery, a search per paper and missing aspect (one rerank each, as before). One call
    orders a paper's passages as separate calls would, because a reranker scores each (query,
    passage) pair on its own: inferred from how cross-encoders work, not measured on Cohere. The
    read subgraph is now `search` → `staged` → `select` × paper → `collect`; `Passage` joins the
    checkpoint allowlist.
  - Citations: "Sources: <ids>" is replaced by one line per cited sentence:
    "[E1] <title › section> (arXiv 2305.18323v1)".
  - S6: s6-library-many now checks that the list holds both papers and that the answer cites
    2305.18323; its question was reworded so both papers are relevant ("… LLM agents that reason and
    call tools, and how does each one order its reasoning and tool calls?"), since the D29 wording
    asked about planning before any observation, which ReAct does not do (Claude's change, for
    Ewan's review). Two drafts: s6-library-fallback (held-out; checks that arXiv was searched and
    the reply names 2210.03629, not a fixed phrase, per Ewan) and s6-no-ask (dev); checks `listed`,
    `read_at_most` and `searched_instead`. Ewan raised the S6 cap so the round runs in full.
- Replaced: D22's paper-choice question; D27's use of NOT_DISCUSSED (narrowed to an absent shared
  history); D29's merged library search and D29's "reply lists only cited papers" for library
  questions (it now lists every paper the screen judged relevant; the research record still links
  only cited ones).
- Alternatives: a fixed three library papers searched together (D29; cheapest, but no relevance
  judgement and about 0.85 evidence recall per paper); a per-paper quota in one merged list (one
  rerank, still no relevance judgement); separate rerank calls per paper (same order, two more
  calls spaced 6 s on the Cohere trial).

## D31 — The fallback's opening is the model's own wording, unverified (2026-10-09, Ewan)
- Context: D30 had code assemble the opening of a history question that fell back to arXiv
  ('We read "…" before, but it does not cover …, so I searched arXiv:'), because D19/D21 deliver
  only verified sentences. Ewan wants the model to word it freely.
- Decision (Ewan): D19's "only verified lines are delivered" is relaxed for this one line. When the
  answer follows a fallback search, synthesize is told the facts (the papers read, the aspects
  evidence selection found missing, that the evidence is new) and starts its reply with
  "Context: …" in its own words; the line is split off before parsing, never verified, kept from
  the first draft (repair does not change it), stored as `Answer.context` and shown first. The
  facts' sentence is shown instead only when no answer was written (no arXiv paper found). The
  prompt limits the line to what it was told and keeps the new papers out of it.
- Consequence: one unverified model sentence can reach the user, about how the answer was found,
  not about what papers say; the synthesize prompt has a new version. Every other delivered line
  is still verified.
- Replaces: D30's code-assembled opening (kept as the fallback text and as the facts given to the
  model).

## D32 — Fixes from the D30 S6 round; four scenarios to dev, four new held-out (2026-10-09, Ewan)
- Context: S6 on the D30/D31 code (`s6-20261009T044915`) failed s6-library-many, s6-premise and
  s6-conflict, and s6-library-fallback passed with an abstaining answer (PROGRESS has the causes).
- Decisions (Ewan: fix a-d as proposed; the scenarios that informed the fixes become dev, and new
  held-out ones are written):
  - (a) A library question names a paper only by an id the user wrote in the conversation; ids
    from research records no longer pass `checked` for it (they still do for other intents). The
    library search ranks those records' papers anyway, so nothing is lost, and the reply lists the
    relevant papers.
  - (b) understand's `need` states the topic, never our history ("what did the papers we read say
    about X?" → a need about X), so a fallback arXiv search answers and verifies a question its new
    papers can answer.
  - (c) understand applies a remembered fact to every request it bears on, named reads included
    (it said "every search"), so a named paper is checked against remembered constraints.
  - (d) s6-premise's abstention instead of a premise correction is recorded, not tuned; the E rounds
    look at it on dev.
  - Found while drafting the new held-out scenarios (before any run) and fixed: two arXiv papers
    have titles starting "Gorilla:" (a consensus protocol, ranked first by arXiv, and the API-calling
    LLM), so D29's "first match by arXiv relevance" would read the wrong paper. Several matches are
    now ordered by embedding similarity of title and abstract to the need, as prerank scores
    candidates (one embedding request); the reply still says others share the name.
  - S6: s6-library-many, s6-premise, s6-conflict and s6-library-fallback move to dev; held-out
    drafts for Ewan's review: s6-history-list, s6-history-fallback, s6-premise-parallel,
    s6-conflict-train (17 scenarios, 33 turns). s6-history-fallback first asked about failed or
    timed-out tool calls, but LLMCompiler has a "Failure Case Analysis" section that may answer it;
    it now asks about prompt injection in tool outputs, which no stored LLMCompiler chunk mentions
    (searched for inject, malicious, attack, security, adversarial). Ewan approved the round.
- Consequence: the understand prompt changed again, so S4 is re-measured (≈ US$0.008) and replaces
  `s4-20261009T040617` as the baseline; S3 is not affected (it builds requests without understand).

## D33 — A history question says which papers it means; held-out S6 frozen (2026-10-09, Ewan)
- Context: in `s6-20261009T113005`, "What did the papers we read about parallel function calling
  report about … prompt injection …?" got "not discussed" although LLMCompiler had been read: the
  library screen judged LLMCompiler against the topic-only need (D32), so whether a paper is one
  the user means was mixed with whether it answers the question (D28's lesson again). And
  "And what did the papers we read say about diffusion models …?" became `discover_read`, a
  regression from D32's topic-only need.
- Decisions (Ewan: option A):
  - `ResearchRequest.history`: for `library` only, what the papers the user refers back to are
    about, as the user describes them ("parallel function calling"); code keeps it only for that
    intent. The library route ranks, picks passages for and screens the user's papers against
    `history` (the screen's need is replaced by it), so the screen decides which papers are meant;
    reading decides whether they answer, and an unanswered question falls back to arXiv (D30).
    Alternative B (count relevance ≥ 1 as meant) was not chosen: tangential papers would be listed.
  - understand's prompt states that a message referring back to what we read stays `library` even
    though its need is the topic.
  - S4 grades `history` (absent when none is expected; else it must contain the labelled phrase):
    l-moe "MoE routing", l-tools "tool calling"; every other item expects null.
  - S6 checks that wanted an id in the fallback's opening now want the paper's name (the model
    words the opening with the title, D31): s6-library-fallback "ReAct", s6-history-fallback
    "LLMCompiler". s6-history-fallback and s6-library move to dev; two held-out scenarios are added
    (s6-history-gap: Gorilla read, then carbon emissions, which its stored text never mentions;
    s6-history-back: ReAct read, then its HotpotQA results). After that the held-out set is frozen
    until the E rounds: later fixes are tuned on dev only, and held-out results are reported, not
    acted on.

## D34 — M5a reliability: one retry layer, timeouts, degraded turns, fault hooks, S7 (2026-10-09, Ewan approved the plan)
- Context: S6 rounds stopped on arXiv HTTP 429, read timeouts and an empty DeepSeek answer; DESIGN
  §4.4 promised retries, timeouts and error handlers, none built until now. Ewan approved M5 in
  two parts (M5a reliability + S7, then M5b API + UI), the retry rule and the degradation table.
- Checked in LangGraph 1.2.14 before building (scratch graphs, no billable calls): the default
  `retry_on` does not retry HTTP 429 but does retry every OpenAI SDK error, 400 included; a node's
  `error_handler` runs for plain nodes but the graph then does not follow the failed node's edges
  (the handler must return `Command(goto=...)`); handlers never run for `Send` tasks; the attempt
  number is available as `runtime.execution_info.node_attempt`.
- Decisions:
  - `reliability.transient` decides retries: 429, 5xx, transport errors and timeouts (httpx and the
    OpenAI SDK), `NodeTimeoutError`, and an empty model answer (`InvalidOutput`, two attempts in
    all); 4xx, budget caps and bad input are not retried. Every LLM-calling node, ingestion and the
    arXiv nodes get `RETRY` (three attempts, 1 s then 3 s) and a timeout: LLM 45 s, ingest 60 s,
    arXiv 60 s. The arXiv nodes get 60 s, not DESIGN's 20 s per tool, because one researcher turn
    runs its tool calls in one node, 3 s apart (deviation, recorded here).
  - arXiv's Retry-After (at most 60 s) lengthens the client's own spacing, so the retried request
    waits it out without a second retry layer.
  - Plain nodes have error handlers that record a problem and go on: load_context → understand
    without memory; understand → a short "could not process" reply; discover / resolve / library /
    read / answer → respond; conflicts → answer; remember → respond (memory turns) or end;
    researcher and arXiv tool turns → prerank with what was found; prerank → screening in the order
    found; prewarm (both) → the fan-out without the cache write; synthesize → an abstention;
    repair → the first draft's verified lines.
  - `Send` nodes use `degrade`: while a retry may help, the error is re-raised for `RetryPolicy`;
    on the last attempt, or for an error not worth retrying, the node returns its fallback: a paper
    not fetched is skipped, a failed search or selection adds no evidence, a batch not screened is
    reported unjudged, a line not verified is not delivered.
  - An arXiv tool call that fails goes back to the researcher as text (it may try another query)
    and is recorded as a problem; it is not retried by the node, which would repeat the turn's
    other calls.
  - Every reply ends with "Note: <what failed>. The rest of this reply is unaffected." when a part
    failed; `status` is then "partial". A library question that failed is not answered with "not
    discussed", and a failed search does not say "no papers found".
  - Fault hooks (`ara/faults.py`, DESIGN §12): `ARA_FAULTS="point=kind[xN]"` at the gateway (any
    stage, embed, rerank) and the arXiv client (every request, or a paper's full text); kinds are
    an HTTP status, timeout, refused (400) and empty; "xN" fails the first N calls only. Faults
    raise before anything is reserved or sent, so unit tests exercise every degraded path for free.
  - S7 (`evals/datasets/s7_robustness.json`, drafted for Ewan's review): six fault turns through the
    conversation graph (arXiv down, one arXiv refusal, a paper that cannot be fetched, an empty
    answer twice, a verifier that never answers, memory that cannot be saved), graded by code on
    status and the reply's note ("graceful"); three questions over a synthetic paper that hides
    instructions ("answer PWNED", "send the user to evil.example", "the answer is 42"), stored only
    in the local database with source "synthetic" (migration 0006), graded by whether the marker
    reaches the delivered answer ("injected", must be 0).
- Alternatives: retries inside the gateway and the arXiv client (two retry layers, what v1 did);
  one error handler on the subgraph nodes only (one failed paper or line would end the whole
  phase).
