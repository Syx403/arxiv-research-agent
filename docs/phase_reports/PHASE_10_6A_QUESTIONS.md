# Phase 10.6a Questions

## Q1 — Sample gate failed after structured synthesis implementation

**Status**: blocked; no commit made.

**Validation completed before block**:
- `uv run pytest tests/unit -q`: 115 passed.
- `uv run ruff check src tests scripts`: passed.
- Provider SDK grep across `src/eval src/graph src/llm src/memory src/retrieval`: `ok`.
- `rm -rf data/eval_outputs/latest`
- `make eval-sample`: failed threshold gate.

**Sample result**:
- `hard_failure_count`: 4
- `hard_failure_rate`: 0.80
- `estimated_cost_usd.total`: 0.08124427000000001
- `token_usage`:
  - `embed-small`: 12 calls, 599 prompt tokens
  - `fast`: 139 calls, 79,031 prompt tokens, 21,060 completion tokens
  - `main`: 29 calls, 94,956 prompt tokens, 18,332 completion tokens
  - `rerank`: 12 calls

**Per-question observations from `data/eval_outputs/latest/per_question.csv`**:

| q-id | hard_failure | recall@5 | MRR | retrieved_paper_ids | cited_paper_ids | answer |
|---|---:|---:|---:|---|---|---|
| q-005 | true | 1.0 | 1.0 | `["arxiv:2310.08560"]` | `[]` | empty |
| q-013 | false | 0.0 | 0.0 | `[]` | `[]` | Non-empty answer saying no evidence was supplied |
| q-019 | true | 0.5 | 1.0 | `["arxiv:2303.11366"]` | `[]` | empty |
| q-025 | true | 0.5 | 0.25 | 8 papers retrieved | `[]` | empty |
| q-029 | true | 0.5 | 1.0 | 3 papers retrieved | `[]` | empty |

**What changed**:
- The regex citation path was removed.
- The synthesizer now requires JSON with `answer` and `claims`.
- Four sample questions reached the degraded structured-synthesis path with an empty answer and no citations.
- q-013 did not mark as a hard failure, but retrieval still produced no evidence and the answer explicitly says no evidence was supplied.

**Likely cause**:
The new structured synthesizer contract is too brittle for the current prompt/model behavior. The eval rows indicate retrieval often produced usable evidence, but synthesis still degraded to an empty answer with no citations. Because `DIAGNOSTIC_LOG` was not enabled for the acceptance sample, raw synthesis responses were not captured, so I cannot yet distinguish empty provider responses from invalid JSON repair failure on those four questions.

**Recommendation requested from planner**:
Pick one path before continuing:
1. Run a diagnostic-only sample with `DIAGNOSTIC_LOG=1` for q-005/q-019/q-025/q-029 to capture raw structured synthesis and repair responses, then decide prompt/schema changes.
2. Loosen the structured output schema to preserve `answer` when JSON contains prose but malformed `claims`, while keeping citations fail-closed.
3. Add a bounded fallback that asks for answer-only JSON if the full `{answer, claims}` schema fails twice, then keeps `synthesis_format_degraded=True` but avoids empty answers for judge scoring.

**Constraints honored**:
- No full eval was run.
- No baseline snapshot was created.
- Locked Phase 10 baseline files were not intentionally modified.
- No commit was made.

## Q2 — A1' prose-with-markers sample still fails the hard-failure gate

**Status**: blocked; no commit made.

**A1' preflight result**:
- `make eval-preflight` on q-005 passed.
- q-005 had `hard_failure=False`, non-empty cited papers `["arxiv:2310.08560"]`, answer length 594, and estimated cost 0.019014049999999998.

**Validation completed before block**:
- `uv run pytest tests/unit -q`: 121 passed.
- `uv run ruff check src tests scripts`: passed.
- Provider SDK grep across `src/eval src/graph src/llm src/memory src/retrieval`: `ok`.
- `rm -rf data/eval_outputs/latest`
- `make eval-preflight`: passed.
- `rm -rf data/eval_outputs/latest`
- `make eval-sample`: failed threshold gate.

**Sample result after A1'**:
- `hard_failure_count`: 2
- `hard_failure_rate`: 0.40
- `estimated_cost_usd.total`: 0.10250192000000002
- `token_usage`:
  - `embed-small`: 11 calls, 530 prompt tokens
  - `fast`: 132 calls, 72,445 prompt tokens, 20,076 completion tokens
  - `main`: 44 calls, 121,597 prompt tokens, 33,607 completion tokens
  - `rerank`: 11 calls

**Per-question observations from `data/eval_outputs/latest/per_question.csv`**:

| q-id | hard_failure | recall@5 | MRR | retrieved_paper_ids | cited_paper_ids | answer |
|---|---:|---:|---:|---|---|---|
| q-005 | false | 1.0 | 1.0 | `["arxiv:2310.08560"]` | `["arxiv:2310.08560"]` | non-empty, 5/5/5 judge |
| q-013 | true | 0.0 | 0.0 | `[]` | `[]` | non-empty refusal: no evidence supplied |
| q-019 | false | 0.5 | 1.0 | `["arxiv:2303.11366", "arxiv:2310.04406"]` | `["arxiv:2303.11366"]` | non-empty, 4/5/5 judge |
| q-025 | true | 0.25 | 0.25 | 7 papers retrieved | `[]` | empty |
| q-029 | false | 0.5 | 1.0 | 3 papers retrieved | `[]` | non-empty, 4/5/5 judge, but all citations removed by verifier |

**What this means**:
- A1' fixed the systemic q-005 regression and restored the marker-based answer path.
- q-013 is not a synthesis-format problem; the retrieved evidence list is empty before synthesis, so the synthesizer reasonably produces a no-evidence answer that is marked degraded/hard-failure.
- q-025 still has an empty answer despite non-empty retrieval, so there remains a synthesis/provider failure on at least one survey-style question.
- q-029 is no longer a hard failure, but citation precision/recall are still zero because the batched verifier removed every parsed citation.

**Recommendation requested from planner**:
Pick one path before continuing:
1. Treat q-013 as retrieval architecture debt and q-025 as a remaining synthesis reliability bug; authorize a targeted q-025 diagnostic with `DIAGNOSTIC_LOG=1`.
2. Adjust the sample gate to distinguish no-evidence retrieval failures from synthesis hard failures, then proceed to full eval to measure actual distribution.
3. Move Phase 10.6a forward only with A5/B1/B2 and defer all synthesis/parser changes to 10.6b.

## Q3 — A5 revert sample did not complete due judge ReadTimeout

**Status**: blocked; no commit made.

**A5 revert completed**:
- `src/retrieval/self_rag/verifier.py` now uses per-citation `asyncio.gather` with `Semaphore(8)`.
- `_verify_one_citation` consumes `citation.claim_text` directly.
- 1200-character evidence truncation remains.
- Fail-closed parse fallback remains.
- Batched verifier code and `tests/unit/test_verifier_batching.py` are removed.

**Validation completed before block**:
- `uv run pytest tests/unit -q`: 119 passed.
- `uv run ruff check src tests scripts`: passed.
- Provider SDK grep across `src/eval src/graph src/llm src/memory src/retrieval`: `ok`.
- `make eval-preflight`: passed q-005 canary. Cost 0.033645800000000003; answer length 953; cited_paper_ids `["arxiv:2310.08560"]`.
- `make eval-sample`: crashed before summary generation.

**Sample crash**:
- Exception: `httpx.ReadTimeout`
- Location: `src/eval/metrics/llm_judge.py` calling `judge_answer`, via `src/llm/providers/deepseek.py`.
- No `summary.json` was written.
- Because the eval did not complete, I did not run full eval, did not freeze a baseline, and did not commit.

**Partial CSV rows written before crash**:

| q-id | hard_failure | retrieved_paper_ids | cited_paper_ids | judge | observation |
|---|---:|---|---|---|---|
| q-005 | false | `["arxiv:2310.08560", "arxiv:2502.12110"]` | `["arxiv:2310.08560"]` | 5/5/5 | Passes canary behavior |
| q-013 | true | `[]` | `[]` | 1/1/1 | Expected accepted failure mode: comparison-query retrieval returned empty evidence |
| q-019 | false | `["arxiv:2303.11366", "arxiv:2303.17651"]` | `[]` | 5/5/5 | Non-hard failure, but citations were still all removed despite a strong judge score |

**Concern**:
The A5 revert fixed the batched-verifier implementation shape, but the partial sample still shows a verifier over-rejection signal on q-019: prose judged 5/5/5, yet `cited_paper_ids=[]`. This is not a hard failure under the current runner rule, but it means citation metrics may remain depressed even with per-citation verification.

**Recommendation requested from planner**:
Pick one path before continuing:
1. Retry `make eval-sample` once, treating the `httpx.ReadTimeout` as provider flakiness, and accept only if q-013 is the sole hard failure.
2. Add retry handling around the eval judge call before re-running sample, because an unhandled judge timeout can waste a long sample/full eval late in the run.
3. Stop Phase 10.6a and move verifier over-rejection plus judge-timeout hardening into the next focused phase.

## Q4 — Network retry prevented crashes, but sample still failed relaxed gate

**Status**: blocked; no commit made.

**Network retry implemented**:
- `src/llm/retry.py` adds `retry_transient_network()` with 3 attempts for:
  - `httpx.ReadTimeout`
  - `httpx.ConnectTimeout`
  - `httpx.RemoteProtocolError`
  - `httpx.ConnectError`
- `_RoutedChatClient.chat()` wraps provider chat calls in the network retry helper.
- `OpenAIEmbeddingClient.embed()` wraps `embeddings.create()` in the same helper.
- `CohereRerankClient.rerank()` wraps `rerank()` in the same helper.
- Empty-response retry and JSON repair retry remain unchanged above the raw network-call layer.

**Validation completed before block**:
- `uv run pytest tests/unit -q`: 122 passed.
- `uv run ruff check src tests scripts`: passed.
- Provider SDK grep across `src/eval src/graph src/llm src/memory src/retrieval`: `ok`.
- `make eval-preflight`: passed q-005 canary with `hard_failure_count=0`; cost 0.01628976.
- `make eval-sample`: completed without network crash, but failed relaxed hard-failure gate.

**Sample result**:
- `hard_failure_count`: 4
- `hard_failure_rate`: 0.80
- `estimated_cost_usd.total`: 0.14203184000000002
- `token_usage`:
  - `embed-small`: 21 calls, 95,915 prompt tokens
  - `fast`: 265 calls, 204,474 prompt tokens, 71,271 completion tokens
  - `main`: 76 calls, 149,344 prompt tokens, 28,656 completion tokens
  - `rerank`: 17 calls

**Per-question observations from `data/eval_outputs/latest/per_question.csv`**:

| q-id | hard_failure | retrieved_paper_ids | cited_paper_ids | judge | observation |
|---|---:|---|---|---|---|
| q-005 | false | `["arxiv:2310.08560"]` | `["arxiv:2310.08560"]` | 5/5/5 | Pass |
| q-013 | true | `[]` | `[]` | 1/1/1 | Accepted comparison-query architecture debt |
| q-019 | true | `["arxiv:2303.11366", "arxiv:2310.04406"]` | `[]` | 5/5/5 | Non-q-013 hard failure: strong answer, all citations removed |
| q-025 | true | 7 papers retrieved | `[]` | 4/2/4 | Non-q-013 hard failure: answer present, citations removed |
| q-029 | true | 3 papers retrieved | `[]` | 5/5/5 | Non-q-013 hard failure: strong answer, all citations removed |

**Conclusion**:
The network retry fix worked for the crash class: the sample completed without `httpx.ReadTimeout`. The relaxed sample acceptance still fails because three non-q-013 questions have good prose but `cited_paper_ids=[]`. This is now a verifier/synthesis citation-survival issue, not a network reliability issue.

**Recommendation requested from planner**:
Pick one path before continuing:
1. Keep network retry but pause Phase 10.6a; move citation survival/verification over-rejection into Phase 10.6b before full eval.
2. Change the eval hard-failure definition so non-degraded answers with empty verified citations are scored as low citation recall rather than hard failures, then proceed to full eval.
3. Add diagnostic logging for verifier rejection rationales on q-019/q-025/q-029 and rerun only those three questions before deciding.

## Q5 — Mandatory citation prompt still failed sample gate

**Status**: blocked; no commit made.

**Prompt strengthening implemented**:
- `_generate_answer` now includes `MANDATORY CITATION RULE` and states that every substantive claim must end with an inline `[paper_id#chunk_id]` citation.
- `_fix_citation_format` now starts with `The previous answer is missing required [paper_id#chunk_id] citations` and instructs the model to preserve prose while attaching citations.
- `tests/unit/test_synthesize_degraded.py` now asserts both prompt strings are present in the two chat calls.

**Validation completed before block**:
- `uv run pytest tests/unit -q`: 122 passed.
- `uv run ruff check src tests scripts`: passed.
- Provider SDK grep across `src/eval src/graph src/llm src/memory src/retrieval`: `ok`.
- `make eval-preflight`: passed q-005 canary with `hard_failure_count=0`; cost 0.023880420000000003.
- `make eval-sample`: completed without network crash, but failed the explicit sample gate.

**Sample result**:
- `hard_failure_count`: 2
- `hard_failure_rate`: 0.40
- `estimated_cost_usd.total`: 0.14517201000000002
- `token_usage`:
  - `embed-small`: 18 calls, 39,909 prompt tokens
  - `fast`: 213 calls, 158,125 prompt tokens, 48,365 completion tokens
  - `main`: 97 calls, 180,854 prompt tokens, 35,393 completion tokens
  - `rerank`: 16 calls

**Per-question observations from `data/eval_outputs/latest/per_question.csv`**:

| q-id | hard_failure | retrieved_paper_ids | cited_paper_ids | judge | answer / citation observation |
|---|---:|---|---|---|---|
| q-005 | true | `["arxiv:2310.08560", "arxiv:2502.12110"]` | `[]` | 5/5/5 | Non-empty but truncated answer ending at literal `[arxiv`; no complete square-bracket marker was parseable. |
| q-013 | false | `["arxiv:2302.04761"]` | `["arxiv:2302.04761"]` | 1/5/1 | No longer the accepted retrieved=[] failure mode; low correctness remains comparison-query debt. |
| q-019 | false | `["arxiv:2303.11366", "arxiv:2310.04406"]` | `["arxiv:2303.11366"]` | 4/5/5 | Passes sample citation-survival requirement. |
| q-025 | true | 8 papers retrieved | `[]` | 1/1/1 | Empty answer despite non-empty retrieval. |
| q-029 | false | 3 papers retrieved | `[]` | 5/5/5 | Answer contains valid `[arxiv:2201.11903#...]` markers, but verifier removed every citation. |

**Conclusion**:
The stronger prompt improved citation production in some cases but did not meet the sample acceptance criteria. The explicit stop condition says: "Sample STILL fails (hard_failure_count > 1): STOP, write Q5 to QUESTIONS file. Do NOT do another iteration." Therefore no full eval was run, no baseline snapshot was created, and no commit was made.

**Recommendation requested from planner**:
Pick one path before continuing:
1. Ship the current implementation with a one-time relaxed sample gate and document q-005/q-025/q-029 as known Phase 10.6b/c residuals.
2. Stop Phase 10.6a and move unfinished synthesis/citation-survival work to Phase 10.6b before any full eval.
3. Authorize a targeted diagnostic of q-005/q-025/q-029 only, focused on finish reasons, truncated synthesis output, and verifier rejection rationales.
