# Phase 10.6 Report

**Phase**: 10.6 — Engineering Rigor + Foundation
**Status**: complete
**Commit**: HEAD (see git log)
**Date**: 2026-05-05
**Owner handoff pending**: none

## Summary

Phase 10.6 folds the uncommitted Phase 10.6a foundation into one rigor pass and implements F1-F8 from `docs/reviews/ENGINEERING_RIGOR_REVIEW_2026-05-05.md`. The full eval passed the typed hard-failure gate: 1 hard failure out of 30 (`0.0333`), no `retrieval_empty` rows, 5 `verifier_all_rejected` warning rows, and estimated cost `$0.8282`.

New baseline frozen at `data/eval_outputs/baseline_2026_05_05_phase10_6/`.

## Foundation Integrated

| Item | Locator | Locking test |
|---|---|---|
| Citation sentence parser populates `Citation.claim_text`; `claim_span` spans the full sentence | `src/core/types.py:65`, `src/graph/nodes/synthesize.py:192` | `tests/unit/test_sentence_around.py`, `tests/unit/test_parse_citations_claim_text.py` |
| Per-citation verifier consumes `claim_text` | `src/retrieval/self_rag/verifier.py:31` | `tests/unit/test_verifier_uses_claim_text.py` |
| `retrieve.py` decomposed into 9 named stages with structured logs | `src/graph/nodes/retrieve.py:30`, `src/graph/nodes/retrieve.py:82`, `src/graph/nodes/retrieve.py:239` | `tests/unit/test_retrieve_stages.py` |
| LLM call structured logging and token usage | `src/llm/client.py:77`, `src/llm/client.py:263` | `tests/unit/test_llm_call_logging.py`, `tests/unit/test_usage.py` |
| Chat network retry + empty response retry | `src/llm/client.py:78`, `src/llm/client.py:91` | `tests/unit/test_chat_network_retry.py`, `tests/unit/test_chat_empty_retry.py` |
| Synthesizer citation mandate and `max_tokens=2000` | `src/graph/nodes/synthesize.py:141`, `src/graph/nodes/synthesize.py:62` | `tests/unit/test_synthesize_degraded.py` |

## F1-F8 Fixes

| Fix | Locator | Locking test |
|---|---|---|
| F1 synthesis evidence cap: 8 chunks x 800 chars; `finish_reason="length"` compact retry then degraded path | `src/core/types.py:19`, `src/graph/nodes/synthesize.py:27`, `src/graph/nodes/synthesize.py:59`, `src/graph/nodes/synthesize.py:167` | `tests/unit/test_synthesize_evidence_cap.py`, `tests/unit/test_synthesize_finish_length.py` |
| F2 sufficiency evidence cap: 5 chunks x 600 chars; non-parseable finish reasons fail closed | `src/core/types.py:21`, `src/retrieval/self_rag/sufficiency_check.py:16`, `src/retrieval/self_rag/sufficiency_check.py:79`, `src/retrieval/self_rag/sufficiency_check.py:96` | `tests/unit/test_sufficiency_evidence_cap.py`, `tests/unit/test_sufficiency_check.py` |
| F3 verifier rubric uses planner-approved "substantial support" standard | `src/retrieval/self_rag/verifier.py:42` | `tests/unit/test_verifier_rubric.py` |
| F4 comparison preservation: entity detection, validation retry, deterministic fallback | `src/retrieval/query/decomposer.py:18`, `src/retrieval/query/decomposer.py:95`, `src/retrieval/query/decomposer.py:123` | `tests/unit/test_decomposer_comparison.py` |
| F5 provider robustness generalized to SDK/network exceptions and empty embed/rerank validation | `src/llm/retry.py:22`, `src/llm/retry.py:96`, `src/llm/providers/openai_embed.py:46`, `src/llm/providers/cohere_rerank.py:51` | `tests/unit/test_provider_sdk_retry.py`, `tests/unit/test_embed_count_mismatch.py`, `tests/unit/test_rerank_empty_validation.py` |
| F6 typed hard-failure counters split hard classes from warning classes; threshold restored to 0.10 | `src/eval/runner.py:32`, `src/eval/runner.py:46`, `src/eval/runner.py:341` | `tests/unit/test_failure_classification.py` |
| F7 json_object risk remediation: multi-hop scorer batch 8 + 300-char fields; extractor 6000-char guard | `src/retrieval/multi_hop/citation_walker.py:27`, `src/retrieval/multi_hop/citation_walker.py:173`, `src/retrieval/multi_hop/citation_walker.py:290`, `src/memory/semantic/extractor.py:19` | `tests/unit/test_multi_hop_score_batch_cap.py` |
| F8 per-question diagnostic columns | `src/eval/runner.py:62`, `src/eval/runner.py:330`, `src/eval/runner.py:341` | `tests/unit/test_eval_row_diagnostics.py` |

## Decisions Encoded

- F3 uses the planner-selected "substantial support" verifier rubric: paraphrase, hedges, and reasonable inference count as supported; unrelated, contradictory, or omitted assertions fail.
- F6 typed counters make `agent_exception`, `synthesis_empty`, `citation_parse_empty`, and `judge_invalid` hard failures. `retrieval_empty` and `verifier_all_rejected` are warning classes and do not count against `hard_failure_rate`.
- Phase 10.6 is a single commit. `eval-preflight` and `eval-sample` are canaries; the full eval is the validation artifact.

## Validation Commands

```text
uv run pytest tests/unit -q
136 passed in 1.86s

uv run ruff check src tests scripts
All checks passed!

grep -RIE "from openai|from cohere|import openai|import cohere" src/eval src/graph src/llm src/memory src/retrieval || echo "ok"
ok

make eval-preflight
q-005 passed; hard_failure_rate=0.0; estimated_cost_usd.total=0.01357911

make eval-sample
5 rows; hard_failure_rate=0.0; verifier_all_rejected=1 warning; estimated_cost_usd.total=0.14901377

make eval
30 rows; hard_failure_rate=0.03333333333333333; estimated_cost_usd.total=0.82823017
```

## Token Usage

| Role | Calls | Prompt tokens | Completion tokens | Cost USD |
|---|---:|---:|---:|---:|
| embed-small | 127 | 370408 | 0 | 0.00740816 |
| fast | 1625 | 924789 | 343332 | 0.16086819 |
| main | 782 | 470696 | 278969 | 0.43395382 |
| rerank | 113 | 0 | 0 | 0.22600000 |
| **total** | 2647 | 1765893 | 622301 | **0.82823017** |

## Aggregate Deltas

| Metric | Phase 10 baseline | Phase 10.6 | Delta |
|---|---:|---:|---:|
| hard_failure_rate | 0.2333 legacy | 0.0333 typed | -0.2000 |
| recall@5 | 0.6250 | 0.7333 | +0.1083 |
| MRR | 0.7167 | 0.9028 | +0.1861 |
| citation_precision | 0.6722 | 0.7333 | +0.0611 |
| citation_recall | 0.5694 | 0.5056 | -0.0639 |
| judge_correctness | 3.6000 | 4.1667 | +0.5667 |
| judge_groundedness | 3.6000 | 4.3667 | +0.7667 |
| judge_completeness | 3.7000 | 4.3667 | +0.6667 |

Failure classes in Phase 10.6: success `24`, `agent_exception=1`, `verifier_all_rejected=5`, all other hard classes `0`.

## Category Deltas

| Category | Metric | Phase 10 | Phase 10.6 | Delta |
|---|---|---:|---:|---:|
| definition | recall@5 | 0.800 | 1.000 | +0.200 |
| definition | cit_recall | 0.800 | 0.900 | +0.100 |
| definition | correctness | 4.10 | 5.00 | +0.90 |
| comparison | recall@5 | 0.625 | 0.750 | +0.125 |
| comparison | cit_recall | 0.625 | 0.375 | -0.250 |
| comparison | correctness | 3.75 | 4.00 | +0.25 |
| multi_hop | recall@5 | 0.417 | 0.417 | +0.000 |
| multi_hop | cit_recall | 0.306 | 0.278 | -0.028 |
| multi_hop | correctness | 3.33 | 3.50 | +0.17 |
| survey | recall@5 | 0.312 | 0.500 | +0.188 |
| survey | cit_recall | 0.188 | 0.250 | +0.062 |
| survey | correctness | 1.75 | 3.00 | +1.25 |
| application | recall@5 | 1.000 | 0.750 | -0.250 |
| application | cit_recall | 0.750 | 0.250 | -0.500 |
| application | correctness | 5.00 | 5.00 | +0.00 |

## What Got Worse And Why

Citation recall dropped overall from `0.5694` to `0.5056`, with the largest drops in comparison and application questions. This is expected from the move to sentence-level `claim_text` and fail-closed verification: Phase 10's citation metrics were inflated by a fail-open verifier and marker-only `claim_span` behavior. Phase 10.6 exposes five `verifier_all_rejected` warning rows rather than hiding them as successful verification.

One comparison question (`q-011`) still failed hard with `EmptyProviderResponseError` after the provider returned empty content twice for `main`. The network and empty-response retries worked as designed, but this is still a provider-boundary residual for Phase 10.7.

## What Got Better And Why

The typed hard-failure rate dropped from the Phase 10 legacy failure rate of `7/30` to `1/30`. Retrieval quality improved materially (`recall@5 +0.1083`, `MRR +0.1861`) after comparison-preserving decomposition and accumulated retrieval stages. Survey quality improved from failing prose/citation behavior to usable outputs, and diagnostics now distinguish agent exceptions, citation parsing failures, retrieval empties, and verifier rejections.

## Backlog Status

Closed in this phase: F1, F2, F3, F4, F5, F6, F7 minimal remediation, and F8. Also closed the uncommitted foundation items: sentence-level `claim_text`, retrieve stages, LLM structured logs, chat network retry, citation mandate, and synthesis `max_tokens=2000`.

Still pending in `docs/ENGINEERING_BACKLOG.md`: full structured JSON synthesis, batched verifier redesign, semantic-memory prompt/provider remediation, survey coverage criteria, relevance `maybe_relevant`, cost breaker, durable resume cost accounting, and concurrency/idempotency production work.

## New Baseline

Frozen snapshot:

```text
data/eval_outputs/baseline_2026_05_05_phase10_6/
  per_question.csv
  summary.json
```

Original Phase 10 baseline files under `data/eval_outputs/baseline_2026_05_04_phase10/` and `docs/eval_baselines/` were not modified.
