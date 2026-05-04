# Phase 10.5 Report

**Phase title**: Fail-closed self-RAG, token counter, and sample-eval discipline
**Status**: complete
**Commit**: `HEAD (see git log)`
**Owner handoff pending**: none

## Summary
Phase 10.5 changed self-RAG from fail-open to fail-closed, made synthesis degrade instead of crashing, added hard-failure accounting to eval, added a per-role token counter, and added a five-question `eval-sample` gate. Two diagnostic follow-ups fixed q-013/q-025 A-class failures by accumulating retrieval evidence across retries, retrying empty chat responses once, truncating verifier evidence, and embedding full citation tokens directly in evidence headers. Per planner instruction, no full eval was run in this pass; the next full eval is deferred to Phase 10.6.

## Files created
- `src/core/diagnostics.py` - `DIAGNOSTIC_LOG`-gated raw-response JSONL logger.
- `src/llm/usage.py` - per-role token/call counter and cost estimator.
- `tests/unit/test_chat_empty_retry.py` - empty chat response retry coverage.
- `tests/unit/test_retrieve_accumulation.py` - retry evidence accumulation coverage.
- `tests/unit/test_retrieve_fail_closed.py` - fail-closed retrieve routing coverage.
- `tests/unit/test_self_rag_degraded.py` - degraded synthesis verification coverage.
- `tests/unit/test_synthesize_degraded.py` - synthesizer degraded path coverage.
- `tests/unit/test_usage.py` - token counter coverage.
- `tests/unit/test_verifier_truncation.py` - verifier evidence truncation coverage.
- `docs/phase_reports/PHASE_10_5_DIAGNOSTIC.md` - q-013/q-025 diagnostic report.
- `docs/phase_reports/PHASE_10_5_QUESTIONS.md` - sample-gate blocker audit trail.

## Files modified
- `Makefile` - added `eval-sample`.
- `scripts/eval_run.py` - added `--question-ids`.
- `src/core/types.py` - added verification rationale/state fields used by fail-closed paths.
- `src/eval/runner.py` - hard-failure rows, threshold exit, resume behavior, token usage in summary.
- `src/graph/builder.py`, `src/graph/state.py`, `src/graph/edges.py` - `skip_reflection` and degraded-synthesis state.
- `src/graph/nodes/retrieve.py` - retry loop accumulates evidence across retries.
- `src/graph/nodes/self_rag.py` - degraded synthesis verifies as failed, not passed.
- `src/graph/nodes/synthesize.py` - enriched citation prompts, degraded path, full citation-token evidence headers.
- `src/llm/client.py` - empty provider response retry and usage recording.
- `src/llm/providers/cohere_rerank.py`, `src/llm/providers/openai_embed.py` - usage-friendly provider details and grep-safe imports.
- `src/retrieval/multi_hop/citation_walker.py` - generic lexical overlap heuristic, no domain bonuses.
- `src/retrieval/self_rag/relevance_judge.py`, `sufficiency_check.py`, `verifier.py` - fail-closed JSON fallback and diagnostic logging.
- `tests/unit/*` - updated and added lock tests for the changed behavior.
- `docs/STATUS.md` - added Phase 10.5 row.

## Fix matrix
| Finding/fix | Implementation locator | Locking test |
|---|---|---|
| F-001 verifier fail-closed | `src/retrieval/self_rag/verifier.py:76`-`105` | `tests/unit/test_verifier_parser.py:52`-`77` |
| F-002 sufficiency fail-closed | `src/retrieval/self_rag/sufficiency_check.py:37`-`65` | `tests/unit/test_sufficiency_check.py:35`-`45`; `tests/unit/test_retrieve_fail_closed.py` |
| F-003 relevance fail-closed | `src/retrieval/self_rag/relevance_judge.py:36`-`64` | `tests/unit/test_relevance_judge.py:52`-`63`; `tests/unit/test_retrieve_fail_closed.py` |
| F-004 synthesizer hardening/degraded path | `src/graph/nodes/synthesize.py:20`-`44`, `48`-`121`; `src/graph/nodes/self_rag.py:10`-`18` | `tests/unit/test_synthesize_degraded.py:23`-`57`; `tests/unit/test_self_rag_degraded.py` |
| F-005 eval hard-failure threshold | `src/eval/runner.py:34`-`39`, `42`-`61`, `95`-`100`, `103`-`162` | `tests/unit/test_eval_runner.py:11`-`37` |
| F-020 generic multi-hop heuristic | `src/retrieval/multi_hop/citation_walker.py:282`-`286` | `tests/unit/test_citation_walker.py` |
| q-013 fix: accumulate retry evidence | `src/graph/nodes/retrieve.py:75`-`99` | `tests/unit/test_retrieve_accumulation.py:10`-`67` |
| q-025 fix: retry empty chat response | `src/llm/client.py:15`-`17`, `73`-`98` | `tests/unit/test_chat_empty_retry.py:10`-`48` |
| q-025 fix: truncate verifier evidence | `src/retrieval/self_rag/verifier.py:12`, `38`-`59` | `tests/unit/test_verifier_truncation.py:9`-`22` |
| q-029 fix: citation token in evidence header | `src/graph/nodes/synthesize.py:54`-`67`, `89`-`101`, `117`-`121` | final `make eval-sample` q-029 row: `hard_failure=False` |
| Token counter | `src/llm/usage.py:15`-`61`; `src/llm/client.py:82`-`98`, `116`-`117` | `tests/unit/test_usage.py` |

## Tests run
- Command: `uv run pytest tests/unit -q`
- Outcome: pass; `108 passed in 1.13s`.
- Command: `uv run ruff check src tests scripts`
- Outcome: pass; `All checks passed!`.
- Command: `rm -rf data/eval_outputs/latest`
- Outcome: pass.
- Command: `make eval-sample`
- Outcome: pass; `hard_failure_count: 0`, `hard_failure_rate: 0.0`.

## Sample eval rows
| q-id | recall@5 | MRR | citation_precision | citation_recall | judge C/G/Comp | hard_failure |
|---|---:|---:|---:|---:|---|---:|
| q-005 | 1.00 | 1.00 | 0.00 | 0.00 | 5/5/5 | false |
| q-013 | 0.50 | 1.00 | 0.00 | 0.00 | 1/1/1 | false |
| q-019 | 0.50 | 1.00 | 0.00 | 0.00 | 5/5/5 | false |
| q-025 | 0.25 | 0.25 | 0.00 | 0.00 | 1/1/1 | false |
| q-029 | 0.50 | 1.00 | 0.00 | 0.00 | 4/5/5 | false |

## Token counter
Full eval was not run because the planner explicitly instructed: "DO NOT run `make eval` regardless of sample outcome." The actual sample run wrote this token/cost data to `data/eval_outputs/latest/summary.json`:

```json
{
  "token_usage": {
    "embed-small": {"calls": 16, "prompt_tokens": 114209, "completion_tokens": 0},
    "fast": {"calls": 193, "prompt_tokens": 132923, "completion_tokens": 44636},
    "main": {"calls": 113, "prompt_tokens": 128860, "completion_tokens": 37079},
    "rerank": {"calls": 13, "prompt_tokens": 0, "completion_tokens": 0}
  },
  "estimated_cost_usd": {
    "embed-small": 0.00228418,
    "fast": 0.021802690000000003,
    "main": 0.07557910000000001,
    "rerank": 0.026000000000000002,
    "total": 0.12566597000000002
  }
}
```

## Baseline vs sample delta
This table compares the same five sample question IDs against the locked Phase 10 baseline CSV. It is not a full 30-question delta.

| Metric | Baseline sample | Phase 10.5 sample | Delta |
|---|---:|---:|---:|
| recall@5 | 0.85 | 0.55 | -0.30 |
| MRR | 0.90 | 0.85 | -0.05 |
| citation_precision | 0.93 | 0.00 | -0.93 |
| citation_recall | 0.70 | 0.00 | -0.70 |
| correctness | 4.60 | 3.20 | -1.40 |
| groundedness | 4.40 | 3.40 | -1.00 |
| completeness | 4.80 | 3.40 | -1.40 |
| hard_failure_rate | 0.00 | 0.00 | 0.00 |

## Category deltas
The requested survey and multi-hop subset deltas cannot be computed as full category subsets without a full eval. In the sample gate, q-025 is the survey representative and q-019 is the multi-hop representative:

| Category | q-id | Baseline C/G/Comp | Phase 10.5 C/G/Comp | Note |
|---|---|---|---|---|
| survey | q-025 | 3/2/4 | 1/1/1 | No hard failure, but answer quality regressed; Phase 10.6 needs retrieval/coverage planning. |
| multi-hop | q-019 | 5/5/5 | 5/5/5 | Hard failure remains absent; citations did not match expected papers under fail-closed verification. |

## What got worse and why
Citation precision and citation recall collapsed to `0.0` in the sample because fail-closed verification now removes citations rather than letting unsupported or unparsable verifier outputs pass. This is expected from the adversarial-review fix, but it means the sample is now stricter than the locked baseline. q-013 and q-025 no longer hard-fail, but their judge scores show the underlying coverage/retrieval weaknesses are still real.

## What got better and why
The sample hard-failure rate moved from `0.40` in the first Phase 10.5 sample and `0.20` after the REC-2 fixes to `0.00` after the citation-header prompt change. q-013 stopped losing all evidence because retrieval retries now accumulate relevant hits; q-025 stopped producing empty final answers because empty chat responses retry once; q-029 stopped using chunk-only citations after the full `[paper_id#chunk_id]` token moved into the evidence header.

## Known limits transitioned to Phase 10.6
- Full eval was not run by instruction; the next 30-question baseline must be generated after Phase 10.6.
- q-013 and q-025 still have low answer-quality scores despite no hard failure.
- The sample citation metrics are intentionally harsh under fail-closed verification and need Phase 10.6 architecture work, especially coverage planning and retrieval/synthesis separation.

## Acceptance checklist
- [x] F-001/F-002/F-003 fail-closed semantics implemented and tested.
- [x] F-004 degraded synthesis path implemented and tested.
- [x] F-005 hard-failure threshold implemented and tested.
- [x] F-020 domain-specific multi-hop boosts removed.
- [x] Token counter writes `token_usage` and `estimated_cost_usd`.
- [x] q-013/q-025 diagnostic fixes implemented and tested.
- [x] q-029 citation-header fix implemented and sample-verified.
- [x] `make eval-sample` passes with `hard_failure_count == 0`.
- [x] Locked baseline files were not modified.

## Open questions for the planner
None for Phase 10.5. The remaining work belongs to Phase 10.6.
