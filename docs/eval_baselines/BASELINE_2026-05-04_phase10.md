# Eval Baseline — Phase 10 ship

**Date**: 2026-05-04
**Commit**: bd43599 (`phase 10: evaluation harness`)
**Snapshot path** (gitignored): `data/eval_outputs/baseline_2026_05_04_phase10/`
**Run config**: `skip_reflection=True`, sequential (no question-level concurrency), DeepSeek V4 Pro/Flash + OpenAI text-embedding-3-small + Cohere rerank-v4.0-pro

This file is the durable record of the Phase 10 baseline. The CSV/JSON snapshot lives outside git in the path above; numbers and per-category breakdown are duplicated here so the baseline survives even if `data/` is wiped.

## Aggregate

| Metric | Value |
|---|---:|
| total_questions | 30 |
| avg_recall_at_5 | 0.6250 |
| avg_mrr | 0.7167 |
| avg_citation_precision | 0.6722 |
| avg_citation_recall | 0.5694 |
| avg_correctness | 3.6000 |
| avg_groundedness | 3.6000 |
| avg_completeness | 3.7000 |

## Per-category

| Category | n | recall@5 | MRR | cit_p | cit_r | corr | grnd | comp |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| definition  | 10 | 0.800 | 0.800 | 0.750 | 0.800 | 4.10 | 4.00 | 4.10 |
| application |  2 | 1.000 | 1.000 | 1.000 | 0.750 | 5.00 | 4.50 | 5.00 |
| comparison  |  8 | 0.625 | 0.688 | 0.625 | 0.625 | 3.75 | 3.62 | 3.88 |
| multi_hop   |  6 | 0.417 | 0.750 | 0.667 | 0.306 | 3.33 | 3.83 | 3.50 |
| survey      |  4 | 0.312 | 0.375 | 0.417 | 0.188 | 1.75 | 1.75 | 2.00 |

## Failures (7 / 30 = 23%)

| q-id | category | error |
|---|---|---|
| q-006 | definition | SynthesizerError (citation format) |
| q-010 | definition | SynthesizerError (citation format) |
| q-014 | comparison | ValidationError (Decomposition JSON) |
| q-018 | comparison | SynthesizerError (citation format) |
| q-023 | multi_hop  | SynthesizerError (citation format) |
| q-026 | survey     | SynthesizerError (citation format) |
| q-028 | survey     | SynthesizerError (citation format) |

6 of 7 failures are the citation-format gate.

## Key observations (planner notes)

- Definition + application work well; survey collapses; multi_hop has high MRR but low recall@5 (finds *one* of N expected papers).
- Multi-hop walker is rarely triggered: sufficiency_check returns sufficient=True after retrieving a single relevant chunk, so the bidirectional citation expansion built in Phase 6 is mostly idle in real eval runs.
- Citation recall (0.57) lags citation precision (0.67) — agent is conservative.
- Verifier silently defaults to `supports=True` when JSON parses fail twice; this artificially inflates groundedness and verification.passed.
- Input Mode (graph_retriever) was never exercised because eval ran with `skip_reflection=True` and the corpus was never seeded with concepts. The Phase 8 retrieval bias has not been measured end-to-end.

## How to compare future runs

1. Run `make eval` (writes to `data/eval_outputs/latest/`).
2. Diff the new summary.json against `data/eval_outputs/baseline_2026_05_04_phase10/summary.json`.
3. Expect category-level deltas: improvements should not regress definition/application above noise (~0.05 metric points).
