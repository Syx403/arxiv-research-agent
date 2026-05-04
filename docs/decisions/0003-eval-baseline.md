# ADR 0003: Evaluation Baseline

## Status
Accepted

## Dataset
Phase 10 defines a 30-question gold set over the seed corpus:

- 10 definition questions
- 8 comparison questions
- 6 multi-hop questions
- 4 survey questions
- 2 application questions

Each question records expected seed-corpus paper IDs, 3-5 required answer terms, and optional negative terms. The runner writes one row per question to `data/eval_outputs/latest/per_question.csv`.

## Metrics
- `recall_at_5`: fraction of expected paper IDs appearing in the deduplicated top-5 retrieved paper IDs.
- `mrr`: reciprocal rank of the first retrieved expected paper ID.
- `citation_precision`: fraction of cited paper IDs that are expected.
- `citation_recall`: fraction of expected paper IDs that are cited.
- LLM judge scores: `correctness`, `groundedness`, and `completeness` on 1-5 integer scales.

## Baseline Numbers
The baseline was measured on 2026-05-04 with `skip_reflection=True` and the resumable CSV checkpoint enabled.

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

Category-level recall@5:

| Category | Count | Recall@5 | MRR | Judge correctness |
|---|---:|---:|---:|---:|
| definition | 10 | 0.800 | 0.800 | 4.10 |
| comparison | 8 | 0.625 | 0.688 | 3.75 |
| multi_hop | 6 | 0.417 | 0.750 | 3.33 |
| survey | 4 | 0.312 | 0.375 | 1.75 |
| application | 2 | 1.000 | 1.000 | 5.00 |

## Known Weak Spots
- Seven questions produced failure rows: q-006, q-010, q-014, q-018, q-023, q-026, and q-028.
- Six failures were citation-format synthesis failures. One failure was a decomposition JSON validation failure.
- Survey questions had the weakest retrieval baseline: average recall@5 was 0.312 and average judge correctness was 1.75.
- Multi-hop questions had lower retrieval recall than definitions and comparisons, even when the final answer sometimes scored adequately.
- LangSmith dataset/example upsert required passing the settings-loaded API key directly to `langsmith.Client`; the initial eval pass ran before that fix, then runs were backfilled from the CSV rows.

## Next-Iteration Hypotheses
- Add a stricter citation-format postprocessor or constrained retry for broad survey answers to reduce `SynthesizerError` failures.
- Improve decomposition robustness by increasing JSON output budget or adding a conservative single-question fallback after repair fails.
- Add category-level eval reporting so survey and multi-hop regressions are visible without manual CSV analysis.
- Track provider token usage centrally so eval cost is estimated from observed usage rather than run-level call counts.
