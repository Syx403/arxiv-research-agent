# Phase 10 Report

**Phase title**: Evaluation Harness
**Status**: complete
**Commit**: `HEAD (see git log)`
**Owner handoff pending**: none

## Summary
Phase 10 added the gold-question evaluation harness, metric implementations, LLM judge, resumable CSV runner, LangSmith dataset/run export, and baseline ADR. It also added two graph/runtime improvements required by the planner: bounded parallel citation verification and `skip_reflection=True` for eval runs. The final resumed baseline over 30 questions is: recall@5 `0.6250`, MRR `0.7167`, citation precision `0.6722`, citation recall `0.5694`, judge correctness `3.6000`, groundedness `3.6000`, completeness `3.7000`.

## Files created
- `src/eval/datasets/gold_questions.py` - 30-question gold dataset.
- `src/eval/metrics/__init__.py` - metrics package marker.
- `src/eval/metrics/retrieval_metrics.py` - recall@k and MRR.
- `src/eval/metrics/citation_metrics.py` - citation precision/recall.
- `src/eval/metrics/llm_judge.py` - LLM judge with JSON repair.
- `src/eval/runner.py` - resumable CSV eval runner and LangSmith export.
- `scripts/eval_run.py` - executable eval entrypoint.
- `docs/decisions/0003-eval-baseline.md` - baseline ADR.
- `tests/unit/test_retrieval_metrics.py` - retrieval metric tests.
- `tests/unit/test_citation_metrics.py` - citation metric tests.
- `tests/unit/test_gold_questions_consistency.py` - dataset consistency tests.
- `tests/unit/test_eval_runner.py` - failure-row coverage.
- `tests/unit/test_llm_judge.py` - judge fallback coverage.
- `tests/unit/test_verifier_parallel.py` - concurrent verifier fan-out test.
- `tests/unit/test_reflect_skip.py` - `skip_reflection` isolation test.
- `tests/unit/test_runner_resume.py` - CSV checkpoint resume test.

## Files modified
- `Makefile` - added `eval` help comment and `eval-fresh`.
- `src/core/types.py` - added `JudgeVerdict`.
- `src/graph/state.py` - added `skip_reflection`.
- `src/graph/builder.py` - added `run(..., skip_reflection=False)`.
- `src/graph/nodes/reflect.py` - short-circuits reflection when requested.
- `src/retrieval/self_rag/verifier.py` - verifies citations with `asyncio.gather` and `Semaphore(8)`.
- `src/memory/semantic/extractor.py` - preserves empty extraction fallback from Phase 9 eval hardening.
- `tests/unit/test_extractor_empty_retry.py` - covers the extractor fallback.
- `docs/STATUS.md` - Phase 10 marked complete.
- `docs/interview_talking_points.md` - Phase 10 talking points appended.

## Tests run
- Command: `uv run pytest tests/unit -q`
- Outcome: pass; `93 passed in 1.37s`.
- Command: `grep -RIE "from openai|from cohere|import openai|import cohere" src/eval src/graph src/memory src/retrieval || echo "ok"`
- Outcome: pass; printed `ok`.
- Command: `uv run ruff check src tests scripts`
- Outcome: pass; `All checks passed!`.
- Command: `make eval`
- Outcome: pass; full run wrote `data/eval_outputs/latest/per_question.csv` and `summary.json`.
- Command: resume sanity sequence (`cp`, truncate first 10 rows, `make eval`, `diff ... | head -20`)
- Outcome: pass; `make eval` printed `Skipping 10 already-completed questions` and restored 30 rows.

## Validation commands
- Command: `make eval`
- Output (final resumed summary): ```
Skipping 10 already-completed questions
{
  "avg_citation_precision": 0.6722222222222223,
  "avg_citation_recall": 0.5694444444444444,
  "avg_completeness": 3.7,
  "avg_correctness": 3.6,
  "avg_groundedness": 3.6,
  "avg_mrr": 0.7166666666666667,
  "avg_recall_at_5": 0.625,
  "cost_note": "Token usage is not centrally tracked; estimates count graph runs and judge calls only.",
  "estimated_judge_calls": 30,
  "estimated_run_level_calls": 30,
  "total_questions": 30,
  "total_skipped_via_resume": 10
}
```
- Command: `ls data/eval_outputs/latest/`
- Output: ```
per_question.csv
summary.json
```
- Command: `test -f docs/decisions/0003-eval-baseline.md && echo "baseline ADR present"`
- Output: ```
baseline ADR present
```
- Command: `diff /tmp/eval_full.csv data/eval_outputs/latest/per_question.csv | head -20`
- Output (truncated): ```
12,18c12,22
< q-011,...judge fallback from first pass...
---
> q-011,...resumed row with fixed judge score...
> q-012,...resumed row with fixed judge score...
> q-013,...resumed row with fixed judge score...
```
- Command: `docker exec arxiv_agent_pg psql -U arxiv_agent -d arxiv_agent -c "SELECT count(*) FROM concepts;"`
- Output: ```
 count
-------
     0
(1 row)
```
- Command: `docker exec arxiv_agent_pg psql -U arxiv_agent -d arxiv_agent -c "SELECT count(*) FROM sessions WHERE session_id LIKE 'eval-%';"`
- Output: ```
 count
-------
     0
(1 row)
```
- Command: `docker exec arxiv_agent_pg psql -U arxiv_agent -d arxiv_agent -c "SELECT count(*) FROM checkpoints WHERE thread_id LIKE 'eval-%';"`
- Output: ```
 count
-------
     0
(1 row)
```

## Acceptance checklist
- [x] `make eval` completes and writes per-question CSV plus aggregate summary.
- [x] LangSmith dataset/example upsert works with the settings-loaded API key; 30 completed CSV rows were backfilled as LangSmith runs.
- [x] Re-running `make eval` against a truncated CSV skips completed question IDs and processes only the remainder.
- [x] Baseline numbers are recorded in ADR 0003.
- [x] Interrupted eval behavior is covered by unit test and command-level resume sanity.
- [x] `skip_reflection=True` keeps eval from writing concepts; observed `concepts` count remains `0`.
- [x] Provider SDK grep returns `ok`.

## Deviations from the plan
- `judge_answer` uses `max_tokens=2000` and repair `max_tokens=1000`. The initial `500` cap caused DeepSeek to return empty JSON for judge calls; a single live probe confirmed the same prompt returned valid JSON at `2000`.
- The runner writes conservative failure rows for graph exceptions instead of aborting the full eval. This preserved the full 30-row baseline and exposed seven agent failures in ADR 0003.
- LangSmith setup initially failed because `langsmith.Client()` did not read the project `.env`; the runner now passes the settings-loaded API key explicitly, then dataset/examples and runs were backfilled from the final CSV.

## Open questions for the planner
None.

## Ready for review
The planner can read these files and rerun the free validation commands above. The live eval outputs are under ignored `data/eval_outputs/latest/`.
