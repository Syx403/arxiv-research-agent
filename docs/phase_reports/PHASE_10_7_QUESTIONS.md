# Phase 10.7 Questions

## Q1 — Eval sample canary became a runaway multi-hop/scorer run

**Status**: blocked; no commit made.

**Validation completed before block**:

- `uv run pytest tests/unit -q`: `167 passed in 1.40s`
- `uv run ruff check src tests scripts`: `All checks passed!`
- Provider SDK grep across `src/eval src/graph src/llm src/memory src/retrieval`: `ok`
- `rm -rf data/eval_outputs/latest`
- `make eval-preflight`: passed q-005 canary.

**Preflight result**:

- `hard_failure_rate`: `0.0`
- q-005 judge: `5/5/5`
- `estimated_cost_usd.total`: `0.02814017`
- `multi_hop_attempts_total`: `1`
- `multi_hop_expansions_total`: `1`

**Sample run status**:

- Command: `make eval-sample`
- I stopped the run with `pkill -f scripts/eval_run.py` after it issued repeated multi-hop/scorer calls well beyond the expected canary envelope.
- Exit: `make: *** [eval-sample] Error 143`
- No `summary.json` was written.
- Partial CSV contains only q-005, which passed:
  - `hard_failure=False`
  - `retrieved_paper_ids=["arxiv:2310.08560"]`
  - `cited_paper_ids=["arxiv:2310.08560"]`
  - judge `5/5/5`
  - `multi_hop_attempts=0`, `multi_hop_expansions=0` in the partial sample row

**Observed during q-013 before stop**:

- Repeated `multi_hop_preemptive_decision` and `retrieve_stage` loops.
- Multiple `multi_hop_skipped_non_arxiv_candidates` messages.
- Multiple `multi_hop_score_parse_failed` and `multi_hop_scorer_failed` warnings.
- Repeated DeepSeek `fast` scoring calls from the G3 per-item fallback path.
- At least one lazy-ingest path was exercised earlier in the preflight/sample session, including arXiv and Semantic Scholar calls.

**Why this is a blocker**:

The G1/G2/G3 mechanisms are wired and unit-tested, but the live canary shows the combined policy can exceed the intended Phase 10.7 sample envelope before the sample completes. The likely interaction is:

1. G1/G2 now allow multiple multi-hop attempts under the dual-counter budget.
2. Sufficiency-driven fallback can still route to multi-hop after the preemptive path and across decomposition sub-questions.
3. G3 converts malformed/empty batch scorer output into up to eight single-item scorer calls per failed batch.
4. q-013 is a comparison question with multiple decomposed sub-questions, so the fallback multiplication happens in the canary rather than only in full eval.

This is not an eval-scoring issue. It is a mechanism/cost-control issue in the approved G1-G3 interaction.

**Planner decision needed**:

Clarify the intended live-call envelope before another sample/full eval:

1. Should sufficiency-driven multi-hop be limited to at most one post-preemptive call per question, despite `MULTI_HOP_MAX_ATTEMPTS=5`?
2. Should sufficiency-driven multi-hop be disabled for `question_type=="comparison"` now that F4 owns comparison recall, or should it remain available as a bottom safety net?
3. For G3, should a batch-level scorer parse failure fall back to the deterministic lexical score instead of up to eight single-item LLM calls when running inside eval, or is the extra fan-out still required?
4. Should `multi_hop_scorer_failed` be surfaced into `EvalRow.failure_class`/warning counters rather than logs only, so the full eval can quantify this failure class if it proceeds?

**Constraints honored**:

- No full eval was run.
- No baseline snapshot was created.
- Locked Phase 10 and Phase 10.6 baseline folders were not intentionally modified.
- No commit was made.
