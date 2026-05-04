# Phase 10.5 Questions

## Q1 — Sample eval gate fails after fail-closed changes

**Status**: blocked, awaiting planner decision

**Context**: Phase 10.5 requires `make eval-sample` to complete with `hard_failure_count == 0` before the full eval can run. I implemented the fail-closed verifier, sufficiency, and relevance behavior; degraded synthesis no longer raises; token usage is recorded; and the sample target was run from a cleared `data/eval_outputs/latest/`.

**Validation result**:

```text
make eval-sample
...
EVAL FAILED: hard failure rate 0.40 exceeds threshold 0.10
hard_failure_count: 2
hard_failure_rate: 0.4
estimated_cost_usd.total: 0.12433626
token_usage:
  embed-small: 20 calls, 102049 prompt tokens
  fast: 207 calls, 132597 prompt tokens, 42904 completion tokens
  main: 88 calls, 138371 prompt tokens, 30582 completion tokens
  rerank: 15 calls
```

**Hard failures in the sample**:

- `q-013` (`How does Toolformer differ from ToolLLM for tool use?`): `hard_failure=True`; retrieved paper IDs were empty. The degraded answer states that no evidence was provided, so no citations could be produced.
- `q-025` (`What approaches exist for tool use in LLM agents?`): `hard_failure=True`; retrieved paper IDs were `["arxiv:2308.11432", "arxiv:2307.16789", "arxiv:2305.15334", "arxiv:2308.00352", "arxiv:2304.08354"]`, but the final degraded answer was empty.

**Interpretation**: The fail-closed changes are doing what the adversarial review requested: invalid or unsupported intermediate judgments no longer masquerade as success. That exposes two existing weaknesses before the full eval:

1. Relevance/sufficiency fail-closed can leave the graph with no usable evidence for a comparison question.
2. Degraded synthesis prevents graph crashes, but the current hard-failure gate still fails when no valid `[paper_id#chunk_id]` citations survive.

**Options**:

1. Keep Phase 10.5 narrow and stop here. Treat this as a successful discovery by the new sample gate, then move q-013/q-025 retrieval/synthesis coverage into Phase 10.6 architecture work.
2. Allow one more Phase 10.5 patch focused only on avoiding empty evidence/empty answer hard failures, for example by adding an explicit no-evidence fallback policy and a second citation-format repair strategy.
3. Relax the sample acceptance gate to allow degraded-but-non-crashing answers with `hard_failure_count <= 2` for this transition phase, while keeping the full eval hard-failure threshold as the enforcement gate.

**Recommendation**: Option 1. The sample gate is doing its job: it stopped the full eval before spending on a known-bad run. Fixing comparison/survey coverage now starts to overlap with the planned Phase 10.6 architecture work (`F-007` retrieve refactor and `F-019` coverage planner).

## Q2 — Sample eval still fails after REC-2 scoped fixes

**Status**: blocked, awaiting planner decision

**Context**: I implemented the three scoped REC-2 fixes:

- retrieve retry loop now accumulates relevant evidence across retries by `chunk_id`;
- chat client retries once on empty non-`length` provider content and raises `EmptyProviderResponseError` if the retry is also empty;
- verifier truncates evidence chunks to 1200 characters before JSON verification.

Unit and lint gates passed before the sample run:

```text
uv run pytest tests/unit -q
108 passed in 1.11s

uv run ruff check src tests scripts
All checks passed!
```

**Validation result**:

```text
rm -rf data/eval_outputs/latest
make eval-sample
...
EVAL FAILED: hard failure rate 0.20 exceeds threshold 0.10
hard_failure_count: 1
hard_failure_rate: 0.2
estimated_cost_usd.total: 0.12691216000000002
token_usage:
  embed-small: 17 calls, 16698 prompt tokens
  fast: 225 calls, 146864 prompt tokens, 48436 completion tokens
  main: 97 calls, 131312 prompt tokens, 32074 completion tokens
  rerank: 16 calls
```

**Sample row outcomes**:

| q-id | hard_failure | recall@5 | citation_recall | judge scores | Notes |
|---|---:|---:|---:|---|---|
| q-005 | false | 1.00 | 1.00 | 5/5/5 | Passed. |
| q-013 | false | 0.50 | 0.00 | 1/1/1 | The q-013 hard failure is gone, but answer quality remains poor. |
| q-019 | false | 0.50 | 0.00 | 5/5/5 | Passed despite no citations matching expected papers. |
| q-025 | false | 0.50 | 0.00 | 2/3/2 | The q-025 empty-answer hard failure is gone, but citation recall remains zero. |
| q-029 | true | 1.00 | 0.00 | 5/5/5 | New remaining hard failure: answer text is good but citations are chunk-only, e.g. `[991]`, `[994]`, `[992]`, so `CITATION_RE` parses zero citations. |

**Hard failure details**:

`q-029` (`How would chain-of-thought prompting apply to mathematical reasoning?`) retrieved the expected papers:

```text
retrieved_paper_ids:
["arxiv:2308.09687", "arxiv:2201.11903", "arxiv:2211.10435", "arxiv:2203.11171"]
cited_paper_ids: []
```

The answer was non-empty and judged highly, but used chunk-only citation syntax:

```text
Chain-of-thought prompting applies to mathematical reasoning by providing few-shot exemplars
that include intermediate reasoning steps ... [991]. This approach enables models to
decompose multi-step arithmetic problems ... [994]. On the GSM8K math word problem benchmark,
chain-of-thought prompting with PaLM 540B achieved a new state-of-the-art solve rate ...
[992].
```

**Interpretation**: The q-013 and q-025 A-class failures identified in `PHASE_10_5_DIAGNOSTIC.md` are materially improved: neither is a hard failure now. The remaining blocker is a citation-format failure on q-029 where the model produced chunk-only citations after the Phase 10.5 hardening. This is not an empty provider response and not missing evidence; it is the synthesizer/repair path still failing to force `[paper_id#chunk_id]` for an otherwise good answer.

**Recommendation**: Stop before full eval, per the acceptance rule. Planner should decide whether to authorize one more narrow citation-repair fix for chunk-only citations, or defer this to the Phase 10.6 architecture work. No full eval was run.
