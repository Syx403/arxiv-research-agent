# Phase 10.5 Diagnostic Report

## A. Baseline vs. new comparison

The "new" values below are the Phase 10.5 sample-gate rows from `data/eval_outputs/latest/per_question.csv` before the diagnostic append. The diagnostic rerun appended duplicate q-013/q-025 rows and is discussed separately below.

| q-id | recall@5 baseline -> new | citation_recall baseline -> new | judge_correctness baseline -> new | hard_failure baseline -> new | Summary |
|---|---:|---:|---:|---:|---|
| q-013 | 1.00 -> 0.00 | 1.00 -> 0.00 | 5 -> 1 | false -> true | Baseline retrieved and cited Toolformer + ToolLLM; Phase 10.5 ended with no evidence and a refusal answer. |
| q-025 | 0.75 -> 0.75 | 0.50 -> 0.00 | 3 -> 1 | false -> true | Retrieval still found relevant papers, but synthesis produced an empty answer and no citations survived. |

Baseline q-013 answer preview: "Toolformer adopts a self-supervised approach that leverages only a few human-written examples to let the language model itself generate and filter API calls..."

New q-013 answer preview: "I am unable to answer how Toolformer differs from ToolLLM for tool use because no evidence has been provided..."

Baseline q-025 answer preview: "Approaches to tool use in LLM agents include integrating external tools such as APIs, databases, and external models to expand action spaces..."

New q-025 answer preview: `<EMPTY>`.

Rewritten query and HyDE outputs are not captured by the eval runner, so they cannot be compared from existing artifacts.

## B. q-025 root cause

Diagnostic rows show the q-025 synthesis prompt was not malformed: `synthesize_initial` had `evidence_count=9` and `available_chunk_ids_count=9`. The initial model response was actually empty twice (`content_length=0`). In the diagnostic rerun, the repair prompt recovered and produced non-empty cited prose, for example: "Approaches for tool use in LLM agents can be broadly grouped into data-driven fine-tuning and prompting-based planning..." with citations such as `[arxiv:2307.16789#534]`. That means the original sample-gate empty answer was an intermittent empty-content path, not empty evidence or an empty citation map.

The diagnostic rerun also exposed a second q-025 failure mode: verifier calls returned empty raw content for every citation. There were 11 verifier `initial` rows and 11 verifier `parse_failure_twice` rows, all with `content_length=0`, so fail-closed verification removed every citation. Classification: **A-class bug we introduced/exposed**. The fail-closed rule is correct, but the chat layer and synthesis/verifier paths currently treat HTTP-200 empty model content as an ordinary response instead of a provider failure that should be observed and handled before scoring.

## C. q-013 root cause

q-013 was not a hybrid-search empty result. The diagnostic log has 60 `relevance_judge` calls, and at least three initial relevance verdicts were valid `relevant=true`, including chunks 469, 466, and 534. One chunk 534 verdict explicitly said: "Toolformer learns tool use via self-supervised learning, while ToolLLM uses instruction tuning with a depth-first search decision tree and API retriever...". However, sufficiency remained false, the retry loop kept rewriting, and the final retrieved evidence written to the eval row was empty. The final synthesizer then received `evidence_count=0` and correctly refused to answer.

This points to the retrieve retry loop discarding earlier useful hits and returning only the last retry's `kept_hits`. After retries and multi-hop, a later empty/insufficient pass can erase earlier relevant evidence. Classification: **A-class bug**. This is not fail-closed behaving as intended; fail-closed made the problem visible, but the architectural issue is evidence loss across retries.

## D. PARSE_FAILURE count summary

CSV substring counts across the original 5-question Phase 10.5 sample rows:

| q-id | `PARSE_FAILURE` in answer | `PARSE_FAILURE` in judge_rationale |
|---|---:|---:|
| q-005 | 0 | 0 |
| q-013 | 0 | 0 |
| q-019 | 0 | 0 |
| q-025 | 0 | 0 |
| q-029 | 0 | 0 |

Diagnostic JSONL parse-failure events:

| Role | Calls logged | Parse-failure-twice events | Rate | Signal |
|---|---:|---:|---:|---|
| relevance_judge | 70 | 4 | 5.7% | Above 5%; prompt/provider output needs attention. |
| sufficiency | 7 | 1 | 14.3% | Above 5%; prompt/provider output needs attention. |
| verifier | 11 | 11 | 100.0% | Critical; verifier JSON path is currently unusable for q-025. |

By question:

| q-id | verifier | sufficiency | relevance_judge |
|---|---:|---:|---:|
| q-013 | 0 | 1 | 3 |
| q-025 | 11 | 0 | 1 |

The diagnostic run cost reported in `summary.json` was `$0.05217787`, under the `$0.10` diagnostic cap.

## E. Recommendation to planner

**REC-2: Both are A-class.** q-025 needs an empty-content/provider-response hardening path around chat responses, especially for synthesis and verifier JSON calls; the prompt had nine chunks, so the empty answer was not caused by missing evidence. q-013 needs retrieval retry semantics that preserve the best/previous relevant evidence instead of letting a later retry erase useful hits. Fail-closed semantics should remain unchanged; they are exposing real reliability bugs rather than causing the underlying failures.
