# Two evaluation groups and public structured answers

Current request contract: `AgentRequest.response_fields`. Outer `AgentResult.schema_version` remains **2**;
structured answers use **ara-answer-3**, while the older optional extraction experiment uses ara-answer-2.
User approved the task split on 2026-09-17. Implementation and offline regression are complete.
V5 calibration exposed value-encoding problems despite bound fields. V6 adds public choice vocabularies
and user-example symbols, with atomic boolean/set conditions. Old inputs and scores remain unchanged.
The first v6 real-model runs match 11/11 groups across Agentless, LLMCompiler and GQA. Final-code
targeted reruns complete GQA (3/3) and the natural comparison without rewriting/recovery. An earlier
timeout and an unreliable repeated-verdict pass are preserved separately.
**This is calibration, not model-profile selection or held-out accuracy.** See the
[current report](../../data/eval_outputs/claim-context-20260917/analysis.md).

## Two different tasks

| Group | Dataset | Public input | Automatic scoring | Source review |
|---|---|---|---|---|
| Natural | `natural_cases_v5.json`: 12 scenarios / 15 turns | Original v4 questions, unchanged; ordinary conversation | Routes, states, identities/versions, applicable constraints, discovery order, source locations | Free facts, explanations, reading recommendations, assumptions |
| Structured | `structured_cases_v6.json`: 6 single-turn tasks | Question plus field names/types/descriptions, exact versions, optional public value vocabularies | 21 fact groups (24 fact fields), 1 paper-order field, type/domain/scope/answer binding, other applicable checks | Explanations and conditional recommendations; validity of labels and task coverage |

The groups are not paired versions of the same input. Compare model profiles **within** a group;
do not call a score difference between groups an accuracy improvement. All labels remain assistant-authored
calibration labels, not a blind held-out dataset. Explicit versions in structured GQA/Mixtral tasks do not
test follow-up reference resolution or topic reset; those behaviors remain in the natural group.

`response_fields` describes what to answer, never the correct choice, ordering, edges, aliases, grades or a rubric.
Optional `allowed_values` declares a vocabulary with alternatives; the chosen answers remain private gold.
For example, task symbols A/B/C do not disclose the directed edges; phase choices include distractors.
Numbers and booleans remain typed open values. Generation and stored-output auditing validate domains.
Old requests/checkpoints without this optional field still work.
All gold data remains in the evaluator. Extra fields such as `expected` are rejected by the API.
Natural requests omit `response_fields`; the web app is not forced to use evaluation field schemas.
The deep-reading default deadline is now 300 seconds, with the existing $0.15 per-turn budget unchanged;
web UI acceptance was not repeated in this batch.

## Runtime path

1. AgentRuntime validates the public request, isolates its immutable run context, and appends the
   field descriptions to the effective research question. The original question remains in AgentResult.
2. Normal planning, version resolution, retrieval and evidence assessment run against those requirements.
3. In the existing **synthesis** call, the model returns each requested value, explanation, basis,
   qualifiers and evidence indices. Stable instructions and JSON Schema precede variable context.
   Synthesis/semantic-repair stages keep the selected model/reasoning policy.
4. Code binds indices to exact paper/chunk identities, validates value types and source scope, and
   renders **those same values** in a Markdown table. No second model extracts facts from the prose.
5. The existing source verifier checks each complete rendered row, including its explanation and
   conditions. Scope assessment can bind earlier cited antecedents; source checks run in parallel,
   and code withholds a dependent claim unless every transitive premise passes and remains delivered.
   Local new mechanisms/conditions still need local sources. Finalization and checkpoint recovery
   retain the original verdict and dependency metadata. Its judgments are part of the agent, not independent evaluation scores.
6. Finalization retains only checked rows. Machine fields use the same retained rows; an old or rejected
   draft cannot supply a hidden correct value. Best-draft checkpoint recovery carries matching fields.
   Missing required fields make the response partial/incomplete; a new turn resets the contract.
7. The offline scorer validates the answer digest, literal row rendering, offsets, declared fields,
   exact source identities, delivery/check records, and then compares values with frozen gold rules.

Recovery does not reroll the source judge on unchanged input. An expanded uncited excerpt must first
be bound through a revised answer; direct re-verification requires a changed excerpt cited by a rejected
claim. Previous rejected checks remain available for exact-input reuse, and scope rationale wording does
not invalidate their cache. Source checks leave a delivery reserve and keep completed siblings; failed
scope checks do not launch a new source-check batch. These are program guards, not proof of semantic truth.
Public field descriptions no longer include an injected missing-evidence control sentence that the
planner had incorrectly turned into an unconditional requested output.

Native output adds `native_fields`, `structured_facts`, `rankings`, `recommendation_order` and
`answer_contract` (requested/missing fields, status, errors, answer hash). Source-card order never
substitutes for the explicitly rendered recommendation order. A paper-order row cites every compared
paper; this makes its comparison premise available to the verifier, but does not prove the model's
recommendation sound or guarantee completion of the natural three-paper comparison.

Malformed/truncated JSON stops this synthesis attempt without a new extraction or unbounded formatting
loop. Invalid individual fields are omitted and recorded. Repairs regenerate typed rows together so
visible prose cannot diverge from machine values. Provider retry/token limits and budget admission
remain in force. Protocol errors are distinct from source factual rejection.

## What the scores mean

Scoring uses no LLM, network, embedding or fuzzy semantic matcher. It applies these fixed rules:

- Booleans must be actual JSON booleans; numbers cannot be strings/booleans and must be finite.
- Numbers compare exactly unless the label predeclares a tolerance.
- Text uses mechanical spelling normalization and frozen aliases when declared, not inferred synonyms.
  V6 category fields use the published vocabulary and no observed-answer alias additions.
- Sequences retain order; sets ignore order but not missing/extra members.
- Relations retain edge direction and symbol case; only declared parallel groups ignore member order.
- All clauses of a composite fact group must match; conflicting values fail the group.

| Metric | Computation and denominator | Limitation |
|---|---|---|
| Strict fact match (`fact_accuracy`) | matched groups / all graded required groups | In structured tasks, missing fields and total response failures are misses, not dropped samples |
| Fact grading coverage | graded / required groups | A scorer/source-audit error is reported separately; unexecuted tasks are not silently graded |
| Observable facts | groups with bound values for every clause | Separates missing values from mismatches |
| Primary selection | correct first recommended ID / applicable tasks | Discovery uses assessed list order; structured reading uses the rendered explicit order |
| nDCG@3 | sum((2^grade−1)/log2(rank+1)) / ideal DCG | Duplicates gain zero; missing requested order scores zero; unlabelled/free reading order is N/A |
| Reference coverage | required paper/version IDs present / required IDs | Not source entailment or recall across all arXiv |
| Program pass | all applicable deterministic checks pass | `task_pass` remains pending where content review is required |
| Cost/cache/latency | all recorded attempts; cache hit tokens / prompt tokens; terminal P50/P95 with n | Small samples do not establish stable latency or profile superiority |

Natural prose receives no automatic fact-accuracy number. A previously unseen but equivalent textual
value in structured output may still miss a frozen alias; mark it for review, preserve the old score,
and only change labels in a new dataset version. V5 removes the old Agentless rule's unwarranted
AST-specific normalization requirement; it asks whether normalization occurs and how selection works.
The v4 dataset and failed scores are preserved, not retrospectively improved.

Human/assistant review is selective: calibrate labels first, inspect disagreements/new failures and
sample accepted answers. Identify the reviewer honestly. An optional independent LLM judge would need
its own rubric, calibration and budget; none is enabled here. The agent's own verifier is not that judge.

## Reproducible entry points

From the repository root, both commands validate inputs and show the execution plan **without API calls**:

```sh
.venv/bin/python scripts/eval_agent.py \
  --dataset docs/eval_design/natural_cases_v5.json --profile semantic_low
.venv/bin/python scripts/eval_agent.py \
  --dataset docs/eval_design/structured_cases_v6.json --profile semantic_low
```

Select `--case s04-agentless-v1-pipeline` for a focused structured trial. For paid execution, use
`--execute`, an explicitly approved cumulative `--budget`, the existing `--ledger`, a **new** output
directory, and optionally the same frozen `--materials` for controlled comparisons. The user approved
a cumulative **$3.00** ceiling for the next step. Before this batch, $1.379880554 was committed,
including $0.123058800 of old uncertain reservations. The same append-only ledger is reused;
current totals are recorded in the batch report. A dry run makes no paid requests. Do not reset the ledger or silently raise it.
The default dataset is now natural v5. `--typed-observations` is rejected for both v5 groups.

Cases run with bounded `--concurrency`; turns within a case remain sequential. Results are persisted
before grading. `manifest.json` records code/dataset/material hashes and the group. `summary.json`,
`summary.csv`, `report.md` and `scores.json` expose denominators and per-field failures.
`--rescore --dataset <same frozen dataset> --output <saved run>` reuses saved results and source audits
without model/DB calls. It rewrites derived reports only, not raw records or budget; use a copied/new
report directory if preserving a historical scorer's output verbatim.

The older `include_typed_observations=True`/ara-answer-2 path remains opt-in for historical experiments.
It appends semantic extraction to the agent's verifier, proved unstable for arbitrary free-field naming,
and is **not** the v5 structured path. Historical evidence is in
`data/eval_outputs/stage-structured-20260917/analysis.md`.

## Real v5 calibration and unresolved measurement issues

The structured group executed S02, S04 and S05 under one frozen code version, `semantic_low`, a fixed
8192 total-output cap and previously indexed exact-version originals. Strict matching is **6/11**,
grading coverage 11/11, observable groups 11/11, all-program passes 1/3. Independent source binding
and route/version checks pass for all three. These are fixed-material calibration tasks, not live recall.

Assistant source review found support for all eleven mechanism groups, while preserving the five
strict mismatches. Agentless used source-consistent wording outside the aliases. LLMCompiler used
Task 1/2/3 in relation values, explained the A/B/C mapping in prose, and returned full sentences for
two text fields. Public field names solved arbitrary names/splitting but did not define the answer
vocabulary. Specify user-example node labels and predeclared category choices where appropriate;
decompose conditions into atomic typed facts. Do not add observed aliases to old labels after scoring.
Keep explanations subject to separate review. This review is not blind human validation or a new LLM judge.

The natural three-paper task first failed name grounding; a narrow ID-label normalization fix was
saved under a new code version and retested. It returned partial at 164.520 seconds using the new
recovery admission guard. Its global fit order passed, but a repeated ranking label caused the
LLMCompiler mechanism paragraph to be withheld; GQA conversion wording also needs more precise
source attribution. Program pass remains false; natural reading nDCG and fact accuracy remain N/A.

Full results, costs, exact saved sources, failed attempts and offline rescore commands:
[`typed-calibration-20260917/analysis.md`](../../data/eval_outputs/typed-calibration-20260917/analysis.md).
No High/Low comparison or final held-out evaluation has been completed. The existing UI was not restarted.
