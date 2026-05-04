# Adversarial Review — 2026-05-04

## Summary
Do not ship this as a production agent without a refactor pass. The top three blockers are: self-RAG fail-open defaults that convert verifier/sufficiency/relevance JSON failures into success, a state machine where multi-hop and Input Mode exist mostly as unexercised demos, and an eval runner that turns hard graph crashes into successful CSV rows. The current baseline already reports 7/30 hard failures, survey collapse, and unmeasured semantic memory; the code contains more failure-masking paths than the eval exposed.

## Findings

### F-001: Citation verifier fails open and marks unverifiable claims as supported
* Severity: critical
* Category: silent-failure
* Location: `src/retrieval/self_rag/verifier.py:56` — `src/retrieval/self_rag/verifier.py:79`; `tests/unit/test_verifier_parser.py:52` — `tests/unit/test_verifier_parser.py:60`
* Evidence: `_parse_or_repair_verdict` catches a second invalid JSON response and returns `CitationVerdict(... supports=True ...)`. The unit test explicitly asserts `report.passed is True` for an empty verifier response. The baseline calls this out as artificially inflating groundedness and `verification.passed` (`docs/eval_baselines/BASELINE_2026-05-04_phase10.md:51` — `docs/eval_baselines/BASELINE_2026-05-04_phase10.md:53`).
* Root cause: The verifier is a trust boundary, but it is implemented as an availability feature. Provider formatting instability is treated as evidence support instead of an unknown or failed verification state, so the system cannot distinguish "supported", "unsupported", and "verifier broke".
* Recommendation: Make invalid verifier output a non-passing state with explicit telemetry. If availability matters, add a separate `unknown`/`unverified` verdict and route it differently from `supports=True`.
* Test debt: A verifier parse-failure test should assert the answer is not marked verified and should prove the graph either retries synthesis or returns a visible verification failure.

### F-002: Sufficiency checker fails open, disabling the multi-hop recovery path
* Severity: critical
* Category: silent-failure
* Location: `src/retrieval/self_rag/sufficiency_check.py:31` — `src/retrieval/self_rag/sufficiency_check.py:52`; `src/graph/nodes/retrieve.py:87` — `src/graph/nodes/retrieve.py:90`
* Evidence: If both the initial sufficiency JSON and repair JSON fail validation, the checker returns `SufficiencyVerdict(sufficient=True, missing_aspects=[])`. The retrieve node breaks as soon as `sufficient` is true. The baseline says multi-hop is rarely triggered because sufficiency returns true after a single relevant chunk (`docs/eval_baselines/BASELINE_2026-05-04_phase10.md:49` — `docs/eval_baselines/BASELINE_2026-05-04_phase10.md:50`). The unit test locks in this bad behavior (`tests/unit/test_sufficiency_check.py:35` — `tests/unit/test_sufficiency_check.py:43`).
* Root cause: The system uses a fail-open LLM classifier to guard the only path that can gather more evidence. That makes the retrieval loop self-satisfied precisely when the classifier is least trustworthy.
* Recommendation: Treat invalid sufficiency output as insufficient or unknown, and require an explicit "why sufficient" signal before cutting off retrieval. For multi-hop questions, use structural coverage signals in addition to the LLM verdict.
* Test debt: There should be a graph-level multi-hop question test proving insufficient local evidence routes to `multi_hop`, not just a direct citation-walker test.

### F-003: Relevance judge fail-open keeps bad chunks in the evidence set
* Severity: high
* Category: silent-failure
* Location: `src/retrieval/self_rag/relevance_judge.py:30` — `src/retrieval/self_rag/relevance_judge.py:51`; `src/graph/nodes/retrieve.py:85` — `src/graph/nodes/retrieve.py:86`
* Evidence: A double JSON parse failure returns `RelevanceVerdict(relevant=True, ...)`. Retrieve filters evidence with `if verdict.relevant`, so broken judging preserves the hit. The unit test asserts this behavior (`tests/unit/test_relevance_judge.py:52` — `tests/unit/test_relevance_judge.py:63`).
* Root cause: The evidence filter has no "unknown" state. It conflates "the model says relevant" with "the model failed to produce valid output", polluting downstream sufficiency and synthesis with unverified chunks.
* Recommendation: Make relevance parse failure observable and non-positive. If recall is the concern, pass unknowns to a cheaper deterministic fallback or keep them with a degraded score, not as fully relevant hits.
* Test debt: Add tests where malformed relevance output cannot make a known irrelevant chunk survive into `kept_hits`.

### F-004: Citation formatting is a brittle regex gate responsible for most hard failures
* Severity: critical
* Category: state-machine
* Location: `src/graph/nodes/synthesize.py:10` — `src/graph/nodes/synthesize.py:27`; `docs/eval_baselines/BASELINE_2026-05-04_phase10.md:33` — `docs/eval_baselines/BASELINE_2026-05-04_phase10.md:45`
* Evidence: Six of seven baseline failures are `SynthesizerError (citation format)`. The node does one prose generation, regex-parses citations, performs one "fix citation format" retry, then raises. `MALFORMED_CITATION_RE` flags any bracketed single-token text as malformed, regardless of whether it was intended as a citation.
* Root cause: Citation correctness is being enforced after prose generation with a fragile text regex rather than being part of a structured answer plan. The model is asked to do two hard things at once: answer the research question and maintain a custom citation grammar.
* Recommendation: Separate citation planning from prose generation or require structured citation spans before producing the final answer. At minimum, log the malformed answer and parsed citation candidates so prompt failures are diagnosable.
* Test debt: The only live E2E graph test asks "What is ReAct?" (`tests/integration/test_graph_e2e.py:18` — `tests/integration/test_graph_e2e.py:35`); it does not cover survey/comparison citation stress cases where the baseline fails.

### F-005: The eval runner masks graph crashes as successful benchmark rows
* Severity: high
* Category: test-theater
* Location: `src/eval/runner.py:89` — `src/eval/runner.py:107`; `src/eval/runner.py:134` — `src/eval/runner.py:151`
* Evidence: `_run_one` catches any exception from `builder.run`, converts it to an `AGENT_RUN_FAILED` row with judge scores of 1, and continues. Phase 10 explicitly documents this as a deviation (`docs/phase_reports/PHASE_10_REPORT.md:121` — `docs/phase_reports/PHASE_10_REPORT.md:124`). The unit test only asserts the failure row shape (`tests/unit/test_eval_runner.py:7` — `tests/unit/test_eval_runner.py:15`).
* Root cause: The eval harness optimizes for completing 30 rows instead of making system failure impossible to miss. This makes `make eval` a poor production gate: a run with a 23% hard failure rate can still be called "pass".
* Recommendation: Split "agent crashed" from normal metric rows and make the command return non-zero above a configured failure threshold. Keep resumability, but do not make crashes indistinguishable from low-scoring answers.
* Test debt: Add a test that a graph exception increments a hard-failure counter and fails acceptance when above threshold.

### F-006: Input Mode is unmeasured and its paper-hint query ignores standalone concept evidence
* Severity: high
* Category: spec-drift
* Location: `src/memory/semantic/graph_retriever.py:103` — `src/memory/semantic/graph_retriever.py:121`; `docs/eval_baselines/BASELINE_2026-05-04_phase10.md:52` — `docs/eval_baselines/BASELINE_2026-05-04_phase10.md:53`
* Evidence: `_paper_hints_for_concepts` derives hints only from `concept_relations.evidence_chunks`. It never reads `concepts.evidence_chunks`, even though concepts store evidence. The baseline says Input Mode was never exercised because eval ran with `skip_reflection=True` and the corpus has no concepts. The integration test seeds a relation explicitly, so it cannot catch standalone concept evidence being ignored (`tests/integration/test_graph_retriever.py:38` — `tests/integration/test_graph_retriever.py:58`).
* Root cause: The semantic memory write model and read model are inconsistent. Reflection can create concepts with evidence, but graph expansion only rewards relation evidence; then eval disables reflection entirely.
* Recommendation: Decide whether Input Mode is relation-only or concept-plus-relation. Then run an end-to-end test where reflection creates the graph and a later query uses it, without hand-seeded relations.
* Test debt: Add a graph retriever test where a matched concept has `evidence_chunks` but no relation, and assert the expected paper hint behavior.

### F-007: Retrieve node is a monolith and already violates the documented boost placement
* Severity: high
* Category: coupling
* Location: `src/graph/nodes/retrieve.py:60` — `src/graph/nodes/retrieve.py:106`; `docs/EXECUTION_PLAN.md:1761` — `docs/EXECUTION_PLAN.md:1766`
* Evidence: `_run_subquestion_pipeline` owns routing, graph expansion, query rewrite, HyDE, embedding, hybrid search, paper hint boosting, dedupe, rerank, relevance filtering, sufficiency, and retry policy. The plan says paper hint biasing is a post-filter ranking boost and places rerank after hybrid search; the Phase 9 report says boost is after reranking (`docs/phase_reports/PHASE_9_REPORT.md:88` — `docs/phase_reports/PHASE_9_REPORT.md:90`), but code applies the boost before dedupe and before Cohere rerank (`src/graph/nodes/retrieve.py:81` — `src/graph/nodes/retrieve.py:84`).
* Root cause: Retrieval policy, ranking mechanics, graph biasing, and self-RAG control flow are fused into one function. That makes small policy changes high blast-radius and lets spec drift hide in sequencing.
* Recommendation: Separate retrieval strategy orchestration from ranker components and make the ranking stages explicit trace events. Paper-hint boost placement should be a tested contract.
* Test debt: Add a test where a hinted paper would only survive if boosting happens at the documented stage.

### F-008: Multi-hop exists, but the production graph rarely reaches it
* Severity: critical
* Category: state-machine
* Location: `src/graph/nodes/retrieve.py:155` — `src/graph/nodes/retrieve.py:161`; `src/graph/edges.py:7` — `src/graph/edges.py:21`; `tests/integration/test_multi_hop_walk.py:20` — `tests/integration/test_multi_hop_walk.py:39`
* Evidence: The graph routes to `multi_hop` only after a `SubQResult` is insufficient. The baseline says the walker is mostly idle because sufficiency returns true too easily. The integration test calls `walk_citations` directly; it does not prove the graph can trigger multi-hop for a real question.
* Root cause: The multi-hop capability is gated behind a single LLM sufficiency boolean and has no retrieval-coverage heuristic, expected-paper diversity heuristic, or comparison/survey awareness. The thing that should recover missing evidence is downstream of the thing that incorrectly says evidence is enough.
* Recommendation: Add explicit graph-level trigger criteria for multi-hop and log every routing decision with evidence count, sufficiency rationale, and budget state.
* Test debt: Add an integration test that runs the full graph on a known ReAct/Reflexion question and asserts `multi_hop_calls > 0`.

### F-009: Multi-hop budget accounting records attempts, not useful work
* Severity: medium
* Category: state-machine
* Location: `src/graph/nodes/multi_hop.py:19` — `src/graph/nodes/multi_hop.py:31`; `src/retrieval/multi_hop/citation_walker.py:71` — `src/retrieval/multi_hop/citation_walker.py:81`
* Evidence: `multi_hop_node` increments `multi_hop_calls` even when there are no seed papers. The walker marks a paper visited before checking whether it is present or successfully ingested; if lazy ingest fails, it logs and continues with that paper already in `visited_order`.
* Root cause: The state model conflates "attempted", "visited", "ingested", and "expanded". That makes budget exhaustion and telemetry misleading, especially when ingestion or seed selection fails.
* Recommendation: Track separate counters for attempted multi-hop, successful expansion, lazy-ingest success/failure, and visited-complete papers. Do not consume the only multi-hop budget on an empty seed set without surfacing that as a failed expansion.
* Test debt: Add tests for empty seed sets and lazy-ingest failure that assert budget and visited semantics are explicit.

### F-010: Ingestion idempotency is not safe under concurrent runs
* Severity: high
* Category: idempotency
* Location: `src/corpus/ingestion.py:43` — `src/corpus/ingestion.py:80`; `src/corpus/ingestion.py:189` — `src/corpus/ingestion.py:223`
* Evidence: The ingestion status is read outside a transaction. `_mark_failed` deletes citations and chunks, then upserts `ingestion_status='failed'`. If two coroutines ingest the same paper and one fails after the other succeeds, the failing path can delete the successful chunks and mark the paper failed.
* Root cause: Idempotency was implemented as "retryable sequential rerun" rather than concurrency-safe ownership. There is no per-paper lock, attempt id, or conditional status update.
* Recommendation: Serialize ingestion per paper with a database advisory lock or an owner token. Failure cleanup should only affect rows written by that attempt, and status transitions should be conditional on the current owner/state.
* Test debt: Add a concurrent duplicate-ingest test where one path fails after another commits, and assert the completed paper remains complete.

### F-011: Semantic Scholar outages silently produce "complete" papers with empty citation graphs
* Severity: high
* Category: silent-failure
* Location: `src/corpus/ingestion.py:60` — `src/corpus/ingestion.py:64`; `src/corpus/ingestion.py:172` — `src/corpus/ingestion.py:185`
* Evidence: Any Semantic Scholar reference failure is logged as a warning and converted to `references=[]`. The paper is still marked `complete`. Later phases assume citation rows power multi-hop traversal.
* Root cause: The ingestion lifecycle has one status for the paper, but paper text readiness and citation graph readiness are different contracts. A citation-fetch failure is downgraded into a complete paper with missing graph edges.
* Recommendation: Track citation fetch status separately and make the walker/reports aware of degraded citation coverage. At minimum, retry citation fetch independently before declaring multi-hop data ready.
* Test debt: Add an integration or unit test that simulates Semantic Scholar failure and asserts the degraded graph state is visible, not silently "complete".

### F-012: Reflection idempotency is sequential-only and can lose evidence under concurrency
* Severity: high
* Category: race
* Location: `src/memory/reflection.py:15` — `src/memory/reflection.py:35`; `src/memory/semantic/linker.py:52` — `src/memory/semantic/linker.py:65`
* Evidence: `reflect_session` reprocesses every assistant message on every call. Existing-concept evidence merge is computed in Python from a stale `StoredConcept`, then written back with `UPDATE concepts SET evidence_chunks = $2`. The safer SQL merge exists in `graph_store.upsert_concept` but linker does not use it for existing concepts (`src/memory/semantic/graph_store.py:27` — `src/memory/semantic/graph_store.py:35`).
* Root cause: Idempotency was validated as repeated deterministic writes, not as concurrent or evolving LLM outputs. There is no reflection watermark and no atomic evidence merge in the primary linker path.
* Recommendation: Add a processed-message watermark or reflection ledger, and use atomic JSONB set-union for every evidence merge path. Treat reflection as an eventually consistent indexer, not a side effect hidden in answer generation.
* Test debt: Add a parallel `reflect_session` test with overlapping concepts and different evidence IDs; assert no evidence is lost and no duplicate concepts appear.

### F-013: Alias collision behavior is inconsistent across semantic write APIs
* Severity: medium
* Category: observability
* Location: `src/memory/semantic/linker.py:68` — `src/memory/semantic/linker.py:97`; `src/memory/semantic/graph_store.py:44` — `src/memory/semantic/graph_store.py:53`
* Evidence: `linker._insert_aliases` preselects the existing alias and logs a warning if it maps to another concept. `graph_store.upsert_concept` inserts aliases with `ON CONFLICT DO NOTHING` and no collision warning. A race between the preselect and insert in linker can also suppress the collision warning.
* Root cause: The globally unique alias invariant is enforced by the database, but collision observability is split across two write paths with different behavior.
* Recommendation: Centralize alias insertion and return an explicit inserted/conflict/different-owner outcome. Collisions should be queryable, not just best-effort log lines.
* Test debt: Add tests for alias collision through both `link_concept` and `upsert_concept`, including a concurrent conflict.

### F-014: Graph-retriever mention detection has no repair or graceful degradation
* Severity: medium
* Category: silent-failure
* Location: `src/memory/semantic/graph_retriever.py:54` — `src/memory/semantic/graph_retriever.py:66`; `src/graph/nodes/retrieve.py:66` — `src/graph/nodes/retrieve.py:70`
* Evidence: `_detect_concept_phrases` directly validates model JSON and raises on malformed output. If router enables concept graph, that exception aborts the retrieval node. Phase 9 hardened several strict-JSON calls, but Input Mode was not included (`docs/phase_reports/PHASE_9_REPORT.md:88` — `docs/phase_reports/PHASE_9_REPORT.md:90`).
* Root cause: Input Mode is treated as a normal dependency, not an optional retrieval bias. Because it was not exercised end-to-end in eval, its provider-failure behavior lagged behind the rest of the graph.
* Recommendation: Decide whether graph expansion failure should abort the query or degrade to no graph hints. Whichever policy is chosen, log the phrase detector output and route decision.
* Test debt: Add a graph retriever test with invalid mention JSON and a graph-level test proving the chosen failure policy.

### F-015: Router is a substring alias check, not a routing policy
* Severity: medium
* Category: coupling
* Location: `src/retrieval/query/router.py:7` — `src/retrieval/query/router.py:13`; `src/retrieval/query/router.py:16` — `src/retrieval/query/router.py:29`
* Evidence: `may_use_semantic_scholar_live` is always true. Concept graph activation is `POSITION(LOWER(alias) IN LOWER($1)) > 0`, so short aliases can match unrelated words. The plan says the retrieve node should decide live expansion based on local hit count after retrieval/rerank (`docs/EXECUTION_PLAN.md:1436` — `docs/EXECUTION_PLAN.md:1436`).
* Root cause: Routing was reduced to a pre-retrieval DB existence check, with no awareness of retrieval quality, query type, or alias token boundaries.
* Recommendation: Make router output an interpretable policy based on normalized phrase matches and later retrieval telemetry. Do not let arbitrary substring aliases flip graph mode.
* Test debt: Add false-positive alias tests such as short aliases embedded inside unrelated words, and tests where low local hit count changes the live-expansion decision.

### F-016: Edge routing can synthesize from malformed or incomplete state
* Severity: medium
* Category: state-machine
* Location: `src/graph/edges.py:7` — `src/graph/edges.py:21`; `src/graph/edges.py:35` — `src/graph/edges.py:46`
* Evidence: `_all_subquestions_complete` returns true when `decomposition is None`. If subquestions are incomplete but there is no current result, `route_after_retrieve` falls through to `synthesize`. There is no explicit "invalid state" route.
* Root cause: State-machine edges prefer forward progress over invariant enforcement. Missing decomposition or missing current result is treated as a best-effort answer path, not a corrupted graph state.
* Recommendation: Add explicit state invariants and make invalid combinations fail loudly with structured diagnostics. Best-effort synthesis should be a named state, not a fallthrough.
* Test debt: Add edge tests for missing decomposition, missing active result, and partially complete decomposition.

### F-017: Decomposer parse failure aborts the whole run instead of degrading to a single question
* Severity: high
* Category: state-machine
* Location: `src/retrieval/query/decomposer.py:18` — `src/retrieval/query/decomposer.py:58`; `docs/eval_baselines/BASELINE_2026-05-04_phase10.md:37` — `docs/eval_baselines/BASELINE_2026-05-04_phase10.md:40`
* Evidence: Nontrivial questions call `main`, attempt one repair, then let `Decomposition.model_validate_json` raise. Baseline q-014 failed with `ValidationError (Decomposition JSON)`.
* Root cause: The first graph node is a hard dependency on strict JSON for questions that could still be answered without decomposition. The code has a trivial-question bypass, but no "decomposition unavailable, use original question" fallback for parse failure.
* Recommendation: Use a degraded single-subquestion path for decomposition failure with a visible warning, or make decomposition failure a hard production error instead of letting eval hide it as a row.
* Test debt: Add an integration test where decomposition returns invalid JSON and assert the graph either degrades explicitly or fails the command, not the hidden CSV row.

### F-018: Cost control is mostly aspirational because token usage is not tracked
* Severity: high
* Category: cost
* Location: `src/graph/nodes/synthesize.py:36` — `src/graph/nodes/synthesize.py:87`; `src/eval/runner.py:226` — `src/eval/runner.py:239`
* Evidence: The synthesizer sends the full evidence block to `main`, and the citation-format repair sends the full evidence block again. Sufficiency sends all kept evidence to `main` (`src/retrieval/self_rag/sufficiency_check.py:10` — `src/retrieval/self_rag/sufficiency_check.py:29`), verifier calls `main` per citation (`src/retrieval/self_rag/verifier.py:34` — `src/retrieval/self_rag/verifier.py:53`), and the eval judge uses `main` with `max_tokens=2000` (`src/eval/metrics/llm_judge.py:23` — `src/eval/metrics/llm_judge.py:42`). The eval summary admits token usage is not centrally tracked.
* Root cause: Cost was managed by choosing roles (`main`/`fast`) and one-off max token caps, not by measuring or budgeting actual prompt/response tokens across graph nodes.
* Recommendation: Add central token accounting per run and per node, then set hard budgets. Evidence prompts should be capped, summarized, or selected by coverage rather than dumped repeatedly.
* Test debt: Add tests or smoke assertions that report token counts and fail when a node exceeds its configured budget on representative evidence sizes.

### F-019: Survey/comparison failures are architectural, not dataset noise
* Severity: high
* Category: coupling
* Location: `src/graph/nodes/retrieve.py:164` — `src/graph/nodes/retrieve.py:173`; `src/graph/nodes/synthesize.py:40` — `src/graph/nodes/synthesize.py:47`; `docs/eval_baselines/BASELINE_2026-05-04_phase10.md:27` — `docs/eval_baselines/BASELINE_2026-05-04_phase10.md:31`
* Evidence: Survey recall@5 is 0.312 and citation recall is 0.188. Comparison recall is 0.625. Evidence assembly dedupes chunks in subquestion-index order and the synthesizer is constrained to one 4-6 sentence paragraph; there is no coverage objective requiring both sides of a comparison or all expected survey clusters to be cited.
* Root cause: The graph optimizes for "some relevant evidence and a concise answer" rather than coverage. Conservative citation behavior is expected when the answer budget is short and no planner forces side-by-side evidence coverage.
* Recommendation: Add a coverage-planning stage for comparison/survey questions that tracks required facets and expected paper diversity before synthesis.
* Test debt: Add integration tests where the answer must cite both sides of a comparison and at least N distinct papers for survey questions.

### F-020: Multi-hop scoring fallback contains test-shaped domain bonuses
* Severity: medium
* Category: test-theater
* Location: `src/retrieval/multi_hop/citation_walker.py:205` — `src/retrieval/multi_hop/citation_walker.py:216`; `src/retrieval/multi_hop/citation_walker.py:282` — `src/retrieval/multi_hop/citation_walker.py:293`
* Evidence: If LLM score parsing fails, the walker falls back to a heuristic that adds `+4.0` when candidate text contains "reflexion" and `+2.0` for "react". The integration test is specifically ReAct -> Reflexion (`tests/integration/test_multi_hop_walk.py:26` — `tests/integration/test_multi_hop_walk.py:39`).
* Root cause: A test-specific canonical demo leaked into production fallback ranking. That makes the walker look robust on the approved example while giving no general guarantee for other citation chains.
* Recommendation: Replace domain-name bonuses with a generic lexical or embedding fallback, or mark score-parse failure as degraded and observable.
* Test debt: Add multi-hop tests for at least two non-ReAct chains and force score-parse fallback to verify general behavior.

## Cross-cutting themes

1. **Fail-open defaults are everywhere the agent needs judgment most.** Verifier, sufficiency, relevance, extractor, and eval all convert model or graph failures into permissive outcomes or quiet rows.
2. **Capabilities are tested as isolated demos, not production paths.** Multi-hop is tested by direct walker invocation; graph retrieval is tested with hand-seeded relations; E2E asks one easy definition question.
3. **The state machine lacks invariant enforcement.** Budget counters, active subquestion state, and best-effort fallthroughs make it possible to move forward from malformed or low-evidence states without clear diagnostics.
4. **Idempotency claims mean "same sequential replay", not concurrent correctness.** Ingestion and semantic memory both have write races that can lose rows or evidence.
5. **Cost and observability are afterthoughts.** The system uses expensive `main` calls across multiple nodes, repeats large evidence prompts, and does not centrally track tokens or preserve enough failure context to debug bad runs.

## What was NOT reviewed

* Did not run the live eval or any provider calls; trusted the documented Phase 10 baseline numbers.
* Did not run integration tests or inspect live database contents.
* Did not review the future UI phase because it is not implemented at HEAD.
* Did not perform dependency security scanning or network/API behavior verification.
* Did not review every seed-paper gold question for academic correctness beyond using the baseline and source line evidence.
