# Engineering Rigor Review — 2026-05-05

## Summary

Do not run another full eval, and do not ship Phase 10.6a, until the LLM/provider boundaries are treated as engineered contracts rather than prompt hopes. The top three implementation risks are: unbounded evidence prompts feeding production-critical synthesis/sufficiency calls, no systematic handling of `finish_reason="length"` despite an observed truncated q-005 answer, and an eval harness whose sample gate and hard-failure definition are too noisy and too coarse to separate provider failures from architecture debt. This review does not repeat the prior architectural findings; it focuses on prompt mandates, token budgets, provider failure handling, and eval correctness.

## Findings

### E-001: Synthesis has a stronger citation mandate, but no length-stop handling
* Severity: critical
* Category: output-budget
* Location: `src/graph/nodes/synthesize.py:51` — `src/graph/nodes/synthesize.py:86`; `src/llm/client.py:285` — `src/llm/client.py:286`
* Evidence: `_generate_answer` now mandates `[paper_id#chunk_id]` citations and caps output at `max_tokens=1200`. The chat client only retries empty responses when `finish_reason != "length"`, and no caller retries or degrades specifically on `finish_reason="length"`. Q5 shows q-005 returned a non-empty answer ending at literal `[arxiv`, leaving no complete citation marker and producing `hard_failure=True`.
* Root cause: Output validity is enforced by regex after generation, but the provider stop reason is not part of the contract. A partial answer can be long enough to look meaningful, short enough to pass "non-empty", and still be syntactically unparsable because the citation token was cut mid-marker.
* Recommendation: Treat `finish_reason="length"` as a first-class synthesis failure mode with a larger retry budget or a specific compact-answer retry. Log the finish reason into per-question eval rows so truncation is not indistinguishable from model noncompliance.
* Iteration trace: Caught by Iter 5/Q5: q-005 produced high-quality but truncated prose ending in `[arxiv`.

### E-002: Synthesis sends all evidence with no chunk cap or truncation
* Severity: critical
* Category: input-size
* Location: `src/graph/nodes/synthesize.py:70` — `src/graph/nodes/synthesize.py:78`; `src/graph/nodes/retrieve.py:305` — `src/graph/nodes/retrieve.py:314`
* Evidence: `_format_evidence(evidence)` includes full `hit.text` for every assembled hit. `retrieve._assemble_evidence` merges all sub-question hits, and each sub-question can retain up to reranked evidence across retries. With `RERANK_TOP_K=10`, `MAX_SUB_QUESTIONS=5`, and chunks around 800 tokens, worst-case synthesis input is roughly 40k chunk tokens before prompt overhead.
* Root cause: Retrieval evidence is treated as a payload dump, not a bounded prompt budget. The synthesizer prompt got stricter over Phase 10.6a, but its input budget was never recalibrated to the number and size of chunks it may receive.
* Recommendation: Define an explicit synthesis evidence budget in chunks and tokens, select evidence by coverage/rank, and truncate each chunk with visible markers. The answer generator should receive a bounded context, not whatever retrieval happened to accumulate.
* Iteration trace: Caught by Iter 1 and Iter 5b: large evidence prompts correlated with empty/invalid provider output on q-025.

### E-003: Citation repair repeats the same oversized evidence prompt with the same budget
* Severity: high
* Category: prompt-design
* Location: `src/graph/nodes/synthesize.py:89` — `src/graph/nodes/synthesize.py:122`
* Evidence: `_fix_citation_format` resends the full invalid answer plus the full evidence block and still uses `max_tokens=1200`. The repair system prompt is a mandate, but the user prompt asks it to preserve meaning and attach citations while also writing a fresh answer when empty.
* Root cause: The repair path is not a smaller, easier task. It repeats the hardest part of the original prompt with extra input, so it is least likely to recover on the exact long-survey cases where the first pass failed.
* Recommendation: Make repair a constrained transformation: either cite an existing answer using a compact evidence map, or regenerate a shorter answer from a capped evidence subset. Do not send the same unbounded evidence twice.
* Iteration trace: Caught by Iter 5b/Q5: q-025 had non-empty retrieval and still produced an empty answer after repair.

### E-004: The synthesis prompt mandates citations but still gives no invalid-output recovery example
* Severity: medium
* Category: prompt-design
* Location: `src/graph/nodes/synthesize.py:55` — `src/graph/nodes/synthesize.py:66`; `src/graph/nodes/synthesize.py:93` — `src/graph/nodes/synthesize.py:101`
* Evidence: The system prompt gives one positive example of `[arxiv:2210.03629#123]` and forbids paper-only/chunk-only citations. It does not show a complete 4-6 sentence answer with every sentence cited, nor a repair example converting uncited prose into cited prose.
* Root cause: The prompt moved from explanation to mandate, but not to a fully anchored output pattern. The model has seen the grammar but not the required global shape of a valid answer.
* Recommendation: Add one compact full-answer exemplar for initial generation and one repair exemplar. Keep examples shorter than the production answer to avoid diluting the mandate.
* Iteration trace: Caught by Iter 5: q-019/q-025/q-029 produced good prose without citation markers before the mandate was strengthened.

### E-005: Sentence-derived `claim_text` includes the citation marker and may overstate the supported claim
* Severity: high
* Category: synth-verify-contract
* Location: `src/graph/nodes/synthesize.py:132` — `src/graph/nodes/synthesize.py:146`; `src/retrieval/self_rag/verifier.py:50` — `src/retrieval/self_rag/verifier.py:54`
* Evidence: `_parse_citations` sets `claim_text` to the entire sentence containing the marker, including the marker itself. The verifier then asks whether the evidence supports that whole sentence. Q5 shows q-029 had valid markers in the answer but `cited_paper_ids=[]` because verifier removed every citation despite judge scores of 5/5/5.
* Root cause: The synthesizer-verifier contract switched from marker spans to sentence spans, but the verifier is still asked a binary support question over a full natural-language sentence. One sentence can contain multiple facts, hedges, or comparisons, only some of which are in the cited chunk.
* Recommendation: Define support granularity explicitly. Either split claim text into atomic clauses before verification, or make the verifier standard "substantially supported by this chunk" rather than literal sentence entailment.
* Iteration trace: Caught by Iter 5c/Q5: q-029 citation markers parsed, then all citations were rejected.

### E-006: Verifier strictness is undefined, so "honest" citation metrics are unstable
* Severity: high
* Category: synth-verify-contract
* Location: `src/retrieval/self_rag/verifier.py:42` — `src/retrieval/self_rag/verifier.py:60`
* Evidence: The system prompt says only: `Verify whether evidence supports the cited claim. Return JSON...`. It does not define whether support means direct statement, reasonable inference, same concept with different wording, or full entailment of every word. The user prompt supplies the full answer, a specific claim, and a chunk, but no decision rubric.
* Root cause: A research verifier is a policy boundary, not just a JSON parser. Without an explicit standard, small wording choices in synthesis decide whether citation recall is measured as zero or nonzero.
* Recommendation: Choose and encode a support rubric: direct quotation, paraphrase-level support, or inferential support. For research-agent production, use a two-tier verdict (`supported`, `partially_supported`/`needs_broader_evidence`) rather than collapsing everything into false.
* Iteration trace: Caught by Iter 4 and Iter 5c: q-019/q-029 strong prose lost verified citations under strict per-citation behavior.

### E-007: Verifier repair loses the original claim and evidence context
* Severity: medium
* Category: provider-robustness
* Location: `src/retrieval/self_rag/verifier.py:71` — `src/retrieval/self_rag/verifier.py:106`
* Evidence: On parse failure, the repair call sends only `Invalid output:\n{content}`. It does not include the original claim, evidence, or answer. If the model returned malformed but semantically useful text, the repair model can only repair syntax from that text; it cannot recompute the verdict.
* Root cause: JSON repair is treated as pure text normalization. That is only safe when the invalid output contains all semantic fields; empty or truncated provider output contains no recoverable verdict.
* Recommendation: For verifier repair, either include the original compact claim/evidence again, or skip repair on empty/truncated output and return a fail-closed parse failure immediately.
* Iteration trace: Related to Iter 4/5c verifier strictness and parse instability.

### E-008: Sufficiency uses `json_object` over unbounded accumulated evidence
* Severity: critical
* Category: input-size
* Location: `src/retrieval/self_rag/sufficiency_check.py:11` — `src/retrieval/self_rag/sufficiency_check.py:30`; `src/graph/nodes/retrieve.py:193` — `src/graph/nodes/retrieve.py:204`
* Evidence: `is_sufficient` concatenates every accumulated hit with full chunk text and uses `response_format={"type": "json_object"}`. After retries, accumulated evidence can grow across attempts. A realistic 10 chunks x 800 tokens is already around 8k tokens before overhead; multi-subquestion or retry accumulation can exceed that.
* Root cause: The system uses a long evidence-bearing JSON-mode prompt despite documented DeepSeek instability around large JSON prompts. The sufficiency check is on the hot path before synthesis, so provider instability here directly controls retrieval retries and multi-hop routing.
* Recommendation: Bound and summarize sufficiency evidence separately from synthesis evidence. For large evidence sets, ask sufficiency over metadata/facet coverage rather than raw chunks, or run per-facet checks.
* Iteration trace: Same class as Iter 1 and Iter 5b: large JSON/evidence prompts caused provider empty/invalid behavior.

### E-009: Sufficiency prompt does not mandate coverage for comparison or survey questions
* Severity: high
* Category: prompt-design
* Location: `src/retrieval/self_rag/sufficiency_check.py:18` — `src/retrieval/self_rag/sufficiency_check.py:25`; `src/eval/datasets/gold_questions.py:109` — `src/eval/datasets/gold_questions.py:113`
* Evidence: The sufficiency prompt asks if evidence is "sufficient to answer the question" and returns `{"sufficient": true, "missing_aspects": []}`. q-013 asks to compare Toolformer and ToolLLM, but Q5 shows retrieval found only Toolformer and the answer explicitly says ToolLLM evidence is absent while the row is not a hard failure.
* Root cause: Sufficiency is not required to identify entities or facets in the question. A single relevant paper can be judged enough for a comparison or survey unless the prompt forces coverage of every named side.
* Recommendation: Add question-type-aware sufficiency criteria: all named entities for comparisons, multiple approach clusters for surveys, and expected evidence diversity for multi-hop.
* Iteration trace: Caught repeatedly by q-013 in Q2-Q5.

### E-010: Relevance judge is a soft one-line classifier with no calibration standard
* Severity: medium
* Category: prompt-design
* Location: `src/retrieval/self_rag/relevance_judge.py:13` — `src/retrieval/self_rag/relevance_judge.py:29`
* Evidence: The system prompt says `Judge retrieval relevance. Return only JSON...` and the user prompt sends a sub-question plus full chunk. There is no definition of relevant, no threshold, no example, and no instruction to preserve ambiguous-but-useful comparison evidence.
* Root cause: Relevance is deciding which chunks survive into synthesis, but the prompt gives no precision/recall policy. Fail-closed parse handling is now conservative, so ambiguous judging can erase recall.
* Recommendation: Specify a relevance rubric and allow `maybe_relevant` or a score rather than one boolean. This is especially important before the sufficiency check, because dropped hits cannot be recovered except through retry.
* Iteration trace: Related to q-013 retrieve-empty / low-coverage failures.

### E-011: One relevance network failure can kill the whole batch despite per-hit independence
* Severity: medium
* Category: provider-robustness
* Location: `src/retrieval/self_rag/relevance_judge.py:67` — `src/retrieval/self_rag/relevance_judge.py:74`; `src/graph/nodes/retrieve.py:177` — `src/graph/nodes/retrieve.py:181`
* Evidence: `judge_batch` uses `asyncio.gather` without `return_exceptions=True`. If one hit still fails after provider retry, the whole retrieve stage raises instead of dropping or degrading that one hit.
* Root cause: The code treats independent per-hit classifiers as an all-or-nothing operation. Network retry reduces the chance of failure, but does not define a post-retry degradation policy.
* Recommendation: Decide a per-hit failure policy after retry exhaustion: drop, keep with degraded flag, or send to deterministic fallback. Log counts per sub-question.
* Iteration trace: Same failure class as Iter 3: one provider call crashed an eval run.

### E-012: Decomposer prompt does not force preservation of all named comparison entities
* Severity: high
* Category: prompt-design
* Location: `src/retrieval/query/decomposer.py:11` — `src/retrieval/query/decomposer.py:37`; `src/eval/datasets/gold_questions.py:109` — `src/eval/datasets/gold_questions.py:113`
* Evidence: The decomposer prompt only asks for retrieval sub-questions. It does not require one sub-question per named system, one comparison sub-question, or coverage of both Toolformer and ToolLLM. q-013 repeatedly surfaced as comparison debt because the pipeline retrieved Toolformer evidence only.
* Root cause: The first LLM boundary is allowed to compress a comparison question into a single retrieval intent. The rest of the pipeline cannot reliably recover a side the decomposer did not preserve.
* Recommendation: For comparison questions, require decompositions to include both entities and a contrast sub-question. Validate output against the original named entities before retrieval.
* Iteration trace: Caught by q-013 in Q2-Q5.

### E-013: Decomposer JSON repair still hard-fails after one retry
* Severity: medium
* Category: provider-robustness
* Location: `src/retrieval/query/decomposer.py:39` — `src/retrieval/query/decomposer.py:58`
* Evidence: After the repair call, `Decomposition.model_validate_json(repair.content or "")` is returned directly. There is no fallback to the original question when the provider returns invalid JSON twice.
* Root cause: The code has a cheap safe fallback for trivial questions but not for provider failure on nontrivial questions. That makes one JSON boundary the availability gate for the whole graph.
* Recommendation: Treat decomposition parse failure as a degraded single-subquestion path with telemetry, or make eval fail loudly before any spend on downstream calls.
* Iteration trace: Phase 10 baseline q-014 failed on decomposition JSON; this is the same implementation boundary, not an architecture finding.

### E-014: Rewriter and HyDE prompts can erase required terms without detection
* Severity: medium
* Category: prompt-design
* Location: `src/retrieval/query/rewriter.py:12` — `src/retrieval/query/rewriter.py:45`; `src/graph/nodes/retrieve.py:135` — `src/graph/nodes/retrieve.py:138`
* Evidence: `rewrite_for_retrieval` asks for a concise keyword-rich query, and `hyde` writes a hypothetical answer paragraph. Neither prompt requires preserving named entities or all sides of a comparison. The retrieve node embeds HyDE text when enabled, but searches lexical with the rewritten query.
* Root cause: Query rewriting is permitted to optimize relevance without a preservation contract. If it drops ToolLLM, Gorilla, SWE-bench, or a second comparison paper, hybrid search will faithfully retrieve the wrong narrower scope.
* Recommendation: Add preservation requirements and validation for named entities, acronyms, arXiv-paper names, and comparison pairs. Reject rewrites that lose terms.
* Iteration trace: q-013 repeatedly shows one-sided comparison evidence.

### E-015: Multi-hop scorer uses JSON mode over batches of 16 abstracts
* Severity: high
* Category: input-size
* Location: `src/retrieval/multi_hop/citation_walker.py:160` — `src/retrieval/multi_hop/citation_walker.py:204`
* Evidence: `_score_candidate_batch` serializes up to `SCORE_BATCH_SIZE=16` candidates, including title and abstract, into one JSON-mode prompt with `max_tokens=800`. Sixteen 200-300 token abstracts plus JSON overhead can exceed 4k input tokens; longer abstracts can exceed 8k.
* Root cause: Batch size was chosen for throughput, not provider prompt reliability. It combines large JSON input with JSON output mode, the same pattern that failed in Phase 10.6a.
* Recommendation: Cap total prompt tokens per score batch, not just candidate count. Split long abstracts or score candidates with truncated title/abstract snippets.
* Iteration trace: Same provider-boundary class as Iter 1; not yet observed because multi-hop rarely triggers.

### E-016: Multi-hop score JSON omissions silently fall back per paper
* Severity: medium
* Category: prompt-design
* Location: `src/retrieval/multi_hop/citation_walker.py:186` — `src/retrieval/multi_hop/citation_walker.py:216`
* Evidence: The prompt says to reply with scores, but does not require exactly one score for every input paper ID. Missing IDs are filled with `_heuristic_score` at lines 212-216 with no explicit parse-failure count by candidate.
* Root cause: The scorer output schema is underspecified. Missing verdicts are not treated as malformed output; they silently invoke a different ranking policy.
* Recommendation: Require and validate complete coverage of every candidate ID. If missing, retry or mark that batch as degraded.
* Iteration trace: Latent; would surface once multi-hop is exercised under realistic candidate batches.

### E-017: Semantic extractor repeats the failed large JSON pattern
* Severity: high
* Category: input-size
* Location: `src/memory/semantic/extractor.py:21` — `src/memory/semantic/extractor.py:82`; `src/graph/nodes/reflect.py:15` — `src/graph/nodes/reflect.py:23`
* Evidence: `extract_concepts_from_text` uses `main`, `response_format={"type":"json_object"}`, a deeply nested concept/relation schema, and `max_tokens=1800`. Reflection feeds answer text plus cited chunk excerpts; multiple 800-token chunks can push the prompt beyond 4k tokens.
* Root cause: The exact provider pattern that failed for structured synthesis is still present in semantic reflection. Eval hides it with `skip_reflection=True`, but production turns it back on.
* Recommendation: Simplify extraction schema, cap reflection text, or split concept extraction per chunk. Do not run complex nested JSON extraction over concatenated answer-plus-evidence blobs.
* Iteration trace: Same class as Iter 1: DeepSeek `json_object` plus complex schema plus large evidence prompt.

### E-018: Concept extraction silently returns no concepts after empty retry failure
* Severity: medium
* Category: provider-robustness
* Location: `src/memory/semantic/extractor.py:58` — `src/memory/semantic/extractor.py:82`
* Evidence: If the first valid JSON contains no concepts, the code retries with the full text. If that retry returns invalid JSON, it returns `[]`. There is no parse-failure report in `ReflectionReport`.
* Root cause: Semantic memory indexing treats provider failure the same as "no concepts exist." That makes memory-quality regressions invisible.
* Recommendation: Return a structured extraction status with `no_concepts`, `parse_failure`, and `provider_empty` separated. Reflection reports should expose skipped/failed messages.
* Iteration trace: Latent; eval currently skips reflection, so this path has not been stress-tested.

### E-019: Linker same-concept judge has no repair retry or fail policy
* Severity: medium
* Category: provider-robustness
* Location: `src/memory/semantic/linker.py:129` — `src/memory/semantic/linker.py:152`
* Evidence: `_judge_same_concept` uses `response_format={"type":"json_object"}` and immediately validates `_SameConceptPayload`. Invalid or empty JSON raises out of linking.
* Root cause: The linker is a write-path classifier, but it has less provider hardening than the retrieval classifiers. One malformed response can abort reflection or concept creation.
* Recommendation: Add the same explicit parse policy used elsewhere: repair once, then either fail closed to `same=False` or surface a write-path error.
* Iteration trace: Same class as Iter 1 and Iter 3, currently masked by skip-reflection eval.

### E-020: Graph phrase detection is a JSON-mode prompt with no mandate strength
* Severity: low
* Category: prompt-design
* Location: `src/memory/semantic/graph_retriever.py:14` — `src/memory/semantic/graph_retriever.py:66`
* Evidence: `MENTION_PROMPT` says "List noun phrases..." and `Reply JSON {"phrases": ["..."]}`. It does not require exact noun phrase strings from the question, does not limit count, and has no repair path.
* Root cause: Input Mode depends on phrase extraction, but this prompt is still written as a quick helper rather than a production classifier.
* Recommendation: Make phrase extraction deterministic where possible, and for LLM extraction require exact substrings plus a bounded phrase count. Add repair or degrade-to-empty explicitly.
* Iteration trace: Latent; baseline says Input Mode has not been exercised end-to-end.

### E-021: Episodic compressor has no input cap and no empty/truncation policy
* Severity: medium
* Category: output-budget
* Location: `src/memory/episodic/compressor.py:92` — `src/memory/episodic/compressor.py:110`
* Evidence: `_summarize_incremental` concatenates all new messages and uses `max_tokens=500`. It returns stripped content without checking empty output or `finish_reason="length"`.
* Root cause: Memory compression is treated as best-effort text generation. In production, long sessions will exceed prompt or summary budgets, and the cache can persist an empty or truncated summary as if it were valid.
* Recommendation: Add input chunking, summary-size targets, and explicit handling for empty/length stop. Persist summary metadata indicating whether compression was complete.
* Iteration trace: Same budget class as Iter 5a, but in memory rather than synthesis.

### E-022: Provider retry does not cover SDK-native OpenAI/Cohere timeout classes
* Severity: high
* Category: provider-robustness
* Location: `src/llm/retry.py:19` — `src/llm/retry.py:24`; `src/llm/providers/openai_embed.py:46` — `src/llm/providers/openai_embed.py:52`; `src/llm/providers/cohere_rerank.py:51` — `src/llm/providers/cohere_rerank.py:58`
* Evidence: `retry_transient_network` retries only `httpx.ReadTimeout`, `httpx.ConnectTimeout`, `httpx.RemoteProtocolError`, and `httpx.ConnectError`. The OpenAI and Cohere integrations call SDK methods, which commonly wrap transport failures in SDK exception types rather than raw `httpx` exceptions.
* Root cause: The network retry was validated against the routed chat wrapper with fake raw `httpx` exceptions, but the embedding/rerank providers are SDK boundaries. Their actual exception surface is not modeled.
* Recommendation: Audit actual SDK exception classes and include them in transient retry policy, or normalize provider exceptions inside each provider adapter.
* Iteration trace: Caught by Iter 3 for chat; the same hole may remain for embed/rerank.

### E-023: Rate-limit retry ignores `Retry-After` and may still burn budget early
* Severity: medium
* Category: provider-robustness
* Location: `src/llm/retry.py:47` — `src/llm/retry.py:58`; `src/llm/providers/deepseek.py:38` — `src/llm/providers/deepseek.py:72`
* Evidence: HTTP 429 is mapped to `LLMRateLimitError`, then retried with exponential jitter using fixed constants. The response headers are discarded in `_raise_for_status`, so `Retry-After` cannot influence the wait.
* Root cause: Retry timing is role-agnostic and provider-agnostic. It is better than the old 150ms budget, but still not tied to actual provider reset windows.
* Recommendation: Preserve retry headers in provider errors and honor `Retry-After` when present. Otherwise, log rate-limit retry schedules so repeated failures are distinguishable from hard provider outages.
* Iteration trace: Related to Phase 1 retry-backoff fix; still not fully provider-aware.

### E-024: Empty-response retry is chat-only; embeddings and rerank have no empty-result guard
* Severity: medium
* Category: provider-robustness
* Location: `src/llm/client.py:94` — `src/llm/client.py:115`; `src/llm/providers/openai_embed.py:53` — `src/llm/providers/openai_embed.py:63`; `src/llm/providers/cohere_rerank.py:59` — `src/llm/providers/cohere_rerank.py:70`
* Evidence: `_should_retry_empty_response` applies only to chat. Embedding returns whatever vectors the SDK data contains; rerank returns whatever result list Cohere returns. There is no validation that returned embedding count equals input count, or that rerank returned any results for non-empty documents.
* Root cause: Provider "HTTP 200 but semantically empty" was fixed for chat after q-025, but not generalized to all provider adapters.
* Recommendation: Add semantic response validation per provider: embedding count must match input count, rerank must return at least one result when documents are non-empty, and empty SDK payloads should be retried or surfaced.
* Iteration trace: Same class as Iter 1/5b empty chat responses.

### E-025: LLM call logging does not make failures diagnosable in eval rows
* Severity: medium
* Category: eval-mechanism
* Location: `src/llm/client.py:266` — `src/llm/client.py:282`; `src/eval/runner.py:217` — `src/eval/runner.py:242`
* Evidence: `llm_call` logs role, tokens, finish reason, and duration, but `EvalRow` persists only metrics, rationale, answer, and hard_failure. Q5 required manual CSV inspection and could not prove q-005 finish reason from the row.
* Root cause: Observability exists in logs, but eval artifacts do not join per-question outputs with provider stop reasons, parse failures, or stage failures. A production incident has to reconstruct state from separate logs.
* Recommendation: Add per-question diagnostic counters to eval rows or sidecars: synthesis finish reason, citation parse count, verifier rejected count, parse-failure count, and degraded reason.
* Iteration trace: Caught by Q1-Q5 repeatedly; each iteration required ad hoc diagnosis.

### E-026: Hard failure definition misses "answer has citations but verifier removed all"
* Severity: high
* Category: eval-mechanism
* Location: `src/eval/runner.py:115` — `src/eval/runner.py:126`; `data/eval_outputs/latest/per_question.csv:6`
* Evidence: `_run_one` marks hard failure only when `synthesis_format_degraded` and no cited papers. In Q5, q-029 had valid `[arxiv:2201.11903#...]` markers in the answer, judge 5/5/5, `cited_paper_ids=[]`, and `hard_failure=False`.
* Root cause: The eval conflates synthesis-format failure with verification failure. From a user standpoint, an answer whose every citation is stripped is not the same as a verified answer, but the hard-failure counter does not distinguish it.
* Recommendation: Split hard failures into typed counters: agent exception, retrieval-empty, synthesis-empty, citation-parse-empty, verifier-all-rejected, and judge-invalid. Gate each class explicitly.
* Iteration trace: Caught by Iter 4 and Iter 5c.

### E-027: Sample gate with N=5 is statistically noisy
* Severity: medium
* Category: eval-mechanism
* Location: `Makefile:33` — `Makefile:37`; `src/eval/runner.py:31` — `src/eval/runner.py:99`
* Evidence: `eval-sample` always runs five fixed questions. With a pass condition of at most one hard failure, a system whose true hard-failure probability is 20% still fails the sample about 26.3% of the time. Even at 10% true failure, the sample fails about 8.1% of the time.
* Root cause: The sample is used as a deterministic release gate, but it is a tiny stochastic probe over provider behavior and question mix. It is useful as a canary, not as a statistically reliable estimate.
* Recommendation: Keep the cheap canary, but treat sample failure as typed diagnostic input. For gating, use either repeated small samples with typed failures or a larger stratified sample with confidence intervals.
* Iteration trace: The five Phase 10.6a sample iterations demonstrate this exact instability.

### E-028: Resume mode underreports cost because usage is reset before loading old rows
* Severity: medium
* Category: eval-mechanism
* Location: `src/eval/runner.py:64` — `src/eval/runner.py:70`; `src/eval/runner.py:245` — `src/eval/runner.py:282`
* Evidence: `run_eval` calls `usage.reset()` before reading completed CSV rows. `_write_summary` includes all selected rows, including skipped rows, but token usage and estimated cost only include calls from the current process.
* Root cause: CSV rows are durable but usage accounting is process-local. A resumed eval can report metrics for 30 rows with cost for only the last resumed segment.
* Recommendation: Persist per-question usage snapshots or accumulate usage sidecars alongside CSV rows. Summary cost should be recomputed from durable per-row usage, not the current in-memory counter.
* Iteration trace: Related to Phase 10.5 resume discipline; not yet fixed for cost correctness.

### E-029: Eval has no mid-run cost breaker
* Severity: medium
* Category: eval-mechanism
* Location: `src/eval/runner.py:80` — `src/eval/runner.py:99`; `src/llm/usage.py:45` — `src/llm/usage.py:61`
* Evidence: The runner computes `usage.estimated_cost_usd()` only when writing the final summary. There is no check after each question against the phase cost cap.
* Root cause: Cost reporting is retrospective. The planner gives hard caps, but the code cannot stop a bad run when it crosses the cap mid-eval.
* Recommendation: Check estimated cost after each question and stop with a typed `cost_cap_exceeded` row or clean abort before spending further.
* Iteration trace: Phase 10.6a cost was manually watched across iterations; the runner did not enforce it.

### E-030: Eval judge is a JSON-mode single point of failure with an under-specified rubric
* Severity: medium
* Category: prompt-design
* Location: `src/eval/metrics/llm_judge.py:12` — `src/eval/metrics/llm_judge.py:70`
* Evidence: The judge uses `main`, `response_format={"type":"json_object"}`, `max_tokens=2000`, and a brief rubric. It asks correctness to match expected papers and core facts but does not say how to score answers with correct prose and zero verified citations. On invalid JSON twice it assigns 1/1/1.
* Root cause: The judge is both metric and failure classifier, but its prompt lacks policy for the exact cases now dominating Phase 10.6a: good prose with unverifiable citations, empty retrieval, and truncated citation markers.
* Recommendation: Separate judge scoring from system-failure classification. Give the judge explicit rubric examples for unverifiable but substantively correct answers.
* Iteration trace: Iter 4/Q4 and Iter 5/Q5: q-019/q-029 received high judge scores while citation metrics were zero.

### E-031: `DIAGNOSTIC_QUESTION_ID` is process-global mutable state
* Severity: low
* Category: eval-mechanism
* Location: `src/eval/runner.py:103` — `src/eval/runner.py:135`; `src/core/diagnostics.py:15` — `src/core/diagnostics.py:30`
* Evidence: `_run_one` writes `os.environ["DIAGNOSTIC_QUESTION_ID"]` before running a question, and `log_raw_response` reads it later. This is safe only because eval is sequential today.
* Root cause: Diagnostic correlation relies on global process state instead of context-local state. If question-level concurrency is added, raw responses can be attributed to the wrong question.
* Recommendation: Use a `contextvars.ContextVar` or pass diagnostic context through call wrappers. Keep the env var only as the on/off switch.
* Iteration trace: Diagnostic logging was added during Phase 10.5; it was not engineered for future concurrency.

### E-032: Tests assert prompt substrings, not provider-boundary behavior
* Severity: medium
* Category: eval-mechanism
* Location: `tests/unit/test_synthesize_degraded.py:25` — `tests/unit/test_synthesize_degraded.py:43`; `tests/unit/test_retrieve_stages.py:12` — `tests/unit/test_retrieve_stages.py:68`
* Evidence: The synthesis test checks that two substrings appear in prompts and that no-citation output degrades. The retrieve-stage test uses fake perfect dependencies and only asserts stages logged. Neither test catches q-005 truncation, q-025 empty provider response on long evidence, or q-029 verifier over-rejection.
* Root cause: Unit tests lock the local control-flow shape, not the LLM boundary contracts. The production failures were about prompt size, finish reasons, and model compliance under realistic payloads.
* Recommendation: Add deterministic contract tests for prompt construction under worst-case evidence counts, finish_reason length, empty response, malformed citation variants, and verifier rejection rationale distribution.
* Iteration trace: Q1-Q5 all exposed failures that unit tests did not model.

### E-033: Rerank cost estimate is likely overstated and not provider-derived
* Severity: low
* Category: eval-mechanism
* Location: `src/llm/usage.py:15` — `src/llm/usage.py:21`; `src/llm/client.py:176` — `src/llm/client.py:187`
* Evidence: The price table charges `"rerank"` as `$0.002` per query and records one call per rerank wrapper invocation. It does not read Cohere usage or document whether this price is per query, per 1k searches, or per document batch for the current model.
* Root cause: The token counter is now visible but not fully provider-accurate. Rerank billing is modeled as a constant without verification.
* Recommendation: Validate the Cohere rerank billing unit and include document count in usage. Until then, label rerank cost as approximate.
* Iteration trace: Token counter added in Phase 10.5; cost visibility is still approximate.

### E-034: DeepSeek `json_object` risk remains in production-critical non-eval paths
* Severity: high
* Category: input-size
* Location: `src/retrieval/self_rag/sufficiency_check.py:16` — `src/retrieval/self_rag/sufficiency_check.py:30`; `src/memory/semantic/extractor.py:21` — `src/memory/semantic/extractor.py:82`; `src/retrieval/multi_hop/citation_walker.py:186` — `src/retrieval/multi_hop/citation_walker.py:204`
* Evidence: The project backed away from structured JSON synthesis because DeepSeek `json_object` plus complex/large prompts was unreliable. Other paths still use the same mode with evidence-heavy or schema-heavy prompts: sufficiency, semantic extraction, and multi-hop scoring.
* Root cause: The lesson from A1 was applied narrowly to synthesis instead of generalized as a provider integration constraint.
* Recommendation: Inventory every `json_object` call by prompt size and schema complexity. Keep JSON mode only for small, simple outputs, and use simpler contracts or chunked calls for large evidence tasks.
* Iteration trace: Directly from Iter 1: structured synthesis failed under DeepSeek provider behavior.

## Cross-cutting themes

1. **Prompt mandates are inconsistent.** Synthesis now has a hard citation mandate; decomposer, rewriter, relevance, sufficiency, verifier, multi-hop scorer, graph retriever, extractor, linker, compressor, and judge mostly still use soft explanations or tiny schema examples.
2. **Token budgets are historical constants, not engineered limits.** `max_tokens` and evidence payload sizes are set per call, but no call computes worst-case input or output size from the number of chunks, citations, candidates, or messages.
3. **DeepSeek JSON-mode instability was treated as a synthesizer-only problem.** The same high-risk pattern remains in sufficiency, semantic extraction, multi-hop scoring, judge, graph phrase detection, and linker decisions.
4. **Provider robustness is layered after failures instead of designed at every adapter boundary.** Chat got empty-response and network retries; SDK embedding/rerank semantics, `finish_reason="length"`, and provider-specific timeout classes remain under-specified.
5. **The eval harness conflates very different failure modes.** Retrieval-empty, syntax degradation, verifier-all-rejected, provider timeout, judge failure, and architecture debt are collapsed into one `hard_failure` boolean plus a few aggregate metrics.
6. **Observability exists but is not connected to decisions.** Structured logs and diagnostic JSONL help during manual debugging, but eval rows do not carry the stop reasons, parse statuses, rejection counts, or prompt sizes that explain failures.
7. **Tests mostly prove plumbing, not LLM-boundary contracts.** The failures in Q1-Q5 were reviewable from prompt size, output schema, and provider behavior, but existing tests use tiny fake payloads and perfect mock responses.

## What was NOT reviewed

* Did not run live tests, `make eval-preflight`, `make eval-sample`, or any provider calls; relied on code reading plus `PHASE_10_6A_QUESTIONS.md` and the current `data/eval_outputs/latest/` sample artifact.
* Did not inspect `.env` or any secrets.
* Did not review dependency security advisories or current provider API docs.
* Did not re-audit the previous architectural findings except where they intersected with prompt/provider/eval mechanics.
* Did not inspect every UI or database path because this review is scoped to LLM boundary rigor and eval correctness.
