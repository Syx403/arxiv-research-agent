# Engineering Backlog

Tracking all known engineering and architecture issues across the project. Each item is sourced from a specific review or iteration and assigned a proposed phase target. This file is the planner's reference for Phase 10.6+ work — pull from here, do not invent.

## Sources

| Tag | Source |
|---|---|
| ADR | `docs/reviews/ADVERSARIAL_REVIEW_2026-05-04.md` (architectural smells) |
| ENG | `docs/reviews/ENGINEERING_RIGOR_REVIEW_2026-05-05.md` (implementation rigor) |
| ITER | `docs/phase_reports/PHASE_10_6A_QUESTIONS.md` Q1-Q5 (sample iteration history) |

## Status legend

- **[DONE]** — completed in a shipped phase
- **[IN PHASE X]** — scheduled for phase X
- **[DEFERRED]** — not scheduled; proposed phase noted
- **[SUPERSEDED]** — made moot by another fix; brief explanation

---

## Phase 10.6 — DONE

Scope: foundation work from uncommitted Phase 10.6a + 8 high-impact engineering rigor fixes. Single commit, single full eval as validation. Decision authority granted: F3=B (substantial support verifier rubric), F6=typed counters with split between hard-failure and warning per planner spec.

Validation: full eval on 2026-05-05 passed typed hard-failure gate with `hard_failure_rate=0.0333`, `retrieval_empty=0`, `verifier_all_rejected=5`, and total estimated cost `$0.8282`. Frozen snapshot: `data/eval_outputs/baseline_2026_05_05_phase10_6/`.

### F1 — Synthesis evidence cap + `finish_reason="length"` handling
- Findings: ENG E-001, E-002
- Iter trace: Iter 5a (q-005 truncation), Iter 5b (q-025 empty on long survey prompt)
- Status: [DONE]

### F2 — Sufficiency evidence cap + same finish_reason handling
- Findings: ENG E-008
- Iter trace: same provider class as Iter 1, 5b
- Status: [DONE]

### F3 — Verifier rubric = "substantial support" + claim granularity
- Findings: ENG E-005, E-006
- Iter trace: Iter 4, Iter 5c (q-019/q-029 strict-rejection of plausibly cited prose)
- Decision: option B (substantial support, single verdict). Tier-3 verdict (option C) deferred.
- Status: [DONE]

### F4 — Decomposer comparison preservation
- Findings: ENG E-012; ADR F-019 (partial — comparison side)
- Iter trace: q-013 across all 5 iterations
- Status: [DONE]

### F5 — Provider robustness generalization (SDK exceptions + embed/rerank empty validation)
- Findings: ENG E-022, E-024
- Iter trace: Iter 3 (chat covered); embed/rerank still uncovered
- Status: [DONE]

### F6 — Typed hard_failure counters
- Findings: ENG E-026
- Iter trace: every iteration required manual diagnosis to distinguish failure modes
- Decision: counters that COUNT toward hard_failure_rate: `agent_exception`, `synthesis_empty`, `citation_parse_empty`, `judge_invalid`. Counters that EMIT WARNING but DO NOT count: `retrieval_empty` (architecture debt), `verifier_all_rejected` (rubric-driven side effect).
- Status: [DONE]

### F7 — `json_object` risk path remediation
- Findings: ENG E-034 (and inherits scope from E-008/E-015/E-017)
- Scope: inventory and remediate `json_object` calls in sufficiency_check, semantic extractor, multi-hop scorer. Either reduce schema complexity, cap inputs, or chunk calls.
- Status: [DONE]

### F8 — Per-question diagnostic columns in EvalRow
- Findings: ENG E-025
- Scope: add columns for synthesis `finish_reason`, citation parse count, verifier rejected count, parse-failure count
- Status: [DONE]

### Phase 10.6 also folds in (existing uncommitted work)
- B1 retrieve.py 9 stages + structured logging
- B2 LLM call structured logging (`llm_call` event)
- A1' Citation.claim_text from sentence containing marker
- Per-citation verifier consuming `claim_text`
- Network retry on transient httpx errors
- Synthesizer citation mandate (Iter 5 fix)
- max_tokens 1200→2000 in synthesizer (Iter 5a fix)

Status: [DONE]

---

## Phase 10.7 — DEFERRED — Provider/contract reliability second pass

### B-101 — Decomposer JSON repair fallback to single-subquestion path
- Source: ENG E-013
- Severity: medium
- Why deferred: comparison-aware decomposer (F4) lifts decomposer importance; once F4 is in place, retest fallback policy with real failure rate data
- Status: [DEFERRED → 10.7]

### B-102 — Rewriter / HyDE entity preservation
- Source: ENG E-014
- Severity: medium
- Why deferred: F4 (decomposer comparison) addresses the worst symptom (q-013); rewriter-side preservation is second-line defense
- Status: [DEFERRED → 10.7]

### B-103 — Multi-hop scorer batch size + completeness validation
- Source: ENG E-015, E-016
- Severity: high (latent — multi-hop rarely fires under current sufficiency thresholds)
- Why deferred: F4 will increase multi-hop trigger rate; assess real load before batch refactor
- Status: [DEFERRED → 10.7]

### B-104 — Sufficiency coverage criteria for survey question type
- Source: ENG E-009; ADR F-019 (survey side)
- Severity: high
- Why deferred: F4 handles comparison; survey iterative coverage was originally planned as Phase 10.6c. Defer to 10.7 so we measure the post-F4 baseline first.
- Status: [DEFERRED → 10.7]

### B-105 — Relevance judge rubric + `maybe_relevant` tier
- Source: ENG E-010
- Severity: medium
- Status: [DEFERRED → 10.7]

### B-106 — Relevance batch error policy (`return_exceptions=True` + per-hit fallback)
- Source: ENG E-011
- Severity: medium
- Status: [DEFERRED → 10.7]

### B-107 — Verifier repair context preservation
- Source: ENG E-007
- Severity: medium
- Why deferred: F1 (evidence cap) and F3 (rubric) reduce repair frequency; revisit after seeing post-F1/F3 repair rate
- Status: [DEFERRED → 10.7]

### B-108 — Eval judge rubric robustness + JSON repair retry
- Source: ENG E-030
- Severity: medium
- Why deferred: judge invalid JSON observed but rare; F8 typed counters surface it for measurement first
- Status: [DEFERRED → 10.7]

### B-109 — Rate-limit retry honoring `Retry-After`
- Source: ENG E-023
- Severity: medium
- Why deferred: not yet observed in eval load; current backoff suffices
- Status: [DEFERRED → 10.7]

### B-110 — Multi-hop graph-level trigger criteria (preemptive for hop questions)
- Source: ADR F-008
- Severity: critical (architectural)
- Why deferred: original Phase 10.6d. Defer until post-F4 multi_hop trigger rate is measured.
- Status: [DEFERRED → 10.7]

### B-111 — Multi-hop budget accounting (attempts vs successful expansions)
- Source: ADR F-009
- Severity: medium
- Status: [DEFERRED → 10.7]

---

## Phase 11+ — DEFERRED — Production readiness (relevant when UI multi-user)

### B-201 — Ingestion idempotency under concurrency (advisory lock)
- Source: ADR F-010
- Severity: high (only matters when multi-user)
- Status: [DEFERRED → 11+]

### B-202 — Reflection idempotency under concurrency
- Source: ADR F-012
- Severity: high (only matters when multi-user with concurrent sessions)
- Status: [DEFERRED → 11+]

### B-203 — Alias collision observability across write paths
- Source: ADR F-013
- Severity: medium
- Status: [DEFERRED → 11+]

### B-204 — Semantic Scholar partial-failure visibility (citation graph readiness ≠ paper readiness)
- Source: ADR F-011
- Severity: high
- Status: [DEFERRED → 11+]

### B-205 — Router as policy emitter, not substring alias check
- Source: ADR F-015
- Severity: medium
- Status: [DEFERRED → 11+]

### B-206 — Graph edges fail loudly on invalid state instead of best-effort fallthrough
- Source: ADR F-016
- Severity: medium
- Status: [DEFERRED → 11+]

### B-207 — Resume mode cost accounting (durable per-row usage)
- Source: ENG E-028
- Severity: medium
- Status: [DEFERRED → 11+]

### B-208 — Mid-run cost breaker
- Source: ENG E-029
- Severity: medium
- Status: [DEFERRED → 11+]

### B-209 — `DIAGNOSTIC_QUESTION_ID` → ContextVar (for future concurrency)
- Source: ENG E-031
- Severity: low
- Status: [DEFERRED → 11+]

### B-210 — Sample gate redesign (stratified or repeated, not single-shot N=5)
- Source: ENG E-027
- Severity: medium
- Status: [DEFERRED → 11+]

---

## Phase 12+ — DEFERRED — Polish

### B-301 — Episodic compressor input cap + empty/length policy
- Source: ENG E-021
- Severity: medium (production sessions only)
- Status: [DEFERRED → 12+]

### B-302 — Semantic memory paths (extractor / linker / phrase detection / concept extraction silent-fail)
- Source: ENG E-017, E-018, E-019, E-020
- Severity: medium-high (currently masked by `skip_reflection=True` in eval)
- Why deferred: reflection is disabled in eval; before enabling we want F3 + B-104 (rubric + survey coverage) so we don't write low-quality concepts into the persistent graph
- Status: [DEFERRED → 12+]

### B-303 — Provider-boundary contract tests (replace prompt-substring tests)
- Source: ENG E-032
- Severity: medium
- Status: [DEFERRED → 12+]

### B-304 — Cohere rerank price model verification + document-count usage
- Source: ENG E-033
- Severity: low
- Status: [DEFERRED → 12+]

---

## Cross-cutting themes (from reviews)

These are not individual fixes but recurring patterns. Each new phase should consider whether it touches a theme.

1. **Prompt mandates inconsistent**: F3, F4 add mandates to verifier and decomposer. After Phase 10.6, audit remaining soft-explanation prompts (rewriter, HyDE, relevance, multi-hop scorer, judge).
2. **Token budgets are historical constants**: F1, F2 calibrate synthesis/sufficiency. After Phase 10.6, audit remaining `max_tokens` values against worst-case prompts (extractor, judge, decomposer).
3. **DeepSeek `json_object` instability is project-wide**: F7 starts the inventory. Each future phase touching an LLM call must check whether `json_object` is appropriate for that call's prompt size and schema complexity.
4. **Provider robustness layered after failures**: F5 closes the embed/rerank gap. Each new provider integration must include retry + empty-response validation at the adapter boundary.
5. **Eval conflates failure modes**: F6 + F8 introduce typed counters and per-question diagnostic columns. Future eval changes should preserve and extend this typed observability.
6. **Tests prove plumbing, not LLM-boundary contracts**: B-303 is the explicit fix; meanwhile, every Phase 10.6+ test must include at least one provider-boundary scenario (worst-case input size, finish_reason=length, empty response, malformed citation variants).

---

## How to use this file

- When starting a new phase, scan the matching section for items to pull
- When closing a phase, mark addressed items as [DONE] with commit hash
- When a new review or iteration adds findings, append them under the appropriate phase target
- Do NOT silently change severity or skip items — explicit deferral with reason only

Last updated: 2026-05-05 (Phase 10.6 launch)
