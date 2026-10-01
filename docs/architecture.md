# Implemented research flow

The agent keeps its existing LangGraph design. This pass adds a small CLI and corrects the hand-off between evidence, answer text, verification, and memory. It does not introduce another agent framework or change the database schema.

## One question, end to end

1. **Plan.** `decompose_node` produces subquestions. The existing trivial-question heuristic may return the original question directly. The schema can record subquestion dependencies; the current executor processes the subquestions in sequence rather than executing a general dependency graph.
2. **Retrieve.** For each subquestion, the router may add concept-graph hints. Query rewriting/optional HyDE precedes vector and lexical retrieval. RRF fuses rankings; deduplication, reranking, and relevance filtering select chunks.
3. **Check coverage.** Insufficient coverage can trigger up to two query rewrites. If the budget and route allow, one citation-walker call can expand the corpus before retrieval runs again. Already useful chunks survive both kinds of retry.
4. **Synthesize.** The model receives the question and available chunks, each headed by a full `[paper_id#chunk_id]` identifier. Empty evidence returns an explicit gap without a generation call.
5. **Verify.** The parser maps references to the preceding sentence. Each reference must resolve to the same paper and chunk used in generation. The verifier sees that sentence and the complete chunk. Unsupported references, missing identifiers, empty claims, and invalid verification responses fail closed.
6. **Finish.** Failed checks can trigger bounded regeneration. The final output keeps an explicit pass/draft state and the per-reference verdicts. A draft can still contain rejected claims: the CLI labels it unverified rather than presenting it as a trusted answer.
7. **Store.** LangGraph checkpoints persist. Unless reflection is skipped, the assistant answer and its verification flag are stored. Newly rejected answers do not enter semantic learning and are skipped by later reflection passes.

## Memory boundaries

| Layer | Implemented behavior | Current boundary |
| --- | --- | --- |
| Working state | Subquestions, retrieved chunks, generated answer, and verdicts | Evidence for one graph run |
| Checkpoints | Postgres-backed LangGraph state | The new CLI creates a fresh thread for each question |
| Episodic utilities | Session records and rolling-summary compression | Compressed history is not currently fed into synthesis |
| Semantic memory | Extract concepts/relations after accepted runs; use matching concepts as retrieval hints | Model-assessed support is not independent truth; older records are not re-audited here |

## What changed and why

| Before | Now | Reason |
| --- | --- | --- |
| A citation's `claim_span` pointed at the marker itself | It points at the preceding sentence; grouped references share that sentence | The verifier needs the assertion to check, not just `[paper#chunk]` |
| Evidence lookup used only `chunk_id` | It uses `paper_id#chunk_id` | A mismatched paper label cannot borrow another paper's chunk |
| The verifier saw only the first 1,200 characters of a chunk | It sees the full retrieved chunk and only the relevant claim | Supporting text later in the chunk must remain visible |
| Missing references could crash, and an empty citation list could pass vacuously | Both return explicit failed verification | Keep failures inspectable and preserve fail-closed behavior |
| JSON repair lacked the original claim/evidence | A repair attempt retains both | Formatting repair must not manufacture an unsupported `true` verdict |
| Re-retrieval after citation traversal could replace earlier evidence | Previously relevant hits seed the next retrieval pass | Preserve useful evidence when expansion finds nothing better |
| Rejected drafts could feed reflection | New drafts carry `verification_passed=false` and bypass semantic learning | Avoid treating an unverified answer as newly learned knowledge |
| README advertised missing UI files | A real single-question CLI, offline example, and explicit UI status | Readers can try a supported entry point |

The evidence-lookup key and `claim_span` meaning changed. Old checkpoints are not migrated; use a fresh thread for this version. The CLI already does that. Neither model routes nor scoring formulas were changed, and historical evaluation records remain intact.

## Reproduce the offline checks

```bash
uv sync --frozen
uv run pytest tests/unit -q
uv run ruff check src tests scripts
uv run python -m src.cli --example --json
```

The regression cases include grouped citations, misplaced paper IDs, empty evidence, evidence after the old character cutoff, malformed verifier output, failed-answer memory handling, and the full synthesis-to-verification hand-off with deterministic provider fixtures.

These checks do not invoke a paid provider or Postgres. Database integration and live quality still require a configured runtime. In particular, fixing a claim span is not evidence that the historical zero citation scores have recovered.
