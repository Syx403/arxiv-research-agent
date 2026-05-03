# Phase 6 Report

**Phase title**: Bounded Multi-hop Citation Walker
**Status**: complete
**Commit**: `HEAD (see git log)`
**Owner handoff pending**: none

## Summary
Phase 6 implemented the bounded citation traversal subsystem with deterministic frontier ordering, arXiv-only candidate handling, LLM-judged frontier scoring, lazy ingestion, and wall-clock partial returns. The walker uses bidirectional local citation expansion as approved in `PHASE_6_QUESTIONS.md`: outbound references can trigger lazy ingestion, while local incoming citers let follow-up papers such as Reflexion be reached from ReAct. Candidate scoring uses the `fast` role through `src/llm/client.py`, batched with `asyncio.Semaphore(8)`, and non-arXiv citation rows are logged by hop depth with skipped `s2` and `title` counts. The live integration run reached Reflexion, lazy-ingested `arxiv:2208.14271`, completed in 20.43 seconds, and removed the lazy-ingested paper during teardown.

## Files created
- `docs/phase_reports/PHASE_6_QUESTIONS.md` - data-shape and traversal-direction analysis with signed-off options.
- `src/retrieval/multi_hop/__init__.py` - multi-hop package marker.
- `src/retrieval/multi_hop/frontier.py` - heap-backed priority frontier with deterministic tie-breaking and dedup.
- `src/retrieval/multi_hop/citation_walker.py` - bounded bidirectional local citation walker with lazy ingestion.
- `tests/unit/test_frontier.py` - frontier priority, tie-ordering, dedup, and empty-pop tests.
- `tests/unit/test_citation_walker.py` - wall-clock partial-return unit test.
- `tests/integration/test_multi_hop_walk.py` - live ReAct to Reflexion walk with lazy-ingest cleanup.

## Files modified
- `src/core/types.py` - added `MULTI_HOP_WALL_CLOCK_BUDGET_S`, changed `MULTI_HOP_FRONTIER_LIMIT` to `5`, and added `FrontierItem`/`WalkResult`.
- `docs/STATUS.md` - Phase 6 marked complete.
- `docs/interview_talking_points.md` - Phase 6 talking points appended.

## Tests run
- Command: `uv run pytest tests/unit/test_frontier.py -q`
- Outcome: pass; `4 passed in 0.02s`.
- Command: `INTEGRATION_TESTS=1 uv run pytest tests/integration/test_multi_hop_walk.py -q -s`
- Outcome: pass; `1 passed in 21.07s`; visited Reflexion and lazy-ingested `arxiv:2208.14271`.
- Command: `uv run pytest tests/unit -q`
- Outcome: pass; `66 passed in 1.02s`.
- Command: `grep -RIE "from openai|from cohere|import openai|import cohere" src/retrieval || echo "ok"`
- Outcome: pass; printed `ok`.
- Command: `uv run ruff check src tests`
- Outcome: pass; `All checks passed!`.
- Command: `docker exec arxiv_agent_pg psql -U arxiv_agent -d arxiv_agent -c "SELECT paper_id FROM papers WHERE paper_id='arxiv:2208.14271';"`
- Outcome: pass; zero rows, confirming integration teardown removed the lazy-ingested paper.

## Validation commands
- Command: `uv run pytest tests/unit/test_frontier.py -q`
- Output (truncated if long): ```
....                                                                     [100%]
4 passed in 0.02s
```
- Command: `INTEGRATION_TESTS=1 uv run pytest tests/integration/test_multi_hop_walk.py -q -s`
- Output (truncated if long): ```
multi_hop_elapsed_s=20.43
visited=['arxiv:2210.03629', 'arxiv:2303.11366', 'arxiv:2208.14271']
newly_ingested=['arxiv:2208.14271']
.
1 passed in 21.07s
```
- Command: `grep -RIE "from openai|from cohere|import openai|import cohere" src/retrieval || echo "ok"`
- Output (truncated if long): ```
ok
```
- Command: `uv run ruff check src tests`
- Output (truncated if long): ```
All checks passed!
```

## Acceptance checklist
- [x] `Frontier` deterministically orders ties by insertion order.
- [x] Frontier deduplicates against visited and already-enqueued paper IDs.
- [x] Walker respects `max_depth` and `frontier_limit`.
- [x] Lazy ingestion path is exercised; `arxiv:2208.14271` was newly ingested in the integration run.
- [x] Wall-clock budget returns a partial result instead of raising; covered by unit test.
- [x] Provider grep returns `ok`.
- [x] Integration teardown removes newly ingested papers so re-runs are reproducible.

## Deviations from the plan
- The walker uses bidirectional local citation expansion, arXiv-only: outbound references via `citing_paper_id = current` plus local incoming citers via `cited_paper_id = current`.
- Why: the §11 outbound-only spec is incompatible with the observed citation reality and the required ReAct to Reflexion integration test. The live corpus has mixed `cited_paper_id` values (`1224 arxiv`, `163 s2`, `69 title`), and Reflexion cites ReAct rather than ReAct citing Reflexion.
- Bidirectional local expansion stays inside the local corpus for incoming edges, so it adds no new external API calls. Lazy ingestion still exercises the Phase 3 pipeline through outbound arXiv references.
- See `docs/phase_reports/PHASE_6_QUESTIONS.md` for the data evidence, options, recommendations, and owner-approved choices.

## Open questions for the planner
None.

## Ready for review
The planner can read these files and run those commands to verify.
