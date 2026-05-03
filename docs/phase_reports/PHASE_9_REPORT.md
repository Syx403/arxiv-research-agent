# Phase 9 Report

**Phase title**: LangGraph Orchestration
**Status**: complete
**Commit**: `HEAD (see git log)`
**Owner handoff pending**: none

## Summary
Phase 9 wired the existing provider, retrieval, multi-hop, episodic, and semantic-memory layers into a checkpointed LangGraph graph. `thread_id` is the session id; `run()` creates the session, invokes the graph with `AsyncPostgresSaver`, and `reflect_node` appends the final assistant answer plus citation metadata before promoting episodic memory into semantic memory. The end-to-end test produced a grounded ReAct answer with citations, verification passed, reflection created concepts, and the resume test proved cross-process checkpoint reuse.

## Files created
- `src/graph/__init__.py` - graph package marker.
- `src/graph/state.py` - `AgentState` typed dict and explicit reducers.
- `src/graph/edges.py` - budget-aware conditional edge routing.
- `src/graph/builder.py` - graph compilation, Postgres checkpointer setup, `run()` and `shutdown()`.
- `src/graph/nodes/__init__.py` - graph nodes package marker.
- `src/graph/nodes/decompose.py` - decomposition node.
- `src/graph/nodes/retrieve.py` - full retrieval/self-RAG evidence pipeline.
- `src/graph/nodes/multi_hop.py` - citation-walker node.
- `src/graph/nodes/synthesize.py` - answer synthesis and citation parsing.
- `src/graph/nodes/self_rag.py` - citation verification node.
- `src/graph/nodes/reflect.py` - answer persistence plus reflection node.
- `tests/unit/test_edge_budgets.py` - pure edge budget tests.
- `tests/integration/test_router_wired.py` - router-first graph retriever gating test.
- `tests/integration/test_graph_e2e.py` - live end-to-end graph test.
- `tests/integration/test_resume.py` - cross-process checkpoint resume test.

## Files modified
- `src/core/types.py` - added Phase 9 constants and `SubQResult`.
- `src/retrieval/self_rag/relevance_judge.py` - added repair/fallback for malformed strict-JSON provider responses.
- `src/retrieval/self_rag/sufficiency_check.py` - added repair/fallback for malformed strict-JSON provider responses.
- `src/retrieval/self_rag/verifier.py` - added repair/fallback for malformed strict-JSON provider responses.
- `src/memory/semantic/extractor.py` - added one bounded retry for empty-but-valid concept extraction.
- `tests/unit/test_relevance_judge.py` - added repair/fallback coverage.
- `tests/unit/test_sufficiency_check.py` - added fallback coverage.
- `tests/unit/test_verifier_parser.py` - added verifier fallback coverage.
- `tests/unit/test_extractor_empty_retry.py` - added empty extraction retry coverage.
- `docs/STATUS.md` - Phase 9 marked complete.
- `docs/interview_talking_points.md` - Phase 9 talking points appended.

## Tests run
- Command: `uv run pytest tests/unit/test_edge_budgets.py -q`
- Outcome: pass; `2 passed in 0.20s`.
- Command: `INTEGRATION_TESTS=1 uv run pytest tests/integration/test_router_wired.py tests/integration/test_resume.py -q -s`
- Outcome: pass; `3 passed in 1.81s`.
- Command: `INTEGRATION_TESTS=1 uv run pytest tests/integration/test_graph_e2e.py -q -s`
- Outcome: pass; `1 passed in 190.21s`.
- Command: `uv run pytest tests/unit -q`
- Outcome: pass; `76 passed in 1.29s`.
- Command: `grep -RIE "from openai|from cohere|import openai|import cohere" src/graph src/memory src/retrieval || echo "ok"`
- Outcome: pass; printed `ok`.
- Command: `uv run ruff check src tests`
- Outcome: pass; `All checks passed!`.
- Command: `docker exec arxiv_agent_pg psql -U arxiv_agent -d arxiv_agent -c "SELECT count(*) FROM concepts;"`
- Outcome: pass; returned `0` after e2e cleanup.
- Command: `docker exec arxiv_agent_pg psql -U arxiv_agent -d arxiv_agent -c "SELECT count(*) FROM checkpoints WHERE thread_id LIKE 'e2e-%' OR thread_id LIKE 't-resume-%';"`
- Outcome: pass; returned `0` after test cleanup.

## Validation commands
- Command: `uv run pytest tests/unit/test_edge_budgets.py -q`
- Output (truncated if long): ```
..                                                                       [100%]
2 passed in 0.20s
```
- Command: `INTEGRATION_TESTS=1 uv run pytest tests/integration/test_router_wired.py tests/integration/test_resume.py -q -s`
- Output (truncated if long): ```
...
3 passed in 1.81s
```
- Command: `INTEGRATION_TESTS=1 uv run pytest tests/integration/test_graph_e2e.py -q -s`
- Output (truncated if long): ```
answer= ReAct is a general paradigm that prompts large language models (LLMs) to generate both verbal reasoning traces (thoughts) and task-specific actions in an interleaved manner, creating a synergy between reasoning and acting [arxiv:2210.03629#1501][arxiv:2210.03629#1496]. It augments the model's action space with a language space, where thoughts compose useful information from the context to guide future actions without affecting the external environment, while actions enable interaction with external sources like Wikipedia to gather new knowledge [arxiv:2210.03629#1502]. The approach uses few-shot prompting with human-annotated trajectories of thoughts, actions, and observations, and has been applied to diverse tasks including multi-hop question answering, fact verification, and interactive decision making [arxiv:2210.03629#1504][arxiv:2210.03629#1496]. ReAct consistently outperforms reasoning- or acting-only baselines, reduces hallucination, and improves interpretability and human controllability over the model's decision process [arxiv:2210.03629#1503][arxiv:2210.03629#1496].
citations= [{'paper_id': 'arxiv:2210.03629', 'chunk_id': 1501, 'quote': None, 'claim_span': (222, 245)}, {'paper_id': 'arxiv:2210.03629', 'chunk_id': 1496, 'quote': None, 'claim_span': (245, 268)}, {'paper_id': 'arxiv:2210.03629', 'chunk_id': 1502, 'quote': None, 'claim_span': (551, 574)}, {'paper_id': 'arxiv:2210.03629', 'chunk_id': 1504, 'quote': None, 'claim_span': (816, 839)}, {'paper_id': 'arxiv:2210.03629', 'chunk_id': 1496, 'quote': None, 'claim_span': (839, 862)}, {'paper_id': 'arxiv:2210.03629', 'chunk_id': 1503, 'quote': None, 'claim_span': (1045, 1068)}, {'paper_id': 'arxiv:2210.03629', 'chunk_id': 1496, 'quote': None, 'claim_span': (1068, 1091)}]
verification= passed=True
reflection= {'concepts_created': 2, 'concepts_reused': 0, 'relations_created': 1, 'relations_reused': 0}
.
1 passed in 190.21s (0:03:10)
```

## Acceptance checklist
- [x] `run("What is ReAct?", thread_id=...)` produced a grounded answer with multiple `[paper_id#chunk_id]` citations and `shutdown()` cleaned clients/pools.
- [x] Resume test passed across a subprocess using the same `thread_id` and an empty question.
- [x] Edge budget tests prove hard stops route to `synthesize` / `reflect` when counters are exhausted.
- [x] Router-first ordering is covered; graph retriever is skipped when `use_concept_graph=False`.
- [x] Provider SDK grep across graph, memory, and retrieval returns `ok`.
- [x] LangSmith trace visibility was not manually observed by Codex; this remains an H-3 owner dashboard check.

## Deviations from the plan
- Paper hint biasing is implemented as the binding 1.1x score multiplier after reranking and before relevance judging. This makes the vague §14.2 "post-filter ranking boost" deterministic.
- Live strict-JSON calls from DeepSeek occasionally returned truncated or empty JSON during e2e. The relevance judge, sufficiency checker, citation verifier, and concept extractor now have bounded repair/fallback paths so provider formatting instability does not crash graph execution.

## Open questions for the planner
None.

## Ready for review
The planner can read these files and run the validation commands above to verify.
