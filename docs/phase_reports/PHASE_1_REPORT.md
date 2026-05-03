# Phase 1 Report

**Phase title**: Provider Adapter Layer
**Status**: complete
**Commit**: `HEAD (see git log --oneline)`
**Owner handoff pending**: none

## Summary
Phase 1 implemented the provider adapter layer with role-based chat aliases `main` and `fast`, both routed to DeepSeek native models by default. OpenRouter is implemented and unit-tested, but is not in the default registry; the fallback line remains commented out as required by v1.6. ADR 0001 records the verified model IDs, the OpenRouter fallback candidate, and the LangGraph Path B upgrade away from the vulnerable 0.2.x line. Handoff H-1 is marked confirmed in `docs/STATUS.md`.

## Files created
- `docs/decisions/0001-model-and-langgraph-selection.md` - model and LangGraph version/security ADR.
- `src/core/__init__.py` - core package marker.
- `src/core/config.py` - pydantic-settings loader and provider secret validation.
- `src/core/logging.py` - secret redaction helpers and logging filter.
- `src/core/types.py` - shared Pydantic contracts and code constants.
- `src/llm/__init__.py` - LLM package marker.
- `src/llm/client.py` - public routed client API and lifecycle cleanup.
- `src/llm/errors.py` - provider-aware error hierarchy and HTTP status mapping.
- `src/llm/registry.py` - role-based model registry.
- `src/llm/retry.py` - tenacity retry decorator.
- `src/llm/providers/__init__.py` - provider package marker.
- `src/llm/providers/openrouter.py` - OpenRouter chat client.
- `src/llm/providers/deepseek.py` - DeepSeek native chat client.
- `src/llm/providers/openai_embed.py` - OpenAI embedding client with dimension validation.
- `src/llm/providers/cohere_rerank.py` - Cohere rerank client.
- `tests/unit/*.py` - Phase 1 unit tests for retry, registry, logging, status mapping, embedding dimension, config, and lifecycle.
- `tests/integration/test_provider_smoke.py` - skipped-by-default cost-minimized provider smoke test.
- `tests/integration/test_openrouter_smoke.py` - skipped-by-default OpenRouter opt-in smoke test.
- `uv.lock` - resolved dependency lock file.

## Files modified
- `docs/STATUS.md` - H-1 confirmed and Phase 1 marked complete.
- `docs/interview_talking_points.md` - Phase 1 talking points added.
- `pyproject.toml` - LangGraph Path B dependency line and compatible `uvicorn` range.
- `docs/EXECUTION_PLAN.md` - v1.6 plan revision was present at phase start and included in this phase commit; Codex did not edit plan content.

## Tests run
- Command: `uv run pytest tests/unit -q`
- Outcome: pass; `36 passed in 0.45s`.
- Command: `uv run ruff check src tests`
- Outcome: pass; `All checks passed!`.
- Command: `uv run python -c "from src.llm.client import get_chat_client, get_embedding_client, get_rerank_client; print('imports ok')"`
- Outcome: pass; `imports ok`.
- Command: `uv run python -c "from src.llm.client import get_chat_client, close_llm_clients; import asyncio; get_chat_client('main'); get_chat_client('fast'); asyncio.run(close_llm_clients()); print('chat clients ok')"`
- Outcome: pass; `chat clients ok`. This did not call provider APIs.

## Validation commands
- Command: `uv run pytest tests/unit -q`
- Output (truncated if long): ```
36 passed in 0.45s
```
- Command: `grep -RIE "from openai|from cohere|import openai|import cohere" src --exclude-dir=llm || echo "ok"`
- Output (truncated if long): ```
ok
```
- Command: `uv run python -c "from src.llm.client import get_chat_client, get_embedding_client, get_rerank_client; print('imports ok')"`
- Output (truncated if long): ```
imports ok
```
- Command: `test -f docs/decisions/0001-model-and-langgraph-selection.md && echo "model+langgraph ADR present"`
- Output (truncated if long): ```
model+langgraph ADR present
```

## Acceptance checklist
- [x] ADR 0001 exists and records selected vendor IDs, LangGraph Path B, source URLs, and advisory references.
- [x] `src/llm/registry.py` default keys are only `main`, `fast`, `embed-small`, and `rerank`.
- [x] OpenRouter fallback is present only as the commented `fallback-haiku` registry line.
- [x] `get_chat_client("main")` and `get_chat_client("fast")` construct successfully with settings-loaded keys and no API calls.
- [x] Unit tests pass.
- [x] Vendor SDK imports are confined to `src/llm/providers/`.
- [x] Logging redacts synthetic API key and bearer-token strings.
- [x] Missing provider secrets raise clear `LLMAuthError`s naming the env var.
- [x] Integration smoke tests are skipped by default and do not call `main`; the live smoke path uses `fast` with `max_tokens=1`.
- [x] `docs/STATUS.md` row 0 reads `H-1 confirmed 2026-05-03`.
- [x] `docs/STATUS.md` row 1 marks Phase 1 complete with `HEAD (see git log)`.

## Deviations from the plan
- Path B was selected for LangGraph because the advisory affects the old 0.2.x line. `pyproject.toml` now uses `langgraph>=1.0.10,<2` and compatible checkpoint/langchain/langsmith ranges.
- `uvicorn` was narrowed from `>=0.32,<1` to `>=0.25,<0.26` because Chainlit 1.3.x depends on `uvicorn>=0.25,<0.26`; the original ranges were unsatisfiable.
- Live integration smoke tests were not run to avoid provider spend during this phase execution. The skipped tests are present and cost-minimized for owner/planner opt-in.

## Open questions for the planner
None.

## Ready for review
The planner can read these files and run those commands to verify.

## Post-review fixes
- Follow-up commit: `HEAD (see git log)`.
- Changed retry backoff from test-speed production values to production-safe constants: `RETRY_WAIT_INITIAL_SECONDS = 1.0` and `RETRY_WAIT_MAX_SECONDS = 30.0` in `src/core/types.py`.
- Updated `src/llm/retry.py` to use those constants in `wait_exponential_jitter`.
- Updated `tests/unit/test_retry.py` to monkeypatch retry waits to `0.01` and `0.05` seconds so unit tests still complete quickly.
- Verification: `uv run pytest tests/unit -q` passes.
