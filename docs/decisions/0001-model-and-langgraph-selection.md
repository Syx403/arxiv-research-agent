# ADR 0001: Model and LangGraph Selection

**Date checked**: 2026-05-03
**Status**: accepted

## Decision

Use a DeepSeek-native default chat tier:

| Logical alias | Provider | Vendor model | Purpose |
|---|---|---|---|
| `main` | DeepSeek native | `deepseek-v4-pro` | Quality-sensitive reasoning and synthesis |
| `fast` | DeepSeek native | `deepseek-v4-flash` | Cheap routing, JSON repair, HyDE, batch judging |
| `embed-small` | OpenAI native | `text-embedding-3-small` | 1536-dimensional embeddings |
| `rerank` | Cohere native | `rerank-v4.0-pro` | Highest-quality rerank default |

OpenRouter remains implemented but is not in the default registry. The reserved fallback candidate is `anthropic/claude-haiku-4.5`; enabling it later requires adding the commented `fallback-haiku` registry line and documenting the switch in a new ADR.

For LangGraph, choose **Path B**:

- `langgraph>=1.0.10,<2`
- `langgraph-checkpoint>=4.0.1,<5`
- `langgraph-checkpoint-postgres>=3.0.5,<4`
- `langchain-core>=1.3,<2`
- `langsmith>=0.3.45,<1`

## Evidence

- DeepSeek official docs list `deepseek-v4-flash` and `deepseek-v4-pro` for the OpenAI-compatible API and mark `deepseek-chat` / `deepseek-reasoner` as deprecated on 2026-07-24: https://api-docs.deepseek.com/
- DeepSeek model-list docs show the available model IDs `deepseek-v4-flash` and `deepseek-v4-pro`: https://api-docs.deepseek.com/api/list-models/
- OpenAI embedding docs state `text-embedding-3-small` defaults to 1536 dimensions: https://platform.openai.com/docs/guides/embeddings
- Cohere rerank docs list `rerank-v4.0-pro`, `rerank-v4.0-fast`, and `rerank-v3.5`: https://docs.cohere.com/docs/rerank
- OpenRouter's model list contains `anthropic/claude-haiku-4.5`: https://openrouter.ai/api/v1/models
- GitHub advisory GHSA-g48c-2wqr-h844 / CVE-2026-28277 affects `langgraph<=1.0.9` and patches in `1.0.10`: https://github.com/advisories/GHSA-g48c-2wqr-h844
- PyPI shows `langgraph` latest stable 1.1.x with `langchain-core>=1.3.0,<2`: https://pypi.org/project/langgraph/
- PyPI shows `langgraph-checkpoint-postgres` latest stable 3.0.5 and documents `AsyncPostgresSaver` import path plus `autocommit=True` and `row_factory=dict_row` pool requirements: https://pypi.org/project/langgraph-checkpoint-postgres/
- LangGraph Python docs still document `Annotated[list, add_messages]` and `MessagesState.messages: Annotated[list[AnyMessage], add_messages]`: https://reference.langchain.com/python/langgraph/graph/message/add_messages

## Security Notes

The previous `langgraph>=0.2.50,<0.3` line is not acceptable after the 2026 msgpack checkpoint deserialization advisory. Upgrading to `langgraph>=1.0.10` clears that advisory. Pinning `langgraph-checkpoint>=4.0.1,<5` also avoids the older `langgraph-checkpoint<3.0` JSON-mode deserialization advisory and the later BaseCache deserialization advisory affecting earlier checkpoint releases.

The checkpoint store remains local Postgres only. The application must never accept checkpoint bytes from untrusted sources, and write access to the checkpoint tables is treated as integrity-sensitive.

## Consequences

All later graph nodes should request chat clients by role alias (`main` or `fast`) rather than by vendor/model name. OpenRouter can be enabled later with a one-line registry edit if DeepSeek-native strict JSON or tool behavior proves unreliable.
