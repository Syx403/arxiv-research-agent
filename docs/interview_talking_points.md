# Interview Talking Points

## Entries

- Provider abstraction: business code calls role aliases like `main` and `fast` through `src/llm/client.py`, never vendor SDKs directly.
- Retry policy: `src/llm/retry.py` retries transient provider failures with jitter while refusing deterministic auth, invalid request, and context-length errors.
- Secret masking: `src/core/logging.py` redacts bearer tokens, API keys, and key-like values at the logging-filter layer.
- Embedding dimension validation: `src/llm/providers/openai_embed.py` rejects vectors that do not match the 1536-dimensional `text-embedding-3-small` contract.
- Client lifecycle: `src/llm/client.py` caches provider clients and closes all underlying async clients through `close_llm_clients()`.
