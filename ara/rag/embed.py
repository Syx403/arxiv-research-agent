"""Content-addressed embeddings (DESIGN §5): a vector is keyed by sha256(model + text), so the same
text is never embedded twice, whichever paper or query it comes from."""

from collections.abc import Sequence
from hashlib import sha256

import numpy as np
from numpy.typing import NDArray

from ara.db.pool import Pool
from ara.llm.gateway import Gateway
from ara.llm.ledger import Scope
from ara.llm.stages import EMBEDDING_MODEL
from ara.tokens import count_tokens

BATCH_TOKENS = 200_000  # the embeddings API takes at most 300k tokens per request
BATCH_ITEMS = 2_048  # and at most 2,048 inputs

type Vector = NDArray[np.float32]


def cache_key(text: str) -> bytes:
    return sha256(f"{EMBEDDING_MODEL}\n{text}".encode()).digest()


async def embed(
    texts: Sequence[str], *, pool: Pool, gateway: Gateway, scope: Scope
) -> list[Vector]:
    """Vectors for `texts`, in order. Only texts missing from the cache are sent, in batches."""
    vectors = await _cached(pool, {cache_key(text) for text in texts})
    missing = list(
        {cache_key(text): text for text in texts if cache_key(text) not in vectors}.items()
    )
    for batch in _batches(missing):
        embedded = await gateway.embed([text for _, text in batch], scope=scope)
        fresh = {
            key: np.asarray(vector, dtype=np.float32)
            for (key, _), vector in zip(batch, embedded, strict=True)
        }
        async with pool.connection() as conn:
            await conn.cursor().executemany(
                "INSERT INTO embedding_cache (key, model, embedding) VALUES (%s, %s, %s)"
                " ON CONFLICT (key) DO NOTHING",
                [(key, EMBEDDING_MODEL, vector) for key, vector in fresh.items()],
            )
        vectors |= fresh
    return [vectors[cache_key(text)] for text in texts]


async def _cached(pool: Pool, keys: set[bytes]) -> dict[bytes, Vector]:
    async with pool.connection() as conn:
        cursor = await conn.execute(
            "SELECT key, embedding FROM embedding_cache WHERE key = ANY(%(keys)s)",
            {"keys": list(keys)},
        )
        return {bytes(row["key"]): row["embedding"] for row in await cursor.fetchall()}


def _batches(items: list[tuple[bytes, str]]) -> list[list[tuple[bytes, str]]]:
    batches: list[list[tuple[bytes, str]]] = []
    size = 0
    for item in items:
        tokens = count_tokens(item[1])
        if not batches or size + tokens > BATCH_TOKENS or len(batches[-1]) == BATCH_ITEMS:
            batches.append([])
            size = 0
        batches[-1].append(item)
        size += tokens
    return batches
