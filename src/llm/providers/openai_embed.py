from __future__ import annotations

import inspect
from typing import Any

from openai import AsyncOpenAI

from src.core.config import get_settings, require_secret
from src.core.types import EMBEDDING_DIM
from src.llm.errors import LLMProviderError, error_for_status
from src.llm.retry import with_retry


class OpenAIEmbeddingClient:
    provider = "openai_native"

    def __init__(self, *, api_key: str | None = None, client: Any | None = None) -> None:
        settings = get_settings()
        resolved_key = api_key or require_secret(
            settings.openai_api_key,
            env_var="OPENAI_API_KEY",
            provider=self.provider,
        )
        self._client = client or AsyncOpenAI(api_key=resolved_key)

    def _raise_for_status(self, status: int, body: str = "") -> None:
        if status >= 400:
            raise error_for_status(self.provider, status, body)

    @with_retry()
    async def embed(
        self,
        texts: list[str],
        *,
        model: str = "text-embedding-3-small",
    ) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), 256):
            batch = texts[start : start + 256]
            if not batch:
                continue
            response = await self._client.embeddings.create(
                model=model,
                input=batch,
                encoding_format="float",
            )
            for item in _get_attr(response, "data"):
                vector = list(_get_attr(item, "embedding"))
                if len(vector) != EMBEDDING_DIM:
                    raise LLMProviderError(
                        f"Expected embedding dimension {EMBEDDING_DIM}, got {len(vector)}",
                        provider=self.provider,
                    )
                vectors.append(vector)
        return vectors

    async def aclose(self) -> None:
        close = getattr(self._client, "close", None) or getattr(self._client, "aclose", None)
        if close is None:
            return
        result = close()
        if inspect.isawaitable(result):
            await result


def _get_attr(obj: Any, name: str) -> Any:
    if isinstance(obj, dict):
        return obj[name]
    return getattr(obj, name)
