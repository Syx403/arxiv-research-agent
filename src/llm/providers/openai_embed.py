from __future__ import annotations

import inspect
import importlib
from typing import Any

from src.core.config import get_settings, require_secret
from src.core.types import EMBEDDING_DIM
from src.llm.errors import EmptyProviderResponseError, LLMProviderError, error_for_status
from src.llm.retry import retry_transient_network, with_retry


class OpenAIEmbeddingClient:
    provider = "openai_native"

    def __init__(self, *, api_key: str | None = None, client: Any | None = None) -> None:
        settings = get_settings()
        resolved_key = api_key or require_secret(
            settings.openai_api_key,
            env_var="OPENAI_API_KEY",
            provider=self.provider,
        )
        self._client = client
        if self._client is None:
            openai_module = importlib.import_module("openai")
            self._client = openai_module.AsyncOpenAI(api_key=resolved_key)
        self.last_prompt_tokens = 0

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
        prompt_tokens = 0
        for start in range(0, len(texts), 256):
            batch = texts[start : start + 256]
            if not batch:
                continue
            response = await retry_transient_network(lambda: self._create_embedding_batch(model=model, batch=batch))
            prompt_tokens += _prompt_tokens(response)
            vectors.extend(_vectors_from_response(response, provider=self.provider))
        if len(vectors) != len(texts):
            raise EmptyProviderResponseError(
                f"OpenAI embedding response count mismatch: expected {len(texts)}, got {len(vectors)}"
            )
        self.last_prompt_tokens = prompt_tokens
        return vectors

    async def aclose(self) -> None:
        close = getattr(self._client, "close", None) or getattr(self._client, "aclose", None)
        if close is None:
            return
        result = close()
        if inspect.isawaitable(result):
            await result

    async def _create_embedding_batch(self, *, model: str, batch: list[str]):
        response = await self._client.embeddings.create(
            model=model,
            input=batch,
            encoding_format="float",
        )
        data = list(_get_attr(response, "data"))
        if len(data) != len(batch):
            raise EmptyProviderResponseError(
                f"OpenAI embedding batch count mismatch: expected {len(batch)}, got {len(data)}"
            )
        return response


def _get_attr(obj: Any, name: str) -> Any:
    if isinstance(obj, dict):
        return obj[name]
    return getattr(obj, name)


def _prompt_tokens(response: Any) -> int:
    usage = response.get("usage") if isinstance(response, dict) else getattr(response, "usage", None)
    if usage is None:
        return 0
    if isinstance(usage, dict):
        return int(usage.get("prompt_tokens") or usage.get("total_tokens") or 0)
    return int(getattr(usage, "prompt_tokens", None) or getattr(usage, "total_tokens", 0) or 0)


def _vectors_from_response(response: Any, *, provider: str) -> list[list[float]]:
    vectors: list[list[float]] = []
    for item in _get_attr(response, "data"):
        vector = list(_get_attr(item, "embedding"))
        if not vector:
            raise EmptyProviderResponseError("OpenAI embedding response contained an empty vector")
        if len(vector) != EMBEDDING_DIM:
            raise LLMProviderError(
                f"Expected embedding dimension {EMBEDDING_DIM}, got {len(vector)}",
                provider=provider,
            )
        vectors.append(vector)
    return vectors
