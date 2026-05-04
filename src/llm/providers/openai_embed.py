from __future__ import annotations

import inspect
import importlib
from typing import Any

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
            response = await self._client.embeddings.create(
                model=model,
                input=batch,
                encoding_format="float",
            )
            prompt_tokens += _prompt_tokens(response)
            for item in _get_attr(response, "data"):
                vector = list(_get_attr(item, "embedding"))
                if len(vector) != EMBEDDING_DIM:
                    raise LLMProviderError(
                        f"Expected embedding dimension {EMBEDDING_DIM}, got {len(vector)}",
                        provider=self.provider,
                    )
                vectors.append(vector)
        self.last_prompt_tokens = prompt_tokens
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


def _prompt_tokens(response: Any) -> int:
    usage = response.get("usage") if isinstance(response, dict) else getattr(response, "usage", None)
    if usage is None:
        return 0
    if isinstance(usage, dict):
        return int(usage.get("prompt_tokens") or usage.get("total_tokens") or 0)
    return int(getattr(usage, "prompt_tokens", None) or getattr(usage, "total_tokens", 0) or 0)
