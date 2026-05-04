from __future__ import annotations

import inspect
import importlib
from typing import Any

from pydantic import BaseModel

from src.core.config import get_settings, require_secret
from src.llm.errors import LLMInvalidRequestError, error_for_status
from src.llm.retry import with_retry


class RerankedDoc(BaseModel):
    index: int
    score: float


class CohereRerankClient:
    provider = "cohere_native"

    def __init__(self, *, api_key: str | None = None, client: Any | None = None) -> None:
        settings = get_settings()
        resolved_key = api_key or require_secret(
            settings.cohere_api_key,
            env_var="COHERE_API_KEY",
            provider=self.provider,
        )
        self._client = client
        if self._client is None:
            cohere_module = importlib.import_module("cohere")
            self._client = cohere_module.AsyncClientV2(api_key=resolved_key)

    def _raise_for_status(self, status: int, body: str = "") -> None:
        if status >= 400:
            raise error_for_status(self.provider, status, body)

    @with_retry()
    async def rerank(
        self,
        query: str,
        documents: list[str],
        *,
        model: str,
        top_n: int,
    ) -> list[RerankedDoc]:
        indexed_documents = [(idx, doc) for idx, doc in enumerate(documents) if doc.strip()]
        if not indexed_documents:
            raise LLMInvalidRequestError("No non-empty documents to rerank", provider=self.provider)

        response = await self._client.rerank(
            model=model,
            query=query,
            documents=[doc for _, doc in indexed_documents],
            top_n=top_n,
        )
        results = _get_attr(response, "results")
        reranked: list[RerankedDoc] = []
        for result in results:
            filtered_index = int(_get_attr(result, "index"))
            original_index = indexed_documents[filtered_index][0]
            reranked.append(
                RerankedDoc(
                    index=original_index,
                    score=float(_get_attr(result, "relevance_score")),
                )
            )
        return reranked

    async def aclose(self) -> None:
        close = (
            getattr(self._client, "close", None)
            or getattr(self._client, "aclose", None)
            or getattr(self._client, "disconnect", None)
        )
        if close is None:
            return
        result = close()
        if inspect.isawaitable(result):
            await result


def _get_attr(obj: Any, name: str) -> Any:
    if isinstance(obj, dict):
        return obj[name]
    return getattr(obj, name)
