from __future__ import annotations

import inspect
import importlib
from typing import Any

from pydantic import BaseModel

from src.core.config import get_settings, require_secret
from src.llm.errors import EmptyProviderResponseError, LLMInvalidRequestError, error_for_status
from src.llm.retry import retry_transient_network, with_retry
from src.llm.budget import admit_request


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

        response = await retry_transient_network(
            lambda: self._rerank_once(
                model=model,
                query=query,
                documents=[doc for _, doc in indexed_documents],
                top_n=top_n,
            )
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

    async def _rerank_once(self, *, model: str, query: str, documents: list[str], top_n: int):
        reservation = await admit_request(self.provider, model, {"query": query, "documents": documents})
        try:
            response = await self._client.rerank(
                model=model, query=query, documents=documents, top_n=top_n,
                request_options={"max_retries": 0},
            )
            if reservation:
                meta = response.get("meta", {}) if isinstance(response, dict) else getattr(response, "meta", None)
                units = meta.get("billed_units", {}) if isinstance(meta, dict) else getattr(meta, "billed_units", None)
                count = units.get("search_units", 0) if isinstance(units, dict) else getattr(units, "search_units", 0)
                reservation.settle({"search_units": count or 0})
        finally:
            if reservation:
                reservation.uncertain()
        results = _get_attr(response, "results")
        if not results:
            raise EmptyProviderResponseError("Cohere rerank returned no results for non-empty documents")
        return response


def _get_attr(obj: Any, name: str) -> Any:
    if isinstance(obj, dict):
        return obj[name]
    return getattr(obj, name)
