from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Protocol

from src.core.types import ChatResponse, Message
from src.llm.errors import EmptyProviderResponseError
from src.llm.providers.cohere_rerank import CohereRerankClient, RerankedDoc
from src.llm.providers.deepseek import DeepSeekChatClient
from src.llm.providers.openai_embed import OpenAIEmbeddingClient
from src.llm.providers.openrouter import OpenRouterChatClient
from src.llm.registry import ProviderRoute, get_route
from src.llm import usage
from src.llm.retry import retry_transient_network


logger = logging.getLogger(__name__)


class ChatClient(Protocol):
    async def chat(
        self,
        messages: list[Message],
        *,
        model: str | None = None,
        tools: list | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        response_format: dict | None = None,
    ) -> ChatResponse: ...

    async def aclose(self) -> None: ...


class EmbeddingClient(Protocol):
    async def embed(
        self,
        texts: list[str],
        *,
        model: str | None = None,
    ) -> list[list[float]]: ...

    async def aclose(self) -> None: ...


class RerankClient(Protocol):
    async def rerank(
        self,
        query: str,
        documents: list[str],
        *,
        model: str | None = None,
        top_n: int,
    ) -> list[RerankedDoc]: ...

    async def aclose(self) -> None: ...


@dataclass
class _RoutedChatClient:
    role: str
    route: ProviderRoute
    provider_client: DeepSeekChatClient | OpenRouterChatClient

    async def chat(
        self,
        messages: list[Message],
        *,
        model: str | None = None,
        tools: list | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        response_format: dict | None = None,
    ) -> ChatResponse:
        started = time.perf_counter()
        response = await retry_transient_network(
            lambda: self.provider_client.chat(
                messages,
                model=model or self.route.vendor_model,
                tools=tools,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=response_format,
            )
        )
        duration_ms = (time.perf_counter() - started) * 1000
        _record_chat_usage(self.role, response)
        _log_llm_call(self.role, response, duration_ms=duration_ms)
        if _should_retry_empty_response(response):
            started = time.perf_counter()
            # Network retry wraps each provider call; empty-response retry remains above it.
            response = await retry_transient_network(
                lambda: self.provider_client.chat(
                    messages,
                    model=model or self.route.vendor_model,
                    tools=tools,
                    temperature=0.1,
                    max_tokens=max_tokens,
                    response_format=response_format,
                )
            )
            duration_ms = (time.perf_counter() - started) * 1000
            _record_chat_usage(self.role, response)
            _log_llm_call(self.role, response, duration_ms=duration_ms)
            if _should_retry_empty_response(response):
                raise EmptyProviderResponseError(
                    f"Provider returned empty content twice for role={self.role!r}, "
                    f"model={(model or self.route.vendor_model)!r}"
                )
        return response

    async def aclose(self) -> None:
        await self.provider_client.aclose()


@dataclass
class _RoutedEmbeddingClient:
    role: str
    route: ProviderRoute
    provider_client: OpenAIEmbeddingClient

    async def embed(
        self,
        texts: list[str],
        *,
        model: str | None = None,
    ) -> list[list[float]]:
        started = time.perf_counter()
        vectors = await self.provider_client.embed(texts, model=model or self.route.vendor_model)
        duration_ms = (time.perf_counter() - started) * 1000
        usage.record(self.role, prompt=getattr(self.provider_client, "last_prompt_tokens", 0))
        logger.info(
            "llm_call",
            extra={
                "role": self.role,
                "prompt_tokens": getattr(self.provider_client, "last_prompt_tokens", 0),
                "completion_tokens": 0,
                "finish_reason": "unknown",
                "duration_ms": duration_ms,
                "parse_status": "ok",
            },
        )
        return vectors

    async def aclose(self) -> None:
        await self.provider_client.aclose()


@dataclass
class _RoutedRerankClient:
    role: str
    route: ProviderRoute
    provider_client: CohereRerankClient

    async def rerank(
        self,
        query: str,
        documents: list[str],
        *,
        model: str | None = None,
        top_n: int,
    ) -> list[RerankedDoc]:
        started = time.perf_counter()
        results = await self.provider_client.rerank(
            query,
            documents,
            model=model or self.route.vendor_model,
            top_n=top_n,
        )
        duration_ms = (time.perf_counter() - started) * 1000
        usage.record(self.role)
        logger.info(
            "llm_call",
            extra={
                "role": self.role,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "finish_reason": "unknown",
                "duration_ms": duration_ms,
                "parse_status": "ok",
            },
        )
        return results

    async def aclose(self) -> None:
        await self.provider_client.aclose()


_provider_clients: dict[str, object] = {}
_routed_clients: dict[str, object] = {}


def get_chat_client(model_name: str) -> ChatClient:
    route = get_route(model_name, capability="chat")
    cached = _routed_clients.get(model_name)
    if cached is not None:
        return cached  # type: ignore[return-value]
    if route.provider == "deepseek_native":
        provider_client = _get_provider_client(route.provider, DeepSeekChatClient)
    elif route.provider == "openrouter":
        provider_client = _get_provider_client(route.provider, OpenRouterChatClient)
    else:
        raise KeyError(f"{model_name!r} is not a chat model")
    routed = _RoutedChatClient(role=model_name, route=route, provider_client=provider_client)
    _routed_clients[model_name] = routed
    return routed


def get_embedding_client(model_name: str = "embed-small") -> EmbeddingClient:
    route = get_route(model_name, capability="embedding")
    cached = _routed_clients.get(model_name)
    if cached is not None:
        return cached  # type: ignore[return-value]
    provider_client = _get_provider_client(route.provider, OpenAIEmbeddingClient)
    routed = _RoutedEmbeddingClient(role=model_name, route=route, provider_client=provider_client)
    _routed_clients[model_name] = routed
    return routed


def get_rerank_client(model_name: str = "rerank") -> RerankClient:
    route = get_route(model_name, capability="rerank")
    cached = _routed_clients.get(model_name)
    if cached is not None:
        return cached  # type: ignore[return-value]
    provider_client = _get_provider_client(route.provider, CohereRerankClient)
    routed = _RoutedRerankClient(role=model_name, route=route, provider_client=provider_client)
    _routed_clients[model_name] = routed
    return routed


async def close_llm_clients() -> None:
    seen: set[int] = set()
    for client in list(_provider_clients.values()):
        if id(client) in seen:
            continue
        seen.add(id(client))
        aclose = getattr(client, "aclose", None)
        if aclose is not None:
            await aclose()
    _provider_clients.clear()
    _routed_clients.clear()


def _get_provider_client(provider: str, factory):
    cached = _provider_clients.get(provider)
    if cached is not None:
        return cached
    client = factory()
    _provider_clients[provider] = client
    return client


def _record_chat_usage(role: str, response: ChatResponse) -> None:
    usage.record(
        role,
        prompt=response.usage.prompt_tokens if response.usage else 0,
        completion=response.usage.completion_tokens if response.usage else 0,
    )


def _log_llm_call(role: str, response: ChatResponse, *, duration_ms: float) -> None:
    logger.info(
        "llm_call",
        extra={
            "role": role,
            "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
            "completion_tokens": response.usage.completion_tokens if response.usage else 0,
            "finish_reason": response.finish_reason or "unknown",
            "duration_ms": duration_ms,
            "parse_status": "ok",
        },
    )


def log_parse_status(role: str, status: str) -> None:
    """Log best-effort parse status for JSON-wrapped LLM calls."""
    logger.info("llm_parse", extra={"role": role, "status": status})


def _should_retry_empty_response(response: ChatResponse) -> bool:
    return (response.content or "").strip() == "" and response.finish_reason != "length"
