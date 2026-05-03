from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from src.core.types import ChatResponse, Message
from src.llm.providers.cohere_rerank import CohereRerankClient, RerankedDoc
from src.llm.providers.deepseek import DeepSeekChatClient
from src.llm.providers.openai_embed import OpenAIEmbeddingClient
from src.llm.providers.openrouter import OpenRouterChatClient
from src.llm.registry import ProviderRoute, get_route


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
        return await self.provider_client.chat(
            messages,
            model=model or self.route.vendor_model,
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
        )

    async def aclose(self) -> None:
        await self.provider_client.aclose()


@dataclass
class _RoutedEmbeddingClient:
    route: ProviderRoute
    provider_client: OpenAIEmbeddingClient

    async def embed(
        self,
        texts: list[str],
        *,
        model: str | None = None,
    ) -> list[list[float]]:
        return await self.provider_client.embed(texts, model=model or self.route.vendor_model)

    async def aclose(self) -> None:
        await self.provider_client.aclose()


@dataclass
class _RoutedRerankClient:
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
        return await self.provider_client.rerank(
            query,
            documents,
            model=model or self.route.vendor_model,
            top_n=top_n,
        )

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
    routed = _RoutedChatClient(route=route, provider_client=provider_client)
    _routed_clients[model_name] = routed
    return routed


def get_embedding_client(model_name: str = "embed-small") -> EmbeddingClient:
    route = get_route(model_name, capability="embedding")
    cached = _routed_clients.get(model_name)
    if cached is not None:
        return cached  # type: ignore[return-value]
    provider_client = _get_provider_client(route.provider, OpenAIEmbeddingClient)
    routed = _RoutedEmbeddingClient(route=route, provider_client=provider_client)
    _routed_clients[model_name] = routed
    return routed


def get_rerank_client(model_name: str = "rerank") -> RerankClient:
    route = get_route(model_name, capability="rerank")
    cached = _routed_clients.get(model_name)
    if cached is not None:
        return cached  # type: ignore[return-value]
    provider_client = _get_provider_client(route.provider, CohereRerankClient)
    routed = _RoutedRerankClient(route=route, provider_client=provider_client)
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
