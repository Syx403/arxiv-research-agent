from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Protocol

from src.core.types import ChatResponse, EMPTY_RESPONSE_RETRY_TEMPERATURE_FLOOR, Message
from src.llm.errors import EmptyProviderResponseError
from src.llm.providers.cohere_rerank import CohereRerankClient, RerankedDoc
from src.llm.providers.deepseek import DeepSeekChatClient
from src.llm.providers.openai_embed import OpenAIEmbeddingClient
from src.llm.providers.openrouter import OpenRouterChatClient
from src.llm.registry import ProviderRoute, get_route
from src.llm import usage
from src.llm.stages import resolve, use_stage, current_settings
from src.core.trace import record_event
from src.llm.retry import reset_retry_role, retry_transient_network, set_retry_role, single_attempt_enabled


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
        thinking: bool | None = None,
        stage: str | None = None,
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
        self, messages, *, model=None, tools=None, temperature=0.0, max_tokens=None,
        response_format=None, thinking=None, stage=None,
    ) -> ChatResponse:
        role, route, thinking = resolve(stage, self.role, self.route, thinking)
        if current_settings().profile != "legacy" and model not in {None, route.vendor_model}:
            raise ValueError("A model override cannot change a frozen stage profile")
        with use_stage(stage):
            record_event("stage_call", stage=stage, role=role, model=route.vendor_model,
                         effort=route.reasoning_effort if thinking is not False else None,
                         thinking=thinking is not False, visible_output_budget=max_tokens,
                         headroom_tokens=route.reasoning_headroom_tokens,
                         total_output_cap=current_settings().total_output_cap)
            scoped = _RoutedChatClient(role, route, self.provider_client)
            return await scoped._chat(messages, model=model, tools=tools, temperature=temperature,
                                      max_tokens=max_tokens, response_format=response_format,
                                      thinking=thinking)

    async def _chat(
        self,
        messages: list[Message],
        *,
        model: str | None = None,
        tools: list | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        response_format: dict | None = None,
        thinking: bool | None = None,
    ) -> ChatResponse:
        token = set_retry_role(self.role)
        try:
            attempt_messages = messages
            attempt_temperature = temperature
            response: ChatResponse | None = None
            profile_kwargs = {}
            if thinking is False and self.route.provider == "deepseek_native":
                profile_kwargs["thinking"] = False
            elif self.route.reasoning_effort:
                profile_kwargs["reasoning_effort"] = self.route.reasoning_effort
                # Provider max_tokens covers reasoning AND the visible answer.
                max_tokens = (max_tokens or 2000) + self.route.reasoning_headroom_tokens
            settings = current_settings()
            semantic = settings.profile != "legacy" and thinking is not False
            if semantic:
                max_tokens = max(max_tokens or 2000, 4096)
            if settings.total_output_cap is not None:
                # A controlled ablation must give Low and High the SAME total
                # output capacity, not merely clip the already smaller Low cap.
                max_tokens = settings.total_output_cap
            attempts = 1 if single_attempt_enabled() else 3
            for attempt in range(1, attempts + 1):
                for capacity_attempt in range(2):
                    started = time.perf_counter()
                    response = await retry_transient_network(
                        lambda: self.provider_client.chat(
                            attempt_messages,
                            model=model or self.route.vendor_model,
                            tools=tools,
                            temperature=attempt_temperature,
                            max_tokens=max_tokens,
                            response_format=response_format,
                            **profile_kwargs,
                        )
                    )
                    duration_ms = (time.perf_counter() - started) * 1000
                    _record_chat_usage(self.role, response)
                    _log_llm_call(self.role, response, duration_ms=duration_ms)
                    if not (semantic and not capacity_attempt and response.finish_reason == "length"
                            and settings.total_output_cap is None and max_tokens < 16384):
                        break
                    expanded = min(max_tokens * 2, 16384)
                    record_event("output_capacity_retry", previous_total=max_tokens,
                                 next_total=expanded, effort=self.route.reasoning_effort,
                                 reason="truncated_completion", unchanged_prompt=True)
                    max_tokens = expanded
                if not _should_retry_empty_response(response):
                    return response
                usage.record_empty_retry(self.role)
                if attempt == 1:
                    attempt_messages = messages
                    attempt_temperature = temperature
                elif attempt == 2:
                    attempt_messages = [
                        *messages,
                        Message(
                            role="system",
                            content="Your previous response was empty. Provide a non-empty response now using the supplied evidence.",
                        ),
                    ]
                    attempt_temperature = max(temperature, EMPTY_RESPONSE_RETRY_TEMPERATURE_FLOOR)
            raise EmptyProviderResponseError(
                f"Provider returned empty content after {attempts} attempt(s) for role={self.role!r}, "
                f"model={(model or self.route.vendor_model)!r}"
            )
        finally:
            reset_retry_role(token)

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
        token = set_retry_role(self.role)
        try:
            vectors = await self.provider_client.embed(texts, model=model or self.route.vendor_model)
        finally:
            reset_retry_role(token)
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
        token = set_retry_role(self.role)
        try:
            results = await self.provider_client.rerank(
                query,
                documents,
                model=model or self.route.vendor_model,
                top_n=top_n,
            )
        finally:
            reset_retry_role(token)
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
_routed_clients: dict[str | tuple[str, ProviderRoute], object] = {}


def get_chat_client(model_name: str) -> ChatClient:
    route = get_route(model_name, capability="chat")
    key = (model_name, route)
    cached = _routed_clients.get(key)
    if cached is not None:
        return cached  # type: ignore[return-value]
    if route.provider == "deepseek_native":
        provider_client = _get_provider_client(route.provider, DeepSeekChatClient)
    elif route.provider == "openrouter":
        provider_client = _get_provider_client(route.provider, OpenRouterChatClient)
    else:
        raise KeyError(f"{model_name!r} is not a chat model")
    routed = _RoutedChatClient(role=model_name, route=route, provider_client=provider_client)
    _routed_clients[key] = routed
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
        cached_prompt=response.usage.prompt_cache_hit_tokens if response.usage else 0,
        reasoning=response.usage.completion_tokens_details.get("reasoning_tokens", 0) if response.usage else 0,
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
    return not response.tool_calls and (response.content or "").strip() == "" and response.finish_reason != "length"
