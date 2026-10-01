from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
from contextlib import contextmanager
from contextvars import ContextVar
import os
from typing import Literal

from pydantic import BaseModel, ConfigDict


ProviderName = Literal["openrouter", "deepseek_native", "openai_native", "cohere_native"]
Capability = Literal["chat", "embedding", "rerank"]
ReasoningEffort = Literal["low", "high", "max"]


class ReasoningSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    main: ReasoningEffort = "high"
    fast: ReasoningEffort = "low"


@dataclass(frozen=True)
class ProviderRoute:
    provider: ProviderName
    vendor_model: str
    capability: Capability
    reasoning_effort: Literal["low", "high", "max"] | None = None
    reasoning_headroom_tokens: int = 0


REGISTRY: dict[str, ProviderRoute] = {
    "main": ProviderRoute("deepseek_native", "deepseek-flash", "chat", "high", 8192),
    "fast": ProviderRoute("deepseek_native", "deepseek-flash", "chat", "low", 2048),
    "embed-small": ProviderRoute("openai_native", "text-embedding-3-small", "embedding"),
    "rerank": ProviderRoute("cohere_native", "rerank-v4.0-pro", "rerank"),
}

# Reserved (NOT registered by default; opt-in only by uncommenting one line):
# "fallback-haiku": ProviderRoute("openrouter", "anthropic/claude-haiku-4.5", "chat"),

_run_routes: ContextVar[dict[str, ProviderRoute]] = ContextVar("ara_run_routes", default={})


def snapshot_chat_routes(settings: ReasoningSettings | None = None) -> dict[str, ProviderRoute]:
    routes = {role: get_route(role, capability="chat") for role in ("main", "fast")}
    if settings is not None:
        for role, effort in settings.model_dump().items():
            routes[role] = replace(routes[role], reasoning_effort=effort,
                                   reasoning_headroom_tokens={"low": 2048, "high": 8192, "max": 16384}[effort])
    return routes


@contextmanager
def use_chat_routes(routes: dict[str, ProviderRoute]):
    """A question and all child tasks/retries keep the same immutable routes."""
    token = _run_routes.set(dict(routes))
    try:
        yield
    finally:
        _run_routes.reset(token)


def get_route(model_name: str, *, capability: Capability | None = None) -> ProviderRoute:
    route = _run_routes.get().get(model_name, REGISTRY[model_name])
    if capability is not None and route.capability != capability:
        raise KeyError(f"{model_name!r} is registered for {route.capability}, not {capability}")
    if model_name in {"main", "fast"} and model_name not in _run_routes.get():
        effort = os.getenv(f"ARA_{model_name.upper()}_REASONING_EFFORT", route.reasoning_effort)
        if effort not in {"low", "high", "max"}:
            raise ValueError("DeepSeek reasoning effort must be low, high, or max")
        headroom = int(os.getenv(f"ARA_{model_name.upper()}_REASONING_HEADROOM", str(route.reasoning_headroom_tokens)))
        if not 0 <= headroom <= 32768:
            raise ValueError("Reasoning headroom must be between 0 and 32768 tokens")
        route = replace(route, reasoning_effort=effort, reasoning_headroom_tokens=headroom)
    return route
