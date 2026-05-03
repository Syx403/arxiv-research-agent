from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


ProviderName = Literal["openrouter", "deepseek_native", "openai_native", "cohere_native"]
Capability = Literal["chat", "embedding", "rerank"]


@dataclass(frozen=True)
class ProviderRoute:
    provider: ProviderName
    vendor_model: str
    capability: Capability


REGISTRY: dict[str, ProviderRoute] = {
    "main": ProviderRoute("deepseek_native", "deepseek-v4-pro", "chat"),
    "fast": ProviderRoute("deepseek_native", "deepseek-v4-flash", "chat"),
    "embed-small": ProviderRoute("openai_native", "text-embedding-3-small", "embedding"),
    "rerank": ProviderRoute("cohere_native", "rerank-v4.0-pro", "rerank"),
}

# Reserved (NOT registered by default; opt-in only by uncommenting one line):
# "fallback-haiku": ProviderRoute("openrouter", "anthropic/claude-haiku-4.5", "chat"),


def get_route(model_name: str, *, capability: Capability | None = None) -> ProviderRoute:
    route = REGISTRY[model_name]
    if capability is not None and route.capability != capability:
        raise KeyError(f"{model_name!r} is registered for {route.capability}, not {capability}")
    return route
