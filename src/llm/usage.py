from __future__ import annotations

from dataclasses import asdict, dataclass
from threading import Lock
from contextlib import contextmanager
from contextvars import ContextVar


@dataclass
class _RoleUsage:
    role: str
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    provider_429_count: int = 0
    provider_empty_retry_count: int = 0
    cached_prompt_tokens: int = 0
    reasoning_tokens: int = 0


PRICE_USD_PER_1M_TOKENS = {
    # Conservative peak-time Flash rates, USD / million tokens, 2026-09-14.
    "main": {"prompt": 0.30, "completion": 1.20},
    "fast": {"prompt": 0.30, "completion": 1.20},
    "embed-small": {"prompt": 0.02, "completion": 0.0},
    "rerank": {"prompt": 0.0, "completion": 0.0},
}
RERANK_PRICE_USD_PER_QUERY = 0.0  # This project currently uses Cohere Trial.

_usage: dict[str, _RoleUsage] = {}
_lock = Lock()
_scoped: ContextVar[dict[str, _RoleUsage] | None] = ContextVar("ara_usage", default=None)


def _current_usage():
    scoped = _scoped.get()
    return scoped if scoped is not None else _usage


@contextmanager
def isolated_usage():
    token = _scoped.set({})
    try:
        yield
    finally:
        _scoped.reset(token)


def record(role: str, prompt: int = 0, completion: int = 0, *, cached_prompt: int = 0, reasoning: int = 0) -> None:
    with _lock:
        current = _current_usage().setdefault(role, _RoleUsage(role=role))
        current.calls += 1
        current.prompt_tokens += max(prompt, 0)
        current.completion_tokens += max(completion, 0)
        current.cached_prompt_tokens += min(max(cached_prompt, 0), max(prompt, 0))
        current.reasoning_tokens += max(reasoning, 0)


def record_provider_429(role: str) -> None:
    with _lock:
        current = _current_usage().setdefault(role, _RoleUsage(role=role))
        current.provider_429_count += 1


def record_empty_retry(role: str) -> None:
    with _lock:
        current = _current_usage().setdefault(role, _RoleUsage(role=role))
        current.provider_empty_retry_count += 1


def snapshot() -> dict[str, dict]:
    with _lock:
        return {role: asdict(data) for role, data in sorted(_current_usage().items())}


def reset() -> None:
    with _lock:
        _current_usage().clear()


def estimated_cost_usd() -> dict[str, float]:
    usage_snapshot = snapshot()
    costs: dict[str, float] = {}
    total = 0.0
    for role, data in usage_snapshot.items():
        if role == "rerank":
            role_cost = data["calls"] * RERANK_PRICE_USD_PER_QUERY
        else:
            price = PRICE_USD_PER_1M_TOKENS.get(role, {"prompt": 0.0, "completion": 0.0})
            role_cost = (
                (data["prompt_tokens"] - data["cached_prompt_tokens"]) * price["prompt"]
                + data["cached_prompt_tokens"] * (0.006 if role in {"main", "fast"} else price["prompt"])
                + data["completion_tokens"] * price["completion"]
            ) / 1_000_000
        costs[role] = role_cost
        total += role_cost
    costs["total"] = total
    return costs
