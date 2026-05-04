from __future__ import annotations

from dataclasses import asdict, dataclass
from threading import Lock


@dataclass
class _RoleUsage:
    role: str
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0


PRICE_USD_PER_1M_TOKENS = {
    "main": {"prompt": 0.27, "completion": 1.10},
    "fast": {"prompt": 0.07, "completion": 0.28},
    "embed-small": {"prompt": 0.02, "completion": 0.0},
    "rerank": {"prompt": 0.0, "completion": 0.0},
}
RERANK_PRICE_USD_PER_QUERY = 0.002

_usage: dict[str, _RoleUsage] = {}
_lock = Lock()


def record(role: str, prompt: int = 0, completion: int = 0) -> None:
    with _lock:
        current = _usage.setdefault(role, _RoleUsage(role=role))
        current.calls += 1
        current.prompt_tokens += max(prompt, 0)
        current.completion_tokens += max(completion, 0)


def snapshot() -> dict[str, dict]:
    with _lock:
        return {role: asdict(data) for role, data in sorted(_usage.items())}


def reset() -> None:
    with _lock:
        _usage.clear()


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
                data["prompt_tokens"] * price["prompt"]
                + data["completion_tokens"] * price["completion"]
            ) / 1_000_000
        costs[role] = role_cost
        total += role_cost
    costs["total"] = total
    return costs
