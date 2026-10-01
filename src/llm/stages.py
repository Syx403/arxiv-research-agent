"""Frozen stage policies. Legacy retains caller overrides as an experiment baseline."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import replace
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from src.llm.registry import get_route

Stage = Literal[
    "intent",
    "query",
    "candidate",
    "candidate_review",
    "relevance",
    "sufficiency",
    "synthesis",
    "scope",
    "verify",
    "claim_audit",
    "semantic_repair",
    "format",
]
Profile = Literal["legacy", "semantic_low", "critical_high", "semantic_high", "ui"]
STAGES = tuple(Stage.__args__)
CRITICAL = {
    "intent",
    "candidate",
    "candidate_review",
    "sufficiency",
    "verify",
    "claim_audit",
    "semantic_repair",
}


class StageSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    profile: Profile = "legacy"
    total_output_cap: int | None = Field(default=None, ge=1024, le=32768)


_settings: ContextVar[StageSettings] = ContextVar("ara_stage_settings", default=StageSettings())
_stage: ContextVar[str | None] = ContextVar("ara_model_stage", default=None)


def current_stage() -> str | None:
    return _stage.get()


def current_settings() -> StageSettings:
    return _settings.get()


@contextmanager
def use_stage_settings(settings: StageSettings):
    token = _settings.set(settings)
    try:
        yield
    finally:
        _settings.reset(token)


@contextmanager
def use_stage(stage: str | None):
    token = _stage.set(stage)
    try:
        yield
    finally:
        _stage.reset(token)


def resolve(stage, legacy_role, legacy_route, thinking):
    settings = current_settings()
    if settings.profile == "legacy":
        return legacy_role, legacy_route, thinking
    if stage not in STAGES:
        raise ValueError(f"Unclassified chat stage: {stage!r}")
    role = "fast" if stage in {"query", "format"} else "main"
    route = get_route(role, capability="chat")
    if route.provider != "deepseek_native" or route.vendor_model != "deepseek-flash":
        raise ValueError("These experiment profiles require DeepSeek Flash for both roles")
    if stage == "format":
        return role, replace(route, reasoning_effort=None, reasoning_headroom_tokens=0), False
    effort = "low"
    if stage != "query" and (
        settings.profile == "semantic_high"
        or settings.profile == "critical_high"
        and stage in CRITICAL
    ):
        effort = "high"
    if settings.profile == "ui":
        effort = get_route("main" if stage in CRITICAL else "fast").reasoning_effort
    return (
        role,
        replace(
            route,
            reasoning_effort=effort,
            reasoning_headroom_tokens={"low": 2048, "high": 8192, "max": 16384}[effort],
        ),
        True,
    )


def signature() -> dict:
    settings = current_settings()
    rows = {}
    if settings.profile != "legacy":
        for stage in STAGES:
            role, route, thinking = resolve(stage, "main", get_route("main"), None)
            rows[stage] = {
                "role": role,
                "model": route.vendor_model,
                "effort": route.reasoning_effort,
                "thinking": thinking,
                "headroom_tokens": route.reasoning_headroom_tokens,
            }
    return {**settings.model_dump(), "version": 2, "stages": rows,
            "semantic_output_floor": 4096, "capacity_retry_ceiling": 16384,
            "capacity_retry": settings.total_output_cap is None and settings.profile != "legacy"}
