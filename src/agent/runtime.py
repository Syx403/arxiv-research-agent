"""Importable agent runtime. No HTTP/UI dependency and no implicit paid allowance."""

from __future__ import annotations

import asyncio
from datetime import date
from pathlib import Path
from time import perf_counter
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from src.agent.results import present_state
from src.core.answer_contract import RequestedField
from src.core.structured_answer import request_text
from src.core.research_policy import ResearchPolicy, use_research_policy
from src.core.run_context import RunContext, use_run_context
from src.core.trace import LocalTrace, record_event
from src.corpus.access import Acquisition, use_acquisition
from src.llm import usage
from src.llm.budget import RequestBudget, use_turn_budget, usage_for_turn
from src.llm.registry import ReasoningSettings, snapshot_chat_routes, use_chat_routes
from src.llm.stages import StageSettings, signature, use_stage_settings


class AgentRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    question: str = Field(min_length=1, max_length=8000)
    session_id: UUID = Field(default_factory=uuid4)
    turn_id: UUID = Field(default_factory=uuid4)
    run_id: UUID = Field(default_factory=uuid4)
    reasoning: ReasoningSettings = Field(default_factory=ReasoningSettings)
    stages: StageSettings = Field(default_factory=StageSettings)
    policy: ResearchPolicy = Field(default_factory=ResearchPolicy)
    as_of: date | None = None
    model_result_cache: bool = True
    structured_context: bool = True
    # Experimental serialization is opt-in until its value/latency is calibrated.
    include_typed_observations: bool = False
    response_fields: tuple[RequestedField, ...] = Field(default=(), max_length=12)

    @model_validator(mode="after")
    def output_contract(self):
        names = [field.name for field in self.response_fields]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate response field names")
        if sum(f.kind == "paper_order" for f in self.response_fields) > 1:
            raise ValueError("Only one recommendation order per response")
        if self.response_fields and self.include_typed_observations:
            raise ValueError("Native structured responses and experimental extraction are separate modes")
        return self

    @field_validator("question")
    @classmethod
    def nonblank(cls, value):
        if not value.strip():
            raise ValueError("Question must not be blank")
        return value.strip()


class AgentResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: int = 2
    run_id: str
    session_id: str
    turn_id: str
    question: str
    route: str
    output: dict[str, Any]
    session_context: dict[str, Any] = Field(default_factory=dict)
    cost: dict[str, Any]
    role_usage: dict[str, Any]
    configuration: dict[str, Any]
    elapsed_s: float
    trace_path: str


class AgentRuntime:
    def __init__(self, directory: Path, budget: RequestBudget, *, runner=None, title_loader=None):
        self.directory, self.budget = directory, budget
        self.runner, self.title_loader = runner, title_loader
        self._busy: set[str] = set()

    async def query(
        self,
        request: AgentRequest,
        *,
        on_event=None,
        routes=None,
        acquisition: Acquisition | None = None,
    ) -> AgentResult:
        sid, tid = str(request.session_id), str(request.turn_id)
        request = AgentRequest.model_validate(request.model_dump(mode="json"))
        if sid in self._busy:
            raise RuntimeError("A turn is already running in this session")
        trace_path = self.directory / "traces" / sid / f"{tid}.jsonl"
        # A preexisting trajectory is an uncertain/completed attempt, never permission to charge twice.
        progress = {}

        def observe(event):
            if event["event"] == "paper_search":
                progress["paper_search"] = event["report"]
                if "papers" in event:
                    progress["paper_results"] = event["papers"]
            if on_event:
                on_event(event)

        routes = dict(routes or snapshot_chat_routes(request.reasoning))
        trace = LocalTrace(trace_path, run_id=tid, mode="agent", on_event=observe)
        self._busy.add(sid)
        started = perf_counter()
        state, configuration, roles = {}, {}, {}
        try:
            context = RunContext(
                run_id=str(request.run_id),
                turn_id=tid,
                as_of=request.as_of,
                model_result_cache=request.model_result_cache,
                structured_context=request.structured_context,
                include_typed_observations=request.include_typed_observations,
                response_fields=request.response_fields,
                deadline_monotonic=started + request.policy.reading_timeout_s,
            )
            with (
                trace.activate(),
                self.budget.activate(),
                use_turn_budget(request.policy.search_budget_usd, identifier=tid),
                use_chat_routes(routes),
                use_research_policy(request.policy),
                use_stage_settings(request.stages),
                use_run_context(context),
                usage.isolated_usage(),
                use_acquisition(acquisition),
            ):
                configuration = {
                    "stages": signature(),
                    "policy": request.policy.model_dump(),
                    "run_context": context.model_dump(mode="json"),
                    "roles": {
                        role: {
                            "model": route.vendor_model,
                            "effort": route.reasoning_effort,
                            "headroom_tokens": route.reasoning_headroom_tokens,
                        }
                        for role, route in routes.items()
                    },
                }
                record_event("run_configuration", configuration=configuration)
                try:
                    runner = self.runner
                    if runner is None:
                        from src.graph.builder import run

                        runner = run
                    async with asyncio.timeout(request.policy.reading_timeout_s):
                        state = await runner(
                            request_text(request.question, request.response_fields),
                            thread_id=f"ara-ui-{sid}", skip_reflection=True
                        )
                except asyncio.CancelledError:
                    state = {
                        "status": "incomplete",
                        "stop_reason": "cancelled",
                        "answer": "本轮已停止，尚未核验的内容未交付。",
                    }
                except TimeoutError:
                    state = {
                        "status": "incomplete",
                        "stop_reason": "turn_timeout",
                        "answer": "本轮达到时间上限，尚未完成的分析未作为结论交付。",
                    }
                except Exception as exc:
                    record_event("runtime_error", error_type=type(exc).__name__)
                    state = {
                        "status": "incomplete",
                        "stop_reason": "runtime_error",
                        "answer": "本轮未完成，请查看运行诊断。",
                        "error_type": type(exc).__name__,
                    }
                if self.runner is None and state.get("stop_reason") in {"turn_timeout", "runtime_error", "cancelled"}:
                    from src.graph.builder import finalize_interrupted_turn
                    try:
                        async with asyncio.timeout(5):
                            recovered = await finalize_interrupted_turn(
                                thread_id=f"ara-ui-{sid}", turn_id=tid, reason=state["stop_reason"])
                        if recovered:
                            if state.get("error_type"):
                                recovered["error_type"] = state["error_type"]
                            state = recovered
                            record_event("interrupted_delivery", status="checkpoint_projected")
                    except Exception as exc:
                        record_event("interrupted_delivery", status="unavailable", error_type=type(exc).__name__)
                roles = usage.snapshot()
            titles = {}
            if self.title_loader is not None:
                try:
                    titles = await self.title_loader(
                        [c.paper_id for c in state.get("citations", [])]
                    )
                except Exception:
                    pass
            output = present_state(state, titles, include_typed_observations=request.include_typed_observations,
                                   response_fields=request.response_fields)
            if output["stop_reason"] in {"turn_timeout", "runtime_error", "cancelled"}:
                output.update(progress)
                if progress.get("paper_results") and output["stop_reason"] == "turn_timeout":
                    output["status"] = "partial"
            if state.get("error_type"):
                output["error_type"] = state["error_type"]
            plan = state.get("paper_plan", {})
            route = (
                "clarify"
                if plan.get("clarification") or output["stop_reason"] == "clarification_required"
                else "read"
                if plan.get("read_full_text") or output["result_kind"] == "evidence"
                else "discover"
            )
            if not plan and output["stop_reason"] in {"runtime_error", "turn_timeout", "cancelled"}:
                route = "unknown"
            return AgentResult(
                run_id=str(request.run_id),
                session_id=sid,
                turn_id=tid,
                question=request.question,
                route=route,
                output=output,
                session_context=state.get("session_context", {}),
                cost=usage_for_turn(self.budget.snapshot(), tid),
                role_usage=roles,
                configuration=configuration,
                elapsed_s=round(perf_counter() - started, 3),
                trace_path=str(trace_path.resolve()),
            )
        finally:
            self._busy.discard(sid)
