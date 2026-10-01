"""Single-user local runs with durable UI reports and one shared API budget."""
from __future__ import annotations

import asyncio
from datetime import UTC, datetime
import json
import logging
from pathlib import Path
from time import perf_counter
from uuid import UUID, uuid4

from src.core.research_policy import ResearchPolicy
from src.llm.budget import RequestBudget
from src.llm.budget import usage_for_turn
from src.agent.runtime import AgentRequest, AgentRuntime
from src.llm.stages import StageSettings
from src.llm.registry import ReasoningSettings, snapshot_chat_routes
from src.ui.presentation import STAGES, STOP_LABELS

logger = logging.getLogger(__name__)


def timestamp() -> str:
    return datetime.now(UTC).isoformat()


class BusyError(RuntimeError):
    pass


class ChatRuntime:
    def __init__(self, directory: Path, budget: RequestBudget, *, runner=None, title_loader=None,
                 research_policy: ResearchPolicy | None = None):
        self.directory, self.budget = directory, budget
        self.runner, self.title_loader = runner, title_loader
        self.agent = AgentRuntime(directory, budget, runner=runner, title_loader=title_loader)
        self.research_policy = research_policy or ResearchPolicy()
        self.sessions: dict[str, dict] = {}
        self.task: asyncio.Task | None = None
        self.active_session: str | None = None
        self.active_start_index = 0
        self.active_started = 0.0
        directory.mkdir(parents=True, exist_ok=True)
        for path in directory.glob("*.json"):
            session = json.loads(path.read_text())
            self.sessions[session["id"]] = session
            for turn in session["turns"]:
                if turn["status"] == "running":
                    turn.update(status="incomplete", stop_reason="interrupted",
                                answer="本地服务在运行中中断。请重新发送问题。",
                                stage=STOP_LABELS["interrupted"], cost_incomplete=True)
            self._save(session)

    def _save(self, session: dict) -> None:
        path = self.directory / f"{UUID(session['id'])}.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(session, ensure_ascii=False, indent=2))
        temporary.replace(path)

    def create_session(self) -> dict:
        session = {"id": str(uuid4()), "title": "新的研究", "updated_at": timestamp(), "turns": []}
        self.sessions[session["id"]] = session
        self._save(session)
        return session

    def list_sessions(self) -> list[dict]:
        return [{k: s[k] for k in ("id", "title", "updated_at")}
                for s in sorted(self.sessions.values(), key=lambda s: s["updated_at"], reverse=True)]

    def get_session(self, session_id: str) -> dict:
        session = self.sessions[session_id]
        if self.active_session == session_id and session["turns"][-1]["status"] == "running":
            turn = session["turns"][-1]
            turn["cost"] = usage_for_turn(self.budget.snapshot(), turn["id"])
            turn["elapsed_s"] = round(perf_counter() - self.active_started, 1)
        return session

    def budget_summary(self) -> dict:
        snapshot = self.budget.snapshot()
        return {key: snapshot[key] for key in (
            "limit_usd", "committed_usd", "estimated_settled_usd", "uncertain_or_reserved_usd",
        )} | {"requests": len(snapshot["requests"]), "max_requests": self.budget.max_requests}

    def start(self, session_id: str, question: str, request_id: str, *, reasoning: ReasoningSettings | None = None) -> dict:
        session = self.get_session(session_id)
        # Reconcile a retried POST without launching a second paid run.
        existing = next((t for t in session["turns"] if t["id"] == request_id), None)
        if existing:
            if existing["question"] != question:
                raise ValueError("同一个请求编号不能对应不同的问题。")
            if reasoning is not None and any(
                existing.get("profiles", {}).get(role, {}).get("effort") != effort
                for role, effort in reasoning.model_dump().items()
            ):
                raise ValueError("同一个请求编号不能更改思考档位。请查看原有运行，或发送一个新问题。")
            return existing
        if self.task is not None and not self.task.done():
            raise BusyError("已有一轮研究正在运行，请等待完成或先停止。")
        routes = snapshot_chat_routes(reasoning)
        policy = self.research_policy
        turn = {
            "id": request_id, "question": question, "created_at": timestamp(), "status": "running",
            "stage": "准备研究", "answer": "", "stop_reason": "", "sources": [], "steps": [],
            "research_question": "", "subquestions": [], "context_messages": 0,
            "research_policy": policy.model_dump(), "external_discovery": {},
            "result_kind": "papers",
            "paper_results": [], "paper_search": {}, "cache_hits": 0,
            "missing_aspects": [], "checks": [], "generation_retries": 0,
            "cost": {"requests": 0, "estimated_usd": 0.0, "uncertain_usd": 0.0}, "elapsed_s": 0,
            "profiles": {role: {"model": route.vendor_model, "effort": route.reasoning_effort,
                                "headroom_tokens": route.reasoning_headroom_tokens}
                         for role, route in routes.items()},
            "retrieval_progress": None,
        }
        session["turns"].append(turn)
        if len(session["turns"]) == 1:
            session["title"] = question[:70]
        session["updated_at"] = timestamp()
        self._save(session)
        self.active_session = session_id
        self.active_start_index = len(self.budget.snapshot()["requests"])
        self.active_started = perf_counter()
        self.task = asyncio.create_task(self._execute(session, turn, routes))
        return turn

    async def _execute(self, session: dict, turn: dict, routes: dict) -> None:
        started = perf_counter()

        def observe(event: dict):
            if event["event"] == "node_start":
                turn["stage"] = STAGES.get(event["node"], event["node"])
                turn["steps"].append({"stage": turn["stage"], "elapsed_s": event["elapsed_s"]})
            if event["event"] == "node_end" and turn["steps"]:
                step = turn["steps"][-1]
                step["duration_s"] = round(event["elapsed_s"] - step["elapsed_s"], 2)
            if event["event"] == "retrieval_progress":
                turn["retrieval_progress"] = {key: event[key] for key in (
                    "subq_index", "stage", "candidate_chunks", "accepted_chunks", "papers", "retry",
                )}
            if event["event"] == "external_progress":
                turn["stage"] = {"searching": "联网发现论文", "found": "已找到论文候选", "screening": "筛选标题与摘要",
                                 "reading": "读取并索引全文"}.get(event["stage"], "联网发现论文")
                if event.get("title"):
                    turn["stage"] += " · " + event["title"]
            if event["event"] == "external_discovery":
                turn["external_discovery"] = event["report"]
            if event["event"] == "paper_search":
                turn["paper_search"] = event["report"]
                if "papers" in event:
                    turn["paper_results"] = event["papers"]
            if event["event"] in {"paper_cache", "external_metadata"} and (event.get("hit") or event.get("cache_hit")):
                turn["cache_hits"] += 1
            if event["event"] == "node_end" and event.get("node") in {"decompose", "plan_papers", "read_papers"}:
                turn["research_question"] = event.get("research_question", "")
                turn["subquestions"] = event.get("subquestions", [])
                turn["context_messages"] = event.get("context_turns", 0)
            turn["elapsed_s"] = round(perf_counter() - started, 1)

        try:
            result = await self.agent.query(AgentRequest(
                question=turn["question"], session_id=session["id"], turn_id=turn["id"], run_id=turn["id"],
                policy=ResearchPolicy.model_validate(turn["research_policy"]), stages=StageSettings(profile="ui"),
            ), on_event=observe, routes=routes)
            turn.update(result.output)
            turn["agent_result"] = result.model_dump(mode="json")
        except asyncio.CancelledError:
            turn.update(status="incomplete", stop_reason="cancelled", answer="本轮已停止。尚未核验完成的内容未作为回答交付。")
        except TimeoutError:
            turn.update(status="partial" if turn.get("paper_results") else "incomplete", stop_reason="turn_timeout",
                        answer="本轮达到时间上限，已保留取得的论文和执行过程；尚未完成的分析不作为结论交付。")
        except Exception as exc:
            # Provider exceptions can include response bodies; never expose them to the browser.
            logger.error("Local chat failed; run=%s error_type=%s", turn["id"], type(exc).__name__)
            turn.update(status="incomplete", stop_reason="runtime_error", answer="本轮未完成。请检查本地数据库和模型配置后重试。",
                        error_type=type(exc).__name__)
        finally:
            turn["cost"] = usage_for_turn(self.budget.snapshot(), turn["id"])
            turn["elapsed_s"] = round(perf_counter() - started, 1)
            turn["stage"] = STOP_LABELS.get(turn["stop_reason"], "本轮结束")
            session["updated_at"] = timestamp()
            self._save(session)
            self.active_session = None

    async def stop(self, session_id: str) -> None:
        if self.active_session == session_id and self.task and not self.task.done():
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
            # Cancellation before the task's first instruction skips its finally block.
            turn = self.sessions[session_id]["turns"][-1]
            if turn["status"] == "running":
                turn.update(status="incomplete", stop_reason="cancelled", answer="本轮已停止。",
                            stage=STOP_LABELS["cancelled"])
                self._save(self.sessions[session_id])
                self.active_session = None

    async def close(self) -> None:
        if self.active_session:
            await self.stop(self.active_session)
