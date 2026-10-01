"""Run with make ui. Local-only UI; no provider secrets are sent to the browser."""
from __future__ import annotations

from contextlib import asynccontextmanager
import os
from pathlib import Path
from uuid import UUID

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, field_validator
from starlette.middleware.trustedhost import TrustedHostMiddleware

from src.llm.budget import RequestBudget
from src.core.research_policy import ResearchPolicy
from src.llm.registry import ReasoningSettings, get_route
from src.ui.runtime import BusyError, ChatRuntime

ROOT = Path(__file__).resolve().parents[2]
STATIC = Path(__file__).parent / "static"


class ChatInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: str = Field(min_length=1, max_length=8000)
    request_id: UUID
    reasoning: ReasoningSettings | None = None

    @field_validator("question")
    @classmethod
    def nonblank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("请输入研究问题。")
        return value


async def load_titles(paper_ids: list[str]) -> dict[str, str]:
    if not paper_ids:
        return {}
    from src.core import db
    async with db.acquire_app() as conn:
        rows = await conn.fetch("SELECT paper_id, title FROM papers WHERE paper_id = ANY($1::text[])", list(set(paper_ids)))
    return {row["paper_id"]: row["title"] for row in rows}


def create_app(runtime: ChatRuntime | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(application):
        if runtime is None:
            # Trace locally; research policy is per turn, not a global env override.
            for key, value in {
                "LANGSMITH_TRACING": "false", "LANGCHAIN_TRACING_V2": "false",
                "ARA_EVAL_LANGSMITH_EXPORT": "false",
            }.items():
                os.environ[key] = value
        actual = runtime or ChatRuntime(
            ROOT / "data/ui/sessions",
            # Necessary calibration/acceptance under the user's delegated budget control;
            # cumulative cap raised to $2.50 on 2026-09-17, with all prior usage retained.
            RequestBudget(2.5, path=ROOT / "data/eval_outputs/round1-budget.json",
                          cohere_trial=True, max_requests=3000),
            title_loader=load_titles,
            research_policy=ResearchPolicy.configured(),
        )
        application.state.runtime = actual
        yield
        await actual.close()
        if runtime is None:
            from src.graph.builder import shutdown
            await shutdown()

    application = FastAPI(title="ARA · 论文研究", lifespan=lifespan, docs_url=None, redoc_url=None)
    application.add_middleware(TrustedHostMiddleware, allowed_hosts=["localhost", "127.0.0.1", "[::1]", "testserver"])

    @application.middleware("http")
    async def local_requests(request: Request, call_next):
        # Disallow cross-origin scripts/forms from using this local paid endpoint.
        if request.method not in {"GET", "HEAD", "OPTIONS"}:
            origin = request.headers.get("origin")
            if (origin and origin != str(request.base_url).rstrip("/")) or request.headers.get("x-ara-client") != "local-ui":
                return JSONResponse({"detail": "请从本地 ARA 页面操作。"}, status_code=403)
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'self'; form-action 'self'"
        return response

    @application.get("/")
    async def index():
        return FileResponse(STATIC / "index.html")

    @application.get("/api/status")
    async def status(request: Request):
        actual = request.app.state.runtime
        return {"budget": actual.budget_summary(), "active_session": actual.active_session,
                "profiles": {role: {"model": get_route(role).vendor_model, "effort": get_route(role).reasoning_effort} for role in ("main", "fast")},
                "research_policy": actual.research_policy.model_dump(), "clarification_enabled": True}

    @application.get("/api/sessions")
    async def sessions(request: Request):
        return request.app.state.runtime.list_sessions()

    @application.post("/api/sessions", status_code=201)
    async def new_session(request: Request):
        return request.app.state.runtime.create_session()

    def get_session(request, session_id):
        try:
            return request.app.state.runtime.get_session(str(session_id))
        except KeyError:
            raise HTTPException(404, "会话不存在，请新建会话。") from None

    @application.get("/api/sessions/{session_id}")
    async def session(request: Request, session_id: UUID):
        return get_session(request, session_id)

    @application.post("/api/sessions/{session_id}/messages", status_code=202)
    async def message(request: Request, session_id: UUID, body: ChatInput):
        get_session(request, session_id)
        try:
            return request.app.state.runtime.start(str(session_id), body.question, str(body.request_id),
                                                   reasoning=body.reasoning)
        except (BusyError, ValueError) as exc:
            raise HTTPException(409, str(exc)) from None

    @application.post("/api/sessions/{session_id}/stop")
    async def stop(request: Request, session_id: UUID):
        get_session(request, session_id)
        await request.app.state.runtime.stop(str(session_id))
        return {"stopped": True}

    application.mount("/static", StaticFiles(directory=STATIC), name="static")
    return application


app = create_app()
