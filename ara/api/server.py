"""The local web app (DESIGN §9, M5b): FastAPI on 127.0.0.1 for one user, no login. A turn streams
as Server-Sent Events: node starts and ends (subgraphs included), each model call from the ledger,
then the finished turn or the question the graph waits on. The built UI (ui/dist) is served at /.

    uv run ara serve        # http://127.0.0.1:8000
"""

import asyncio
from collections import defaultdict
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.sse import EventSourceResponse, ServerSentEvent
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import HumanMessage
from langgraph.graph.state import CompiledStateGraph
from langgraph.store.base import BaseStore
from langgraph.types import Command
from pydantic import BaseModel

from ara.api import events
from ara.arxiv.client import ArxivClient
from ara.db.migrate import migrate
from ara.db.pool import Pool, make_pool
from ara.graph import app as conversation
from ara.graph.state import Context
from ara.llm.gateway import Gateway
from ara.llm.ledger import Ledger, Scope
from ara.memory import library, store
from ara.rag.ingest import PIPELINE_VERSION
from ara.rag.sources import ParsedPaper
from ara.settings import ROOT, configure_tracing, get_settings
from evals import report

HOST, PORT = "127.0.0.1", 8000
USER = "local"  # single user, no login (DESIGN §9)
UI = ROOT / "ui" / "dist"

DOCUMENT = """
SELECT p.id, p.title, p.source, c.id AS chunk_id, c.heading_path, c.text, c.sentences
FROM papers p JOIN documents d ON d.paper_id = p.id
JOIN chunks c ON c.document_id = d.id
WHERE p.id = %s AND d.pipeline_version = %s ORDER BY c.ord
"""


@dataclass
class Services:
    pool: Pool
    gateway: Gateway
    arxiv: ArxivClient
    memory: BaseStore
    graph: CompiledStateGraph[Any, Any, Any, Any]
    locks: defaultdict[str, asyncio.Lock]

    def context(self, turn_id: str) -> Context:
        async def fetch(reference: str) -> ParsedPaper:
            return await self.arxiv.paper(reference.removeprefix("arxiv:"))

        return Context(self.pool, self.gateway, Scope(turn_id=turn_id), fetch, self.arxiv)


services: Services | None = None


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """One pool, gateway, arXiv client, Store and checkpointer for the server's life."""
    global services
    settings = get_settings()
    configure_tracing(settings)
    migrate(settings.database_url)
    async with AsyncExitStack() as stack:
        pool = await stack.enter_async_context(make_pool(settings.database_url))
        arxiv = await stack.enter_async_context(ArxivClient())
        ledger = Ledger(
            pool, global_cap_usd=settings.global_cap_usd, turn_cap_usd=settings.turn_cap_usd
        )
        gateway = Gateway(settings, ledger)
        stack.push_async_callback(gateway.aclose)
        memory = await store.open_store(pool, gateway)
        graph = conversation.build(await conversation.checkpointer(pool), memory)
        services = Services(pool, gateway, arxiv, memory, graph, defaultdict(asyncio.Lock))
        yield
        services = None


api = FastAPI(title="ARA", lifespan=lifespan)


def _services() -> Services:
    if services is None:
        raise HTTPException(503, "the server is starting")
    return services


class Message(BaseModel):
    text: str


class Resume(BaseModel):
    answer: str


@api.get("/api/graph")
def graph() -> dict[str, str]:
    """Mermaid for the main graph and the three subgraphs its nodes run, drawn from the compiled
    graphs, so the picture always matches the code (xray cannot see subgraphs called inside a
    node, D35)."""
    return {
        "main": conversation.build().get_graph().draw_mermaid(),
        "discover": conversation.DISCOVER.get_graph().draw_mermaid(),
        "read": conversation.READ.get_graph().draw_mermaid(),
        "answer": conversation.ANSWER.get_graph().draw_mermaid(),
    }


@api.post("/api/threads")
def new_thread() -> dict[str, str]:
    return {"id": uuid4().hex}


@api.get("/api/threads/{thread_id}")
async def thread(thread_id: str) -> dict[str, Any]:
    s = _services()
    snapshot = await s.graph.aget_state(_config(thread_id))
    values = snapshot.values or {}
    waiting = [i.value for i in snapshot.interrupts]
    return {**events.turn(values), "waiting": waiting[0] if waiting else None}


@api.post("/api/threads/{thread_id}/messages", response_class=EventSourceResponse)
async def message(thread_id: str, body: Message) -> AsyncIterator[ServerSentEvent]:
    async for event in _turn(thread_id, {"messages": [HumanMessage(body.text)]}):
        yield event


@api.post("/api/threads/{thread_id}/resume", response_class=EventSourceResponse)
async def resume(thread_id: str, body: Resume) -> AsyncIterator[ServerSentEvent]:
    async for event in _turn(thread_id, Command(resume=body.answer)):
        yield event


async def _turn(thread_id: str, payload: Any) -> AsyncIterator[ServerSentEvent]:
    """Run one turn (or its resumption) and stream what happens. One turn at a time per thread."""
    s = _services()
    turn_id = f"ui:{thread_id}:{uuid4().hex[:8]}"
    config = _config(thread_id)
    last_call = 0
    async with s.locks[thread_id]:
        stream = s.graph.astream(
            payload,
            config,
            context=s.context(turn_id),
            stream_mode="tasks",
            subgraphs=True,
            version="v2",
        )
        try:
            async for part in stream:
                if (node := events.node_event(dict(part))) is not None:
                    yield ServerSentEvent(data=node, event="node")
                if node is not None and node["phase"] != "start":
                    async with s.pool.connection() as conn:
                        for call in await events.new_calls(conn, turn_id, last_call):
                            last_call = call["id"]
                            yield ServerSentEvent(data=call, event="call")
        except Exception as error:
            # LangGraph 1.2.14 streaming with subgraphs=True re-raises, at the end, an error a
            # node's error handler already handled, after the run has finished and been
            # checkpointed (reproduced on a two-node graph, D35). A finished run is a turn.
            snapshot = await s.graph.aget_state(config)
            if snapshot.next and not snapshot.interrupts:
                yield ServerSentEvent(data={"error": repr(error)}, event="error")
                return
        snapshot = await s.graph.aget_state(config)
    if snapshot.interrupts:
        yield ServerSentEvent(data=snapshot.interrupts[0].value, event="interrupt")
    else:
        yield ServerSentEvent(data=events.turn(snapshot.values), event="done")


def _config(thread_id: str) -> Any:
    return {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": conversation.RECURSION_LIMIT,
    }


@api.get("/api/papers/{paper_id}/document")
async def document(paper_id: str) -> dict[str, Any]:
    """A stored paper's passages with sentence offsets, for the evidence viewer."""
    async with _services().pool.connection() as conn:
        rows = await (await conn.execute(DOCUMENT, (paper_id, PIPELINE_VERSION))).fetchall()
    if not rows:
        raise HTTPException(404, f"no stored document for {paper_id}")
    passages = [
        {k: row[k] for k in ("chunk_id", "heading_path", "text", "sentences")} for row in rows
    ]
    return {
        "id": paper_id,
        "title": rows[0]["title"],
        "source": rows[0]["source"],
        "passages": passages,
    }


@api.get("/api/evals")
async def evals() -> list[dict[str, Any]]:
    return await report.runs(_services().pool)


@api.get("/api/evals/{run_id}")
async def evaluation(run_id: str) -> dict[str, Any]:
    try:
        return await report.summary(_services().pool, run_id)
    except LookupError as error:
        raise HTTPException(404, str(error)) from error


@api.get("/api/memory")
async def memory() -> dict[str, Any]:
    """What is remembered about the user (with each fact's source turn) and the papers read."""
    s = _services()
    async with s.pool.connection() as conn:
        papers = await library.papers(conn, USER)
    facts = await store.profile(s.memory, USER)
    return {
        "facts": [f.model_dump() for f in facts],
        "library": [
            p.model_dump(include={"arxiv_id", "version", "title", "published"}) for p in papers
        ],
    }


@api.delete("/api/memory/{key}")
async def forget(key: str) -> dict[str, str]:
    await store.forget(_services().memory, USER, key)
    return {"forgotten": key}


if UI.exists():  # built by `make ui`; during development Vite serves the UI instead
    api.mount("/", StaticFiles(directory=UI, html=True), name="ui")
