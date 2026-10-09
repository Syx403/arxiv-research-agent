"""The local web app (DESIGN §9, M5b): FastAPI on 127.0.0.1 for one user, no login. A turn streams
as Server-Sent Events: node starts and ends (subgraphs included), each model call from the ledger,
then the finished turn or the question the graph waits on. The built UI (ui/dist) is served at /.

    uv run ara serve        # http://127.0.0.1:8000
"""

import asyncio
import re
from collections import defaultdict
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.sse import EventSourceResponse, ServerSentEvent
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.store.base import BaseStore
from langgraph.types import Command
from pydantic import BaseModel

from ara.api import conversations as chats
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
    checkpointer: AsyncPostgresSaver
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
        saver = await conversation.checkpointer(pool)
        graph = conversation.build(saver, memory)
        services = Services(pool, gateway, arxiv, memory, saver, graph, defaultdict(asyncio.Lock))
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


class Rename(BaseModel):
    title: str


@api.get("/api/graph")
def graph() -> dict[str, str]:
    """Mermaid for the main graph and the three subgraphs its nodes run, drawn from the compiled
    graphs, so the picture always matches the code (xray cannot see subgraphs called inside a
    node, D35)."""
    graphs: dict[str, CompiledStateGraph[Any, Any, Any, Any]] = {
        "main": conversation.build(),
        "discover": conversation.DISCOVER,
        "read": conversation.READ,
        "answer": conversation.ANSWER,
    }
    return {name: _drawn(g.get_graph().draw_mermaid()) for name, g in graphs.items()}


def _drawn(mermaid: str) -> str:
    """LangGraph draws each error handler as an unconnected node; the UI colours the node it
    handles instead (degraded), so the handlers are left out. Its colour classes are left out
    too, so the UI's own palette applies."""
    lines = [
        re.sub(r":::\w+", "", line)
        for line in mermaid.splitlines()
        if events.HANDLER not in line and not line.strip().startswith("classDef")
    ]
    return "\n".join(lines)


@api.get("/api/conversations")
async def conversations() -> list[dict[str, Any]]:
    """The user's conversations, most recent first (the sidebar)."""
    async with _services().pool.connection() as conn:
        return await chats.listed(conn, USER)


@api.post("/api/conversations")
def new_conversation() -> dict[str, str]:
    """An id for a new conversation; it is stored, and listed, from its first message."""
    return {"id": uuid4().hex}


@api.get("/api/conversations/{conversation_id}")
async def opened(conversation_id: str) -> dict[str, Any]:
    """Every turn of a conversation, each with its reply, answer, evidence, papers, path through
    the graph and model calls, and the question it waits on, if any."""
    async with _services().pool.connection() as conn:
        found = await chats.opened(conn, conversation_id, USER)
    if found is None:
        raise HTTPException(404, f"no conversation {conversation_id}")
    return found


@api.patch("/api/conversations/{conversation_id}")
async def renamed(conversation_id: str, body: Rename) -> dict[str, str]:
    async with _services().pool.connection() as conn:
        if not await chats.rename(conn, conversation_id, USER, body.title):
            raise HTTPException(404, f"no conversation {conversation_id}")
    return {"id": conversation_id, "title": body.title}


@api.delete("/api/conversations/{conversation_id}")
async def deleted(conversation_id: str) -> dict[str, str]:
    """The conversation, its turns and its checkpoints; the ledger keeps its spend (D36)."""
    s = _services()
    async with s.pool.connection() as conn:
        if not await chats.delete(conn, conversation_id, USER):
            raise HTTPException(404, f"no conversation {conversation_id}")
    await s.checkpointer.adelete_thread(conversation_id)
    return {"deleted": conversation_id}


@api.post("/api/conversations/{conversation_id}/messages", response_class=EventSourceResponse)
async def message(conversation_id: str, body: Message) -> AsyncIterator[ServerSentEvent]:
    async with _services().pool.connection() as conn:
        await chats.touch(conn, conversation_id, USER, body.text)
    payload = {"messages": [HumanMessage(body.text)]}
    async for event in _turn(conversation_id, body.text, payload):
        yield event


@api.post("/api/conversations/{conversation_id}/resume", response_class=EventSourceResponse)
async def resume(conversation_id: str, body: Resume) -> AsyncIterator[ServerSentEvent]:
    async with _services().pool.connection() as conn:
        await chats.touch(conn, conversation_id, USER, body.answer)
    async for event in _turn(conversation_id, body.answer, Command(resume=body.answer)):
        yield event


async def _turn(conversation_id: str, said: str, payload: Any) -> AsyncIterator[ServerSentEvent]:
    """Run one turn (or its resumption), stream what happens, then record it as a turn of the
    conversation and send that record. One turn at a time per conversation."""
    s = _services()
    turn_id = f"ui:{conversation_id}:{uuid4().hex[:8]}"
    config = _config(conversation_id)
    started, trace, sent = datetime.now(UTC), list[dict[str, Any]](), set[int]()
    async with s.locks[conversation_id]:
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
                    trace.append(node)
                    yield ServerSentEvent(data=node, event="node")
                if node is not None and node["phase"] != "start":
                    async for event in _calls(s, turn_id, sent):
                        yield event
        except Exception as error:
            # LangGraph 1.2.14 streaming with subgraphs=True re-raises, at the end, an error a
            # node's error handler already handled, after the run has finished and been
            # checkpointed (reproduced on a two-node graph, D35). A finished run is a turn.
            snapshot = await s.graph.aget_state(config)
            if snapshot.next and not snapshot.interrupts:
                yield ServerSentEvent(data={"error": repr(error)}, event="error")
                return
        snapshot = await s.graph.aget_state(config)
        waiting = snapshot.interrupts[0].value if snapshot.interrupts else None
        turn = {
            **events.outcome(snapshot.values),
            "turn_id": turn_id,
            "message": said,
            "trace": trace,
            "waiting": waiting,
        }
        if waiting is not None:  # the turn stops at a question: that question is its reply
            turn |= {"reply": waiting["question"], "status": "needs_input", "answer": None}
        async with s.pool.connection() as conn:
            await chats.record(conn, conversation_id, turn, started)
            calls = (await chats.turn_calls(conn, [turn_id])).get(turn_id, [])
    for row in calls:
        if row["id"] not in sent:  # the last calls settle as the turn ends
            yield ServerSentEvent(data=row, event="call")
    yield ServerSentEvent(
        data={**turn, "calls": calls, "started_at": started, "finished_at": datetime.now(UTC)},
        event="done",
    )


async def _calls(s: Services, turn_id: str, sent: set[int]) -> AsyncIterator[ServerSentEvent]:
    async with s.pool.connection() as conn:
        found = await events.new_calls(conn, turn_id, sent)
    for call in found:
        sent.add(call["id"])
        yield ServerSentEvent(data=call, event="call")


def _config(conversation_id: str) -> Any:
    return {
        "configurable": {"thread_id": conversation_id},
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
