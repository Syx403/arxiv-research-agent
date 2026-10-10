"""The local web app (DESIGN §9, M5b): FastAPI on 127.0.0.1 for one user, no login. A turn streams
as Server-Sent Events: node starts and ends (subgraphs included), each model call from the ledger,
then the finished turn or the question the graph waits on. The built UI (ui/dist) is served at /.

    uv run ara serve        # http://127.0.0.1:8000
"""

import asyncio
import logging
import re
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import AsyncExitStack, aclosing, asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Annotated, Any, cast
from uuid import uuid4

from fastapi import Depends, FastAPI, HTTPException
from fastapi.sse import EventSourceResponse, ServerSentEvent
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.store.base import BaseStore
from langgraph.types import Command, StateSnapshot
from pydantic import BaseModel
from starlette.middleware.trustedhost import TrustedHostMiddleware

from ara.api import conversations as chats
from ara.api import events
from ara.api.runs import Run
from ara.arxiv.client import ArxivClient
from ara.db.migrate import migrate
from ara.db.pool import Pool, make_pool
from ara.graph import app as conversation
from ara.graph.reliability import why
from ara.graph.state import Context
from ara.llm.gateway import Gateway
from ara.llm.ledger import Ledger, Scope
from ara.memory import library, store
from ara.rag.ingest import PIPELINE_VERSION
from ara.rag.sources import ParsedPaper
from ara.settings import ROOT, configure_tracing, get_settings
from evals import report
from evals.catalog import catalog

HOST, PORT = "127.0.0.1", 8000
USER = "local"  # single user, no login (DESIGN §9)
COMMIT = ("respond", "remember")  # past these a turn has replied: Stop lets it finish (D39)
UI = ROOT / "ui" / "dist"
log = logging.getLogger(__name__)

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
    runs: dict[str, Run]  # the running turn of each conversation (D38)

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
        services = Services(pool, gateway, arxiv, memory, saver, graph, {})
        if repaired := await repair(services):
            log.warning("repaired conversations left in the middle of a turn: %s", repaired)
        try:
            yield
        finally:
            await _stop_all(services)
            services = None


async def _stop_all(s: Services) -> None:
    """Shutting down stops running turns as the stop button does."""
    tasks = [r.task for r in s.runs.values() if r.task is not None]
    for task in tasks:
        task.cancel()
    if tasks:
        await asyncio.wait(tasks)


api = FastAPI(title="ARA", lifespan=lifespan)
# Only this machine's own names: a web page that rebinds its domain to 127.0.0.1 cannot drive the
# API (and spend on it) from the user's browser (D39).
api.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost"])


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
    """The user's conversations, most recent first (the sidebar), each marked if a turn is
    running in it."""
    s = _services()
    async with s.pool.connection() as conn:
        found = await chats.listed(conn, USER)
    return [{**c, "running": c["id"] in s.runs} for c in found]


@api.post("/api/conversations")
def new_conversation() -> dict[str, str]:
    """An id for a new conversation; it is stored, and listed, from its first message."""
    return {"id": uuid4().hex}


@api.get("/api/conversations/{conversation_id}")
async def opened(conversation_id: str) -> dict[str, Any]:
    """Every turn of a conversation, each with its reply, answer, evidence, papers, path through
    the graph and model calls, and the question it waits on, if any."""
    s = _services()
    async with s.pool.connection() as conn:
        found = await chats.opened(conn, conversation_id, USER)
    run = s.runs.get(conversation_id)
    if found is None:
        if run is None:
            raise HTTPException(404, f"no conversation {conversation_id}")
        found = {"id": conversation_id, "title": chats.title(run.message), "turns": []}
    return {**found, "running": run.summary() if run else None}


@api.patch("/api/conversations/{conversation_id}")
async def renamed(conversation_id: str, body: Rename) -> dict[str, str]:
    async with _services().pool.connection() as conn:
        if not await chats.rename(conn, conversation_id, USER, body.title):
            raise HTTPException(404, f"no conversation {conversation_id}")
    return {"id": conversation_id, "title": body.title}


@api.delete("/api/conversations/{conversation_id}")
async def deleted(conversation_id: str) -> dict[str, str]:
    """The conversation, its turns and its checkpoints; the ledger keeps its spend (D36). A turn
    running in it is stopped first."""
    s = _services()
    if (run := s.runs.get(conversation_id)) is not None and run.task is not None:
        run.task.cancel()
        await asyncio.wait({run.task})
    async with s.pool.connection() as conn:
        if not await chats.delete(conn, conversation_id, USER):
            raise HTTPException(404, f"no conversation {conversation_id}")
    await s.checkpointer.adelete_thread(conversation_id)
    return {"deleted": conversation_id}


async def _sent(conversation_id: str, body: Message) -> Run:
    return _start(conversation_id, "messages", body.text, {"messages": [HumanMessage(body.text)]})


async def _answered(conversation_id: str, body: Resume) -> Run:
    return _start(conversation_id, "resume", body.answer, Command(resume=body.answer))


async def _running(conversation_id: str) -> Run:
    run = _services().runs.get(conversation_id)
    if run is None:
        raise HTTPException(404, f"no running turn in {conversation_id}")
    return run


# The run is started (or found) in a dependency, before the stream begins, so a refusal is an
# ordinary HTTP error rather than a broken stream.
@api.post("/api/conversations/{conversation_id}/messages", response_class=EventSourceResponse)
async def message(run: Annotated[Run, Depends(_sent)]) -> AsyncIterator[ServerSentEvent]:
    async for event in _follow(run):
        yield event


@api.post("/api/conversations/{conversation_id}/resume", response_class=EventSourceResponse)
async def resume(run: Annotated[Run, Depends(_answered)]) -> AsyncIterator[ServerSentEvent]:
    async for event in _follow(run):
        yield event


@api.get("/api/conversations/{conversation_id}/live", response_class=EventSourceResponse)
async def live(run: Annotated[Run, Depends(_running)]) -> AsyncIterator[ServerSentEvent]:
    """The running turn's events from its start: how a page rejoins a turn it left (D38)."""
    async for event in _follow(run):
        yield event


@api.post("/api/conversations/{conversation_id}/stop")
async def stop(conversation_id: str) -> dict[str, Any]:
    """End the running turn now. It is not recorded and the conversation is put back as it was
    before it, so the message can be edited and sent again; its spend stays in the ledger (D38)."""
    run = await _running(conversation_id)
    assert run.task is not None  # set when the run is claimed
    if not run.committing:  # the reply is written and being committed: let it finish (D39)
        run.task.cancel()
    await asyncio.wait({run.task})
    stopped = next((data for event, data in run.events if event == "stopped"), None)
    if stopped is None:  # it finished before the stop arrived
        raise HTTPException(409, "the turn had already finished")
    return dict(stopped)


def _start(conversation_id: str, kind: str, said: str, payload: Any) -> Run:
    """One turn at a time per conversation; the check and the claim happen with no await between
    them, so two requests cannot both start one."""
    s = _services()
    if conversation_id in s.runs:
        raise HTTPException(409, "a turn is already running in this conversation")
    run = Run(conversation_id, f"ui:{conversation_id}:{uuid4().hex[:8]}", said, kind)
    s.runs[conversation_id] = run
    run.task = asyncio.create_task(_run(s, run, payload))
    return run


async def _follow(run: Run) -> AsyncIterator[ServerSentEvent]:
    async for event, data in run.follow():
        yield ServerSentEvent(data=data, event=event)


async def _run(s: Services, run: Run, payload: Any) -> None:
    """Run the turn, keep what happens as events, record the finished turn and send its record.
    A turn that is stopped, or fails in any way, is undone: the conversation goes back to its
    state before the turn and the message to the composer (D38, D39). Every run ends with one
    of done, stopped or error."""
    config = _config(run.conversation)
    started, trace, sent, noted = datetime.now(UTC), list[dict[str, Any]](), set[int](), set[str]()
    before: StateSnapshot | None = None
    try:
        async with s.pool.connection() as conn:
            await chats.touch(conn, run.conversation, USER, run.message)
        before = await s.graph.aget_state(config)
        await run.emit("started", {**run.summary(), "started_at": started})
        try:
            async with aclosing(
                cast(
                    AsyncGenerator[Any],
                    s.graph.astream(
                        payload,
                        config,
                        context=s.context(run.turn_id),
                        stream_mode="tasks",
                        subgraphs=True,
                        version="v2",
                    ),
                )
            ) as stream:
                async for part in stream:
                    await _observe(s, run, dict(part), trace, sent, noted)
        except Exception:
            # LangGraph 1.2.14 streaming with subgraphs=True re-raises, at the end, an error a
            # node's error handler already handled, after the run has finished and been
            # checkpointed (reproduced on a two-node graph, D35). A finished run is a turn.
            snapshot = await s.graph.aget_state(config)
            if snapshot.next and not snapshot.interrupts:
                raise
        await _record(s, run, config, started, trace, sent)
    except asyncio.CancelledError:
        await _undone(s, run, before, "stopped", {})
    except Exception as error:
        log.exception("turn %s failed", run.turn_id)
        await _undone(s, run, before, "error", {"error": "The turn failed: " + why(error)})
    finally:
        s.runs.pop(run.conversation, None)
        await run.finish()


async def _observe(
    s: Services,
    run: Run,
    part: dict[str, Any],
    trace: list[dict[str, Any]],
    sent: set[int],
    noted: set[str],
) -> None:
    """A task event: the node's start or end, what it reported failed, and model calls settled."""
    if (node := events.node_event(part)) is None:
        return
    trace.append(node)
    if node["graph"] == "main" and node["node"] in COMMIT and node["phase"] == "start":
        run.committing = True
    await run.emit("node", node)
    for text in events.problems(part):
        if text not in noted:
            noted.add(text)
            await run.emit("problem", {"text": text})
    if node["phase"] != "start":
        async with s.pool.connection() as conn:
            found = await events.new_calls(conn, run.turn_id, sent)
        for call in found:
            sent.add(call["id"])
            await run.emit("call", call)


async def _record(
    s: Services,
    run: Run,
    config: Any,
    started: datetime,
    trace: list[dict[str, Any]],
    sent: set[int],
) -> None:
    snapshot = await s.graph.aget_state(config)
    waiting = snapshot.interrupts[0].value if snapshot.interrupts else None
    turn = {
        **events.outcome(snapshot.values),
        "turn_id": run.turn_id,
        "message": run.message,
        "trace": trace,
        "waiting": waiting,
    }
    if waiting is not None:  # the turn stops at a question: that question is its reply
        turn |= {"reply": waiting["question"], "status": "needs_input", "answer": None, "parts": []}
    async with s.pool.connection() as conn:
        await chats.record(conn, run.conversation, turn, started)
        calls = (await chats.turn_calls(conn, [run.turn_id])).get(run.turn_id, [])
    for row in calls:
        if row["id"] not in sent:  # the last calls settle as the turn ends
            await run.emit("call", row)
    finished = datetime.now(UTC)
    await run.emit("done", {**turn, "calls": calls, "started_at": started, "finished_at": finished})


async def _undone(
    s: Services, run: Run, before: StateSnapshot | None, event: str, extra: dict[str, Any]
) -> None:
    """Undo the turn and say so; if undoing fails too, say what failed."""
    try:
        removed = await restore(s, run.conversation, before, run.turn_id)
        async with s.pool.connection() as conn:
            spent = await chats.charged(conn, run.turn_id)
    except Exception as error:
        log.exception("undoing turn %s failed", run.turn_id)
        extra = {"error": f"The turn could not be undone: {why(error)}"}
        event, removed, spent = "error", False, 0.0
    await run.emit(
        event,
        {**run.summary(), **extra, "spent_usd": spent, "conversation_removed": removed},
    )


async def restore(
    s: Services, conversation_id: str, before: StateSnapshot | None, turn_id: str
) -> bool:
    """Put a conversation back as it was at `before`: that checkpoint becomes the latest again (a
    question it was waiting on is asked again, by clarify, which calls no model). With nothing
    before (the turn started the conversation), the thread goes, and so does the conversation if
    it has no recorded turn; returns whether it went. `before` None with a thread also means the
    graph never ran: nothing to undo."""
    config = _config(conversation_id)
    if before is not None and before.config["configurable"].get("checkpoint_id"):
        await s.graph.aupdate_state(before.config, None, as_node="__copy__")
        if before.interrupts:
            await s.graph.ainvoke(None, config, context=s.context(turn_id))
        return False
    if before is not None:
        await s.checkpointer.adelete_thread(conversation_id)
    async with s.pool.connection() as conn:
        return await chats.forget_if_empty(conn, conversation_id)


async def repair(s: Services) -> list[str]:
    """At startup: a conversation whose thread stopped in the middle of a turn (the server was
    killed while it ran) goes back to the last checkpoint between turns, or to a question it
    was waiting on (D39). Returns the conversations repaired."""
    async with s.pool.connection() as conn:
        ids = [c["id"] for c in await chats.listed(conn, USER)]
    repaired = []
    for conversation_id in ids:
        latest = await s.graph.aget_state(_config(conversation_id))
        if not latest.next or latest.interrupts:
            continue
        settled = None
        async for snapshot in s.graph.aget_state_history(_config(conversation_id)):
            if not snapshot.next or snapshot.interrupts:
                settled = snapshot
                break
        await restore(s, conversation_id, settled, f"ui:{conversation_id}:repair")
        repaired.append(conversation_id)
    return repaired


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


@api.get("/api/evals/suites")
def suites() -> dict[str, Any]:
    """What each suite tests, its data, grading and metrics, and the shared protocol (D38)."""
    return catalog()


@api.get("/api/evals/{run_id}")
async def evaluation(run_id: str) -> dict[str, Any]:
    try:
        return await report.summary(_services().pool, run_id)
    except LookupError as error:
        raise HTTPException(404, str(error)) from error


PAGE = 50  # items per page of a memory list


@api.get("/api/memory/facts")
async def facts(offset: int = 0, limit: int = PAGE) -> dict[str, Any]:
    """What is remembered about the user, oldest first (new facts are added at the end), each
    with the conversation it came from when that conversation is still kept (D38)."""
    s = _services()
    items = await s.memory.asearch(store.profile_namespace(USER), limit=store.ALL)
    ordered = sorted(items, key=lambda i: i.created_at)
    page = [store.Remembered(**i.value) for i in ordered[offset : offset + limit]]
    async with s.pool.connection() as conn:
        named = await chats.titles(conn, [f.thread for f in page])
    return {
        "items": [
            {
                **f.model_dump(),
                "conversation": {"id": f.thread, "title": named[f.thread]}
                if f.thread in named
                else None,
            }
            for f in page
        ],
        "total": len(ordered),
    }


@api.get("/api/memory/library")
async def shelf(q: str = "", offset: int = 0, limit: int = PAGE) -> dict[str, Any]:
    """The papers read, in the order first read, matched by title or id."""
    async with _services().pool.connection() as conn:
        items, total = await library.shelf(conn, USER, q, offset, min(limit, 200))
    return {"items": items, "total": total}


@api.get("/api/memory/library/{arxiv_id}")
async def paper(arxiv_id: str) -> dict[str, Any]:
    """One paper read: its abstract, dates, and the conversations that read it."""
    async with _services().pool.connection() as conn:
        found = await library.entry(conn, USER, arxiv_id)
        if found is None:
            raise HTTPException(404, f"{arxiv_id} is not in the library")
        return {**found, "read_in": await chats.reading(conn, USER, arxiv_id)}


@api.delete("/api/memory/{key}")
async def forget(key: str) -> dict[str, str]:
    await store.forget(_services().memory, USER, key)
    return {"forgotten": key}


if UI.exists():  # built by `make ui`; during development Vite serves the UI instead
    api.mount("/", StaticFiles(directory=UI, html=True), name="ui")
