"""The web API (M5b) against the real test database: the server's own lifespan on `ara_test`, called
in-process over ASGI. Model calls are blocked with injected faults (ARA_FAULTS), so a whole turn
streams without a billable request."""

import asyncio
import json
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any

import httpx
import pytest
from langchain_core.messages import HumanMessage

from ara import faults
from ara.api import conversations, events, server
from ara.api.runs import Run
from ara.db import prune
from ara.db.pool import Pool
from ara.graph import app
from ara.memory import library, store
from ara.memory.store import Remembered
from tests.unit.test_app import request
from tests.unit.test_memory import cache
from tests.unit.test_search import setup


@pytest.fixture
async def client(
    pool: Pool, test_database: str, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[httpx.AsyncClient]:
    for key in ("OPENAI_API_KEY", "DEEPSEEK_API_KEY", "COHERE_API_KEY"):  # never used: faults
        monkeypatch.setenv(key, "unused")
    monkeypatch.setenv("ARA_DATABASE_URL", test_database)
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setenv("ARA_FAULTS", "embed=refused,understand=refused")
    faults.reset()
    async with server.lifespan(server.api):
        transport = httpx.ASGITransport(app=server.api)
        async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1") as http:
            yield http


def sse(body: str) -> list[tuple[str, Any]]:
    """(event, data) pairs from a text/event-stream body."""
    found = []
    for block in body.strip().split("\n\n"):
        fields = dict(line.split(": ", 1) for line in block.splitlines() if ": " in line)
        found.append((fields.get("event", "message"), json.loads(fields["data"])))
    return found


def test_task_events_name_the_graph_the_node_and_the_phase() -> None:
    start = {"ns": (), "data": {"id": "1", "name": "understand", "input": {}, "triggers": []}}
    assert events.node_event(start) == {"graph": "main", "node": "understand", "phase": "start"}
    ended = {
        "ns": ("read:1f0",),
        "data": {"id": "2", "name": "select", "error": None, "result": {}},
    }
    assert events.node_event(ended) == {"graph": "read", "node": "select", "phase": "end"}
    handled = {"ns": ("discover:9a",), "data": {"id": "3", "name": "__error_handler__researcher"}}
    assert events.node_event(handled) == {
        "graph": "discover",
        "node": "researcher",
        "phase": "degraded",
    }
    assert events.node_event({"ns": (), "data": {"id": "4", "name": "__start__"}}) is None
    assert events.problems({"ns": (), "data": {"result": {"problems": ["p"]}}}) == ["p"]
    assert events.problems(start) == []


async def test_the_graphs_are_drawn_from_the_compiled_code(client: httpx.AsyncClient) -> None:
    drawn = (await client.get("/api/graph")).json()
    assert set(drawn) == {"main", "discover", "read", "answer"}
    assert "researcher" in drawn["discover"] and "search_instead" in drawn["main"]


async def test_a_conversation_keeps_each_turn_with_its_path(client: httpx.AsyncClient) -> None:
    """Memory and understand are refused by injected faults: the turn degrades, never raises, and
    is recorded with the nodes it went through (D36)."""
    conversation = (await client.post("/api/conversations")).json()["id"]
    assert (await client.get("/api/conversations")).json() == [], "listed from its first message"
    response = await client.post(
        f"/api/conversations/{conversation}/messages",
        json={"text": "Find papers on KV-cache eviction for long-context inference, please"},
    )
    assert response.headers["content-type"].startswith("text/event-stream")
    streamed = sse(response.text)
    nodes = [(d["node"], d["phase"]) for kind, d in streamed if kind == "node"]
    assert ("load_context", "degraded") in nodes and ("understand", "degraded") in nodes
    kind, done = streamed[-1]
    assert kind == "done" and done["status"] == "failed" and done["waiting"] is None
    assert done["reply"].startswith("I could not process that message just now")
    assert ("understand", "degraded") in [(t["node"], t["phase"]) for t in done["trace"]]

    [listed] = (await client.get("/api/conversations")).json()
    assert listed["id"] == conversation
    assert listed["title"] == "Find papers on KV-cache eviction for long-context…"
    kept = (await client.get(f"/api/conversations/{conversation}")).json()
    [turn] = kept["turns"]
    assert turn["message"].startswith("Find papers") and turn["reply"] == done["reply"]
    assert turn["trace"] == done["trace"] and turn["calls"] == []

    renamed = await client.patch(f"/api/conversations/{conversation}", json={"title": "KV cache"})
    assert renamed.json()["title"] == "KV cache"
    assert (await client.delete(f"/api/conversations/{conversation}")).status_code == 200
    assert (await client.get(f"/api/conversations/{conversation}")).status_code == 404
    assert (await client.delete(f"/api/conversations/{conversation}")).status_code == 404


def test_a_title_is_the_first_message_cut_at_a_word() -> None:
    assert conversations.title("  Read   2210.03629 ") == "Read 2210.03629"
    long = "word " * 20
    assert conversations.title(long).endswith("word…") and len(conversations.title(long)) <= 61


async def test_documents_memory_and_evaluations_read_the_database(
    client: httpx.AsyncClient, pool: Pool
) -> None:
    gateway, _, second = await setup(pool)
    async with pool.connection() as conn:
        await conn.execute("TRUNCATE store, store_vectors")  # created by the server's lifespan
        await library.record_read(conn, server.USER, [second])
    document = (await client.get("/api/papers/arxiv:2401.00002v1/document")).json()
    assert document["title"] == "Cache" and len(document["passages"]) == 3
    assert document["passages"][1]["text"] == "Evicting keys by attention score reduces memory."
    assert (await client.get("/api/papers/arxiv:9999.99999v1/document")).status_code == 404
    assert (await client.get("/api/memory/facts")).json() == {"items": [], "total": 0}
    shelf = (await client.get("/api/memory/library")).json()
    assert shelf["total"] == 1 and shelf["items"][0]["arxiv_id"] == "2401.00002"
    assert (await client.get("/api/memory/library", params={"q": "cach"})).json()["total"] == 1
    assert (await client.get("/api/memory/library", params={"q": "2401.0000"})).json()["total"] == 1
    assert (await client.get("/api/memory/library", params={"q": "zebra"})).json()["total"] == 0
    paper = (await client.get("/api/memory/library/2401.00002")).json()
    assert paper["title"] == "Cache" and paper["read_in"] == []
    assert (await client.get("/api/memory/library/9999.99999")).status_code == 404
    assert (await client.delete("/api/memory/unknown")).json() == {"forgotten": "unknown"}
    assert (await client.get("/api/evals")).json() == []
    described = (await client.get("/api/evals/suites")).json()
    assert [x["id"] for x in described["suites"]] == ["s1", "s2", "s3", "s4", "s5", "s6", "s7"]
    s6 = described["suites"][5]
    assert s6["items"] == {"dev": 11, "test": 8} and "passed" in s6["metrics"]
    assert (await client.get("/api/evals/nope")).status_code == 404
    await gateway.aclose()


async def test_a_turn_runs_on_when_its_page_leaves_and_can_be_rejoined() -> None:
    """Followers read a run's events from the start, however late they come (D38)."""
    run = Run("c", "ui:c:1", "hello", "messages")
    early = run.follow()
    await run.emit("node", {"node": "understand"})
    late = run.follow()
    await run.emit("done", {"reply": "hi"})
    await run.finish()
    assert [e async for e in early] == [e async for e in late] == run.events


async def wait_running(client: httpx.AsyncClient, conversation: str) -> dict[str, Any]:
    for _ in range(200):
        opened = await client.get(f"/api/conversations/{conversation}")
        if opened.status_code == 200 and (running := opened.json()["running"]):
            return dict(running)
        await asyncio.sleep(0.02)
    raise AssertionError("the turn never started")


async def test_a_stopped_turn_leaves_the_conversation_as_it_was(
    client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """understand fails with 503 and is retried with backoff (no request is sent: the fault trips
    before the gateway reserves anything), so the turn is still running when it is stopped. The
    first turn of a conversation, stopped, removes the conversation; a later one is undone back
    to the checkpoint before it (D38)."""
    conversation = (await client.post("/api/conversations")).json()["id"]
    monkeypatch.setenv("ARA_FAULTS", "embed=refused,understand=503")
    sending = asyncio.create_task(
        client.post(f"/api/conversations/{conversation}/messages", json={"text": "first try"})
    )
    running = await wait_running(client, conversation)
    assert running["message"] == "first try" and running["kind"] == "messages"
    while not (listed := (await client.get("/api/conversations")).json()):
        await asyncio.sleep(0.02)  # stored as the run starts, a moment after it is claimed
    assert [c["running"] for c in listed] == [True]
    second = await client.post(f"/api/conversations/{conversation}/messages", json={"text": "x"})
    assert second.status_code == 409, "one turn at a time"
    stopped = (await client.post(f"/api/conversations/{conversation}/stop")).json()
    assert stopped["message"] == "first try" and stopped["conversation_removed"]
    assert stopped["spent_usd"] == 0
    assert sse((await sending).text)[-1][0] == "stopped"
    assert (await client.get(f"/api/conversations/{conversation}")).status_code == 404
    assert (await client.get(f"/api/conversations/{conversation}/live")).status_code == 404
    assert (await client.post(f"/api/conversations/{conversation}/stop")).status_code == 404

    monkeypatch.setenv("ARA_FAULTS", "embed=refused,understand=refused")
    await client.post(f"/api/conversations/{conversation}/messages", json={"text": "kept turn"})
    config: Any = {"configurable": {"thread_id": conversation}}
    graph = server._services().graph
    kept = (await graph.aget_state(config)).values["messages"]
    monkeypatch.setenv("ARA_FAULTS", "embed=refused,understand=503")
    sending = asyncio.create_task(
        client.post(f"/api/conversations/{conversation}/messages", json={"text": "undone"})
    )
    await wait_running(client, conversation)
    while len((await graph.aget_state(config)).values["messages"]) == len(kept):
        await asyncio.sleep(0.02)  # the graph has taken the message: there is something to undo
    rejoined = asyncio.create_task(client.get(f"/api/conversations/{conversation}/live"))
    stopped = (await client.post(f"/api/conversations/{conversation}/stop")).json()
    assert not stopped["conversation_removed"]
    assert sse((await rejoined).text)[0][0] == "started"
    await sending
    assert (await graph.aget_state(config)).values["messages"] == kept
    turns = (await client.get(f"/api/conversations/{conversation}")).json()["turns"]
    assert [t["message"] for t in turns] == ["kept turn"]


async def test_a_stopped_answer_leaves_the_question_waiting(
    client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Stopping the turn that answers a question puts the question back: it waits again, asked
    by re-running clarify, which calls no model (D38)."""
    s = server._services()
    conversation = (await client.post("/api/conversations")).json()["id"]
    config: Any = {"configurable": {"thread_id": conversation}}
    await s.graph.aupdate_state(
        config,
        {
            "messages": [HumanMessage("find the best paper to cut my agent's cost")],
            **app.fresh_turn(),
            "request": request(clarification="Which cost?"),
        },
        as_node="understand",
    )
    await s.graph.ainvoke(None, config, context=s.context("ui:test"))
    monkeypatch.setenv("ARA_FAULTS", "embed=refused,understand=503")
    answering = asyncio.create_task(
        client.post(f"/api/conversations/{conversation}/resume", json={"answer": "API spend"})
    )
    await wait_running(client, conversation)
    while (await s.graph.aget_state(config)).interrupts:
        await asyncio.sleep(0.02)  # the answer has been taken
    stopped = (await client.post(f"/api/conversations/{conversation}/stop")).json()
    assert stopped["kind"] == "resume" and stopped["message"] == "API spend"
    await answering
    [waiting] = (await s.graph.aget_state(config)).interrupts
    assert waiting.value == {"kind": "clarify", "question": "Which cost?"}


async def test_memory_links_facts_and_papers_to_their_conversations(
    client: httpx.AsyncClient, pool: Pool
) -> None:
    gateway, _, second = await setup(pool)
    async with pool.connection() as conn:
        await conn.execute("TRUNCATE store, store_vectors")
        await library.record_read(conn, server.USER, [second])
        await conversations.touch(conn, "c1", server.USER, "Read the cache paper")
        turn = {
            "turn_id": "ui:c1:1",
            "message": "Read the cache paper",
            "reply": "r",
            "status": "complete",
            "intent": "read",
            "answer": None,
            "papers": [],
            "read": [{"arxiv_id": "2401.00002", "title": "Cache"}],
            "problems": [],
            "trace": [],
            "waiting": None,
            "parts": [],
        }
        await conversations.record(conn, "c1", turn, datetime.now(UTC))
    s = server._services()
    await cache(pool, {"s": 1})  # facts are indexed by statement (D39): no embedding request
    for n, thread in enumerate(["c1", "gone"]):
        fact = Remembered(
            key=f"k{n}", statement="s", quote="q", thread=thread, turn=0, day="2026-10-10"
        )
        await store.remember(s.memory, server.USER, fact)
    facts = (await client.get("/api/memory/facts")).json()
    assert [f["key"] for f in facts["items"]] == ["k0", "k1"], "oldest first"
    assert [f["conversation"] for f in facts["items"]] == [
        {"id": "c1", "title": "Read the cache paper"},
        None,
    ]
    assert (await client.get("/api/memory/facts", params={"offset": 1})).json()["total"] == 2
    [read_in] = (await client.get("/api/memory/library/2401.00002")).json()["read_in"]
    assert (read_in["conversation_id"], read_in["title"]) == ("c1", "Read the cache paper")
    await gateway.aclose()


async def test_a_turn_that_fails_outside_the_graph_is_undone(
    client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Recording the turn fails: the turn is undone like a stopped one and the stream ends with
    an error that carries the message back (D39)."""

    async def broken(*_: Any) -> None:
        raise RuntimeError("disk full")

    monkeypatch.setattr(conversations, "record", broken)
    conversation = (await client.post("/api/conversations")).json()["id"]
    sent = await client.post(f"/api/conversations/{conversation}/messages", json={"text": "hi"})
    kind, data = sse(sent.text)[-1]
    assert kind == "error" and data["message"] == "hi" and data["conversation_removed"]
    assert data["error"] == "The turn failed: an unexpected error"
    assert (await client.get(f"/api/conversations/{conversation}")).status_code == 404
    assert server._services().runs == {}


async def test_stop_lets_a_turn_that_has_replied_finish(
    client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Past respond the reply is written: Stop waits for the commit and the turn is kept (D39)."""
    monkeypatch.setenv("ARA_FAULTS", "embed=refused,understand=503")
    conversation = (await client.post("/api/conversations")).json()["id"]
    sending = asyncio.create_task(
        client.post(f"/api/conversations/{conversation}/messages", json={"text": "kept"})
    )
    await wait_running(client, conversation)
    server._services().runs[conversation].committing = True  # as if respond had started
    assert (await client.post(f"/api/conversations/{conversation}/stop")).status_code == 409
    assert sse((await sending).text)[-1][0] == "done"
    turns = (await client.get(f"/api/conversations/{conversation}")).json()["turns"]
    assert [t["message"] for t in turns] == ["kept"]


async def test_startup_repairs_a_conversation_left_in_the_middle_of_a_turn(
    client: httpx.AsyncClient,
) -> None:
    """The server was killed mid-turn: the thread goes back to its last turn's end; a
    conversation whose first turn never finished is removed (D39)."""
    s = server._services()
    kept = (await client.post("/api/conversations")).json()["id"]
    await client.post(f"/api/conversations/{kept}/messages", json={"text": "first"})
    lost = (await client.post("/api/conversations")).json()["id"]
    async with s.pool.connection() as conn:
        await conversations.touch(conn, lost, server.USER, "never answered")
    for thread in (kept, lost):
        config: Any = {"configurable": {"thread_id": thread}}
        await s.graph.aupdate_state(
            config,
            {
                "messages": [HumanMessage("killed mid-turn")],
                **app.fresh_turn(),
                "request": request(intent="discover"),
            },
            as_node="understand",
        )
        assert (await s.graph.aget_state(config)).next == ("discover",)
    assert sorted(await server.repair(s)) == sorted([kept, lost])
    state = await s.graph.aget_state({"configurable": {"thread_id": kept}})
    assert state.next == () and state.values["messages"][-1].text.startswith("I could not")
    assert (await client.get(f"/api/conversations/{lost}")).status_code == 404
    assert await server.repair(s) == []


async def test_only_this_machines_names_reach_the_api(client: httpx.AsyncClient) -> None:
    """A page that rebinds its own domain to 127.0.0.1 is refused (D39)."""
    assert (await client.get("/api/conversations")).status_code == 200
    rebound = await client.get("/api/conversations", headers={"Host": "evil.example"})
    assert rebound.status_code == 400


async def test_prune_removes_evaluation_leftovers_only(
    client: httpx.AsyncClient, pool: Pool
) -> None:
    """Threads no conversation holds, and simulated users' memory and library, go; the app's
    own conversations stay (D39)."""
    s = server._services()
    kept = (await client.post("/api/conversations")).json()["id"]
    await client.post(f"/api/conversations/{kept}/messages", json={"text": "mine"})
    gateway, first, _ = await setup(pool)
    async with pool.connection() as conn:
        await prune.prune(conn)  # what other tests left
        await library.record_read(conn, "s6-run:scenario", [first])
        await library.record_read(conn, server.USER, [first])
    await s.graph.aupdate_state(
        {"configurable": {"thread_id": "s6-run:scenario:0"}},
        {"messages": [HumanMessage("eval")], **app.fresh_turn(), "request": request()},
        as_node="understand",
    )
    await s.memory.aput(("users", "s6-run:scenario", "episodes"), "e", {"need": "x"}, index=False)
    await s.memory.aput(("users", server.USER, "episodes"), "e", {"need": "x"}, index=False)
    async with pool.connection() as conn:
        assert await prune.leftovers(conn) == {"threads": 1, "memory items": 1, "library rows": 1}
        await prune.prune(conn)
        assert await prune.leftovers(conn) == {"threads": 0, "memory items": 0, "library rows": 0}
        assert [p.arxiv_id for p in await library.papers(conn, server.USER)] == ["2401.00001"]
    assert (await s.graph.aget_state({"configurable": {"thread_id": kept}})).values["messages"]
    assert await s.memory.aget(("users", server.USER, "episodes"), "e") is not None
    await gateway.aclose()
