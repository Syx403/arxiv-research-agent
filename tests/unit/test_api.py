"""The web API (M5b) against the real test database: the server's own lifespan on `ara_test`, called
in-process over ASGI. Model calls are blocked with injected faults (ARA_FAULTS), so a whole turn
streams without a billable request."""

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest

from ara import faults
from ara.api import conversations, events, server
from ara.db.pool import Pool
from ara.memory import library
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
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:
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
        await library.record_read(conn, server.USER, [second])
    document = (await client.get("/api/papers/arxiv:2401.00002v1/document")).json()
    assert document["title"] == "Cache" and len(document["passages"]) == 3
    assert document["passages"][1]["text"] == "Evicting keys by attention score reduces memory."
    assert (await client.get("/api/papers/arxiv:9999.99999v1/document")).status_code == 404
    remembered = (await client.get("/api/memory")).json()
    assert remembered["facts"] == [] and remembered["library"][0]["arxiv_id"] == "2401.00002"
    assert (await client.delete("/api/memory/unknown")).json() == {"forgotten": "unknown"}
    assert (await client.get("/api/evals")).json() == []
    assert (await client.get("/api/evals/nope")).status_code == 404
    await gateway.aclose()
