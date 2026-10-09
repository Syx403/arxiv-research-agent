"""What the UI is told during and after a turn (DESIGN §9, M5b): node starts and ends from
LangGraph's task stream (subgraphs included), model calls from the ledger, a question the graph
waits on, and the finished turn. Plain functions of their inputs, unit-tested without a model."""

from typing import Any

from langchain_core.messages import AIMessage, HumanMessage

from ara.db.pool import Connection

HANDLER = "__error_handler__"
CALLS = """
SELECT id, stage, model, status, latency_ms, input_tokens, cached_tokens, output_tokens,
       reasoning_tokens, cost_usd
FROM llm_calls WHERE turn_id = %s AND status <> 'reserved' AND NOT id = ANY(%s) ORDER BY id
"""


def node_event(part: dict[str, Any]) -> dict[str, Any] | None:
    """A task start or result: which graph (the main graph, or the subgraph a main node runs), which
    node, and whether it started, ended, failed, or was handled by its error handler (degraded)."""
    data, ns = part["data"], part["ns"]
    name: str = data["name"]
    if name.startswith("__"):
        if not name.startswith(HANDLER):
            return None  # LangGraph's own start and end tasks
        return {"graph": _graph(ns), "node": name.removeprefix(HANDLER), "phase": "degraded"}
    if "result" not in data:
        return {"graph": _graph(ns), "node": name, "phase": "start"}
    return {
        "graph": _graph(ns),
        "node": name,
        "phase": "failed" if data.get("error") else "end",
    }


def _graph(ns: tuple[str, ...]) -> str:
    """ "discover:1f0…" → "discover": a subgraph is named by the main node that runs it."""
    return ns[-1].split(":")[0] if ns else "main"


async def new_calls(conn: Connection, turn_id: str, sent: set[int]) -> list[dict[str, Any]]:
    """This turn's model calls that have finished since the ones already `sent` (ledger rows:
    stage, model, latency, tokens, cache hits, cost). A call in flight waits for its result."""
    cursor = await conn.execute(CALLS, (turn_id, list(sent)))
    return [
        {**row, "cost_usd": float(row["cost_usd"]) if row["cost_usd"] is not None else None}
        for row in await cursor.fetchall()
    ]


def messages(values: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {"role": "user" if isinstance(m, HumanMessage) else "assistant", "text": m.text}
        for m in values.get("messages", [])
        if isinstance(m, HumanMessage | AIMessage)
    ]


def turn(values: dict[str, Any]) -> dict[str, Any]:
    """The finished turn: the reply, and what the evidence page shows (papers listed and read,
    the answer with its cited evidence and the lines the verifier rejected)."""
    request, delivered = values.get("request"), values.get("answer")
    return {
        "messages": messages(values),
        "status": values.get("status", "complete"),
        "problems": values.get("problems", []),
        "intent": request.intent if request else None,
        "papers": [p.model_dump() for p in values.get("papers", [])],
        "read": [p.model_dump() for p in values.get("read", [])],
        "answer": delivered.model_dump() if delivered else None,
    }
