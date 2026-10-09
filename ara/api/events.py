"""What the UI is told during and after a turn (DESIGN §9, M5b, D36): node starts and ends from
LangGraph's task stream (subgraphs included), model calls from the ledger, and what the turn
produced. Plain functions of their inputs, unit-tested without a model."""

from typing import Any

from langchain_core.messages import AIMessage

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


def problems(part: dict[str, Any]) -> list[str]:
    """What a finished task recorded as failed, so the page can say so while the turn runs."""
    result = part["data"].get("result")
    return list(result.get("problems", [])) if isinstance(result, dict) else []


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


def outcome(values: dict[str, Any]) -> dict[str, Any]:
    """What a finished turn (or a turn waiting on a question) produced, as the UI shows it: the
    reply, the papers listed and read, the answer with its cited evidence and the lines the
    verifier rejected, and anything that failed."""
    request, delivered = values.get("request"), values.get("answer")
    replies = [m for m in values.get("messages", []) if isinstance(m, AIMessage)]
    return {
        "reply": replies[-1].text if replies else "",
        "status": values.get("status", "complete"),
        "intent": request.intent if request else None,
        "papers": [p.model_dump() for p in values.get("papers", [])],
        "read": [p.model_dump() for p in values.get("read", [])],
        "answer": delivered.model_dump() if delivered else None,
        "problems": values.get("problems", []),
    }
