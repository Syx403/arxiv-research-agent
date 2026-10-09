"""The app's conversations and their recorded turns (D36), in SQL. A conversation's id is its
LangGraph thread id; it is created by its first message, titled from it, and listed newest first.
"""

import json
from datetime import datetime
from typing import Any

from ara.db.pool import Connection

TITLE_CHARS = 60

TURN_COLUMNS = (
    "turn_id",
    "message",
    "reply",
    "status",
    "intent",
    "answer",
    "papers",
    "read",
    "problems",
    "trace",
    "waiting",
    "started_at",
    "finished_at",
)
CALLS = """
SELECT turn_id, id, stage, model, status, latency_ms, input_tokens, cached_tokens, output_tokens,
       reasoning_tokens, cost_usd
FROM llm_calls WHERE turn_id = ANY(%s) AND status <> 'reserved' ORDER BY id
"""


def title(message: str) -> str:
    """The first message, cut at a word near TITLE_CHARS."""
    text = " ".join(message.split())
    if len(text) <= TITLE_CHARS:
        return text
    cut = text[:TITLE_CHARS].rsplit(" ", 1)[0]
    return f"{cut}…"


async def touch(conn: Connection, conversation: str, user: str, message: str) -> None:
    """Create the conversation on its first message (titled from it), or mark it as recent."""
    await conn.execute(
        "INSERT INTO conversations (id, user_id, title) VALUES (%s, %s, %s)"
        " ON CONFLICT (id) DO UPDATE SET updated_at = now()",
        (conversation, user, title(message)),
    )


async def listed(conn: Connection, user: str) -> list[dict[str, Any]]:
    cursor = await conn.execute(
        "SELECT id, title, created_at, updated_at FROM conversations WHERE user_id = %s"
        " ORDER BY updated_at DESC",
        (user,),
    )
    return list(await cursor.fetchall())


async def rename(conn: Connection, conversation: str, user: str, new_title: str) -> bool:
    cursor = await conn.execute(
        "UPDATE conversations SET title = %s WHERE id = %s AND user_id = %s",
        (new_title.strip() or "Untitled", conversation, user),
    )
    return cursor.rowcount == 1


async def delete(conn: Connection, conversation: str, user: str) -> bool:
    """The conversation and its turns; its model calls stay in the ledger (spend is kept)."""
    cursor = await conn.execute(
        "DELETE FROM conversations WHERE id = %s AND user_id = %s", (conversation, user)
    )
    return cursor.rowcount == 1


async def record(
    conn: Connection, conversation: str, turn: dict[str, Any], started_at: datetime
) -> None:
    """One turn as the UI will show it again."""
    values = {k: turn[k] for k in TURN_COLUMNS[:-2]}
    jsonb = {"answer", "papers", "read", "problems", "trace", "waiting"}
    await conn.execute(
        "INSERT INTO turns (conversation_id, started_at, "
        + ", ".join(values)
        + ") VALUES (%s, %s, "
        + ", ".join("%s" for _ in values)
        + ")",
        (
            conversation,
            started_at,
            *(json.dumps(v) if k in jsonb else v for k, v in values.items()),
        ),
    )


async def opened(conn: Connection, conversation: str, user: str) -> dict[str, Any] | None:
    """The conversation with every turn, each with its settled model calls."""
    cursor = await conn.execute(
        "SELECT id, title, created_at, updated_at FROM conversations"
        " WHERE id = %s AND user_id = %s",
        (conversation, user),
    )
    head = await cursor.fetchone()
    if head is None:
        return None
    cursor = await conn.execute(
        "SELECT " + ", ".join(TURN_COLUMNS) + " FROM turns WHERE conversation_id = %s ORDER BY id",
        (conversation,),
    )
    turns = list(await cursor.fetchall())
    calls = await turn_calls(conn, [t["turn_id"] for t in turns])
    return {**head, "turns": [{**t, "calls": calls.get(t["turn_id"], [])} for t in turns]}


async def turn_calls(conn: Connection, turn_ids: list[str]) -> dict[str, list[dict[str, Any]]]:
    cursor = await conn.execute(CALLS, (turn_ids,))
    found: dict[str, list[dict[str, Any]]] = {}
    for row in await cursor.fetchall():
        cost = row["cost_usd"]
        found.setdefault(row["turn_id"], []).append(
            {**row, "cost_usd": float(cost) if cost is not None else None}
        )
    return found
