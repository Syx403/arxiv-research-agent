from __future__ import annotations

from typing import NamedTuple

from src.core import db
from src.core.types import EPISODIC_VERBATIM_TURNS, Message, StoredMessage
from src.llm.client import get_chat_client
from src.memory.episodic.session_store import _row_to_message


class CachedSummary(NamedTuple):
    summary: str
    up_to_message_id: int


async def compressed_history(
    session_id: str,
    *,
    verbatim_turns: int = EPISODIC_VERBATIM_TURNS,
) -> list[Message]:
    messages = await _fetch_all_messages(session_id)
    if not messages:
        return []

    split_at = max(len(messages) - max(verbatim_turns, 0), 0)
    older = messages[:split_at]
    verbatim = messages[split_at:]
    if not older:
        return [_to_llm_message(message) for message in verbatim]

    cutoff_id = older[-1].message_id
    cached = await _get_cached_summary(session_id, cutoff_id)
    if cached is not None and cached.up_to_message_id == cutoff_id:
        summary = cached.summary
    else:
        cached_summary = cached.summary if cached is not None else None
        cached_up_to = cached.up_to_message_id if cached is not None else 0
        delta = [message for message in older if message.message_id > cached_up_to]
        summary = await _summarize_incremental(cached_summary, delta)
        await _upsert_summary(session_id, cutoff_id, summary)

    return [Message(role="system", content=summary), *[_to_llm_message(message) for message in verbatim]]


async def _fetch_all_messages(session_id: str) -> list[StoredMessage]:
    async with db.acquire_app() as conn:
        rows = await conn.fetch(
            """
            SELECT message_id, session_id, role, content, metadata, created_at
            FROM session_messages
            WHERE session_id = $1
            ORDER BY message_id ASC
            """,
            session_id,
        )
    return [_row_to_message(row) for row in rows]


async def _get_cached_summary(session_id: str, cutoff_id: int) -> CachedSummary | None:
    async with db.acquire_app() as conn:
        row = await conn.fetchrow(
            """
            SELECT summary, up_to_message_id
            FROM session_summaries
            WHERE session_id = $1
              AND up_to_message_id <= $2
            ORDER BY up_to_message_id DESC
            LIMIT 1
            """,
            session_id,
            cutoff_id,
        )
    if row is None:
        return None
    return CachedSummary(summary=row["summary"], up_to_message_id=row["up_to_message_id"])


async def _upsert_summary(session_id: str, cutoff_id: int, summary: str) -> None:
    async with db.acquire_app() as conn:
        await conn.execute(
            """
            INSERT INTO session_summaries (session_id, up_to_message_id, summary)
            VALUES ($1, $2, $3)
            ON CONFLICT (session_id, up_to_message_id) DO NOTHING
            """,
            session_id,
            cutoff_id,
            summary,
        )


async def _summarize_incremental(
    cached_summary: str | None,
    messages: list[StoredMessage],
) -> str:
    client = get_chat_client("fast")
    prefix = f"Existing summary:\n{cached_summary}\n\n" if cached_summary else ""
    transcript = "\n".join(f"{message.role}: {message.content}" for message in messages)
    response = await client.chat(
        [
            Message(
                role="system",
                content="Summarize older conversation history for future retrieval. Preserve user goals, decisions, and unresolved tasks.",
            ),
            Message(role="user", content=f"{prefix}New messages to incorporate:\n{transcript}"),
        ],
        temperature=0.0,
        max_tokens=500,
    )
    return (response.content or "").strip()


def _to_llm_message(message: StoredMessage) -> Message:
    return Message(role=message.role, content=message.content)
