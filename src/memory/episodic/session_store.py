from __future__ import annotations

import json
from typing import Any

from src.core import db
from src.core.types import Session, StoredMessage


async def get_or_create_session(session_id: str, title: str | None = None) -> Session:
    async with db.acquire_app() as conn:
        await conn.execute(
            """
            INSERT INTO sessions (session_id, title)
            VALUES ($1, $2)
            ON CONFLICT (session_id) DO NOTHING
            """,
            session_id,
            title,
        )
        row = await conn.fetchrow(
            """
            SELECT session_id, title, created_at, last_active
            FROM sessions
            WHERE session_id = $1
            """,
            session_id,
        )
    if row is None:
        raise LookupError(f"Session not found after insert: {session_id}")
    return _row_to_session(row)


async def append_message(
    session_id: str,
    role: str,
    content: str,
    metadata: dict | None = None,
) -> int:
    async with db.acquire_app() as conn:
        async with conn.transaction():
            message_id = await conn.fetchval(
                """
                INSERT INTO session_messages (session_id, role, content, metadata)
                VALUES ($1, $2, $3, $4::jsonb)
                RETURNING message_id
                """,
                session_id,
                role,
                content,
                json.dumps(metadata or {}),
            )
            await conn.execute(
                "UPDATE sessions SET last_active = NOW() WHERE session_id = $1",
                session_id,
            )
    return int(message_id)


async def get_recent(session_id: str, limit: int) -> list[StoredMessage]:
    async with db.acquire_app() as conn:
        rows = await conn.fetch(
            """
            SELECT message_id, session_id, role, content, metadata, created_at
            FROM session_messages
            WHERE session_id = $1
            ORDER BY message_id DESC
            LIMIT $2
            """,
            session_id,
            limit,
        )
    return [_row_to_message(row) for row in reversed(rows)]


async def get_messages(session_id: str, *, role: str | None = None) -> list[StoredMessage]:
    async with db.acquire_app() as conn:
        rows = await conn.fetch(
            """
            SELECT message_id, session_id, role, content, metadata, created_at
            FROM session_messages
            WHERE session_id = $1
              AND ($2::text IS NULL OR role = $2)
            ORDER BY message_id ASC
            """,
            session_id,
            role,
        )
    return [_row_to_message(row) for row in rows]


def _row_to_session(row: Any) -> Session:
    return Session(
        session_id=row["session_id"],
        title=row["title"],
        created_at=row["created_at"],
        last_active=row["last_active"],
    )


def _row_to_message(row: Any) -> StoredMessage:
    metadata = row["metadata"]
    if isinstance(metadata, str):
        metadata = json.loads(metadata)
    return StoredMessage(
        message_id=row["message_id"],
        session_id=row["session_id"],
        role=row["role"],
        content=row["content"],
        metadata=metadata or {},
        created_at=row["created_at"],
    )
