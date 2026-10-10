"""Evaluation leftovers in the app database (D39): the threads that evaluation rounds and live
checks ran outside the app (no conversation holds them), and the libraries and memories of the
simulated users they read and remembered for. Results, the ledger and the shared corpus (papers,
chunks, embeddings) stay: they are the record and a cache. `ara db prune` lists; --execute
deletes."""

from ara.db.pool import Connection

USER = "local"  # the app's own user (ara.api.server.USER)

COUNTS = {
    "threads": """
        SELECT count(DISTINCT thread_id) AS n FROM checkpoints
        WHERE thread_id NOT IN (SELECT id FROM conversations)""",
    "memory items": """
        SELECT count(*) AS n FROM store
        WHERE prefix LIKE 'users.%%' AND prefix NOT LIKE %(own)s""",
    "library rows": "SELECT count(*) AS n FROM library_items WHERE user_id <> %(user)s",
}
DELETES = [
    *(
        f"DELETE FROM {table} WHERE thread_id NOT IN (SELECT id FROM conversations)"
        for table in ("checkpoint_writes", "checkpoint_blobs", "checkpoints")
    ),
    "DELETE FROM store WHERE prefix LIKE 'users.%%' AND prefix NOT LIKE %(own)s",
    "DELETE FROM library_items WHERE user_id <> %(user)s",
]


def _params() -> dict[str, str]:
    return {"user": USER, "own": f"users.{USER}.%"}


async def leftovers(conn: Connection) -> dict[str, int]:
    counts = {}
    for name, sql in COUNTS.items():
        row = await (await conn.execute(sql, _params())).fetchone()
        counts[name] = int(row["n"]) if row else 0
    return counts


async def prune(conn: Connection) -> dict[str, int]:
    """Delete the leftovers in one transaction; returns what there was."""
    async with conn.transaction():
        counts = await leftovers(conn)
        for sql in DELETES:
            await conn.execute(sql, _params())
    return counts
