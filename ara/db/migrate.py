"""Apply ara/db/migrations/*.sql in name order, each once: `python -m ara.db.migrate`."""

from pathlib import Path

import psycopg

MIGRATIONS = Path(__file__).with_name("migrations")
LOCK_ID = 7_042_001  # pg_advisory_lock key: one migrator at a time


def migrate(url: str) -> list[str]:
    """Apply pending migrations, each in its own transaction; return the names applied."""
    with psycopg.connect(url, autocommit=True) as conn:
        conn.execute("SELECT pg_advisory_lock(%s)", (LOCK_ID,))
        conn.execute(
            "CREATE TABLE IF NOT EXISTS schema_migrations"
            " (name text PRIMARY KEY, applied_at timestamptz NOT NULL DEFAULT now())"
        )
        done = {name for (name,) in conn.execute("SELECT name FROM schema_migrations")}
        pending = [path for path in sorted(MIGRATIONS.glob("*.sql")) if path.stem not in done]
        for path in pending:
            with conn.transaction():
                conn.execute(path.read_bytes())
                conn.execute("INSERT INTO schema_migrations (name) VALUES (%s)", (path.stem,))
        return [path.stem for path in pending]


if __name__ == "__main__":
    from ara.settings import get_settings

    applied = migrate(get_settings().database_url)
    print("applied:", ", ".join(applied) if applied else "nothing (up to date)")
