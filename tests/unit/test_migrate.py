import psycopg

from ara.db.migrate import MIGRATIONS, migrate


def test_each_migration_applies_once(test_database: str) -> None:
    assert migrate(test_database) == []  # the session fixture has applied everything
    with psycopg.connect(test_database) as conn:
        applied = [name for (name,) in conn.execute("SELECT name FROM schema_migrations")]
    assert sorted(applied) == sorted(path.stem for path in MIGRATIONS.glob("*.sql"))


def test_search_extensions_are_installed(test_database: str) -> None:
    with psycopg.connect(test_database) as conn:
        extensions = {name for (name,) in conn.execute("SELECT extname FROM pg_extension")}
    assert {"vector", "pg_search"} <= extensions
