"""Dependency contract: aiosqlite (import name ``aiosqlite``).

aiosqlite is the *async SQLite driver* behind Open WebUI's default database.
``internal/db.py`` rewrites any ``sqlite://`` / ``sqlite:///`` URL to
``sqlite+aiosqlite://`` for the async engine (see ``handle_peewee_migration``
/ the ``DATABASE_URL`` normalization), so on the default config every async
DB call in every request path executes through aiosqlite under SQLAlchemy's
``create_async_engine``. The backend never imports aiosqlite directly; it
relies entirely on SQLAlchemy's ``sqlite+aiosqlite`` dialect, which in turn
relies on aiosqlite exposing the standard ``sqlite3``-shaped async API
(``connect`` -> ``Connection`` with awaitable ``execute`` / ``executemany`` /
``commit`` / ``cursor``, awaitable ``Cursor.fetchone`` / ``fetchall``, a
``Row`` factory, and the ``Error`` exception hierarchy).

This module pins exactly that surface so an aiosqlite bump that broke the
async connection protocol fails loudly here instead of as a runtime failure
on the first DB query of every request. Two layers, mirroring
test_sqlalchemy.py's async section: symbol-existence + signature checks, plus
offline BEHAVIOURAL contracts that drive a real in-memory async SQLite
connection (``:memory:`` — no file, no network) through the DDL/DML/commit/
read lifecycle, AND a full SQLAlchemy ``sqlite+aiosqlite:///:memory:``
round-trip (the actual runtime path). aiosqlite is installed in this env, so
everything runs for real.

Uses the ``depcheck`` fixture from unit/deps/conftest.py.
"""

from __future__ import annotations

import inspect

import pytest

pytestmark = pytest.mark.depcheck

IMPORT_NAME = "aiosqlite"
DIST_NAME = "aiosqlite"

# The dotted symbols SQLAlchemy's aiosqlite dialect (and any direct user)
# resolves on the package.
TOP_LEVEL_SYMBOLS = [
    "connect",  # the async connection factory
    "Connection",  # the awaitable connection class
    "Cursor",  # the awaitable cursor class
    "Row",  # row factory (SQLAlchemy sets connection.row_factory)
    "Error",  # base DB-API exception
    "DatabaseError",  # operational error base
    "register_adapter",  # type adaptation hooks (DB-API parity)
    "register_converter",
    "sqlite_version",  # underlying libsqlite version string
    "paramstyle",  # DB-API 2.0 attribute SQLAlchemy inspects
]

# Awaitable methods SQLAlchemy's async dialect drives on the Connection.
CONNECTION_METHODS = [
    "execute",
    "executemany",
    "executescript",
    "commit",
    "rollback",
    "cursor",
    "close",
]

# Cursor methods consumed when fetching results.
CURSOR_METHODS = ["execute", "fetchone", "fetchall", "fetchmany", "close"]


# ---------------------------------------------------------------------------
# Import + version
# ---------------------------------------------------------------------------


def test_import(depcheck):
    """`aiosqlite` must import (skip cleanly if absent in this env)."""
    mod = depcheck.load(IMPORT_NAME)
    assert mod.__name__ == "aiosqlite"


def test_version_reported(depcheck):
    """The installed distribution version must be resolvable so bump tooling
    and this suite agree on what's under test."""
    depcheck.load(IMPORT_NAME)
    assert depcheck.dist_version(DIST_NAME) is not None


# ---------------------------------------------------------------------------
# Symbol-existence + signature checks (API surface SQLAlchemy depends on)
# ---------------------------------------------------------------------------


def test_top_level_symbols_exist(depcheck):
    """Every top-level `aiosqlite.*` symbol the sqlite+aiosqlite dialect (and
    DB-API consumers) resolves must exist."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_symbols(mod, TOP_LEVEL_SYMBOLS)


def test_connect_is_callable(depcheck):
    """SQLAlchemy's dialect calls aiosqlite.connect(database, ...). It must
    remain a callable factory."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_callable(mod, "connect")


def test_connect_accepts_database_arg(depcheck):
    """connect(':memory:') / connect(<path>) — the `database` parameter must
    remain accepted (it is the first positional)."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_params(mod.connect, ["database"])


def test_dbapi_paramstyle_is_qmark(depcheck):
    """aiosqlite is DB-API 2.0 with the sqlite3 `qmark` paramstyle (the `?`
    placeholders used below). SQLAlchemy's dialect keys on this; pin it."""
    mod = depcheck.load(IMPORT_NAME)
    assert mod.paramstyle == "qmark", (
        f"aiosqlite.paramstyle changed to {mod.paramstyle!r}; SQLAlchemy's "
        "sqlite paramstyle assumption (qmark '?') would break."
    )


def test_error_hierarchy(depcheck):
    """SQLAlchemy wraps driver errors; DatabaseError must remain an Error
    subclass so broad exception handling stays correct."""
    mod = depcheck.load(IMPORT_NAME)
    assert issubclass(mod.DatabaseError, mod.Error), (
        "aiosqlite.DatabaseError no longer subclasses Error"
    )


# ---------------------------------------------------------------------------
# Method-surface checks on the connection/cursor CLASSES (no I/O)
# ---------------------------------------------------------------------------


def test_connection_methods_exist(depcheck):
    """The async Connection must expose the awaitable methods the dialect drives."""
    mod = depcheck.load(IMPORT_NAME)
    names = set(dir(mod.Connection))
    missing = [m for m in CONNECTION_METHODS if m not in names]
    assert not missing, f"aiosqlite.Connection missing method(s): {missing}"


def test_cursor_methods_exist(depcheck):
    """The async Cursor must expose execute + the fetch family."""
    mod = depcheck.load(IMPORT_NAME)
    names = set(dir(mod.Cursor))
    missing = [m for m in CURSOR_METHODS if m not in names]
    assert not missing, f"aiosqlite.Cursor missing method(s): {missing}"


# ---------------------------------------------------------------------------
# Behavioural: real in-memory async SQLite lifecycle (no file, no network).
# This is the driver-level contract SQLAlchemy's async engine builds upon.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_behaviour_connect_is_async_context_manager(depcheck):
    """`async with aiosqlite.connect(':memory:') as db:` is the idiom the dialect
    uses. The connect() result must be an async context manager yielding a live
    connection."""
    mod = depcheck.load(IMPORT_NAME)
    async with mod.connect(":memory:") as db:
        assert db is not None
        assert hasattr(db, "execute"), "connection has no execute()"


@pytest.mark.asyncio
async def test_behaviour_ddl_insert_commit_select_roundtrip(depcheck):
    """Full driver lifecycle: CREATE TABLE -> parametrised INSERT -> commit ->
    SELECT -> fetchall. This is exactly what SQLAlchemy emits under the hood for
    an Open WebUI write+read, just expressed at the DB-API level."""
    mod = depcheck.load(IMPORT_NAME)
    async with mod.connect(":memory:") as db:
        await db.execute("CREATE TABLE item(id INTEGER PRIMARY KEY, name TEXT, count INTEGER)")
        await db.execute("INSERT INTO item(name, count) VALUES(?, ?)", ("alpha", 3))
        await db.execute("INSERT INTO item(name, count) VALUES(?, ?)", ("beta", 7))
        await db.commit()

        cur = await db.execute("SELECT name, count FROM item ORDER BY count")
        rows = await cur.fetchall()
        assert rows == [("alpha", 3), ("beta", 7)]


@pytest.mark.asyncio
async def test_behaviour_parametrised_query_fetchone(depcheck):
    """Parametrised SELECT with `?` placeholders + fetchone — the row-at-a-time
    path. Pin that bound params filter correctly and fetchone yields one tuple."""
    mod = depcheck.load(IMPORT_NAME)
    async with mod.connect(":memory:") as db:
        await db.execute("CREATE TABLE u(id TEXT PRIMARY KEY, role TEXT)")
        await db.executemany(
            "INSERT INTO u(id, role) VALUES(?, ?)",
            [("u1", "admin"), ("u2", "user"), ("u3", "user")],
        )
        await db.commit()

        cur = await db.execute("SELECT id FROM u WHERE role = ?", ("admin",))
        row = await cur.fetchone()
        assert row == ("u1",)


@pytest.mark.asyncio
async def test_behaviour_executemany_bulk_insert(depcheck):
    """executemany batches inserts (used by SQLAlchemy for multi-row writes).
    Pin that every row lands."""
    mod = depcheck.load(IMPORT_NAME)
    async with mod.connect(":memory:") as db:
        await db.execute("CREATE TABLE n(v INTEGER)")
        await db.executemany("INSERT INTO n(v) VALUES(?)", [(i,) for i in range(10)])
        await db.commit()
        cur = await db.execute("SELECT COUNT(*) FROM n")
        (total,) = await cur.fetchone()
        assert total == 10


@pytest.mark.asyncio
async def test_behaviour_rollback_discards_uncommitted(depcheck):
    """rollback() must discard uncommitted work (SQLAlchemy session rollback maps
    onto this). Pin the transactional semantics."""
    mod = depcheck.load(IMPORT_NAME)
    async with mod.connect(":memory:") as db:
        await db.execute("CREATE TABLE t(v INTEGER)")
        await db.commit()
        await db.execute("INSERT INTO t(v) VALUES(1)")
        await db.rollback()
        cur = await db.execute("SELECT COUNT(*) FROM t")
        (total,) = await cur.fetchone()
        assert total == 0


@pytest.mark.asyncio
async def test_behaviour_row_factory_named_access(depcheck):
    """SQLAlchemy sets `connection.row_factory = aiosqlite.Row` so result rows
    support both index and column-name access. Pin that Row gives keyed access."""
    mod = depcheck.load(IMPORT_NAME)
    async with mod.connect(":memory:") as db:
        db.row_factory = mod.Row
        await db.execute("CREATE TABLE p(id TEXT, name TEXT)")
        await db.execute("INSERT INTO p(id, name) VALUES(?, ?)", ("x", "alpha"))
        await db.commit()
        cur = await db.execute("SELECT id, name FROM p")
        row = await cur.fetchone()
        assert row["id"] == "x"
        assert row["name"] == "alpha"
        assert row[0] == "x"  # positional still works


@pytest.mark.asyncio
async def test_behaviour_cursor_as_async_iterator(depcheck):
    """`async for row in cursor` — aiosqlite cursors are async-iterable, which
    SQLAlchemy's streaming result path can use. Pin the protocol."""
    mod = depcheck.load(IMPORT_NAME)
    async with mod.connect(":memory:") as db:
        await db.execute("CREATE TABLE s(v INTEGER)")
        await db.executemany("INSERT INTO s(v) VALUES(?)", [(i,) for i in range(5)])
        await db.commit()
        seen = []
        async with db.execute("SELECT v FROM s ORDER BY v") as cur:
            async for row in cur:
                seen.append(row[0])
        assert seen == [0, 1, 2, 3, 4]


# ---------------------------------------------------------------------------
# Behavioural: the ACTUAL runtime path — SQLAlchemy create_async_engine over
# the sqlite+aiosqlite dialect (what internal/db.py builds on the default DB).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_behaviour_sqlalchemy_async_engine_uses_aiosqlite(depcheck):
    """internal/db.py rewrites sqlite URLs to `sqlite+aiosqlite://` and runs the
    async engine on them. Build that exact engine over an in-memory DB, create a
    table via run_sync, and execute a text() SELECT through aiosqlite — the real
    end-to-end driver path for every Open WebUI request. Skips cleanly if
    SQLAlchemy is absent (aiosqlite itself is still covered above)."""
    depcheck.load(IMPORT_NAME)
    sa = depcheck.try_load("sqlalchemy")
    if sa is None:
        pytest.skip("sqlalchemy not installed; direct-driver tests cover aiosqlite")

    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import create_async_engine

    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    try:
        async with engine.begin() as conn:
            await conn.run_sync(
                lambda sync_conn: sync_conn.exec_driver_sql(
                    "CREATE TABLE demo(id INTEGER PRIMARY KEY, name TEXT)"
                )
            )
            await conn.exec_driver_sql("INSERT INTO demo(name) VALUES('owui')")
        async with engine.connect() as conn:
            value = (await conn.execute(text("SELECT name FROM demo"))).scalar()
            assert value == "owui"
            one = (await conn.execute(text("SELECT 1"))).scalar()
            assert one == 1
    finally:
        await engine.dispose()


def test_dialect_driver_name_matches(depcheck):
    """SQLAlchemy resolves the dialect by the `aiosqlite` driver name. Confirm
    SQLAlchemy can build that dialect (i.e. the entry point still maps to this
    package), so internal/db.py's URL rewrite resolves a real driver."""
    depcheck.load(IMPORT_NAME)
    sa = depcheck.try_load("sqlalchemy")
    if sa is None:
        pytest.skip("sqlalchemy not installed")
    from sqlalchemy.engine import make_url

    url = make_url("sqlite+aiosqlite:///:memory:")
    assert url.get_driver_name() == "aiosqlite"
    # get_dialect() imports the dialect class, which imports aiosqlite — proves
    # the wiring resolves without instantiating an engine.
    dialect_cls = url.get_dialect()
    assert dialect_cls is not None
    assert inspect.isclass(dialect_cls)
