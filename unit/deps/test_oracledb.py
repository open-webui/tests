"""Dependency contract: oracledb (python-oracledb).

``oracledb`` is the driver behind Open WebUI's Oracle 23ai vector store
(``retrieval/vector/dbs/oracle23ai.py``). The backend uses it directly:

  * ``oracledb.create_pool(user=, password=, dsn=, min=, max=, increment=,
    config_dir=, wallet_location=, wallet_password=)`` — both the
    wallet-authenticated (ADB) and basic-auth (DBCS) connection pools.
  * ``oracledb.DatabaseError`` — caught around pool acquisition retries.
  * ``oracledb.DB_TYPE_VECTOR`` — the vector column type, compared against
    ``cursor`` metadata in the output-type handler.
  * ``oracledb.LOB`` — ``isinstance(row[i], oracledb.LOB)`` to decide
    whether to ``.read()`` a CLOB/BLOB result.

A breaking bump (renamed ``create_pool`` kwargs, removed the
``DB_TYPE_VECTOR`` type constant added for 23ai vectors, or a moved
``LOB`` / ``DatabaseError``) would break the Oracle vector store. This
module pins those exact symbols + the ``create_pool`` keyword surface, and
verifies pool *construction* offline (a ``min=0`` pool is lazy — it opens
no connection — so we never contact a database). NO network, NO real
Oracle, NO wallet on disk.

Pattern mirrors test_requests.py. Uses ``depcheck`` from conftest.py.
"""

from __future__ import annotations

import inspect

import pytest

pytestmark = pytest.mark.depcheck

IMPORT_NAME = "oracledb"
DIST_NAME = "oracledb"

# Symbols oracle23ai.py resolves on the package.
USED_SYMBOLS = [
    "create_pool",  # pool factory (both ADB + DBCS paths)
    "DatabaseError",  # caught on pool.acquire() retry
    "DB_TYPE_VECTOR",  # 23ai vector column type
    "LOB",  # isinstance() check for CLOB/BLOB results
]

# Wider stable surface a driver consumer relies on.
SURFACE_SYMBOLS = [
    "create_pool",
    "connect",
    "Connection",
    "ConnectionPool",
    "Cursor",
    "DatabaseError",
    "Error",
    "OperationalError",
    "InterfaceError",
    "DB_TYPE_VECTOR",
    "DB_TYPE_CLOB",
    "DB_TYPE_BLOB",
    "DB_TYPE_NUMBER",
    "LOB",
]

# create_pool kwargs the backend actually passes (both code paths combined).
CREATE_POOL_KWARGS = [
    "user",
    "password",
    "dsn",
    "min",
    "max",
    "increment",
    "config_dir",
    "wallet_location",
    "wallet_password",
]


# ---------------------------------------------------------------------------
# Import + version
# ---------------------------------------------------------------------------


def test_import(depcheck):
    """`oracledb` must import (skip cleanly if absent)."""
    mod = depcheck.load(IMPORT_NAME)
    assert mod.__name__ == "oracledb"


def test_version_reported(depcheck):
    """The installed distribution version must resolve."""
    assert depcheck.dist_version(DIST_NAME) is not None


# ---------------------------------------------------------------------------
# Symbol-existence checks (API surface).
# ---------------------------------------------------------------------------


def test_used_symbols_exist(depcheck):
    """Every oracledb symbol oracle23ai.py resolves must exist."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_symbols(mod, USED_SYMBOLS)


def test_surface_symbols_exist(depcheck):
    """The wider driver surface (connection/cursor/type-constants/errors) the
    vector store builds on must remain present."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_symbols(mod, SURFACE_SYMBOLS)


def test_create_pool_is_callable(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    assert callable(mod.create_pool)


# ---------------------------------------------------------------------------
# Type-constant + exception-hierarchy contracts.
# ---------------------------------------------------------------------------


def test_db_type_vector_is_a_dbtype(depcheck):
    """oracle23ai.py compares `metadata.type_code is oracledb.DB_TYPE_VECTOR`
    (identity). It must remain a stable singleton DbType object — not a plain
    int that could collide — so the identity check stays correct."""
    mod = depcheck.load(IMPORT_NAME)
    vec = mod.DB_TYPE_VECTOR
    assert vec is not None
    # Identity-stability: re-resolving yields the same object.
    assert mod.DB_TYPE_VECTOR is vec
    # It must be distinguishable from the other column types via identity.
    assert mod.DB_TYPE_VECTOR is not mod.DB_TYPE_NUMBER
    assert mod.DB_TYPE_VECTOR is not mod.DB_TYPE_CLOB


def test_lob_is_a_class(depcheck):
    """`isinstance(row[i], oracledb.LOB)` requires LOB to be a class/type."""
    mod = depcheck.load(IMPORT_NAME)
    assert isinstance(mod.LOB, type)


def test_database_error_hierarchy(depcheck):
    """oracle23ai.py catches oracledb.DatabaseError; it must subclass the
    driver's base Error (and Exception), so the retry handler stays sound."""
    mod = depcheck.load(IMPORT_NAME)
    assert issubclass(mod.DatabaseError, mod.Error)
    assert issubclass(mod.DatabaseError, Exception)


def test_operational_error_subclasses_database_error(depcheck):
    """Connection-loss errors surface as OperationalError; pin its placement
    under DatabaseError so the broad except keeps catching transient failures."""
    mod = depcheck.load(IMPORT_NAME)
    assert issubclass(mod.OperationalError, mod.DatabaseError)


# ---------------------------------------------------------------------------
# create_pool keyword contract.
# ---------------------------------------------------------------------------


def test_create_pool_accepts_our_kwargs(depcheck):
    """create_pool must accept every keyword both connection paths pass
    (user/password/dsn/min/max/increment for DBCS; plus config_dir/
    wallet_location/wallet_password for the ADB wallet path)."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_params(mod.create_pool, CREATE_POOL_KWARGS)


# ---------------------------------------------------------------------------
# Pool construction contract (OFFLINE) — a min=0 pool is lazy, opens no socket.
# We never call .acquire() (that would connect to a database).
# ---------------------------------------------------------------------------


def test_create_pool_lazy_constructs_without_connecting(depcheck):
    """A pool with min=0 creates no initial sessions, so construction performs
    no network I/O. Build one with the basic-auth kwarg shape and assert it
    exposes the acquire/release/close surface the backend uses. The DSN is
    never contacted because we never acquire a connection."""
    mod = depcheck.load(IMPORT_NAME)
    pool = mod.create_pool(
        user="owui",
        password="placeholder",
        dsn="localhost:1521/vectorpdb",
        min=0,
        max=2,
        increment=1,
    )
    try:
        assert pool is not None
        for name in ("acquire", "release", "close"):
            assert callable(getattr(pool, name, None)), f"pool.{name} missing"
        # Pool sizing knobs the backend configures must be readable.
        assert pool.max == 2
    finally:
        _safe_close_pool(pool)


def test_pool_acquire_signature(depcheck):
    """get_connection() calls pool.acquire() with no args (then sets
    outputtypehandler). acquire must be callable with no required args."""
    mod = depcheck.load(IMPORT_NAME)
    pool = mod.create_pool(
        user="owui",
        password="placeholder",
        dsn="localhost:1521/vectorpdb",
        min=0,
        max=1,
        increment=1,
    )
    try:
        sig = inspect.signature(pool.acquire)
        required = [
            p
            for p in sig.parameters.values()
            if p.default is inspect.Parameter.empty
            and p.kind
            in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        ]
        assert not required, f"pool.acquire now requires args: {sig}"
    finally:
        _safe_close_pool(pool)


def test_connection_has_outputtypehandler_attr(depcheck):
    """get_connection() sets `connection.outputtypehandler = self._output_type_
    handler`. The Connection class must expose that attribute name (the hook the
    vector store uses to map DB_TYPE_VECTOR columns to python lists)."""
    mod = depcheck.load(IMPORT_NAME)
    # Class-level attribute name presence (no instance / no connection needed).
    assert "outputtypehandler" in dir(mod.Connection), (
        "Connection.outputtypehandler hook removed — vector output mapping breaks"
    )


def test_cursor_var_method_exists(depcheck):
    """_output_type_handler calls cursor.var(type_code, arraysize=,
    outconverter=). The Cursor class must expose `var`."""
    mod = depcheck.load(IMPORT_NAME)
    assert "var" in dir(mod.Cursor), "Cursor.var removed — output type handler breaks"


# ---------------------------------------------------------------------------
# Local helper (no cross-file imports).
# ---------------------------------------------------------------------------


def _safe_close_pool(pool) -> None:
    """Close a lazily-built pool best-effort (it opened no connections)."""
    try:
        pool.close(force=True)
    except Exception:
        pass
