"""Dependency contract: psycopg2-binary (import name ``psycopg2``).

``psycopg2-binary`` is the Open WebUI backend's **sync** PostgreSQL driver,
used for startup migrations, config loading, Alembic, the peewee migration,
and health checks (``internal/db.py``'s SYNC ENGINE). It is also the dialect
base for the openGauss vector store
(``sqlalchemy.dialects.postgresql.psycopg2.PGDialect_psycopg2`` subclassed in
``retrieval/vector/dbs/opengauss.py``).

The backend doesn't call ``psycopg2.connect`` directly — SQLAlchemy does —
but ``db.py`` builds the connection string psycopg2 receives, and its SSL
handling is written specifically around psycopg2/libpq:

  - ``extract_ssl_params_from_url`` strips SSL query params and
    ``reattach_ssl_params_to_url`` re-appends them in canonical libpq form
    (``sslmode``, ``sslrootcert``, ``sslcert``, ``sslkey``, ``sslcrl``)
    *"for psycopg2/libpq consumers that expect ``sslmode``"*. The whole
    reattach step exists because psycopg2 reads ``sslmode`` from the
    connection string.

So the load-bearing contract is: psycopg2 (via libpq) parses ``sslmode`` and
the cert-file keys out of a connection string. This module pins that, plus
the DBAPI 2.0 surface and the exception hierarchy SQLAlchemy maps onto, all
OFFLINE with NO real database connection (uses ``extensions.parse_dsn``,
which parses without connecting).

NOTE (version drift): the pin is ``psycopg2-binary==2.9.12`` but the test
venv has ``2.9.11`` installed. Tests validate whatever is importable.

A psycopg2 bump that dropped a DBAPI symbol, reshuffled the exception tree,
or stopped recognising ``sslmode`` would fail here instead of surfacing as a
migration-time connection failure (or a silently-unencrypted sync
connection).

Uses the ``depcheck`` fixture from unit/deps/conftest.py.
"""

from __future__ import annotations

import inspect

import pytest

pytestmark = pytest.mark.depcheck

IMPORT_NAME = "psycopg2"
DIST_NAME = "psycopg2-binary"

# DBAPI 2.0 surface + the bits the SSL/DSN handling and SQLAlchemy rely on.
USED_SYMBOLS = [
    "connect",
    "extensions",
    "extensions.connection",
    "extensions.cursor",
    "extensions.parse_dsn",
    "Error",
]

DBAPI_MODULE_ATTRS = ["apilevel", "threadsafety", "paramstyle", "Binary"]

DBAPI_EXCEPTIONS = [
    "Warning",
    "Error",
    "InterfaceError",
    "DatabaseError",
    "DataError",
    "OperationalError",
    "IntegrityError",
    "InternalError",
    "ProgrammingError",
    "NotSupportedError",
]


def _parse_dsn(depcheck):
    return depcheck.resolve(depcheck.load(IMPORT_NAME), "extensions.parse_dsn")


# --------------------------------------------------------------------------- #
# Import / version
# --------------------------------------------------------------------------- #


def test_import(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    assert mod.__name__ == "psycopg2"


def test_is_version_2(depcheck):
    """The sync engine assumes psycopg2 (v2). Pin the major version."""
    mod = depcheck.load(IMPORT_NAME)
    ver = getattr(mod, "__version__", "")
    assert ver.startswith("2."), f"expected psycopg2 v2, got {ver!r}"


def test_version_reported(depcheck):
    """The distribution name is psycopg2-binary (the wheel), even though the
    import name is psycopg2."""
    assert depcheck.dist_version(DIST_NAME) is not None


# --------------------------------------------------------------------------- #
# Symbol existence (API surface)
# --------------------------------------------------------------------------- #


def test_used_symbols_exist(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_symbols(mod, USED_SYMBOLS)


def test_dbapi_module_attrs_exist(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_symbols(mod, DBAPI_MODULE_ATTRS)


def test_dbapi_level_values(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    assert mod.apilevel == "2.0"
    assert mod.paramstyle == "pyformat"


def test_connect_is_callable(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_callable(mod, "connect")


def test_connect_accepts_dsn_and_kwargs(depcheck):
    """psycopg2.connect(dsn, **kwargs). SQLAlchemy passes the DSN; libpq
    params flow through **kwargs."""
    mod = depcheck.load(IMPORT_NAME)
    sig = inspect.signature(mod.connect)
    params = sig.parameters
    assert "dsn" in params
    has_var_kw = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())
    assert has_var_kw, "psycopg2.connect no longer accepts **kwargs"


def test_extensions_connection_and_cursor_are_classes(depcheck):
    """SQLAlchemy's psycopg2 dialect references extensions.connection /
    extensions.cursor as the base types."""
    mod = depcheck.load(IMPORT_NAME)
    assert inspect.isclass(mod.extensions.connection)
    assert inspect.isclass(mod.extensions.cursor)


# --------------------------------------------------------------------------- #
# Exception hierarchy (SQLAlchemy maps every one of these)
# --------------------------------------------------------------------------- #


def test_dbapi_exceptions_exist(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_symbols(mod, DBAPI_EXCEPTIONS)


def test_exception_hierarchy(depcheck):
    """The DBAPI 2.0 tree: every concrete error subclasses DatabaseError which
    subclasses Error. SQLAlchemy's error translation relies on this layering."""
    mod = depcheck.load(IMPORT_NAME)
    assert issubclass(mod.DatabaseError, mod.Error)
    assert issubclass(mod.InterfaceError, mod.Error)
    for name in (
        "DataError",
        "OperationalError",
        "IntegrityError",
        "InternalError",
        "ProgrammingError",
        "NotSupportedError",
    ):
        exc = getattr(mod, name)
        assert issubclass(exc, mod.DatabaseError), f"{name} no longer subclasses DatabaseError"


# --------------------------------------------------------------------------- #
# Behavioural: DSN parsing (the SSL contract db.py builds around)
# --------------------------------------------------------------------------- #


def test_parse_dsn_basic_url(depcheck):
    """parse_dsn decomposes a postgresql:// URL into libpq keyword fields —
    parses, does NOT connect."""
    parse_dsn = _parse_dsn(depcheck)
    d = parse_dsn("postgresql://user:pw@dbhost:5432/mydb")
    assert d.get("user") == "user"
    assert d.get("host") == "dbhost"
    assert d.get("port") == "5432"
    assert d.get("dbname") == "mydb"


def test_parse_dsn_recognises_sslmode(depcheck):
    """THE load-bearing contract: db.py reattaches ``sslmode`` to the URL
    *because psycopg2 reads it*. Prove parse_dsn recognises ``sslmode`` as a
    valid libpq key and preserves its value — i.e. the reattach actually
    results in an enforced TLS mode on the sync connection."""
    parse_dsn = _parse_dsn(depcheck)
    d = parse_dsn("postgresql://u:p@h:5432/db?sslmode=require")
    assert d.get("sslmode") == "require", (
        "psycopg2/libpq no longer parses sslmode from the connection string — "
        "db.py's reattach_ssl_params_to_url would then be silently ineffective "
        "and the sync DB connection could fall back to plaintext"
    )


def test_parse_dsn_recognises_cert_file_keys(depcheck):
    """The exact cert-file keys reattach_ssl_params_to_url emits
    (sslrootcert / sslcert / sslkey) must be recognised by libpq, else
    client-cert auth on the sync engine silently breaks."""
    parse_dsn = _parse_dsn(depcheck)
    d = parse_dsn(
        "postgresql://u@h/db"
        "?sslmode=verify-full"
        "&sslrootcert=/etc/ssl/root.crt"
        "&sslcert=/etc/ssl/client.crt"
        "&sslkey=/etc/ssl/client.key"
    )
    assert d.get("sslmode") == "verify-full"
    assert d.get("sslrootcert") == "/etc/ssl/root.crt"
    assert d.get("sslcert") == "/etc/ssl/client.crt"
    assert d.get("sslkey") == "/etc/ssl/client.key"


def test_parse_dsn_rejects_garbage(depcheck):
    """A malformed DSN must raise a ProgrammingError (psycopg2's DBAPI error),
    not silently return a partial dict — so a bad DATABASE_URL fails loudly."""
    mod = depcheck.load(IMPORT_NAME)
    with pytest.raises(mod.ProgrammingError):
        mod.extensions.parse_dsn("this is not a valid dsn !!!")


def test_pgdialect_psycopg2_importable_from_sqlalchemy(depcheck):
    """opengauss.py subclasses
    sqlalchemy.dialects.postgresql.psycopg2.PGDialect_psycopg2. Pin that the
    SQLAlchemy<->psycopg2 dialect bridge still resolves (this is what ties the
    sync engine and the openGauss vector store to psycopg2)."""
    sa = depcheck.try_load("sqlalchemy.dialects.postgresql.psycopg2")
    if sa is None:
        pytest.skip("sqlalchemy psycopg2 dialect not importable in this env")
    assert hasattr(sa, "PGDialect_psycopg2")
    assert inspect.isclass(sa.PGDialect_psycopg2)
