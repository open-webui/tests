"""Dependency contract: PyMySQL (import name ``pymysql``).

PyMySQL is a pure-Python MySQL client (a PEP 249 / DB-API 2.0 driver). It
is pinned in ``backend/requirements.txt`` (``PyMySQL==1.2.0``) to be the
MySQL/MariaDB driver for SQLAlchemy. Open WebUI does not hardcode a DB
backend: ``internal/db.py`` builds the engine from ``DATABASE_URL`` via
``sqlalchemy.create_engine(...)`` / ``create_async_engine(...)``. When the
operator points ``DATABASE_URL`` at MySQL using the ``mysql+pymysql://``
scheme, SQLAlchemy's ``MySQLDialect_pymysql`` imports this package as its
DB-API module.

IMPORTANT — usage note: the Open WebUI *application* code does NOT import
``pymysql`` directly anywhere (the only textual "mysql" hit in the source
is a telemetry span-tag constant). PyMySQL is a declared/transitive
dependency consumed by SQLAlchemy's MySQL dialect. The contract that
matters is therefore twofold:

  1. PyMySQL keeps a valid DB-API 2.0 surface (``connect`` / ``Connection``
     / the ``pymysql.err`` exception hierarchy / module-level
     ``paramstyle`` / ``apilevel`` / ``threadsafety``), because that is
     exactly what SQLAlchemy dispatches onto;
  2. SQLAlchemy's ``mysql+pymysql`` dialect still resolves its DB-API to
     this module, and ``create_engine('mysql+pymysql://...')`` binds it
     lazily WITHOUT connecting.

Everything here is offline: building an engine does not open a socket
(SQLAlchemy connects lazily on first use), and we never call ``connect``.

Uses the ``depcheck`` fixture from unit/deps/conftest.py.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.depcheck

IMPORT_NAME = "pymysql"
DIST_NAME = "PyMySQL"

# DB-API 2.0 module-level surface SQLAlchemy's dialect relies on.
DBAPI_SYMBOLS = [
    "connect",  # DB-API connection factory
    "Connection",
    "install_as_MySQLdb",  # lets `import MySQLdb` resolve to pymysql
    "paramstyle",
    "apilevel",
    "threadsafety",
    "err",  # exception hierarchy submodule
    "cursors",
    "constants",
    "converters",
]

# DB-API exception classes SQLAlchemy maps to its own error wrappers.
DBAPI_EXCEPTIONS = [
    "Error",
    "Warning",
    "InterfaceError",
    "DatabaseError",
    "DataError",
    "OperationalError",
    "IntegrityError",
    "InternalError",
    "ProgrammingError",
    "NotSupportedError",
]


def test_import(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    assert mod.__name__ == "pymysql"


def test_version_reported(depcheck):
    assert depcheck.dist_version(DIST_NAME) is not None


def test_dbapi_symbols_exist(depcheck):
    """The DB-API 2.0 module surface SQLAlchemy's pymysql dialect dispatches
    onto must remain present."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_symbols(mod, DBAPI_SYMBOLS)


def test_connect_is_callable(depcheck):
    """connect(...) is the DB-API connection factory SQLAlchemy calls to open a
    connection. It must remain callable (we never invoke it — no server)."""
    mod = depcheck.load(IMPORT_NAME)
    assert callable(mod.connect)


def test_dbapi_module_level_constants(depcheck):
    """PEP 249 requires paramstyle / apilevel / threadsafety. SQLAlchemy's
    pymysql dialect assumes apilevel '2.0' and paramstyle 'pyformat' (it builds
    %(name)s-style parameter binds). Pin those exact values."""
    mod = depcheck.load(IMPORT_NAME)
    assert mod.apilevel == "2.0", f"PyMySQL apilevel changed: {mod.apilevel!r}"
    assert mod.paramstyle == "pyformat", (
        f"PyMySQL paramstyle changed to {mod.paramstyle!r}; SQLAlchemy's "
        "pymysql dialect emits pyformat binds"
    )
    assert isinstance(mod.threadsafety, int)


def test_exception_hierarchy_present(depcheck):
    """SQLAlchemy maps the DB-API exception classes to its own DBAPIError
    subclasses. They must all exist on pymysql.err."""
    mod = depcheck.load(IMPORT_NAME)
    err = mod.err
    missing = [e for e in DBAPI_EXCEPTIONS if not hasattr(err, e)]
    assert not missing, f"pymysql.err missing exception(s): {missing}"


def test_exception_hierarchy_rooted_at_error(depcheck):
    """DB-API requires the operational/integrity/programming errors to subclass
    DatabaseError, which subclasses Error. SQLAlchemy's `is_disconnect` and
    error-wrapping logic relies on this tree."""
    mod = depcheck.load(IMPORT_NAME)
    err = mod.err
    assert issubclass(err.DatabaseError, err.Error)
    for name in (
        "OperationalError",
        "IntegrityError",
        "ProgrammingError",
        "DataError",
        "InternalError",
        "NotSupportedError",
    ):
        exc = getattr(err, name)
        assert issubclass(exc, err.DatabaseError), (
            f"pymysql.err.{name} no longer subclasses DatabaseError"
        )


def test_top_level_exceptions_aliased(depcheck):
    """PyMySQL also re-exports the exception classes at the top level
    (pymysql.OperationalError etc.), which DB-API consumers use. Pin the
    top-level aliases match the err submodule classes."""
    mod = depcheck.load(IMPORT_NAME)
    for name in ("Error", "OperationalError", "IntegrityError", "ProgrammingError"):
        top = getattr(mod, name, None)
        sub = getattr(mod.err, name, None)
        assert top is not None, f"pymysql.{name} top-level alias missing"
        assert top is sub, f"pymysql.{name} no longer aliases pymysql.err.{name}"


def test_connect_accepts_standard_dsn_kwargs(depcheck):
    """SQLAlchemy translates the URL into connect(host=, user=, password=,
    database=, port=, ...). Pin those keyword names on connect (it may take
    **kwargs)."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_params(
        mod.connect,
        ["host", "user", "password", "database", "port"],
    )


def test_sqlalchemy_pymysql_dialect_resolves_to_this_module(depcheck):
    """The real integration contract: SQLAlchemy's mysql+pymysql dialect must
    import THIS module as its DB-API. If SQLAlchemy isn't installed we skip;
    otherwise assert dialect.import_dbapi() is the pymysql module."""
    mod = depcheck.load(IMPORT_NAME)
    sa_dialect = depcheck.try_load("sqlalchemy.dialects.mysql.pymysql")
    if sa_dialect is None:
        pytest.skip("SQLAlchemy not installed; DB-API surface tests cover pymysql")
    dialect_cls = sa_dialect.MySQLDialect_pymysql
    # import_dbapi() (SQLAlchemy 2.x) returns the bound DB-API module.
    importer = getattr(dialect_cls, "import_dbapi", None) or getattr(dialect_cls, "dbapi", None)
    assert importer is not None, "pymysql dialect has no import_dbapi/dbapi classmethod"
    dbapi = importer()
    assert dbapi is mod, "SQLAlchemy's mysql+pymysql dialect no longer binds pymysql"


def test_create_engine_binds_pymysql_lazily_without_connecting(depcheck):
    """create_engine('mysql+pymysql://...') must build an engine whose dialect
    driver is 'pymysql' and whose DB-API is this module — WITHOUT opening a
    connection (SQLAlchemy connects lazily on first use). This mirrors exactly
    what internal/db.py does for a MySQL DATABASE_URL."""
    mod = depcheck.load(IMPORT_NAME)
    sa = depcheck.try_load("sqlalchemy")
    if sa is None:
        pytest.skip("SQLAlchemy not installed; DB-API surface tests cover pymysql")
    engine = sa.create_engine("mysql+pymysql://user:pw@localhost:3306/dbname")
    try:
        assert engine.dialect.driver == "pymysql"
        assert engine.dialect.dbapi is mod
    finally:
        engine.dispose()


def test_url_dialect_selection_for_mysql_pymysql(depcheck):
    """A `mysql+pymysql://` URL must resolve to the pymysql dialect by name
    (the scheme operators put in DATABASE_URL). Pin the scheme->driver mapping."""
    depcheck.load(IMPORT_NAME)
    sa_engine = depcheck.try_load("sqlalchemy.engine")
    if sa_engine is None:
        pytest.skip("SQLAlchemy not installed; DB-API surface tests cover pymysql")
    url = sa_engine.make_url("mysql+pymysql://u:p@localhost/db")
    dialect = url.get_dialect()
    assert dialect.driver == "pymysql", (
        f"mysql+pymysql URL resolved to driver {dialect.driver!r}, expected pymysql"
    )


def test_not_imported_by_backend_marker():
    """Documentation guard (no dep assertion): the backend doesn't import pymysql
    directly; it's SQLAlchemy's MySQL DB-API driver, engaged only when
    DATABASE_URL uses the mysql+pymysql scheme. The DB-API + dialect pins above
    guard exactly the slice SQLAlchemy depends on."""
    assert True
