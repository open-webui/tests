"""Dependency contract: psycopg (psycopg v3, import name ``psycopg``).

psycopg v3 is the Open WebUI backend's **async** PostgreSQL driver. The
backend never calls ``psycopg.connect`` itself; it goes through SQLAlchemy.
What it relies on is documented explicitly in ``internal/db.py``:

  - ``_make_async_url`` rewrites ``postgresql://`` / ``postgres://`` /
    ``postgresql+psycopg2://`` URLs to ``postgresql+psycopg://`` so the
    async engine uses this driver;
  - the comments state the design assumption — *"psycopg (v3) speaks libpq
    natively, so all standard connection-string parameters (``sslmode``,
    ``sslrootcert``, ``options``, ``target_session_attrs``, …) are passed
    through without any translation"*. That is why the SSL-param
    normalisation that ``extract_ssl_params_from_url`` does for the
    **psycopg2** sync engine is deliberately *skipped* for the async URL:
    psycopg v3 understands ``sslmode=require`` in the URL directly.

That load-bearing assumption is exactly what this module pins, OFFLINE and
with NO real database connection:

  - the DBAPI 2.0 module surface (``connect``, ``Connection``,
    ``AsyncConnection``, ``Cursor``/``AsyncCursor``, ``apilevel`` etc.);
  - the full DBAPI exception hierarchy SQLAlchemy maps onto its own
    exceptions (``Error`` → ``DatabaseError`` → ``OperationalError`` …);
  - the ``psycopg.conninfo`` parser round-trips a libpq URL — crucially,
    that ``sslmode`` and the cert-file keys survive
    ``conninfo_to_dict`` / ``make_conninfo`` unchanged, which is precisely
    why ``db.py`` can leave them in the async URL untouched.

A psycopg bump that dropped a DBAPI symbol, reshuffled the exception tree,
or changed how libpq connection strings parse would fail here instead of
surfacing as a connection failure (or a silently-dropped ``sslmode``,
i.e. an unencrypted DB connection) at runtime.

Uses the ``depcheck`` fixture from unit/deps/conftest.py.
"""

from __future__ import annotations

import inspect

import pytest

pytestmark = pytest.mark.depcheck

IMPORT_NAME = "psycopg"
DIST_NAME = "psycopg"

# DBAPI 2.0 module-level surface + the connection/cursor classes the async
# SQLAlchemy dialect drives.
USED_SYMBOLS = [
    "connect",
    "Connection",
    "AsyncConnection",
    "Cursor",
    "AsyncCursor",
    "ClientCursor",
    "Error",
    "conninfo.make_conninfo",
    "conninfo.conninfo_to_dict",
    "errors",
]

# The DBAPI module-level dunder constants SQLAlchemy / the DBAPI spec read.
DBAPI_MODULE_ATTRS = ["apilevel", "threadsafety", "paramstyle", "Binary"]

# The standard DBAPI exception classes (SQLAlchemy maps each of these).
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


def _conninfo(depcheck):
    return depcheck.resolve(depcheck.load(IMPORT_NAME), "conninfo")


# --------------------------------------------------------------------------- #
# Import / version
# --------------------------------------------------------------------------- #


def test_import(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    assert mod.__name__ == "psycopg"


def test_is_version_3(depcheck):
    """db.py's whole async story assumes psycopg *v3* (libpq-native). Pin the
    major version so an accidental v2 alias would fail loudly."""
    mod = depcheck.load(IMPORT_NAME)
    ver = getattr(mod, "__version__", None) or depcheck.dist_version(DIST_NAME)
    assert ver is not None
    assert ver.split(".")[0] == "3", f"expected psycopg v3, got {ver!r}"


def test_version_reported(depcheck):
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
    """SQLAlchemy's psycopg dialect treats this as a DBAPI 2.0 module."""
    mod = depcheck.load(IMPORT_NAME)
    assert mod.apilevel == "2.0"
    # psycopg v3 uses pyformat paramstyle; pin it.
    assert mod.paramstyle == "pyformat"


def test_connect_is_callable(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_callable(mod, "connect")


def test_connection_classes_are_classes(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    assert inspect.isclass(mod.Connection)
    assert inspect.isclass(mod.AsyncConnection)
    assert inspect.isclass(mod.Cursor)
    assert inspect.isclass(mod.AsyncCursor)


def test_connection_classmethod_connect_exists(depcheck):
    """SQLAlchemy's async dialect uses AsyncConnection.connect(conninfo, ...);
    the sync one uses Connection.connect. Pin both classmethods exist."""
    mod = depcheck.load(IMPORT_NAME)
    for cls in (mod.Connection, mod.AsyncConnection):
        names = set(dir(cls))
        assert "connect" in names, f"{cls.__name__}.connect missing"
        assert callable(getattr(cls, "connect"))


# --------------------------------------------------------------------------- #
# connect() signature (what SQLAlchemy / a libpq URL flow through)
# --------------------------------------------------------------------------- #


def test_connect_accepts_conninfo_first(depcheck):
    """psycopg.connect(conninfo, **kwargs). The first parameter is the libpq
    connection string and must be positional."""
    mod = depcheck.load(IMPORT_NAME)
    sig = inspect.signature(mod.connect)
    params = list(sig.parameters.values())
    assert params, "psycopg.connect has no parameters"
    first = params[0]
    assert first.name == "conninfo"
    assert first.kind in (
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    )


def test_connect_accepts_libpq_kwargs(depcheck):
    """connect must accept arbitrary libpq connection params via **kwargs
    (autocommit/row_factory are explicit; sslmode/host/etc go through kwargs)."""
    mod = depcheck.load(IMPORT_NAME)
    sig = inspect.signature(mod.connect)
    has_var_kw = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    assert has_var_kw, "psycopg.connect no longer accepts **kwargs libpq params"
    # autocommit is the one explicit kwarg SQLAlchemy commonly toggles.
    assert "autocommit" in sig.parameters


# --------------------------------------------------------------------------- #
# Exception hierarchy (SQLAlchemy maps every one of these)
# --------------------------------------------------------------------------- #


def test_dbapi_exceptions_exist(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_symbols(mod, DBAPI_EXCEPTIONS)


def test_exception_hierarchy(depcheck):
    """The DBAPI 2.0 tree: every concrete error subclasses DatabaseError which
    subclasses Error. SQLAlchemy's error translation relies on this layering;
    if OperationalError stopped subclassing DatabaseError, connection-drop
    handling would silently change."""
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


def test_errors_module_reexports_base(depcheck):
    """psycopg.errors.* is where SQLAlchemy reaches for SQLSTATE-specific
    classes; the base Error must be reachable there too."""
    mod = depcheck.load(IMPORT_NAME)
    assert mod.errors.Error is mod.Error


# --------------------------------------------------------------------------- #
# Behavioural: conninfo parsing (the libpq-native contract db.py relies on)
# --------------------------------------------------------------------------- #


def test_conninfo_to_dict_parses_basic_url(depcheck):
    """A standard postgresql:// URL decomposes into the libpq keyword fields."""
    ci = _conninfo(depcheck)
    d = ci.conninfo_to_dict("postgresql://user:pw@dbhost:5432/mydb")
    assert d.get("user") == "user"
    assert d.get("host") == "dbhost"
    assert d.get("port") == "5432"
    assert d.get("dbname") == "mydb"


def test_conninfo_preserves_sslmode(depcheck):
    """THE load-bearing contract: db.py leaves ``sslmode`` in the async URL
    untouched because psycopg v3 understands it natively. Prove a
    ``sslmode=require`` URL parses to sslmode=require (i.e. it is NOT
    dropped or renamed), so the async connection really is TLS-enforced."""
    ci = _conninfo(depcheck)
    d = ci.conninfo_to_dict("postgresql://user:pw@dbhost:5432/mydb?sslmode=require")
    assert d.get("sslmode") == "require", (
        "psycopg no longer parses sslmode from the URL — db.py's async path "
        "assumes libpq-native sslmode handling; dropping it would silently "
        "downgrade the DB connection to plaintext"
    )


def test_conninfo_preserves_cert_file_params(depcheck):
    """The cert-file keys db.py knows about (sslrootcert/sslcert/sslkey) must
    also survive parsing so client-cert auth keeps working over the async
    driver."""
    ci = _conninfo(depcheck)
    url = (
        "postgresql://u:p@h:5432/db"
        "?sslmode=verify-full"
        "&sslrootcert=/etc/ssl/root.crt"
        "&sslcert=/etc/ssl/client.crt"
        "&sslkey=/etc/ssl/client.key"
    )
    d = ci.conninfo_to_dict(url)
    assert d.get("sslmode") == "verify-full"
    assert d.get("sslrootcert") == "/etc/ssl/root.crt"
    assert d.get("sslcert") == "/etc/ssl/client.crt"
    assert d.get("sslkey") == "/etc/ssl/client.key"


def test_make_conninfo_roundtrips_sslmode(depcheck):
    """make_conninfo builds a libpq string from kwargs; conninfo_to_dict must
    read it back with sslmode intact (full round-trip of the SSL contract)."""
    ci = _conninfo(depcheck)
    s = ci.make_conninfo("", host="h", port=5432, dbname="d", user="u", sslmode="require")
    assert isinstance(s, str)
    back = ci.conninfo_to_dict(s)
    assert back.get("sslmode") == "require"
    assert back.get("host") == "h"
    assert back.get("dbname") == "d"


def test_make_conninfo_merges_string_and_kwargs(depcheck):
    """make_conninfo(base_url, **overrides) merges a string with keyword
    params — the shape SQLAlchemy uses to layer options onto a URL."""
    ci = _conninfo(depcheck)
    s = ci.make_conninfo("postgresql://u@h:5432/db", connect_timeout=10, application_name="owui")
    back = ci.conninfo_to_dict(s)
    assert back.get("host") == "h"
    assert str(back.get("connect_timeout")) == "10"
    assert back.get("application_name") == "owui"


def test_conninfo_to_dict_empty_string(depcheck):
    """An empty conninfo is valid (all-default libpq connection) and yields a
    dict (possibly empty), never an error."""
    ci = _conninfo(depcheck)
    d = ci.conninfo_to_dict("")
    assert isinstance(d, dict)


def test_make_conninfo_rejects_unknown_keyword(depcheck):
    """make_conninfo validates keywords against libpq; a bogus key must raise
    a ProgrammingError (psycopg's DBAPI error), not silently pass an invalid
    option through to the connection."""
    mod = depcheck.load(IMPORT_NAME)
    ci = mod.conninfo
    with pytest.raises(mod.ProgrammingError):
        ci.make_conninfo("", definitely_not_a_libpq_keyword="x")
