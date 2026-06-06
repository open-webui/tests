"""Dependency contract: SQLAlchemy.

SQLAlchemy is the Open WebUI backend's entire persistence layer. Every
model in `open_webui/models/*` is a `declarative_base()` ORM class built
from `Column` + the SQLAlchemy type system; all runtime reads/writes go
through the 2.0-style `select()/insert()/update()/delete()` constructs
executed on an `AsyncSession` (`sqlalchemy.ext.asyncio`), while a sync
`Session` / `scoped_session` handles startup config + Alembic. The custom
`JSONField(types.TypeDecorator)` in `internal/db.py` stores arbitrary JSON
as TEXT for SQLite/Postgres portability. A SQLAlchemy bump (e.g.
2.0.48 -> 2.0.50) that renamed/removed any of this would otherwise surface
as an AttributeError deep in a request path or a silent ORM behaviour
change; these tests pin the slice the codebase actually relies on.

Pattern mirrors unit/deps/test_requests.py: symbol-existence checks (the
API surface) plus offline behavioural contracts. Behavioural contracts use
only in-memory SQLite (`sqlite:///:memory:` and, when aiosqlite is present,
`sqlite+aiosqlite:///:memory:`) — never an external database. Uses the
`depcheck` fixture from unit/deps/conftest.py.
"""

from __future__ import annotations

import asyncio
import json

import pytest

pytestmark = pytest.mark.depcheck

IMPORT_NAME = "sqlalchemy"
DIST_NAME = "SQLAlchemy"


# ── Symbol inventory (dotted paths resolved against the package) ──────
#
# Each entry is referenced somewhere under open_webui/. Grouped by the
# import site the codebase uses, so a failure points at what broke.

# Top-level constructors / SQL expression helpers used across models/*.
TOPLEVEL_SYMBOLS = [
    # engine / metadata / schema objects
    "create_engine",
    "engine_from_config",
    "MetaData",
    "Table",
    "Column",
    "Index",
    "ForeignKey",
    "UniqueConstraint",
    "PrimaryKeyConstraint",
    "Engine",
    "Dialect",
    "inspect",
    "event",
    # the column type system (models/* build columns from these)
    "String",
    "Text",
    "Integer",
    "BigInteger",
    "Boolean",
    "JSON",
    "DateTime",
    "Date",
    "LargeBinary",
    "types",
    # 2.0-style statement constructors
    "select",
    "insert",
    "update",
    "delete",
    # expression-language helpers
    "func",
    "text",
    "cast",
    "and_",
    "or_",
    "case",
    "exists",
    "column",
    "literal",
    "values",
    "pool",
]

# sqlalchemy.orm — the sync session machinery + declarative base.
ORM_SYMBOLS = [
    "orm.declarative_base",
    "orm.Session",
    "orm.scoped_session",
    "orm.sessionmaker",
]

# sqlalchemy.ext.* — async engine/session + the legacy declarative_base
# location internal/db.py imports, and the mutable extension used by the
# pgvector / opengauss vector stores.
EXT_SYMBOLS = [
    "ext.asyncio.create_async_engine",
    "ext.asyncio.AsyncSession",
    "ext.asyncio.async_sessionmaker",
    "ext.declarative.declarative_base",
    "ext.mutable.MutableDict",
]

# The custom type system: JSONField subclasses types.TypeDecorator with
# impl = types.UnicodeText.
TYPES_SYMBOLS = [
    "types.TypeDecorator",
    "types.UnicodeText",
    "types.String",
    "types.Text",
    "types.Integer",
    "types.BigInteger",
    "types.Boolean",
    "types.JSON",
    "types.DateTime",
]

# Connection pools selected explicitly in internal/db.py + vector stores.
POOL_SYMBOLS = [
    "pool.NullPool",
    "pool.QueuePool",
]

# Lower-level SQL / exception / reflection imports used by models + migrations.
MISC_SYMBOLS = [
    "sql.exists",
    "sql.true",
    "sql.table",
    "sql.column",
    "sql.select",
    "sql.update",
    "sql.expression.bindparam",
    "exc.NoSuchTableError",
    "engine.reflection.Inspector",
]

ALL_SYMBOL_GROUPS = {
    "toplevel": TOPLEVEL_SYMBOLS,
    "orm": ORM_SYMBOLS,
    "ext": EXT_SYMBOLS,
    "types": TYPES_SYMBOLS,
    "pool": POOL_SYMBOLS,
    "misc": MISC_SYMBOLS,
}


# ── Import + version ─────────────────────────────────────────────────


def test_import(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    assert mod.__name__ == "sqlalchemy"


def test_is_v2(depcheck):
    """The codebase is written against the 2.0 API (select()-as-statement,
    async engine, async_sessionmaker). Guard the major version."""
    mod = depcheck.load(IMPORT_NAME)
    assert mod.__version__.split(".")[0] == "2", (
        f"Expected SQLAlchemy 2.x, got {mod.__version__}. The Open WebUI "
        "backend relies on 2.0 semantics (Result.scalars(), async engine, "
        "async_sessionmaker)."
    )


def test_version_reported(depcheck):
    """Sanity: the installed distribution version is resolvable (so bump
    tooling and this suite agree on what's under test)."""
    assert depcheck.dist_version(DIST_NAME) is not None


# ── Symbol-existence: every used name must still resolve ──────────────


def test_toplevel_symbols_exist(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_symbols(mod, TOPLEVEL_SYMBOLS)


def test_orm_symbols_exist(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_symbols(mod, ORM_SYMBOLS)


def test_ext_symbols_exist(depcheck):
    """The async API + legacy declarative_base + mutable extension."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_symbols(mod, EXT_SYMBOLS)


def test_async_symbols_exist_even_without_driver(depcheck):
    """The async symbols must exist regardless of whether an async DB
    driver is installed — internal/db.py imports them unconditionally at
    module load, so a missing name breaks every backend entry point."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_symbols(
        mod,
        [
            "ext.asyncio.create_async_engine",
            "ext.asyncio.AsyncSession",
            "ext.asyncio.async_sessionmaker",
        ],
    )


def test_types_symbols_exist(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_symbols(mod, TYPES_SYMBOLS)


def test_pool_symbols_exist(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_symbols(mod, POOL_SYMBOLS)


def test_misc_symbols_exist(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_symbols(mod, MISC_SYMBOLS)


def test_all_symbol_groups_resolve(depcheck):
    """Belt-and-suspenders: resolve the full union in one pass so a bump
    that drops several names reports them all together."""
    mod = depcheck.load(IMPORT_NAME)
    everything = [name for group in ALL_SYMBOL_GROUPS.values() for name in group]
    depcheck.assert_symbols(mod, everything)


# ── Callability of the statement constructors / engine factory ───────


def test_statement_constructors_callable(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    for name in ("select", "insert", "update", "delete", "text", "cast"):
        depcheck.assert_callable(mod, name)


def test_engine_factories_callable(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_callable(mod, "create_engine")
    depcheck.assert_callable(mod, "ext.asyncio.create_async_engine")
    depcheck.assert_callable(mod, "orm.sessionmaker")
    depcheck.assert_callable(mod, "ext.asyncio.async_sessionmaker")
    depcheck.assert_callable(mod, "orm.scoped_session")
    depcheck.assert_callable(mod, "orm.declarative_base")


def test_func_and_logical_helpers_callable(depcheck):
    """func.now()/func.count()/func.max()/func.lower() and the and_/or_/
    case/exists boolean helpers are used throughout the model query layer."""
    mod = depcheck.load(IMPORT_NAME)
    for name in ("and_", "or_", "case", "exists", "column", "literal"):
        depcheck.assert_callable(mod, name)
    # func is a special factory object; its members are produced lazily.
    assert callable(mod.func.now)
    assert callable(mod.func.count)
    assert callable(mod.func.max)
    assert callable(mod.func.lower)
    assert callable(mod.func.coalesce)


# ── Signature contracts for the engine / sessionmaker kwargs we pass ──


def test_create_engine_accepts_our_kwargs(depcheck):
    """internal/db.py calls create_engine(url, pool_size=, max_overflow=,
    pool_timeout=, pool_recycle=, pool_pre_ping=, poolclass=, connect_args=,
    creator=, echo=). Those kwargs must remain accepted (or **kwargs)."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_params(
        mod.create_engine,
        [
            "pool_size",
            "max_overflow",
            "pool_timeout",
            "pool_recycle",
            "pool_pre_ping",
            "poolclass",
            "connect_args",
            "creator",
            "echo",
        ],
    )


def test_create_async_engine_accepts_our_kwargs(depcheck):
    """internal/db.py calls create_async_engine(url, connect_args=,
    pool_size=, pool_timeout=, pool_recycle=, pool_pre_ping=, poolclass=,
    max_overflow=)."""
    mod = depcheck.load(IMPORT_NAME)
    create_async_engine = depcheck.resolve(mod, "ext.asyncio.create_async_engine")
    depcheck.assert_params(
        create_async_engine,
        [
            "connect_args",
            "pool_size",
            "pool_timeout",
            "pool_recycle",
            "pool_pre_ping",
            "poolclass",
            "max_overflow",
        ],
    )


def test_sessionmaker_accepts_our_kwargs(depcheck):
    """SessionLocal = sessionmaker(autocommit=, autoflush=, bind=,
    expire_on_commit=)."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_params(
        mod.orm.sessionmaker,
        ["autocommit", "autoflush", "bind", "expire_on_commit"],
    )


def test_async_sessionmaker_accepts_our_kwargs(depcheck):
    """AsyncSessionLocal = async_sessionmaker(bind=, class_=, autocommit=,
    autoflush=, expire_on_commit=)."""
    mod = depcheck.load(IMPORT_NAME)
    async_sessionmaker = depcheck.resolve(mod, "ext.asyncio.async_sessionmaker")
    depcheck.assert_params(
        async_sessionmaker,
        ["bind", "class_", "autocommit", "autoflush", "expire_on_commit"],
    )


def test_metadata_accepts_schema_kwarg(depcheck):
    """metadata_obj = MetaData(schema=DATABASE_SCHEMA)."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_params(mod.MetaData, ["schema"])


def test_declarative_base_accepts_metadata_kwarg(depcheck):
    """Base = declarative_base(metadata=metadata_obj)."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_params(mod.orm.declarative_base, ["metadata"])


def test_event_listen_api(depcheck):
    """internal/db.py wires SQLite PRAGMAs via event.listen(engine,
    'connect', fn) and the @event.listens_for(...) decorator."""
    mod = depcheck.load(IMPORT_NAME)
    assert callable(mod.event.listen)
    assert callable(mod.event.listens_for)


# ── Column type instantiation (models build columns from these) ──────


def test_column_types_instantiate(depcheck):
    """Each scalar type used in a model column must instantiate. A removed
    type would break model class construction at import time."""
    mod = depcheck.load(IMPORT_NAME)
    # length-taking types
    assert mod.String(255) is not None
    assert mod.String() is not None
    # parameterless scalar types
    for tname in (
        "Text",
        "Integer",
        "BigInteger",
        "Boolean",
        "JSON",
        "DateTime",
        "Date",
        "LargeBinary",
    ):
        t = getattr(mod, tname)
        assert t() is not None, f"{tname}() failed to instantiate"


def test_column_construction_with_constraints(depcheck):
    """Mirror the model column idioms: primary_key/unique/nullable/default
    and a ForeignKey('table.col', ondelete=...)."""
    mod = depcheck.load(IMPORT_NAME)
    col = mod.Column(mod.Text, primary_key=True, unique=True)
    assert col.primary_key is True
    assert col.unique is True

    nullable_col = mod.Column(mod.BigInteger, nullable=True)
    assert nullable_col.nullable is True

    fk_col = mod.Column(mod.Text, mod.ForeignKey("channel.id", ondelete="CASCADE"), nullable=False)
    assert any(isinstance(fk, mod.ForeignKey) for fk in fk_col.foreign_keys)


def test_server_default_and_onupdate_with_func(depcheck):
    """internal/config.py: created_at = Column(DateTime,
    server_default=func.now()) / updated_at = Column(DateTime,
    onupdate=func.now()). The func.now() expression must be acceptable."""
    mod = depcheck.load(IMPORT_NAME)
    created = mod.Column(mod.DateTime, nullable=False, server_default=mod.func.now())
    assert created.server_default is not None
    updated = mod.Column(mod.DateTime, nullable=True, onupdate=mod.func.now())
    assert updated.onupdate is not None


def test_constraint_objects_construct(depcheck):
    """UniqueConstraint / PrimaryKeyConstraint / Index are passed in
    __table_args__ across models."""
    mod = depcheck.load(IMPORT_NAME)
    assert mod.UniqueConstraint("a", "b") is not None
    assert mod.PrimaryKeyConstraint("id") is not None
    assert mod.Index("ix_demo_a", "a") is not None


# ── Behavioural: full declarative model lifecycle on in-memory SQLite ─


def _build_demo_base(mod):
    """Construct a fresh declarative Base + a model that exercises the
    column types and the JSONField-style TypeDecorator the codebase uses.

    Returns (Base, Item, JSONField). A new Base per call keeps each test's
    table registry isolated (no cross-test metadata collisions).
    """
    declarative_base = mod.orm.declarative_base

    class JSONField(mod.types.TypeDecorator):
        """Mirror of internal/db.py JSONField: JSON stored as TEXT."""

        impl = mod.types.UnicodeText
        cache_ok = True

        def process_bind_param(self, value, dialect):
            return json.dumps(value) if value is not None else None

        def process_result_value(self, value, dialect):
            return json.loads(value) if value is not None else None

    Base = declarative_base()

    class Item(Base):
        __tablename__ = "demo_item"

        id = mod.Column(mod.Text, primary_key=True, unique=True)
        name = mod.Column(mod.String(255), nullable=False)
        count = mod.Column(mod.BigInteger, nullable=True)
        active = mod.Column(mod.Boolean, default=True)
        data = mod.Column(JSONField, nullable=True)
        meta = mod.Column(mod.JSON, nullable=True)

    return Base, Item, JSONField


def test_create_all_and_session_insert_select(depcheck):
    """End-to-end: declarative model -> create_all on in-memory SQLite ->
    insert an ORM instance via a Session -> read it back with select()."""
    mod = depcheck.load(IMPORT_NAME)
    Base, Item, _ = _build_demo_base(mod)

    engine = mod.create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)

    Session = mod.orm.sessionmaker(bind=engine, expire_on_commit=False)
    with Session() as session:
        session.add(Item(id="a1", name="alpha", count=3, active=True))
        session.commit()

        result = session.execute(mod.select(Item).where(Item.id == "a1"))
        row = result.scalars().first()
        assert row is not None
        assert row.name == "alpha"
        assert row.count == 3
        assert row.active is True

    engine.dispose()


def test_session_update_and_delete(depcheck):
    """2.0-style bulk update()/delete() executed on a Session, mirroring
    models/* (e.g. `await session.execute(delete(Auth).where(...))`)."""
    mod = depcheck.load(IMPORT_NAME)
    Base, Item, _ = _build_demo_base(mod)

    engine = mod.create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = mod.orm.sessionmaker(bind=engine, expire_on_commit=False)

    with Session() as session:
        session.add(Item(id="x", name="orig", count=1))
        session.add(Item(id="y", name="orig", count=2))
        session.commit()

        session.execute(mod.update(Item).where(Item.id == "x").values(name="updated"))
        session.commit()
        updated = session.execute(mod.select(Item).where(Item.id == "x")).scalars().first()
        assert updated.name == "updated"

        session.execute(mod.delete(Item).where(Item.id == "y"))
        session.commit()
        gone = session.execute(mod.select(Item).where(Item.id == "y")).scalars().first()
        assert gone is None

    engine.dispose()


def test_insert_construct_executes(depcheck):
    """The `insert()` construct (used in migrations + add flows) executes
    against a core connection."""
    mod = depcheck.load(IMPORT_NAME)
    Base, Item, _ = _build_demo_base(mod)

    engine = mod.create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with engine.begin() as conn:
        conn.execute(mod.insert(Item).values(id="i1", name="ins", count=9))
        count = conn.execute(mod.select(mod.func.count()).select_from(Item)).scalar()
        assert count == 1
    engine.dispose()


def test_typedecorator_json_roundtrip(depcheck):
    """The JSONField TypeDecorator must round-trip a nested Python object
    through TEXT storage unchanged — this is internal/db.py's portability
    mechanism and the contract every JSON column depends on."""
    mod = depcheck.load(IMPORT_NAME)
    Base, Item, _ = _build_demo_base(mod)

    engine = mod.create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = mod.orm.sessionmaker(bind=engine, expire_on_commit=False)

    payload = {"k": [1, 2, 3], "nested": {"flag": True, "s": "v"}}
    with Session() as session:
        session.add(Item(id="j1", name="json", data=payload))
        session.commit()

        row = session.execute(mod.select(Item).where(Item.id == "j1")).scalars().first()
        assert row.data == payload
        assert isinstance(row.data, dict)

    engine.dispose()


def test_typedecorator_stores_none(depcheck):
    """JSONField round-trips None (NULL) rather than the string 'null' —
    process_bind_param/process_result_value short-circuit on None."""
    mod = depcheck.load(IMPORT_NAME)
    Base, Item, _ = _build_demo_base(mod)

    engine = mod.create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = mod.orm.sessionmaker(bind=engine, expire_on_commit=False)

    with Session() as session:
        session.add(Item(id="n1", name="none", data=None))
        session.commit()
        row = session.execute(mod.select(Item).where(Item.id == "n1")).scalars().first()
        assert row.data is None

    engine.dispose()


def test_native_json_column_roundtrip(depcheck):
    """Some models use the native JSON type directly (Column(JSON)). On
    SQLite it must still round-trip a dict."""
    mod = depcheck.load(IMPORT_NAME)
    Base, Item, _ = _build_demo_base(mod)

    engine = mod.create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = mod.orm.sessionmaker(bind=engine, expire_on_commit=False)

    with Session() as session:
        session.add(Item(id="m1", name="meta", meta={"a": 1}))
        session.commit()
        row = session.execute(mod.select(Item).where(Item.id == "m1")).scalars().first()
        assert row.meta == {"a": 1}

    engine.dispose()


# ── Behavioural: SQL expression-language constructs used in queries ──


def test_text_construct_executes(depcheck):
    """`from sqlalchemy import text` + `session.execute(text('SELECT 1'))`
    is used for health checks and raw queries."""
    mod = depcheck.load(IMPORT_NAME)
    engine = mod.create_engine("sqlite:///:memory:")
    with engine.connect() as conn:
        value = conn.execute(mod.text("SELECT 1")).scalar()
        assert value == 1
    engine.dispose()


def test_text_bound_params(depcheck):
    """text() with bound params (:name) — the safe parametrised form."""
    mod = depcheck.load(IMPORT_NAME)
    engine = mod.create_engine("sqlite:///:memory:")
    with engine.connect() as conn:
        value = conn.execute(mod.text("SELECT :n + 1 AS r"), {"n": 41}).scalar()
        assert value == 42
    engine.dispose()


def test_func_count_and_aggregates(depcheck):
    """select(func.count()).select_from(...) and func.max() label() forms
    appear in the pagination/aggregation code paths."""
    mod = depcheck.load(IMPORT_NAME)
    Base, Item, _ = _build_demo_base(mod)
    engine = mod.create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = mod.orm.sessionmaker(bind=engine, expire_on_commit=False)

    with Session() as session:
        for i in range(5):
            session.add(Item(id=f"c{i}", name="x", count=i))
        session.commit()

        total = session.execute(mod.select(mod.func.count()).select_from(Item)).scalar()
        assert total == 5

        biggest = session.execute(mod.select(mod.func.max(Item.count).label("mx"))).scalar()
        assert biggest == 4

    engine.dispose()


def test_where_and_or_in_filters(depcheck):
    """and_/or_/.in_/.is_/.ilike — the boolean filter primitives the model
    query layer composes (e.g. groups.py / auths.py / automations.py)."""
    mod = depcheck.load(IMPORT_NAME)
    Base, Item, _ = _build_demo_base(mod)
    engine = mod.create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = mod.orm.sessionmaker(bind=engine, expire_on_commit=False)

    with Session() as session:
        session.add(Item(id="p1", name="alpha", count=1, active=True))
        session.add(Item(id="p2", name="beta", count=2, active=False))
        session.add(Item(id="p3", name="gamma", count=3, active=True))
        session.commit()

        # and_
        rows = (
            session.execute(mod.select(Item).where(mod.and_(Item.active.is_(True), Item.count > 1)))
            .scalars()
            .all()
        )
        assert {r.id for r in rows} == {"p3"}

        # or_
        rows = (
            session.execute(
                mod.select(Item).where(mod.or_(Item.name == "alpha", Item.name == "beta"))
            )
            .scalars()
            .all()
        )
        assert {r.id for r in rows} == {"p1", "p2"}

        # .in_
        rows = session.execute(mod.select(Item).where(Item.id.in_(["p1", "p3"]))).scalars().all()
        assert {r.id for r in rows} == {"p1", "p3"}

        # .ilike (case-insensitive LIKE) — used for search filters
        rows = session.execute(mod.select(Item).where(Item.name.ilike("ALP%"))).scalars().all()
        assert {r.id for r in rows} == {"p1"}

    engine.dispose()


def test_order_by_limit_offset(depcheck):
    """select().order_by().limit().offset() — the pagination shape used
    across list endpoints."""
    mod = depcheck.load(IMPORT_NAME)
    Base, Item, _ = _build_demo_base(mod)
    engine = mod.create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = mod.orm.sessionmaker(bind=engine, expire_on_commit=False)

    with Session() as session:
        for i in range(10):
            session.add(Item(id=f"o{i:02d}", name="x", count=i))
        session.commit()

        rows = (
            session.execute(mod.select(Item).order_by(Item.count.desc()).limit(3).offset(1))
            .scalars()
            .all()
        )
        assert [r.count for r in rows] == [8, 7, 6]

    engine.dispose()


def test_exists_subquery(depcheck):
    """select(exists().where(...)) — used by chats.py to test existence."""
    mod = depcheck.load(IMPORT_NAME)
    Base, Item, _ = _build_demo_base(mod)
    engine = mod.create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = mod.orm.sessionmaker(bind=engine, expire_on_commit=False)

    with Session() as session:
        session.add(Item(id="e1", name="x", count=1))
        session.commit()

        present = session.execute(mod.select(mod.exists().where(Item.id == "e1"))).scalar()
        assert present is True

        absent = session.execute(mod.select(mod.exists().where(Item.id == "nope"))).scalar()
        assert absent is False

    engine.dispose()


def test_cast_expression(depcheck):
    """cast(col, String) — automations.py / users.py cast JSON columns to
    text for ilike search. The cast construct must compile + execute."""
    mod = depcheck.load(IMPORT_NAME)
    engine = mod.create_engine("sqlite:///:memory:")
    with engine.connect() as conn:
        value = conn.execute(mod.select(mod.cast(mod.literal(123), mod.String))).scalar()
        assert str(value) == "123"
    engine.dispose()


def test_scalars_all_and_first(depcheck):
    """Result.scalars().all()/.first() and Result.first() are the 2.0 row
    consumption idioms the codebase uses everywhere."""
    mod = depcheck.load(IMPORT_NAME)
    Base, Item, _ = _build_demo_base(mod)
    engine = mod.create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = mod.orm.sessionmaker(bind=engine, expire_on_commit=False)

    with Session() as session:
        session.add(Item(id="s1", name="x", count=1))
        session.add(Item(id="s2", name="y", count=2))
        session.commit()

        result = session.execute(mod.select(Item).order_by(Item.id))
        rows = result.scalars().all()
        assert len(rows) == 2

        # Multi-entity select -> Result.first() yields a Row tuple.
        row = session.execute(mod.select(Item.id, Item.name).where(Item.id == "s1")).first()
        assert tuple(row) == ("s1", "x")

    engine.dispose()


# ── Behavioural: inspection + metadata reflection ────────────────────


def test_inspect_reports_tables_and_columns(depcheck):
    """`from sqlalchemy import inspect` (migrations/util.py) is used to read
    table/column metadata off a live connection."""
    mod = depcheck.load(IMPORT_NAME)
    Base, Item, _ = _build_demo_base(mod)
    engine = mod.create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)

    insp = mod.inspect(engine)
    assert "demo_item" in insp.get_table_names()
    colnames = {c["name"] for c in insp.get_columns("demo_item")}
    assert {"id", "name", "count", "active"}.issubset(colnames)
    engine.dispose()


def test_metadata_table_and_core_column(depcheck):
    """Core Table()/Column() (pgvector/opengauss + migrations build tables
    this way, not via the ORM)."""
    mod = depcheck.load(IMPORT_NAME)
    md = mod.MetaData()
    tbl = mod.Table(
        "core_demo",
        md,
        mod.Column("id", mod.Integer, primary_key=True),
        mod.Column("blob", mod.LargeBinary),
        mod.Column("body", mod.Text),
    )
    engine = mod.create_engine("sqlite:///:memory:")
    md.create_all(engine)

    with engine.begin() as conn:
        conn.execute(tbl.insert().values(id=1, body="hello"))
        got = conn.execute(mod.select(tbl.c.body).where(tbl.c.id == 1)).scalar()
        assert got == "hello"
    engine.dispose()


# ── Behavioural: async engine + AsyncSession (skips if no driver) ─────


def _aiosqlite_available(depcheck):
    return depcheck.try_load("aiosqlite") is not None


def test_async_roundtrip_with_aiosqlite(depcheck):
    """Full async path: create_async_engine('sqlite+aiosqlite:///:memory:')
    -> run_sync(create_all) -> async_sessionmaker -> AsyncSession add/commit
    -> await execute(select()).scalars(). This is the runtime DB path for
    every Open WebUI request. Skips cleanly if aiosqlite isn't installed
    (the async *symbols* are still asserted by the symbol-existence tests)."""
    mod = depcheck.load(IMPORT_NAME)
    if not _aiosqlite_available(depcheck):
        pytest.skip("aiosqlite not installed; async behavioural path skipped")

    create_async_engine = depcheck.resolve(mod, "ext.asyncio.create_async_engine")
    AsyncSession = depcheck.resolve(mod, "ext.asyncio.AsyncSession")
    async_sessionmaker = depcheck.resolve(mod, "ext.asyncio.async_sessionmaker")
    Base, Item, _ = _build_demo_base(mod)

    async def run() -> str:
        engine = create_async_engine("sqlite+aiosqlite:///:memory:")
        try:
            async with engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)

            maker = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)
            async with maker() as session:
                session.add(Item(id="async1", name="zeta", count=7, data={"x": 1}))
                await session.commit()

                result = await session.execute(mod.select(Item).where(Item.id == "async1"))
                obj = result.scalars().first()
                assert obj is not None
                assert obj.count == 7
                assert obj.data == {"x": 1}
                return obj.name
        finally:
            await engine.dispose()

    assert asyncio.run(run()) == "zeta"


def test_async_session_context_manager_shape(depcheck):
    """AsyncSessionLocal() is used as `async with ... as db:` and exposes
    async commit()/close()/execute(). Pin that shape (offline)."""
    mod = depcheck.load(IMPORT_NAME)
    if not _aiosqlite_available(depcheck):
        pytest.skip("aiosqlite not installed; async session shape skipped")

    create_async_engine = depcheck.resolve(mod, "ext.asyncio.create_async_engine")
    AsyncSession = depcheck.resolve(mod, "ext.asyncio.AsyncSession")
    async_sessionmaker = depcheck.resolve(mod, "ext.asyncio.async_sessionmaker")

    async def run() -> None:
        engine = create_async_engine("sqlite+aiosqlite:///:memory:")
        try:
            maker = async_sessionmaker(bind=engine, class_=AsyncSession)
            async with maker() as session:
                assert isinstance(session, AsyncSession)
                for meth in ("execute", "commit", "rollback", "close", "add"):
                    assert hasattr(session, meth), f"AsyncSession.{meth} missing"
                value = (await session.execute(mod.text("SELECT 1"))).scalar()
                assert value == 1
        finally:
            await engine.dispose()

    asyncio.run(run())


def test_async_engine_begin_run_sync(depcheck):
    """engine.begin() async context + conn.run_sync(metadata.create_all) is
    how schema is created on the async engine — pin both methods exist and
    work."""
    mod = depcheck.load(IMPORT_NAME)
    if not _aiosqlite_available(depcheck):
        pytest.skip("aiosqlite not installed; async engine path skipped")

    create_async_engine = depcheck.resolve(mod, "ext.asyncio.create_async_engine")
    Base, _, _ = _build_demo_base(mod)

    async def run() -> None:
        engine = create_async_engine("sqlite+aiosqlite:///:memory:")
        try:
            assert hasattr(engine, "begin")
            assert hasattr(engine, "dispose")
            async with engine.begin() as conn:
                assert hasattr(conn, "run_sync")
                await conn.run_sync(Base.metadata.create_all)
        finally:
            await engine.dispose()

    asyncio.run(run())


# ── Behavioural: connection-pool classes are accepted by create_engine ─


def test_nullpool_accepted_by_create_engine(depcheck):
    """internal/db.py passes poolclass=NullPool for non-pooled configs."""
    mod = depcheck.load(IMPORT_NAME)
    engine = mod.create_engine("sqlite:///:memory:", poolclass=mod.pool.NullPool)
    try:
        with engine.connect() as conn:
            assert conn.execute(mod.text("SELECT 1")).scalar() == 1
    finally:
        engine.dispose()


def test_queuepool_is_a_pool_class(depcheck):
    """QueuePool is selected for pooled Postgres/SQLCipher configs; assert
    it's a Pool subclass so poolclass=QueuePool stays valid."""
    mod = depcheck.load(IMPORT_NAME)
    from sqlalchemy.pool import Pool

    assert issubclass(mod.pool.QueuePool, Pool)
    assert issubclass(mod.pool.NullPool, Pool)


# ── Behavioural: scoped_session (ScopedSession in internal/db.py) ────


def test_scoped_session_proxies_session(depcheck):
    """ScopedSession = scoped_session(SessionLocal); the registry proxies
    Session methods (execute/add/commit/remove)."""
    mod = depcheck.load(IMPORT_NAME)
    Base, Item, _ = _build_demo_base(mod)
    engine = mod.create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)

    SessionLocal = mod.orm.sessionmaker(bind=engine, expire_on_commit=False)
    Scoped = mod.orm.scoped_session(SessionLocal)
    try:
        Scoped.add(Item(id="sc1", name="scoped", count=1))
        Scoped.commit()
        row = Scoped.execute(mod.select(Item).where(Item.id == "sc1")).scalars().first()
        assert row.name == "scoped"
    finally:
        Scoped.remove()
        engine.dispose()


# ── Behavioural: the TypeDecorator class contract itself ─────────────


def test_typedecorator_subclass_attributes(depcheck):
    """A TypeDecorator subclass must accept impl + cache_ok and expose the
    process_bind_param/process_result_value hooks JSONField overrides."""
    mod = depcheck.load(IMPORT_NAME)

    class MyJSON(mod.types.TypeDecorator):
        impl = mod.types.UnicodeText
        cache_ok = True

        def process_bind_param(self, value, dialect):
            return json.dumps(value) if value is not None else None

        def process_result_value(self, value, dialect):
            return json.loads(value) if value is not None else None

    inst = MyJSON()
    assert inst.cache_ok is True
    # impl_instance / impl resolve to the backing type
    assert isinstance(inst.impl_instance, mod.types.UnicodeText) or isinstance(inst.impl, type)
    # The bind/result hooks behave as the JSONField contract requires.
    assert inst.process_bind_param({"a": 1}, None) == '{"a": 1}'
    assert inst.process_result_value('{"a": 1}', None) == {"a": 1}
    assert inst.process_bind_param(None, None) is None
    assert inst.process_result_value(None, None) is None
