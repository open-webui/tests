"""Dependency contract: pgvector (import name ``pgvector``).

pgvector provides the SQLAlchemy column types Open WebUI uses for its
PostgreSQL vector store (and the opengauss variant). Both
``retrieval/vector/dbs/pgvector.py`` and ``.../opengauss.py`` do:

    from pgvector.sqlalchemy import HALFVEC, Vector

and then:

  * pick a column type by dimensionality —
    ``VECTOR_TYPE_FACTORY = HALFVEC if USE_HALFVEC else Vector`` — and declare
    the embedding column as
    ``Column(VECTOR_TYPE_FACTORY(dim=VECTOR_LENGTH), nullable=True)``;
  * rank search results by cosine distance on that column —
    ``DocumentChunk.vector.cosine_distance(query_vector)`` in the ORDER BY /
    SELECT (this compiles to PostgreSQL's ``<=>`` operator).

So the contract is: ``Vector`` and ``HALFVEC`` import from
``pgvector.sqlalchemy``, are SQLAlchemy types constructible with a ``dim``
argument, compile to the ``VECTOR(n)`` / ``HALFVEC(n)`` column DDL, and expose
the distance comparators (``cosine_distance`` -> ``<=>``, plus ``l2_distance``
and ``max_inner_product``) used to build the similarity query.

A real query needs a PostgreSQL server with the pgvector extension, which this
offline suite must NOT touch. But everything above can be verified by *compiling
SQL* (never executing it) against SQLAlchemy's PostgreSQL dialect — no
connection, no server. This module pins that surface and those compilations, so
a pgvector bump that removed/renamed the types, changed the ``dim`` constructor,
or altered the distance-operator SQL fails loudly here instead of at vector-
search time.

Pattern mirrors the unit/deps/ exemplar: symbol-existence + signature checks +
offline SQL-compilation contracts (NO database, NO network). Uses the
`depcheck` fixture from unit/deps/conftest.py.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.depcheck

IMPORT_NAME = "pgvector"
DIST_NAME = "pgvector"

# The two column types the backend imports from pgvector.sqlalchemy.
USED_SYMBOLS = ["Vector", "HALFVEC"]

# Distance comparators the search query builds on a Vector/HALFVEC column.
DISTANCE_OPS = ["cosine_distance", "l2_distance", "max_inner_product"]


def _pg_dialect(depcheck):
    """SQLAlchemy PostgreSQL dialect used to COMPILE (not run) the vector SQL."""
    depcheck.load("sqlalchemy")
    dialects = depcheck.load("sqlalchemy.dialects")
    from sqlalchemy.dialects import postgresql

    assert dialects is not None
    return postgresql.dialect()


def _vector_table(depcheck, vector_type, dim=3):
    """Build a SQLAlchemy table with one vector column of the given type — the
    offline analogue of the backend's DocumentChunk table. No engine/connection
    is created."""
    sa = depcheck.load("sqlalchemy")
    md = sa.MetaData()
    table = sa.Table(
        "document_chunk",
        md,
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("vector", vector_type(dim=dim)),
    )
    return table


# --------------------------------------------------------------------------- #
# Import + version + API surface
# --------------------------------------------------------------------------- #
def test_import(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    assert mod.__name__ == "pgvector"


def test_version_reported(depcheck):
    """The installed distribution version must be resolvable so bump tooling
    and this suite agree on what's under test."""
    depcheck.load(IMPORT_NAME)
    assert depcheck.dist_version(DIST_NAME) is not None


def test_sqlalchemy_submodule_importable(depcheck):
    """The backend does ``from pgvector.sqlalchemy import HALFVEC, Vector`` — the
    sqlalchemy submodule must import cleanly."""
    depcheck.load(IMPORT_NAME)
    mod = depcheck.try_load("pgvector.sqlalchemy")
    assert mod is not None, "pgvector.sqlalchemy no longer importable"


def test_used_symbols_exist(depcheck):
    """``Vector`` and ``HALFVEC`` must both resolve on pgvector.sqlalchemy."""
    depcheck.load(IMPORT_NAME)
    mod = depcheck.load("pgvector.sqlalchemy")
    depcheck.assert_symbols(mod, USED_SYMBOLS)


def test_types_are_sqlalchemy_types(depcheck):
    """Vector/HALFVEC are used as SQLAlchemy column types, so both must subclass
    SQLAlchemy's ``TypeEngine`` (the base every column type derives from)."""
    depcheck.load(IMPORT_NAME)
    mod = depcheck.load("pgvector.sqlalchemy")
    sa = depcheck.load("sqlalchemy")
    from sqlalchemy.types import TypeEngine

    assert sa is not None
    assert issubclass(mod.Vector, TypeEngine), "Vector is not a SQLAlchemy type"
    assert issubclass(mod.HALFVEC, TypeEngine), "HALFVEC is not a SQLAlchemy type"


# --------------------------------------------------------------------------- #
# Constructor — Column(VECTOR_TYPE_FACTORY(dim=VECTOR_LENGTH), ...)
# --------------------------------------------------------------------------- #
def test_vector_constructs_with_dim(depcheck):
    """The backend builds the column type as ``Vector(dim=VECTOR_LENGTH)``. Pin
    that ``dim`` is the constructor parameter and an instance constructs."""
    depcheck.load(IMPORT_NAME)
    mod = depcheck.load("pgvector.sqlalchemy")
    depcheck.assert_params(mod.Vector.__init__, ["dim"])
    assert mod.Vector(dim=384) is not None


def test_halfvec_constructs_with_dim(depcheck):
    """HALFVEC is the >2000-dim path (``VECTOR_TYPE_FACTORY = HALFVEC if
    USE_HALFVEC``). Pin its ``dim`` constructor too."""
    depcheck.load(IMPORT_NAME)
    mod = depcheck.load("pgvector.sqlalchemy")
    depcheck.assert_params(mod.HALFVEC.__init__, ["dim"])
    assert mod.HALFVEC(dim=4096) is not None


# --------------------------------------------------------------------------- #
# Column DDL compilation — VECTOR(n) / HALFVEC(n) (offline, no DB)
# --------------------------------------------------------------------------- #
def test_vector_column_type_compiles_to_vector_ddl(depcheck):
    """``Vector(dim=N)`` must compile to the PostgreSQL ``VECTOR(N)`` column DDL
    — the type the embedding column is created as. Compiled, never executed."""
    depcheck.load(IMPORT_NAME)
    mod = depcheck.load("pgvector.sqlalchemy")
    dialect = _pg_dialect(depcheck)
    ddl = mod.Vector(dim=3).compile(dialect=dialect)
    assert "VECTOR(3)" in str(ddl).upper()


def test_halfvec_column_type_compiles_to_halfvec_ddl(depcheck):
    """``HALFVEC(dim=N)`` compiles to ``HALFVEC(N)`` DDL."""
    depcheck.load(IMPORT_NAME)
    mod = depcheck.load("pgvector.sqlalchemy")
    dialect = _pg_dialect(depcheck)
    ddl = mod.HALFVEC(dim=5).compile(dialect=dialect)
    assert "HALFVEC(5)" in str(ddl).upper()


# --------------------------------------------------------------------------- #
# Distance comparators — the similarity query operators (offline SQL compile)
# --------------------------------------------------------------------------- #
def test_vector_column_has_distance_operators(depcheck):
    """A Vector column must expose cosine_distance / l2_distance /
    max_inner_product — the comparator methods the search query calls. Build a
    table with a Vector column and assert each operator is callable."""
    depcheck.load(IMPORT_NAME)
    mod = depcheck.load("pgvector.sqlalchemy")
    table = _vector_table(depcheck, mod.Vector)
    col = table.c.vector
    for op in DISTANCE_OPS:
        assert callable(getattr(col, op, None)), f"Vector column missing {op}()"


def test_cosine_distance_compiles_to_pg_operator(depcheck):
    """The headline contract: ``column.cosine_distance(query_vector)`` must
    compile to PostgreSQL's cosine-distance operator ``<=>`` — exactly what the
    backend's ORDER BY relies on for similarity ranking. Compiled against the
    PostgreSQL dialect, never executed.
    """
    depcheck.load(IMPORT_NAME)
    mod = depcheck.load("pgvector.sqlalchemy")
    dialect = _pg_dialect(depcheck)
    table = _vector_table(depcheck, mod.Vector)
    expr = table.c.vector.cosine_distance([1.0, 2.0, 3.0])
    sql = str(expr.compile(dialect=dialect))
    assert "<=>" in sql, f"cosine_distance no longer compiles to <=>: {sql}"


def test_l2_distance_compiles_to_pg_operator(depcheck):
    """``l2_distance`` compiles to the Euclidean-distance operator ``<->``."""
    depcheck.load(IMPORT_NAME)
    mod = depcheck.load("pgvector.sqlalchemy")
    dialect = _pg_dialect(depcheck)
    table = _vector_table(depcheck, mod.Vector)
    sql = str(table.c.vector.l2_distance([1.0, 2.0, 3.0]).compile(dialect=dialect))
    assert "<->" in sql, f"l2_distance no longer compiles to <->: {sql}"


def test_max_inner_product_compiles_to_pg_operator(depcheck):
    """``max_inner_product`` compiles to the inner-product operator ``<#>``."""
    depcheck.load(IMPORT_NAME)
    mod = depcheck.load("pgvector.sqlalchemy")
    dialect = _pg_dialect(depcheck)
    table = _vector_table(depcheck, mod.Vector)
    sql = str(table.c.vector.max_inner_product([1.0, 2.0, 3.0]).compile(dialect=dialect))
    assert "<#>" in sql, f"max_inner_product no longer compiles to <#>: {sql}"


def test_halfvec_column_supports_cosine_distance(depcheck):
    """The HALFVEC path (large embeddings) must support the same cosine_distance
    ranking — the backend uses ``halfvec_cosine_ops`` and the same operator on
    HALFVEC columns. Pin that a HALFVEC column compiles cosine_distance to <=>.
    """
    depcheck.load(IMPORT_NAME)
    mod = depcheck.load("pgvector.sqlalchemy")
    dialect = _pg_dialect(depcheck)
    table = _vector_table(depcheck, mod.HALFVEC)
    sql = str(table.c.vector.cosine_distance([1.0, 2.0, 3.0]).compile(dialect=dialect))
    assert "<=>" in sql


def test_order_by_cosine_distance_compiles(depcheck):
    """End-to-end shape of the search query: a SELECT ordered by
    ``cosine_distance`` must compile to valid PostgreSQL containing the column,
    the <=> operator and ORDER BY — the exact structure
    ``DocumentChunk.vector.cosine_distance(...)`` in the loader produces. No
    execution, just compilation.
    """
    depcheck.load(IMPORT_NAME)
    mod = depcheck.load("pgvector.sqlalchemy")
    sa = depcheck.load("sqlalchemy")
    dialect = _pg_dialect(depcheck)
    table = _vector_table(depcheck, mod.Vector)

    distance = table.c.vector.cosine_distance([0.1, 0.2, 0.3]).label("distance")
    stmt = sa.select(table.c.id, distance).order_by(distance).limit(5)
    sql = str(stmt.compile(dialect=dialect)).upper()
    assert "ORDER BY" in sql
    assert "<=>" in sql
    assert "DOCUMENT_CHUNK" in sql or "DOCUMENT_CHUNK.VECTOR" in sql
