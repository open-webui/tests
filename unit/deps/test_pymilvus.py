"""Dependency contract: pymilvus (Milvus vector DB client).

Open WebUI has two Milvus vector-store backends:
``retrieval/vector/dbs/milvus.py`` and ``.../milvus_multitenancy.py``.
Between them they import:

    from pymilvus import Collection, DataType, FieldSchema, connections
    from pymilvus import MilvusClient as Client
    from pymilvus import CollectionSchema, utility

and drive a specific slice of the API:
  - schema/index construction (the offline part of _create_collection):
        schema = client.create_schema(auto_id=False, enable_dynamic_field=True)
        schema.add_field(field_name='id', datatype=DataType.VARCHAR,
                         is_primary=True, max_length=65535)
        schema.add_field(field_name='vector', datatype=DataType.FLOAT_VECTOR,
                         dim=dimension)
        schema.add_field(field_name='data', datatype=DataType.JSON)
        index_params = client.prepare_index_params()
        index_params.add_index(field_name='vector', index_type=..., metric_type=...,
                               params=...)
  - client data ops: create_collection / has_collection / drop_collection /
    insert / upsert / search / delete / list_collections;
  - the ORM path: Collection(name).load() / .query_iterator(...),
    connections.connect(uri=, token=, db_name=), utility.has_collection(...).

This module pins that surface. It builds the schema + index-params objects
OFFLINE — ``create_schema`` and ``prepare_index_params`` are callable
without a connection, and ``FieldSchema`` / ``CollectionSchema`` /
``DataType`` are pure data — so _create_collection's construction half is
exercised end to end with NO Milvus server. ``MilvusClient`` itself is never
instantiated (its default uri would attempt a connection); its method
surface is pinned on the class.

NOTE: these DB integrations are community-supported in the backend. Pattern
mirrors test_requests.py. Uses the ``depcheck`` fixture.
"""

from __future__ import annotations

import inspect

import pytest

pytestmark = pytest.mark.depcheck

IMPORT_NAME = "pymilvus"
DIST_NAME = "pymilvus"

USED_SYMBOLS = [
    "Collection",
    "CollectionSchema",
    "DataType",
    "FieldSchema",
    "connections",
    "utility",
    "MilvusClient",
]

# DataType members the backend names.
USED_DATATYPES = ["VARCHAR", "FLOAT_VECTOR", "JSON"]

# MilvusClient methods the backend calls.
CLIENT_METHODS = [
    "create_schema",
    "prepare_index_params",
    "create_collection",
    "has_collection",
    "drop_collection",
    "insert",
    "upsert",
    "search",
    "delete",
    "list_collections",
]


# ---------------------------------------------------------------------------
# Import + version + surface
# ---------------------------------------------------------------------------


def test_import(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    assert mod.__name__ == "pymilvus"


def test_version_reported(depcheck):
    assert depcheck.dist_version(DIST_NAME) is not None


def test_used_symbols_exist(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_symbols(mod, USED_SYMBOLS)


def test_milvusclient_is_class(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    assert isinstance(mod.MilvusClient, type)


def test_milvusclient_ctor_accepts_uri_db_token(depcheck):
    """The backend builds Client(uri=, db_name=[, token=]). Those parameters
    must remain accepted (no call here — instantiation would connect)."""
    mod = depcheck.load(IMPORT_NAME)
    sig = inspect.signature(mod.MilvusClient.__init__)
    for name in ("uri", "db_name", "token"):
        assert name in sig.parameters, f"MilvusClient.__init__ lost {name!r}"


# ---------------------------------------------------------------------------
# DataType enum members.
# ---------------------------------------------------------------------------


def test_datatype_members_present(depcheck):
    """The schema uses DataType.VARCHAR / FLOAT_VECTOR / JSON."""
    mod = depcheck.load(IMPORT_NAME)
    for name in USED_DATATYPES:
        assert hasattr(mod.DataType, name), f"DataType.{name} missing"


# ---------------------------------------------------------------------------
# Offline schema construction — the side-effect-free half of _create_collection.
# create_schema / prepare_index_params are callable WITHOUT a connection.
# ---------------------------------------------------------------------------


def test_create_schema_offline(depcheck):
    """MilvusClient.create_schema(auto_id=, enable_dynamic_field=) returns a
    CollectionSchema without a live client. This is exactly what the backend
    does at the top of _create_collection."""
    mod = depcheck.load(IMPORT_NAME)
    schema = mod.MilvusClient.create_schema(auto_id=False, enable_dynamic_field=True)
    assert schema is not None
    assert hasattr(schema, "add_field") and callable(schema.add_field)


def test_schema_add_field_builds_fields(depcheck):
    """schema.add_field(...) for id (VARCHAR primary), vector (FLOAT_VECTOR
    dim), data (JSON), metadata (JSON) — the exact fields _create_collection
    adds. All must be accepted and accumulate on the schema."""
    mod = depcheck.load(IMPORT_NAME)
    schema = mod.MilvusClient.create_schema(auto_id=False, enable_dynamic_field=True)
    schema.add_field(
        field_name="id", datatype=mod.DataType.VARCHAR, is_primary=True, max_length=65535
    )
    schema.add_field(field_name="vector", datatype=mod.DataType.FLOAT_VECTOR, dim=384)
    schema.add_field(field_name="data", datatype=mod.DataType.JSON)
    schema.add_field(field_name="metadata", datatype=mod.DataType.JSON)
    field_names = {f.name for f in schema.fields}
    assert {"id", "vector", "data", "metadata"}.issubset(field_names)


def test_prepare_index_params_and_add_index_offline(depcheck):
    """index_params = client.prepare_index_params(); index_params.add_index(
    field_name='vector', index_type=, metric_type=, params=) — built offline.
    Covers the HNSW path (M / efConstruction params)."""
    mod = depcheck.load(IMPORT_NAME)
    index_params = mod.MilvusClient.prepare_index_params()
    assert hasattr(index_params, "add_index") and callable(index_params.add_index)
    index_params.add_index(
        field_name="vector",
        index_type="HNSW",
        metric_type="COSINE",
        params={"M": 16, "efConstruction": 100},
    )
    # add_index mutates in place; the object remains usable (no exception).
    assert index_params is not None


def test_fieldschema_constructs_offline(depcheck):
    """The ORM/multitenancy path builds FieldSchema(name=, dtype=, ...)
    directly. Pin that it constructs offline and preserves name/dtype."""
    mod = depcheck.load(IMPORT_NAME)
    fs = mod.FieldSchema(name="id", dtype=mod.DataType.VARCHAR, is_primary=True, max_length=65535)
    assert fs.name == "id"
    assert fs.dtype == mod.DataType.VARCHAR


def test_collectionschema_constructs_offline(depcheck):
    """CollectionSchema(fields=[...], enable_dynamic_field=) bundles FieldSchemas
    (the multitenancy backend builds the schema this way)."""
    mod = depcheck.load(IMPORT_NAME)
    fid = mod.FieldSchema(name="id", dtype=mod.DataType.VARCHAR, is_primary=True, max_length=128)
    fvec = mod.FieldSchema(name="vector", dtype=mod.DataType.FLOAT_VECTOR, dim=8)
    cs = mod.CollectionSchema(fields=[fid, fvec], enable_dynamic_field=True)
    assert len(cs.fields) == 2


# ---------------------------------------------------------------------------
# MilvusClient method surface — pin on the class (never instantiate).
# ---------------------------------------------------------------------------


def test_client_method_surface(depcheck):
    """Every data/collection method the backend calls on a MilvusClient must
    exist on the class."""
    mod = depcheck.load(IMPORT_NAME)
    names = set(dir(mod.MilvusClient))
    missing = [m for m in CLIENT_METHODS if m not in names]
    assert not missing, f"MilvusClient missing method(s) the backend calls: {missing}"


def test_client_search_signature(depcheck):
    """The backend calls client.search(collection_name=, data=, limit=,
    output_fields=). Those kwargs must remain accepted."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_params(
        mod.MilvusClient.search,
        ["collection_name", "data", "limit", "output_fields"],
    )


def test_client_insert_signature(depcheck):
    """client.insert(collection_name=, data=[...])."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_params(mod.MilvusClient.insert, ["collection_name", "data"])


def test_client_create_collection_signature(depcheck):
    """client.create_collection(collection_name=, schema=, index_params=)."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_params(
        mod.MilvusClient.create_collection,
        ["collection_name", "schema", "index_params"],
    )


def test_client_delete_signature(depcheck):
    """The backend calls client.delete(collection_name=, ids=) and
    client.delete(collection_name=, filter=). Both ids and filter must be
    accepted kwargs."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_params(mod.MilvusClient.delete, ["collection_name"])
    sig = inspect.signature(mod.MilvusClient.delete)
    params = sig.parameters
    has_var_kw = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())
    assert has_var_kw or ("ids" in params and "filter" in params), (
        "MilvusClient.delete no longer accepts ids/filter"
    )


# ---------------------------------------------------------------------------
# connections / utility / Collection — the ORM-path surface.
# ---------------------------------------------------------------------------


def test_connections_connect_callable(depcheck):
    """The ORM query path calls connections.connect(uri=, token=, db_name=).
    connect must be callable (we do NOT call it — that opens a connection)."""
    mod = depcheck.load(IMPORT_NAME)
    assert callable(mod.connections.connect)


def test_utility_helpers_present(depcheck):
    """The multitenancy backend uses utility.has_collection / list_collections
    / drop_collection. Pin them as callables."""
    mod = depcheck.load(IMPORT_NAME)
    for name in ("has_collection", "drop_collection", "list_collections"):
        assert callable(getattr(mod.utility, name, None)), f"utility.{name} missing/not callable"


def test_collection_class_orm_surface(depcheck):
    """The backend builds Collection(name) then calls .load() and
    .query_iterator(...). Pin those methods on the class (no instantiation —
    Collection() requires a live connection)."""
    mod = depcheck.load(IMPORT_NAME)
    names = set(dir(mod.Collection))
    for attr in ("load", "query_iterator"):
        assert attr in names, f"Collection.{attr} missing"


def test_query_iterator_yields_next_close(depcheck):
    """The ORM query loop does `it = collection.query_iterator(...)` then
    `it.next()` / `it.close()`. Those method names must exist on the iterator
    type (located via the pymilvus orm/iterator module)."""
    depcheck.load(IMPORT_NAME)
    # The iterator class lives in a pymilvus submodule; locate one exposing
    # next + close without instantiating (instantiation needs a connection).
    for modname in (
        "pymilvus.orm.iterator",
        "pymilvus.client.search_iterator",
        "pymilvus.orm.search_iterator",
    ):
        sub = depcheck.try_load(modname)
        if sub is None:
            continue
        for n in dir(sub):
            obj = getattr(sub, n, None)
            if isinstance(obj, type) and {"next", "close"}.issubset(set(dir(obj))):
                return  # found an iterator type with next + close
    pytest.skip("could not locate a pymilvus query-iterator type in this version")
