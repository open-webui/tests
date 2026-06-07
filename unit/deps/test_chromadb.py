"""Dependency contract: chromadb (import name ``chromadb``).

ChromaDB is Open WebUI's *default* vector store. The whole retrieval /
RAG layer routes through ``ChromaClient`` in
``retrieval/vector/dbs/chroma.py``, which constructs a chromadb client
(``PersistentClient`` for the embedded on-disk default, ``HttpClient``
when ``CHROMA_HTTP_HOST`` is set) and then drives a small, fixed slice of
the collection API: ``get_or_create_collection`` / ``get_collection`` /
``delete_collection`` / ``list_collections`` / ``reset`` on the client,
and ``add`` / ``query`` / ``get`` / ``upsert`` / ``delete`` / ``count``
on the collection. It also relies on ``chromadb.Settings`` (allow_reset,
anonymized_telemetry, the two ``chroma_client_auth_*`` fields) and on
``chromadb.utils.batch_utils.create_batches`` to chunk large inserts.

This module pins exactly that surface so the chromadb 1.5.2 -> 1.5.9 bump
(and any future bump) fails loudly here instead of as a runtime
``AttributeError`` / ``KeyError`` / ``TypeError`` deep in an ingest or
search request. Two layers, mirroring test_requests.py / test_redis.py:

  * symbol-existence + signature checks for the API surface, and
  * offline BEHAVIOURAL contracts driven by an in-memory EphemeralClient
    (no disk path, no server, no HttpClient, no network, no embedding
    model) that exercise the create -> add -> query -> get -> count ->
    delete lifecycle and assert the exact result-dict shapes the backend
    consumes (``result['ids']``, ``result['distances'][0]``,
    ``result['documents']``, ``result['metadatas']``).

Embeddings are supplied as plain Python float lists; the backend passes
model-produced vectors of the same shape, so no real embedder is loaded.

Uses the ``depcheck`` fixture from unit/deps/conftest.py.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.depcheck

IMPORT_NAME = "chromadb"
DIST_NAME = "chromadb"

# A valid collection name: chromadb enforces 3-512 chars from
# [a-zA-Z0-9._-], starting/ending alphanumeric. Open WebUI's collection
# names (knowledge/file ids) satisfy this; keep the test fixtures valid too.
COLL = "owui_dep_contract_col"


# ---------------------------------------------------------------------------
# Symbol inventory — every dotted name the Open WebUI backend resolves on
# the `chromadb` package and its submodules.
# ---------------------------------------------------------------------------

# Top-level package symbols (retrieval/vector/dbs/chroma.py + config.py).
TOP_LEVEL_SYMBOLS = [
    "Client",  # in-memory/embedded client factory (and our test driver)
    "PersistentClient",  # chroma.py default (embedded, on-disk)
    "HttpClient",  # chroma.py CHROMA_HTTP_HOST path (remote server)
    "EphemeralClient",  # pure in-memory client (used by these tests)
    "Settings",  # chroma.py: `from chromadb import Settings`
    "config.Settings",  # chromadb.Settings is re-exported from config
]

# chromadb.utils.batch_utils.create_batches — chroma.py chunks inserts.
BATCH_UTILS_SYMBOLS = ["create_batches"]

# Result TypedDicts the collection methods return (api.types).
API_TYPES_SYMBOLS = ["QueryResult", "GetResult"]

# Client methods chroma.py calls on the constructed client.
CLIENT_METHODS = [
    "get_or_create_collection",  # insert() / upsert()
    "get_collection",  # search() / query() / get() / delete()
    "create_collection",  # (general API surface)
    "delete_collection",  # delete_collection()
    "list_collections",  # has_collection()
    "reset",  # reset()
]

# Collection methods chroma.py calls on a collection object.
COLLECTION_METHODS = [
    "add",  # insert() (via create_batches)
    "query",  # search()
    "get",  # query() / get()
    "upsert",  # upsert()
    "delete",  # delete()
    "count",  # (general API surface)
]

# Settings fields chroma.py constructs Settings(**...) with.
SETTINGS_FIELDS = [
    "allow_reset",
    "anonymized_telemetry",
    "chroma_client_auth_provider",
    "chroma_client_auth_credentials",
]


# ---------------------------------------------------------------------------
# Import + version
# ---------------------------------------------------------------------------


def test_import(depcheck):
    """`chromadb` must import (skip cleanly if absent in this env)."""
    mod = depcheck.load(IMPORT_NAME)
    assert mod.__name__ == "chromadb"


def test_version_reported(depcheck):
    """The installed distribution version must be resolvable so bump tooling
    and this suite agree on what's under test."""
    depcheck.load(IMPORT_NAME)
    assert depcheck.dist_version(DIST_NAME) is not None


# ---------------------------------------------------------------------------
# Symbol-existence checks (API surface)
# ---------------------------------------------------------------------------


def test_top_level_symbols_exist(depcheck):
    """Every top-level `chromadb.*` symbol the codebase resolves must exist."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_symbols(mod, TOP_LEVEL_SYMBOLS)


def test_client_factories_callable(depcheck):
    """chroma.py calls chromadb.PersistentClient(...) / chromadb.HttpClient(...);
    these (and Client/EphemeralClient) must be callable factories."""
    mod = depcheck.load(IMPORT_NAME)
    for name in ("Client", "PersistentClient", "HttpClient", "EphemeralClient"):
        depcheck.assert_callable(mod, name)


def test_batch_utils_symbols_exist(depcheck):
    """chroma.py: `from chromadb.utils.batch_utils import create_batches`."""
    depcheck.load(IMPORT_NAME)
    batch_utils = depcheck.load("chromadb.utils.batch_utils")
    depcheck.assert_symbols(batch_utils, BATCH_UTILS_SYMBOLS)
    depcheck.assert_callable(batch_utils, "create_batches")


def test_api_types_result_symbols_exist(depcheck):
    """query()/get() return QueryResult/GetResult TypedDicts; chroma.py keys
    into them by ['ids'] / ['distances'] / ['documents'] / ['metadatas']."""
    depcheck.load(IMPORT_NAME)
    api_types = depcheck.load("chromadb.api.types")
    depcheck.assert_symbols(api_types, API_TYPES_SYMBOLS)


def test_default_tenant_database_constants_exist(depcheck):
    """chroma.py passes tenant=CHROMA_TENANT, database=CHROMA_DATABASE whose
    defaults are chromadb.DEFAULT_TENANT / DEFAULT_DATABASE (config.py)."""
    depcheck.load(IMPORT_NAME)
    api = depcheck.load("chromadb.api")
    depcheck.assert_symbols(api, ["DEFAULT_TENANT", "DEFAULT_DATABASE"])
    assert isinstance(api.DEFAULT_TENANT, str)
    assert isinstance(api.DEFAULT_DATABASE, str)


# ---------------------------------------------------------------------------
# Settings contract
# ---------------------------------------------------------------------------


def test_settings_is_config_settings(depcheck):
    """`from chromadb import Settings` and `from chromadb.config import Settings`
    must resolve to the same class (chroma.py imports the former)."""
    mod = depcheck.load(IMPORT_NAME)
    config = depcheck.load("chromadb.config")
    assert mod.Settings is config.Settings


def test_settings_accepts_our_fields(depcheck):
    """chroma.py builds Settings(allow_reset=, anonymized_telemetry=,
    chroma_client_auth_provider=, chroma_client_auth_credentials=) and reads
    them back. Constructing with all four must not raise and must round-trip."""
    mod = depcheck.load(IMPORT_NAME)
    s = mod.Settings(
        allow_reset=True,
        anonymized_telemetry=False,
        chroma_client_auth_provider="chromadb.auth.basic_authn.BasicAuthClientProvider",
        chroma_client_auth_credentials="user:pass",
    )
    assert s.allow_reset is True
    assert s.anonymized_telemetry is False
    assert s.chroma_client_auth_provider is not None
    assert s.chroma_client_auth_credentials is not None


def test_settings_declares_our_fields(depcheck):
    """The four Settings fields chroma.py sets must remain declared fields
    (pydantic-style), so a rename surfaces here rather than being silently
    swallowed by an over-permissive constructor."""
    mod = depcheck.load(IMPORT_NAME)
    fields = getattr(mod.Settings, "__fields__", None)
    if not fields:
        pytest.skip("chromadb.Settings exposes no introspectable field set")
    declared = set(fields)
    missing = [f for f in SETTINGS_FIELDS if f not in declared]
    assert not missing, f"chromadb.Settings no longer declares field(s): {missing}"


# ---------------------------------------------------------------------------
# Client constructor signatures (we never connect HttpClient — signature only)
# ---------------------------------------------------------------------------


def test_persistent_client_accepts_our_kwargs(depcheck):
    """chroma.py: PersistentClient(path=, settings=, tenant=, database=)."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_params(mod.PersistentClient, ["path", "settings", "tenant", "database"])


def test_http_client_accepts_our_kwargs(depcheck):
    """chroma.py: HttpClient(host=, port=, headers=, ssl=, tenant=, database=,
    settings=). Pin the kwargs; do NOT instantiate (it would touch the network
    / a server)."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_params(
        mod.HttpClient,
        ["host", "port", "headers", "ssl", "tenant", "database", "settings"],
    )


# ---------------------------------------------------------------------------
# create_batches signature contract
# ---------------------------------------------------------------------------


def test_create_batches_signature(depcheck):
    """chroma.py: create_batches(api=self.client, documents=, embeddings=,
    ids=, metadatas=) then `collection.add(*batch)`. Those kwarg names and the
    positional `api` must remain accepted."""
    depcheck.load(IMPORT_NAME)
    batch_utils = depcheck.load("chromadb.utils.batch_utils")
    depcheck.assert_params(
        batch_utils.create_batches,
        ["api", "documents", "embeddings", "ids", "metadatas"],
    )


# ---------------------------------------------------------------------------
# Client / Collection method-signature contracts (resolved off real classes)
# ---------------------------------------------------------------------------


def _ephemeral(mod):
    """An in-memory chromadb client mirroring chroma.py's Settings.

    EphemeralClient keeps everything in RAM: no persist_directory, no server,
    no socket — safe and deterministic for offline contract tests. The
    Settings match what ChromaClient.__init__ builds (sans auth, which only
    applies to the remote HttpClient path)."""
    settings = mod.Settings(allow_reset=True, anonymized_telemetry=False)
    return mod.EphemeralClient(settings=settings)


def test_client_methods_exist(depcheck):
    """Every client method chroma.py calls must exist on a real client."""
    mod = depcheck.load(IMPORT_NAME)
    client = _ephemeral(mod)
    try:
        names = set(dir(client))
        missing = [m for m in CLIENT_METHODS if m not in names]
        assert not missing, f"chromadb client missing method(s) chroma.py calls: {missing}"
        for m in CLIENT_METHODS:
            assert callable(getattr(client, m)), f"client.{m} is not callable"
    finally:
        _safe_reset(client)


def test_collection_methods_exist(depcheck):
    """Every collection method chroma.py calls must exist on a real
    collection object."""
    mod = depcheck.load(IMPORT_NAME)
    client = _ephemeral(mod)
    try:
        col = client.get_or_create_collection(name=COLL, metadata={"hnsw:space": "cosine"})
        names = set(dir(col))
        missing = [m for m in COLLECTION_METHODS if m not in names]
        assert not missing, f"chromadb collection missing method(s): {missing}"
        for m in COLLECTION_METHODS:
            assert callable(getattr(col, m)), f"collection.{m} is not callable"
    finally:
        _safe_reset(client)


def test_get_or_create_collection_accepts_name_and_metadata(depcheck):
    """chroma.py: get_or_create_collection(name=, metadata={'hnsw:space':
    'cosine'}). Both kwargs must remain accepted."""
    mod = depcheck.load(IMPORT_NAME)
    client = _ephemeral(mod)
    try:
        depcheck.assert_params(client.get_or_create_collection, ["name", "metadata"])
    finally:
        _safe_reset(client)


def test_collection_method_kwargs(depcheck):
    """Pin the exact collection-method kwargs chroma.py passes:
    add(ids=, embeddings=, documents=, metadatas=)
    query(query_embeddings=, n_results=, where=)
    get(ids=, where=, limit=)
    upsert(ids=, documents=, embeddings=, metadatas=)
    delete(ids=, where=)"""
    mod = depcheck.load(IMPORT_NAME)
    client = _ephemeral(mod)
    try:
        col = client.get_or_create_collection(name=COLL, metadata={"hnsw:space": "cosine"})
        depcheck.assert_params(col.add, ["ids", "embeddings", "documents", "metadatas"])
        depcheck.assert_params(col.query, ["query_embeddings", "n_results", "where"])
        depcheck.assert_params(col.get, ["ids", "where", "limit"])
        depcheck.assert_params(col.upsert, ["ids", "documents", "embeddings", "metadatas"])
        depcheck.assert_params(col.delete, ["ids", "where"])
    finally:
        _safe_reset(client)


def test_query_supports_include_and_where_document(depcheck):
    """search() relies on the default include returning distances/documents/
    metadatas; pin that `include` (and `where_document`) remain accepted query
    kwargs so a future signature change is caught."""
    mod = depcheck.load(IMPORT_NAME)
    client = _ephemeral(mod)
    try:
        col = client.get_or_create_collection(name=COLL)
        depcheck.assert_params(col.query, ["include", "where_document"])
    finally:
        _safe_reset(client)


# ---------------------------------------------------------------------------
# Behavioural contracts — in-memory EphemeralClient, NO network / NO model.
# These exercise the create -> add -> query -> get -> count -> delete flow
# and pin the exact result-dict shapes chroma.py consumes.
# ---------------------------------------------------------------------------


def test_behaviour_create_and_list_collections(depcheck):
    """insert()/upsert() call get_or_create_collection; has_collection() calls
    list_collections(). Verify a created collection shows up.

    NOTE on shape: in chromadb 1.5.x list_collections() returns a list of
    Collection OBJECTS, not names. chroma.py's `collection_name in
    list_collections()` therefore matches by object, so membership must be
    checked via the `.name` attribute — pin that the objects expose `.name`."""
    mod = depcheck.load(IMPORT_NAME)
    client = _ephemeral(mod)
    try:
        client.get_or_create_collection(name=COLL, metadata={"hnsw:space": "cosine"})
        cols = client.list_collections()
        assert isinstance(cols, list)
        names = [getattr(c, "name", c) for c in cols]
        assert COLL in names, f"created collection not listed (got {names!r})"
    finally:
        _safe_reset(client)


def test_behaviour_add_then_count(depcheck):
    """insert() adds ids/embeddings/documents/metadatas; count() reflects the
    number of stored items."""
    mod = depcheck.load(IMPORT_NAME)
    client = _ephemeral(mod)
    try:
        col = client.get_or_create_collection(name=COLL, metadata={"hnsw:space": "cosine"})
        col.add(
            ids=["a", "b", "c"],
            embeddings=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            documents=["doc a", "doc b", "doc c"],
            metadatas=[{"k": "1"}, {"k": "2"}, {"k": "3"}],
        )
        assert col.count() == 3
    finally:
        _safe_reset(client)


def test_behaviour_query_result_shape(depcheck):
    """search() consumes result['distances'][0] (a flat list), result['ids'],
    result['documents'], result['metadatas']. Pin that query() returns those
    keys and that ids/distances/documents/metadatas are nested one level per
    query embedding (the [0] index chroma.py uses)."""
    mod = depcheck.load(IMPORT_NAME)
    client = _ephemeral(mod)
    try:
        col = client.get_or_create_collection(name=COLL, metadata={"hnsw:space": "cosine"})
        col.add(
            ids=["a", "b"],
            embeddings=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            documents=["doc a", "doc b"],
            metadatas=[{"k": "1"}, {"k": "2"}],
        )
        result = col.query(query_embeddings=[[1.0, 0.0, 0.0]], n_results=2, where=None)

        for key in ("ids", "distances", "documents", "metadatas"):
            assert key in result, f"query result missing {key!r} key chroma.py reads"

        # One query embedding -> one inner list per key.
        assert len(result["ids"]) == 1
        assert isinstance(result["ids"][0], list)
        assert set(result["ids"][0]) == {"a", "b"}

        # chroma.py: distances = result['distances'][0]; iterate as floats.
        dists = result["distances"][0]
        assert isinstance(dists, list) and len(dists) == 2
        for d in dists:
            assert isinstance(d, (int, float))

        assert isinstance(result["documents"][0], list)
        assert isinstance(result["metadatas"][0], list)
        assert result["metadatas"][0][0]["k"] in {"1", "2"}
    finally:
        _safe_reset(client)


def test_behaviour_query_cosine_distance_ordering(depcheck):
    """search() rescales cosine distance (2 -> 0 worst..best) into 0..1; this
    assumes chroma cosine distances land in [0, 2] with the nearest neighbour
    smallest. Pin that the exact-match vector comes back first with ~0 distance
    and the orthogonal one with a larger distance."""
    mod = depcheck.load(IMPORT_NAME)
    client = _ephemeral(mod)
    try:
        col = client.get_or_create_collection(name=COLL, metadata={"hnsw:space": "cosine"})
        col.add(
            ids=["same", "orth"],
            embeddings=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            documents=["same", "orth"],
            metadatas=[{"k": "s"}, {"k": "o"}],
        )
        result = col.query(query_embeddings=[[1.0, 0.0, 0.0]], n_results=2)
        ids = result["ids"][0]
        dists = result["distances"][0]
        assert ids[0] == "same", "nearest-neighbour ordering changed"
        # Nearest distance ~0; all cosine distances within [0, 2] (chroma.py's
        # `2 - dist` rescale depends on this upper bound).
        assert dists[0] == pytest.approx(0.0, abs=1e-5)
        assert all(0.0 <= d <= 2.0 + 1e-6 for d in dists)
        assert dists[1] > dists[0]
    finally:
        _safe_reset(client)


def test_behaviour_query_n_results_caps_output(depcheck):
    """search() passes n_results=limit; chroma must return at most that many
    (and no more than the stored count). Pin the cap semantics."""
    mod = depcheck.load(IMPORT_NAME)
    client = _ephemeral(mod)
    try:
        col = client.get_or_create_collection(name=COLL, metadata={"hnsw:space": "cosine"})
        col.add(
            ids=["a", "b", "c"],
            embeddings=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            documents=["a", "b", "c"],
            metadatas=[{"k": "1"}, {"k": "2"}, {"k": "3"}],
        )
        capped = col.query(query_embeddings=[[1.0, 0.0, 0.0]], n_results=2)
        assert len(capped["ids"][0]) == 2
        # n_results larger than the data set is clamped to what exists.
        over = col.query(query_embeddings=[[1.0, 0.0, 0.0]], n_results=99)
        assert len(over["ids"][0]) == 3
    finally:
        _safe_reset(client)


def test_behaviour_query_where_filter(depcheck):
    """search() forwards `where=filter` to restrict by metadata. Verify the
    metadata equality filter actually narrows the result set."""
    mod = depcheck.load(IMPORT_NAME)
    client = _ephemeral(mod)
    try:
        col = client.get_or_create_collection(name=COLL, metadata={"hnsw:space": "cosine"})
        col.add(
            ids=["a", "b"],
            embeddings=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            documents=["doc a", "doc b"],
            metadatas=[{"group": "x"}, {"group": "y"}],
        )
        result = col.query(
            query_embeddings=[[1.0, 0.0, 0.0]],
            n_results=10,
            where={"group": "x"},
        )
        assert result["ids"][0] == ["a"]
        assert result["metadatas"][0][0]["group"] == "x"
    finally:
        _safe_reset(client)


def test_behaviour_get_by_ids_shape(depcheck):
    """query()/get() in chroma.py call collection.get(...) and read a FLAT
    result['ids'] / ['documents'] / ['metadatas'] (then wrap each in a list).
    Pin that get() returns those keys as flat (un-nested) lists."""
    mod = depcheck.load(IMPORT_NAME)
    client = _ephemeral(mod)
    try:
        col = client.get_or_create_collection(name=COLL, metadata={"hnsw:space": "cosine"})
        col.add(
            ids=["a", "b"],
            embeddings=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            documents=["doc a", "doc b"],
            metadatas=[{"k": "1"}, {"k": "2"}],
        )
        got = col.get(ids=["a"])
        for key in ("ids", "documents", "metadatas"):
            assert key in got, f"get result missing {key!r} key chroma.py reads"
        # Flat, not nested: chroma.py does GetResult(ids=[result['ids']], ...).
        assert got["ids"] == ["a"]
        assert got["documents"] == ["doc a"]
        assert got["metadatas"][0]["k"] == "1"
    finally:
        _safe_reset(client)


def test_behaviour_get_all_and_where(depcheck):
    """get() with no args returns every item (chroma.py's get()); get(where=)
    filters by metadata (chroma.py's query())."""
    mod = depcheck.load(IMPORT_NAME)
    client = _ephemeral(mod)
    try:
        col = client.get_or_create_collection(name=COLL, metadata={"hnsw:space": "cosine"})
        col.add(
            ids=["a", "b", "c"],
            embeddings=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            documents=["a", "b", "c"],
            metadatas=[{"g": "x"}, {"g": "x"}, {"g": "y"}],
        )
        all_items = col.get()
        assert set(all_items["ids"]) == {"a", "b", "c"}

        filtered = col.get(where={"g": "y"})
        assert filtered["ids"] == ["c"]
    finally:
        _safe_reset(client)


def test_behaviour_get_limit(depcheck):
    """query() forwards limit= to collection.get(limit=...). Pin that it caps
    the number of returned rows."""
    mod = depcheck.load(IMPORT_NAME)
    client = _ephemeral(mod)
    try:
        col = client.get_or_create_collection(name=COLL, metadata={"hnsw:space": "cosine"})
        col.add(
            ids=["a", "b", "c"],
            embeddings=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            documents=["a", "b", "c"],
            metadatas=[{"k": "1"}, {"k": "2"}, {"k": "3"}],
        )
        limited = col.get(limit=2)
        assert len(limited["ids"]) == 2
    finally:
        _safe_reset(client)


def test_behaviour_upsert_inserts_and_updates(depcheck):
    """upsert() inserts new ids and overwrites existing ones in place (count
    unchanged on overwrite, document/metadata replaced)."""
    mod = depcheck.load(IMPORT_NAME)
    client = _ephemeral(mod)
    try:
        col = client.get_or_create_collection(name=COLL, metadata={"hnsw:space": "cosine"})
        col.upsert(
            ids=["a"],
            documents=["original"],
            embeddings=[[1.0, 0.0, 0.0]],
            metadatas=[{"k": "1"}],
        )
        assert col.count() == 1
        # Overwrite the same id: count stays 1, content replaced.
        col.upsert(
            ids=["a"],
            documents=["updated"],
            embeddings=[[0.0, 1.0, 0.0]],
            metadatas=[{"k": "2"}],
        )
        assert col.count() == 1
        got = col.get(ids=["a"])
        assert got["documents"] == ["updated"]
        assert got["metadatas"][0]["k"] == "2"
        # Insert a brand-new id via upsert: count grows. (chromadb rejects an
        # empty metadata dict, so supply a real key.)
        col.upsert(
            ids=["b"], documents=["new"], embeddings=[[0.0, 0.0, 1.0]], metadatas=[{"k": "3"}]
        )
        assert col.count() == 2
    finally:
        _safe_reset(client)


def test_behaviour_delete_by_ids(depcheck):
    """delete() with ids= removes exactly those items."""
    mod = depcheck.load(IMPORT_NAME)
    client = _ephemeral(mod)
    try:
        col = client.get_or_create_collection(name=COLL, metadata={"hnsw:space": "cosine"})
        col.add(
            ids=["a", "b", "c"],
            embeddings=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            documents=["a", "b", "c"],
            metadatas=[{"k": "1"}, {"k": "2"}, {"k": "3"}],
        )
        col.delete(ids=["b"])
        assert col.count() == 2
        assert set(col.get()["ids"]) == {"a", "c"}
    finally:
        _safe_reset(client)


def test_behaviour_delete_by_where(depcheck):
    """delete() with where= removes items matching the metadata filter (the
    chroma.py `elif filter:` branch)."""
    mod = depcheck.load(IMPORT_NAME)
    client = _ephemeral(mod)
    try:
        col = client.get_or_create_collection(name=COLL, metadata={"hnsw:space": "cosine"})
        col.add(
            ids=["a", "b", "c"],
            embeddings=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            documents=["a", "b", "c"],
            metadatas=[{"g": "x"}, {"g": "x"}, {"g": "y"}],
        )
        col.delete(where={"g": "x"})
        assert col.count() == 1
        assert col.get()["ids"] == ["c"]
    finally:
        _safe_reset(client)


def test_behaviour_delete_collection(depcheck):
    """delete_collection(name=) drops a whole collection (chroma.py's
    delete_collection())."""
    mod = depcheck.load(IMPORT_NAME)
    client = _ephemeral(mod)
    try:
        client.get_or_create_collection(name=COLL, metadata={"hnsw:space": "cosine"})
        assert COLL in [getattr(c, "name", c) for c in client.list_collections()]
        client.delete_collection(name=COLL)
        assert COLL not in [getattr(c, "name", c) for c in client.list_collections()]
    finally:
        _safe_reset(client)


def test_behaviour_get_collection_roundtrip(depcheck):
    """search()/query()/get()/delete() fetch the collection via
    get_collection(name=) before operating. Pin that it returns the same
    (named) collection an insert created, with the data intact."""
    mod = depcheck.load(IMPORT_NAME)
    client = _ephemeral(mod)
    try:
        created = client.get_or_create_collection(name=COLL, metadata={"hnsw:space": "cosine"})
        created.add(
            ids=["a"], embeddings=[[1.0, 0.0, 0.0]], documents=["doc"], metadatas=[{"k": "1"}]
        )
        fetched = client.get_collection(name=COLL)
        assert fetched.name == COLL
        assert fetched.count() == 1
        assert fetched.get(ids=["a"])["documents"] == ["doc"]
    finally:
        _safe_reset(client)


def test_behaviour_reset_clears_everything(depcheck):
    """reset() (chroma.py's reset()) drops all collections + items. Requires
    Settings(allow_reset=True), which chroma.py always sets."""
    mod = depcheck.load(IMPORT_NAME)
    client = _ephemeral(mod)
    try:
        client.get_or_create_collection(name=COLL, metadata={"hnsw:space": "cosine"})
        assert client.list_collections()  # non-empty
        client.reset()
        assert client.list_collections() == []
    finally:
        _safe_reset(client)


def test_behaviour_create_batches_chunks_inserts(depcheck):
    """insert() funnels every add through create_batches(api=client, ...) then
    `collection.add(*batch)`. Verify create_batches yields tuples in
    (ids, embeddings, metadatas, documents) order such that unpacking them
    positionally into collection.add reconstructs the full set."""
    mod = depcheck.load(IMPORT_NAME)
    batch_utils = depcheck.load("chromadb.utils.batch_utils")
    client = _ephemeral(mod)
    try:
        col = client.get_or_create_collection(name=COLL, metadata={"hnsw:space": "cosine"})
        ids = ["a", "b", "c"]
        embeddings = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        documents = ["doc a", "doc b", "doc c"]
        metadatas = [{"k": "1"}, {"k": "2"}, {"k": "3"}]

        batches = list(
            batch_utils.create_batches(
                api=client,
                documents=documents,
                embeddings=embeddings,
                ids=ids,
                metadatas=metadatas,
            )
        )
        assert batches, "create_batches yielded nothing"
        for batch in batches:
            # chroma.py relies on positional unpacking: collection.add(*batch).
            assert len(batch) == 4
            col.add(*batch)

        assert col.count() == 3
        assert set(col.get()["ids"]) == {"a", "b", "c"}
    finally:
        _safe_reset(client)


def test_behaviour_full_lifecycle(depcheck):
    """End-to-end mirror of chroma.py: get_or_create -> add -> query (search)
    -> get (query/get) -> upsert -> delete -> count, on one in-memory client,
    asserting the result shapes each backend method consumes hold together."""
    mod = depcheck.load(IMPORT_NAME)
    client = _ephemeral(mod)
    try:
        col = client.get_or_create_collection(name=COLL, metadata={"hnsw:space": "cosine"})
        col.add(
            ids=["a", "b"],
            embeddings=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            documents=["doc a", "doc b"],
            metadatas=[{"k": "1"}, {"k": "2"}],
        )
        # search()
        s = col.query(query_embeddings=[[1.0, 0.0, 0.0]], n_results=1, where=None)
        assert s["ids"][0][0] == "a"
        assert isinstance(s["distances"][0][0], (int, float))
        # query() / get()
        g = col.get(ids=["b"])
        assert g["documents"] == ["doc b"]
        # upsert()
        col.upsert(
            ids=["a"], documents=["a2"], embeddings=[[0.0, 0.0, 1.0]], metadatas=[{"k": "9"}]
        )
        assert col.get(ids=["a"])["documents"] == ["a2"]
        # delete()
        col.delete(ids=["a"])
        assert col.count() == 1
        assert col.get()["ids"] == ["b"]
    finally:
        _safe_reset(client)


# ---------------------------------------------------------------------------
# Local helpers (no cross-file imports — conftest exposes only fixtures).
# ---------------------------------------------------------------------------


def _safe_reset(client) -> None:
    """Best-effort teardown for an in-memory client: reset() drops all state so
    leftover collections never leak between tests sharing the same process
    (chromadb caches embedded clients by settings). Never raises."""
    try:
        client.reset()
    except Exception:
        pass
