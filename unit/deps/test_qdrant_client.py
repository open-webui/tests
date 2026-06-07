"""Dependency contract: qdrant-client (import name ``qdrant_client``).

Qdrant is one of Open WebUI's pluggable vector stores. ``QdrantClient`` in
``retrieval/vector/dbs/qdrant.py`` (and the multitenancy variant) constructs a
``qdrant_client.QdrantClient`` (either ``url=`` REST or ``host=``/``grpc_port=``
gRPC) and drives a fixed slice of the API:

  * client lifecycle: ``create_collection`` (with ``models.VectorParams`` /
    ``models.Distance.COSINE`` / ``models.HnswConfigDiff``),
    ``create_payload_index`` (``models.KeywordIndexParams`` /
    ``models.KeywordIndexType``), ``collection_exists``, ``delete_collection``,
    ``get_collections`` (``.collections`` -> objects with ``.name``);
  * data ops: ``upload_points`` / ``upsert`` of ``PointStruct(id, vector,
    payload)``, ``query_points(collection_name=, query=, limit=)`` returning
    ``.points`` (each with ``.id`` / ``.score`` / ``.payload``), ``scroll(...)``
    returning a ``(points, next_offset)`` tuple, and ``delete`` with either
    ``models.PointIdsList`` or ``models.FilterSelector(models.Filter(...))``
    built from ``models.FieldCondition`` / ``models.MatchValue``.

This module pins exactly that surface so a qdrant-client bump that
removed/renamed any of it fails loudly here instead of as a runtime
``AttributeError`` / ``TypeError`` deep in an ingest or search request. Two
layers, mirroring test_chromadb.py: symbol/signature checks, plus FULL
offline BEHAVIOURAL contracts driven by ``QdrantClient(':memory:')`` — qdrant's
embedded local mode keeps everything in RAM with no server and no socket, so
the create -> upsert -> query_points -> scroll -> delete lifecycle runs for
real and the exact result shapes ``qdrant.py`` consumes (``.points[i].score``,
``.payload['text']``, ``scroll(...)[0]``) are asserted. Vectors are plain float
lists (the backend passes model vectors of the same shape); no embedder is
loaded.

Uses the ``depcheck`` fixture from unit/deps/conftest.py.
"""

from __future__ import annotations

import time

import pytest

pytestmark = pytest.mark.depcheck

IMPORT_NAME = "qdrant_client"
DIST_NAME = "qdrant-client"

# models.* symbols qdrant.py + qdrant_multitenancy.py resolve.
MODELS_SYMBOLS = [
    "VectorParams",
    "Distance",
    "HnswConfigDiff",
    "KeywordIndexParams",
    "KeywordIndexType",
    "FieldCondition",
    "MatchValue",
    "Filter",
    "PointIdsList",
    "FilterSelector",
    "HasIdCondition",
]

# Client methods the backend calls.
CLIENT_METHODS = [
    "create_collection",
    "create_payload_index",
    "collection_exists",
    "delete_collection",
    "query_points",
    "scroll",
    "upload_points",
    "upsert",
    "delete",
    "get_collections",
    "count",
    "retrieve",
]


# ---------------------------------------------------------------------------
# Import + version
# ---------------------------------------------------------------------------


def test_import(depcheck):
    """`qdrant_client` must import (skip cleanly if absent in this env)."""
    mod = depcheck.load(IMPORT_NAME)
    assert mod.__name__ == "qdrant_client"


def test_version_reported(depcheck):
    """The installed distribution version must be resolvable so bump tooling and
    this suite agree on what's under test."""
    depcheck.load(IMPORT_NAME)
    assert depcheck.dist_version(DIST_NAME) is not None


# ---------------------------------------------------------------------------
# Symbol-existence checks (API surface)
# ---------------------------------------------------------------------------


def test_qdrant_client_callable(depcheck):
    """qdrant.py: `from qdrant_client import QdrantClient`. Must be callable."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_callable(mod, "QdrantClient")


def test_point_struct_importable(depcheck):
    """qdrant.py: `from qdrant_client.http.models import PointStruct`."""
    depcheck.load(IMPORT_NAME)
    http_models = depcheck.load("qdrant_client.http.models")
    assert hasattr(http_models, "PointStruct"), "qdrant_client.http.models.PointStruct is gone"


def test_models_namespace_importable(depcheck):
    """qdrant.py: `from qdrant_client.models import models`. The `models`
    namespace must be importable and carry every type the backend builds."""
    depcheck.load(IMPORT_NAME)
    models_ns = depcheck.load("qdrant_client.models")
    assert hasattr(models_ns, "models"), "qdrant_client.models.models namespace is gone"
    depcheck.assert_symbols(models_ns.models, MODELS_SYMBOLS)


def test_unexpected_response_importable(depcheck):
    """qdrant_multitenancy.py: `from qdrant_client.http.exceptions import
    UnexpectedResponse`."""
    depcheck.load(IMPORT_NAME)
    exc = depcheck.load("qdrant_client.http.exceptions")
    assert hasattr(exc, "UnexpectedResponse"), (
        "qdrant_client.http.exceptions.UnexpectedResponse is gone"
    )


def test_distance_cosine_exists(depcheck):
    """qdrant.py builds VectorParams(distance=models.Distance.COSINE). The COSINE
    member must exist (the similarity metric the whole store assumes)."""
    depcheck.load(IMPORT_NAME)
    models_ns = depcheck.load("qdrant_client.models")
    assert hasattr(models_ns.models.Distance, "COSINE"), "models.Distance.COSINE is gone"


def test_keyword_index_type_keyword_exists(depcheck):
    """create_payload_index uses models.KeywordIndexType.KEYWORD."""
    depcheck.load(IMPORT_NAME)
    models_ns = depcheck.load("qdrant_client.models")
    assert hasattr(models_ns.models.KeywordIndexType, "KEYWORD"), (
        "models.KeywordIndexType.KEYWORD is gone"
    )


# ---------------------------------------------------------------------------
# Constructor signature contract
# ---------------------------------------------------------------------------


def test_qdrant_client_init_accepts_our_kwargs(depcheck):
    """qdrant.py constructs with url=/host=/port=/grpc_port=/prefer_grpc=/
    api_key=/timeout=. Those kwargs must remain accepted on the constructor."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_params(
        mod.QdrantClient.__init__,
        ["url", "host", "port", "grpc_port", "prefer_grpc", "api_key", "timeout"],
    )


# ---------------------------------------------------------------------------
# Model constructor contracts (the exact kwargs qdrant.py passes)
# ---------------------------------------------------------------------------


def test_vector_params_accepts_size_distance_on_disk(depcheck):
    """qdrant.py: VectorParams(size=, distance=, on_disk=). Constructing with
    those kwargs must not raise."""
    depcheck.load(IMPORT_NAME)
    m = depcheck.load("qdrant_client.models").models
    vp = m.VectorParams(size=384, distance=m.Distance.COSINE, on_disk=False)
    assert vp.size == 384


def test_point_struct_accepts_id_vector_payload(depcheck):
    """qdrant.py: PointStruct(id=, vector=, payload={'text','metadata'})."""
    depcheck.load(IMPORT_NAME)
    PointStruct = depcheck.load("qdrant_client.http.models").PointStruct
    p = PointStruct(id=1, vector=[1.0, 0.0, 0.0], payload={"text": "a", "metadata": {"k": "1"}})
    assert p.id == 1
    assert p.payload["text"] == "a"


def test_field_condition_and_match_value_construct(depcheck):
    """query/delete build FieldCondition(key='metadata.x', match=MatchValue(
    value=...)) inside a Filter. Pin those constructors."""
    depcheck.load(IMPORT_NAME)
    m = depcheck.load("qdrant_client.models").models
    cond = m.FieldCondition(key="metadata.k", match=m.MatchValue(value="1"))
    flt = m.Filter(should=[cond])
    assert flt is not None
    must_flt = m.Filter(must=[cond])
    assert must_flt is not None


# ---------------------------------------------------------------------------
# Behavioural: FULL lifecycle on an in-memory (local) client. NO server.
# ---------------------------------------------------------------------------


def _client(mod):
    """An embedded in-memory qdrant client.

    QdrantClient(':memory:') runs qdrant entirely in-process (RAM), opening no
    socket and contacting no server — safe and deterministic for offline
    contract tests. It speaks the same client API qdrant.py uses against a
    remote server."""
    return mod.QdrantClient(":memory:")


def _models(mod):
    """The `models` namespace off the imported package (local helper)."""
    import importlib

    return importlib.import_module("qdrant_client.models").models


def _make_collection(mod, client, name="owui_dep_qdrant"):
    """Mirror qdrant.py's _create_collection: VectorParams(COSINE) + HnswConfigDiff."""
    m = _models(mod)
    client.create_collection(
        collection_name=name,
        vectors_config=m.VectorParams(size=3, distance=m.Distance.COSINE, on_disk=False),
        hnsw_config=m.HnswConfigDiff(m=16),
    )
    return name


def test_behaviour_create_collection_and_exists(depcheck):
    """_create_collection_if_not_exists calls create_collection then
    has_collection -> collection_exists. Verify a created collection reports
    existing and an absent one does not."""
    mod = depcheck.load(IMPORT_NAME)
    client = _client(mod)
    try:
        name = _make_collection(mod, client)
        assert client.collection_exists(name) is True
        assert client.collection_exists("does_not_exist") is False
    finally:
        client.close()


def test_behaviour_create_payload_index_is_accepted(depcheck):
    """qdrant.py creates keyword payload indexes on metadata.hash / metadata.file_id
    via create_payload_index(KeywordIndexParams(KEYWORD,...)). In local mode this
    is a documented no-op, but the call (and the model constructor) must be
    accepted without raising."""
    mod = depcheck.load(IMPORT_NAME)
    m = _models(mod)
    client = _client(mod)
    try:
        name = _make_collection(mod, client)
        client.create_payload_index(
            collection_name=name,
            field_name="metadata.hash",
            field_schema=m.KeywordIndexParams(
                type=m.KeywordIndexType.KEYWORD, is_tenant=False, on_disk=False
            ),
        )
    finally:
        client.close()


def test_behaviour_upload_points_and_query_points_shape(depcheck):
    """insert() uploads PointStructs; search() calls query_points(...).points and
    reads point.score + point.payload. Pin the result shape: query_points returns
    an object with .points, each carrying .id / .score / .payload."""
    mod = depcheck.load(IMPORT_NAME)
    PointStruct = depcheck.load("qdrant_client.http.models").PointStruct
    client = _client(mod)
    try:
        name = _make_collection(mod, client)
        client.upload_points(
            name,
            [
                PointStruct(id=1, vector=[1.0, 0.0, 0.0], payload={"text": "a", "metadata": {}}),
                PointStruct(id=2, vector=[0.0, 1.0, 0.0], payload={"text": "b", "metadata": {}}),
            ],
        )
        time.sleep(0.1)  # local index settle
        resp = client.query_points(collection_name=name, query=[1.0, 0.0, 0.0], limit=2)
        assert hasattr(resp, "points"), "query_points result has no .points"
        points = resp.points
        assert len(points) == 2
        # Nearest neighbour first; score is a float; payload carries 'text'.
        top = points[0]
        assert top.id == 1, "nearest-neighbour ordering changed"
        assert isinstance(top.score, float)
        assert top.payload["text"] == "a"
    finally:
        client.close()


def test_behaviour_query_points_score_normalization_range(depcheck):
    """search() normalizes qdrant's cosine score via (score + 1)/2, which assumes
    cosine scores land in [-1, 1] with the best match largest. Pin that the
    exact-match score is ~1.0 and the orthogonal one is lower."""
    mod = depcheck.load(IMPORT_NAME)
    PointStruct = depcheck.load("qdrant_client.http.models").PointStruct
    client = _client(mod)
    try:
        name = _make_collection(mod, client)
        client.upload_points(
            name,
            [
                PointStruct(id=1, vector=[1.0, 0.0, 0.0], payload={"text": "same", "metadata": {}}),
                PointStruct(id=2, vector=[0.0, 1.0, 0.0], payload={"text": "orth", "metadata": {}}),
            ],
        )
        time.sleep(0.1)
        resp = client.query_points(collection_name=name, query=[1.0, 0.0, 0.0], limit=2)
        scores = [p.score for p in resp.points]
        assert scores[0] == pytest.approx(1.0, abs=1e-4), "cosine self-similarity should be ~1.0"
        assert all(-1.0 - 1e-6 <= s <= 1.0 + 1e-6 for s in scores), (
            "cosine scores outside [-1, 1]; the (score+1)/2 normalization assumption breaks"
        )
        assert scores[0] > scores[1]
    finally:
        client.close()


def test_behaviour_scroll_returns_points_tuple(depcheck):
    """query()/get() call client.scroll(...) and read `points[0]` (the first
    tuple element). Pin that scroll returns a (records, next_offset) tuple and
    that element 0 holds the point records."""
    mod = depcheck.load(IMPORT_NAME)
    PointStruct = depcheck.load("qdrant_client.http.models").PointStruct
    client = _client(mod)
    try:
        name = _make_collection(mod, client)
        client.upsert(
            name,
            [
                PointStruct(
                    id=i,
                    vector=[float(i == 0), float(i == 1), float(i == 2)],
                    payload={"text": str(i), "metadata": {"k": str(i)}},
                )
                for i in range(3)
            ],
        )
        result = client.scroll(collection_name=name, limit=10)
        assert isinstance(result, tuple) and len(result) == 2, (
            "scroll no longer returns a (records, next_offset) tuple"
        )
        records = result[0]
        assert {r.id for r in records} == {0, 1, 2}
        assert records[0].payload["text"] in {"0", "1", "2"}
    finally:
        client.close()


def test_behaviour_scroll_with_filter(depcheck):
    """query() builds scroll_filter=Filter(should=[FieldCondition(...)]) to
    restrict by metadata. Verify the metadata filter narrows the scroll result."""
    mod = depcheck.load(IMPORT_NAME)
    m = _models(mod)
    PointStruct = depcheck.load("qdrant_client.http.models").PointStruct
    client = _client(mod)
    try:
        name = _make_collection(mod, client)
        client.upsert(
            name,
            [
                PointStruct(
                    id=1, vector=[1.0, 0.0, 0.0], payload={"text": "a", "metadata": {"k": "x"}}
                ),
                PointStruct(
                    id=2, vector=[0.0, 1.0, 0.0], payload={"text": "b", "metadata": {"k": "y"}}
                ),
            ],
        )
        records, _ = client.scroll(
            collection_name=name,
            scroll_filter=m.Filter(
                should=[m.FieldCondition(key="metadata.k", match=m.MatchValue(value="x"))]
            ),
            limit=10,
        )
        assert [r.id for r in records] == [1]
    finally:
        client.close()


def test_behaviour_delete_by_point_ids(depcheck):
    """delete(ids=...) uses points_selector=PointIdsList(points=ids). Verify it
    removes exactly those points."""
    mod = depcheck.load(IMPORT_NAME)
    m = _models(mod)
    PointStruct = depcheck.load("qdrant_client.http.models").PointStruct
    client = _client(mod)
    try:
        name = _make_collection(mod, client)
        client.upsert(
            name,
            [
                PointStruct(
                    id=i,
                    vector=[float(i == 0), float(i == 1), float(i == 2)],
                    payload={"text": str(i), "metadata": {}},
                )
                for i in range(3)
            ],
        )
        client.delete(collection_name=name, points_selector=m.PointIdsList(points=[0]))
        records, _ = client.scroll(collection_name=name, limit=10)
        assert {r.id for r in records} == {1, 2}
    finally:
        client.close()


def test_behaviour_delete_by_filter_selector(depcheck):
    """delete(filter=...) uses points_selector=FilterSelector(filter=Filter(
    must=[FieldCondition(...)])). Verify the metadata filter deletes the right
    points."""
    mod = depcheck.load(IMPORT_NAME)
    m = _models(mod)
    PointStruct = depcheck.load("qdrant_client.http.models").PointStruct
    client = _client(mod)
    try:
        name = _make_collection(mod, client)
        client.upsert(
            name,
            [
                PointStruct(
                    id=1, vector=[1.0, 0.0, 0.0], payload={"text": "a", "metadata": {"k": "x"}}
                ),
                PointStruct(
                    id=2, vector=[0.0, 1.0, 0.0], payload={"text": "b", "metadata": {"k": "y"}}
                ),
            ],
        )
        client.delete(
            collection_name=name,
            points_selector=m.FilterSelector(
                filter=m.Filter(
                    must=[m.FieldCondition(key="metadata.k", match=m.MatchValue(value="x"))]
                )
            ),
        )
        records, _ = client.scroll(collection_name=name, limit=10)
        assert {r.id for r in records} == {2}
    finally:
        client.close()


def test_behaviour_get_collections_and_delete_collection(depcheck):
    """reset() iterates get_collections().collections and reads each `.name`,
    then delete_collection(name). Pin that get_collections returns an object with
    a `.collections` list of name-bearing objects, and delete_collection drops
    one."""
    mod = depcheck.load(IMPORT_NAME)
    client = _client(mod)
    try:
        name = _make_collection(mod, client)
        listing = client.get_collections()
        assert hasattr(listing, "collections"), "get_collections() has no .collections"
        names = [c.name for c in listing.collections]
        assert name in names
        client.delete_collection(collection_name=name)
        names_after = [c.name for c in client.get_collections().collections]
        assert name not in names_after
    finally:
        client.close()


def test_behaviour_full_lifecycle(depcheck):
    """End-to-end mirror of qdrant.py: create -> upsert -> query_points (search)
    -> scroll (get) -> delete-by-id, on one in-memory client, asserting the
    result shapes each backend method consumes hold together."""
    mod = depcheck.load(IMPORT_NAME)
    m = _models(mod)
    PointStruct = depcheck.load("qdrant_client.http.models").PointStruct
    client = _client(mod)
    try:
        name = _make_collection(mod, client)
        client.upsert(
            name,
            [
                PointStruct(
                    id=1, vector=[1.0, 0.0, 0.0], payload={"text": "a", "metadata": {"k": "1"}}
                ),
                PointStruct(
                    id=2, vector=[0.0, 1.0, 0.0], payload={"text": "b", "metadata": {"k": "2"}}
                ),
            ],
        )
        time.sleep(0.1)
        # search()
        resp = client.query_points(collection_name=name, query=[1.0, 0.0, 0.0], limit=1)
        assert resp.points[0].id == 1
        assert resp.points[0].payload["text"] == "a"
        # get() via scroll
        records, _ = client.scroll(collection_name=name, limit=10)
        assert {r.id for r in records} == {1, 2}
        # delete()
        client.delete(collection_name=name, points_selector=m.PointIdsList(points=[1]))
        records, _ = client.scroll(collection_name=name, limit=10)
        assert {r.id for r in records} == {2}
    finally:
        client.close()
