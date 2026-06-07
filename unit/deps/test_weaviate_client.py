"""Dependency contract: weaviate-client (import name ``weaviate``, v4 API).

Open WebUI's Weaviate vector store
(``retrieval/vector/dbs/weaviate.py``) is built entirely on the
weaviate-client **v4** API and touches a wide, specific slice of it:

    weaviate.connect_to_custom(http_host=, http_port=, http_secure=,
                               grpc_host=, grpc_port=, grpc_secure=,
                               skip_init_checks=, auth_credentials=)
    weaviate.classes.init.Auth.api_key(KEY)
    weaviate.classes.config.Configure.Vectors.self_provided()
    weaviate.classes.config.Property(name='text',
                                     data_type=weaviate.classes.config.DataType.TEXT)
    weaviate.classes.query.MetadataQuery(distance=True)
    weaviate.classes.query.Filter.by_property(name=k).equal(v)
    weaviate.classes.query.Filter.all_of([f1, f2])

plus the client object graph it then drives:
``client.collections.{exists,create,get,delete,list_all}`` and a
collection's ``query.near_vector`` / ``query.fetch_objects`` /
``data.delete_by_id`` / ``data.delete_many`` / ``batch.fixed_size`` /
``iterator``.

This module pins that surface and constructs every offline artifact the
backend builds (filters, metadata query, property/datatype, api-key auth,
self-provided vector config) WITHOUT connecting to any Weaviate server.
``connect_to_custom`` itself is only checked for existence + signature; it
is never *called*, because the backend follows it with ``client.connect()``
and we must not open a socket. A v4 bump that renamed any of these would
break the integration; this fails loudly instead.

NOTE: this DB integration is community-supported in the backend. Pattern
mirrors test_requests.py. Uses the ``depcheck`` fixture.
"""

from __future__ import annotations

import inspect

import pytest

pytestmark = pytest.mark.depcheck

IMPORT_NAME = "weaviate"
DIST_NAME = "weaviate-client"

# Dotted symbols the backend resolves on the `weaviate` package.
USED_SYMBOLS = [
    "connect_to_custom",
    "WeaviateClient",
    "classes",
    "classes.init",
    "classes.init.Auth",
    "classes.config",
    "classes.config.Configure",
    "classes.config.Property",
    "classes.config.DataType",
    "classes.query",
    "classes.query.MetadataQuery",
    "classes.query.Filter",
]


# ---------------------------------------------------------------------------
# Import + version + surface
# ---------------------------------------------------------------------------


def test_import(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    assert mod.__name__ == "weaviate"


def test_version_reported(depcheck):
    assert depcheck.dist_version(DIST_NAME) is not None


def test_is_v4_client(depcheck):
    """The backend uses the v4 client API (connect_to_custom + classes.*).
    Guard against a slip back to a v3 line, which had a totally different
    (Client(url=...)) surface."""
    mod = depcheck.load(IMPORT_NAME)
    version = depcheck.dist_version(DIST_NAME)
    assert version is not None
    major = int(str(version).split(".")[0])
    assert major >= 4, f"expected weaviate-client v4+, got {version}"
    # v4 hallmark: the top-level connect_to_custom helper.
    assert hasattr(mod, "connect_to_custom")


def test_used_symbols_exist(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_symbols(mod, USED_SYMBOLS)


def test_classes_submodules_present(depcheck):
    """`weaviate.classes` must expose the init/config/query namespaces the
    backend reaches into."""
    mod = depcheck.load(IMPORT_NAME)
    wc = depcheck.resolve(mod, "classes")
    for sub in ("init", "config", "query"):
        assert hasattr(wc, sub), f"weaviate.classes.{sub} missing"


# ---------------------------------------------------------------------------
# connect_to_custom — signature only (NEVER call it: it precedes a real
# client.connect() in the backend and would open a socket).
# ---------------------------------------------------------------------------


def test_connect_to_custom_is_callable(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_callable(mod, "connect_to_custom")


def test_connect_to_custom_signature(depcheck):
    """The backend passes exactly these kwargs into connect_to_custom; all
    must remain accepted parameter names."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_params(
        mod.connect_to_custom,
        [
            "http_host",
            "http_port",
            "http_secure",
            "grpc_host",
            "grpc_port",
            "grpc_secure",
            "skip_init_checks",
            "auth_credentials",
        ],
    )


def test_weaviateclient_class_has_connect_close(depcheck):
    """The backend calls self.client.connect() and (implicitly) close(). Pin
    those methods on the class without instantiating (instantiation needs a
    connection). `collections` is an instance attribute set in __init__, so it
    is covered separately by the collections-manager test below."""
    mod = depcheck.load(IMPORT_NAME)
    names = set(dir(mod.WeaviateClient))
    for attr in ("connect", "close"):
        assert attr in names, f"WeaviateClient.{attr} missing"


# ---------------------------------------------------------------------------
# Auth.api_key — offline credential object.
# ---------------------------------------------------------------------------


def test_auth_api_key_constructs_offline(depcheck):
    """When WEAVIATE_API_KEY is set the backend builds
    weaviate.classes.init.Auth.api_key(KEY). That must produce a credential
    object offline (no auth round-trip on construction)."""
    mod = depcheck.load(IMPORT_NAME)
    Auth = depcheck.resolve(mod, "classes.init.Auth")
    assert callable(Auth.api_key)
    cred = Auth.api_key("unit-test-key-not-real")
    assert cred is not None


# ---------------------------------------------------------------------------
# Collection-creation artifacts: Configure.Vectors.self_provided(), Property,
# DataType.TEXT — all built offline in _create_collection.
# ---------------------------------------------------------------------------


def test_configure_vectors_self_provided(depcheck):
    """_create_collection passes vector_config=Configure.Vectors.self_provided()
    (the backend supplies its own embeddings). That factory must exist and
    return a config object offline."""
    mod = depcheck.load(IMPORT_NAME)
    Configure = depcheck.resolve(mod, "classes.config.Configure")
    assert hasattr(Configure, "Vectors")
    assert callable(Configure.Vectors.self_provided)
    cfg = Configure.Vectors.self_provided()
    assert cfg is not None


def test_datatype_text_member(depcheck):
    """DataType.TEXT is used for the 'text' property; the enum member must
    exist and carry the 'text' value."""
    mod = depcheck.load(IMPORT_NAME)
    DataType = depcheck.resolve(mod, "classes.config.DataType")
    assert hasattr(DataType, "TEXT")
    assert DataType.TEXT.value == "text"


def test_property_constructs_with_name_and_datatype(depcheck):
    """Property(name='text', data_type=DataType.TEXT) is built in
    _create_collection. The `data_type` kwarg must be accepted and the name
    preserved."""
    mod = depcheck.load(IMPORT_NAME)
    config = depcheck.resolve(mod, "classes.config")
    prop = config.Property(name="text", data_type=config.DataType.TEXT)
    assert prop.name == "text"
    # v4 stores it under the `dataType` alias; assert the value round-trips.
    assert prop.dataType == config.DataType.TEXT


# ---------------------------------------------------------------------------
# MetadataQuery — search() requests distance metadata.
# ---------------------------------------------------------------------------


def test_metadata_query_distance(depcheck):
    """search() passes return_metadata=MetadataQuery(distance=True) so it can
    read obj.metadata.distance back. The flag must construct and stick."""
    mod = depcheck.load(IMPORT_NAME)
    MetadataQuery = depcheck.resolve(mod, "classes.query.MetadataQuery")
    mq = MetadataQuery(distance=True)
    assert mq.distance is True


# ---------------------------------------------------------------------------
# Filter — by_property().equal() and all_of() chains (query/delete paths).
# ---------------------------------------------------------------------------


def test_filter_by_property_equal_constructs(depcheck):
    """query()/delete() build Filter.by_property(name=key).equal(value). The
    fluent chain must construct a filter object offline."""
    mod = depcheck.load(IMPORT_NAME)
    Filter = depcheck.resolve(mod, "classes.query.Filter")
    assert callable(Filter.by_property)
    f = Filter.by_property(name="source").equal("doc-1")
    assert f is not None


def test_filter_all_of_combines(depcheck):
    """Multiple filters are AND-combined via Filter.all_of([f1, f2]) (the
    backend folds a metadata dict into a conjunction)."""
    mod = depcheck.load(IMPORT_NAME)
    Filter = depcheck.resolve(mod, "classes.query.Filter")
    assert callable(Filter.all_of)
    f1 = Filter.by_property(name="a").equal("1")
    f2 = Filter.by_property(name="b").equal("2")
    combined = Filter.all_of([f1, f2])
    assert combined is not None


def test_filter_by_property_signature(depcheck):
    """by_property is called with name= as a keyword."""
    mod = depcheck.load(IMPORT_NAME)
    Filter = depcheck.resolve(mod, "classes.query.Filter")
    depcheck.assert_params(Filter.by_property, ["name"])


def test_filter_value_has_equal(depcheck):
    """The object returned by by_property must expose .equal (the operator the
    backend uses); pin the fluent method by building it."""
    mod = depcheck.load(IMPORT_NAME)
    Filter = depcheck.resolve(mod, "classes.query.Filter")
    fp = Filter.by_property(name="k")
    assert hasattr(fp, "equal") and callable(fp.equal)


# ---------------------------------------------------------------------------
# Collection-manager / collection method surface — pin the names the backend
# calls on a connected client without ever connecting.
# ---------------------------------------------------------------------------


def test_collections_manager_method_names(depcheck):
    """The backend calls client.collections.{exists,create,get,delete,
    list_all}. The sync collections-manager class lives under
    weaviate.collections.*; locate the class that exposes the full set without
    instantiating a client (no connection)."""
    depcheck.load(IMPORT_NAME)
    wanted = {"exists", "create", "get", "delete", "list_all"}

    # Try the known sync module first, then fall back to scanning candidates so
    # the test survives a minor reshuffle of the private module path.
    candidates = []
    for modname in (
        "weaviate.collections.collections.sync",
        "weaviate.collections.collections",
        "weaviate.collections",
    ):
        sub = depcheck.try_load(modname)
        if sub is None:
            continue
        candidates += [getattr(sub, n) for n in dir(sub) if isinstance(getattr(sub, n, None), type)]

    matched = [c for c in candidates if wanted.issubset(set(dir(c)))]
    assert matched, (
        "no weaviate collections-manager class exposes the methods "
        f"{sorted(wanted)} the backend calls (client.collections.*)"
    )


def test_collection_class_exposes_iterator(depcheck):
    """The backend iterates a collection via `collection.iterator()` (get()).
    The sync Collection class must expose that method. (query/data/batch are
    instance attributes set with a live connection, so they can't be pinned on
    the class offline; the manager-method test above and the filter/metadata
    construction tests cover the call shapes that reach them.)"""
    depcheck.load(IMPORT_NAME)
    coll_mod = depcheck.try_load("weaviate.collections.collection")
    if coll_mod is None:
        coll_mod = depcheck.try_load("weaviate.collections")
    if coll_mod is None:
        pytest.skip("weaviate.collections submodule not importable in this env")
    Collection = getattr(coll_mod, "Collection", None)
    assert Collection is not None, "weaviate sync Collection class not found"
    assert "iterator" in set(dir(Collection)), "Collection.iterator missing (get() relies on it)"


def test_connect_to_custom_returns_weaviateclient_type(depcheck):
    """The return annotation of connect_to_custom must be WeaviateClient (the
    type the backend stores as self.client and drives). We read the
    annotation, not call the function."""
    mod = depcheck.load(IMPORT_NAME)
    sig = inspect.signature(mod.connect_to_custom)
    ret = sig.return_annotation
    # Accept either the class object or its string form across typing styles.
    assert ret is mod.WeaviateClient or "WeaviateClient" in str(ret)
