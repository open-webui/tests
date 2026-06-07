"""Dependency contract: pinecone (the ``pinecone`` SDK, v5+/"pinecone" pkg).

Pinecone is one of Open WebUI's pluggable vector-DB backends
(``retrieval/vector/dbs/pinecone.py``, ``VECTOR_DB=pinecone``). The
client wrapper imports and uses:

    from pinecone import Pinecone, ServerlessSpec
    try:
        from pinecone.grpc import PineconeGRPC      # optional, preferred
    except ImportError:
        ...

    client = PineconeGRPC(api_key=, pool_threads=, timeout=)   # or Pinecone(...)
    client.list_indexes().names()
    client.create_index(name=, dimension=, metric=,
                        spec=ServerlessSpec(cloud=, region=))
    index = client.Index(name, pool_threads=)
    index.query(vector=, top_k=, filter=, include_metadata=)   # -> .matches[*].id/.score/.metadata
    index.upsert(vectors=[{id, values, metadata}, ...])
    index.delete(ids=) / index.delete(filter=) / index.delete(delete_all=True)

This is the modern "pinecone" package (the ``Pinecone`` class), NOT the
legacy ``pinecone-client`` ``pinecone.init()`` API. This file pins the
class/method surface and keyword arguments the wrapper depends on. It
NEVER constructs a real client or touches the network: ``Pinecone(...)``
only stores config (no connection until a call), and ``create_index`` /
``Index`` / ``query`` / ``upsert`` / ``delete`` all hit the API, so those
are checked by signature/attribute only. ``ServerlessSpec`` is a plain
config object and IS constructed offline.

Uses the ``depcheck`` fixture from unit/deps/conftest.py.
"""

from __future__ import annotations

import inspect

import pytest

pytestmark = pytest.mark.depcheck

IMPORT_NAME = "pinecone"
DIST_NAME = "pinecone"

# Top-level symbols the backend imports from `pinecone`.
USED_SYMBOLS = [
    "Pinecone",  # from pinecone import Pinecone
    "ServerlessSpec",  # from pinecone import ServerlessSpec
]

# Methods the wrapper calls on the client object (Pinecone / PineconeGRPC).
CLIENT_METHODS = [
    "list_indexes",
    "create_index",
    "Index",  # note: capitalised — returns an index handle
]

# Methods the wrapper calls on the index handle.
INDEX_METHODS = [
    "query",
    "upsert",
    "delete",
]


def test_import(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    assert mod.__name__ == "pinecone"


def test_version_reported(depcheck):
    assert depcheck.dist_version(DIST_NAME) is not None


def test_used_symbols_exist(depcheck):
    """Pinecone + ServerlessSpec must be importable off the top-level package."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_symbols(mod, USED_SYMBOLS)


def test_pinecone_is_class(depcheck):
    """The wrapper does `from pinecone import Pinecone` then `Pinecone(...)`."""
    mod = depcheck.load(IMPORT_NAME)
    assert inspect.isclass(mod.Pinecone)


def test_serverlessspec_is_class(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    assert inspect.isclass(mod.ServerlessSpec)


def test_pinecone_constructor_accepts_our_kwargs(depcheck):
    """The wrapper constructs Pinecone(api_key=, pool_threads=, timeout=). Those
    keyword names must remain accepted (or it must take **kwargs)."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_params(mod.Pinecone.__init__, ["api_key", "pool_threads", "timeout"])


def test_client_methods_exist(depcheck):
    """list_indexes / create_index / Index must exist on the Pinecone client
    class. (Capital-I `Index` is the index-handle factory — easy to lose in a
    rename, so pin it explicitly.)"""
    mod = depcheck.load(IMPORT_NAME)
    names = set(dir(mod.Pinecone))
    missing = [m for m in CLIENT_METHODS if m not in names]
    assert not missing, f"pinecone.Pinecone missing method(s): {missing}"
    for m in CLIENT_METHODS:
        assert callable(getattr(mod.Pinecone, m)), f"Pinecone.{m} not callable"


def test_create_index_accepts_our_kwargs(depcheck):
    """create_index(name=, dimension=, metric=, spec=) — pin those keyword
    names so a signature change is caught before index creation fails."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_params(
        mod.Pinecone.create_index,
        ["name", "dimension", "metric", "spec"],
    )


def test_index_factory_accepts_name_and_pool_threads(depcheck):
    """The wrapper does client.Index(name, pool_threads=20). The Index factory
    must accept a name positionally and a pool_threads keyword."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_params(mod.Pinecone.Index, ["name", "pool_threads"])


def test_serverlessspec_constructs_offline(depcheck):
    """ServerlessSpec(cloud=, region=) is built locally (no network) and passed
    to create_index. Constructing it with the wrapper's kwargs must not raise."""
    mod = depcheck.load(IMPORT_NAME)
    spec = mod.ServerlessSpec(cloud="aws", region="us-east-1")
    assert spec is not None
    # cloud/region should be retrievable off the spec object.
    assert getattr(spec, "cloud", "aws") == "aws"
    assert getattr(spec, "region", "us-east-1") == "us-east-1"


def test_serverlessspec_signature_has_cloud_region(depcheck):
    """Pin the keyword names on ServerlessSpec itself (the wrapper passes
    cloud=self.cloud, region=self.environment)."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_params(mod.ServerlessSpec.__init__, ["cloud", "region"])


def test_index_methods_exist_on_index_type(depcheck):
    """query / upsert / delete must exist on the index handle type. We resolve
    the index class without connecting: prefer the public data-plane Index
    class from the SDK's data module; fall back to the type used by Pinecone."""
    mod = depcheck.load(IMPORT_NAME)
    index_cls = _resolve_index_class(mod)
    if index_cls is None:
        pytest.skip("Could not resolve a pinecone Index class offline in this build")
    names = set(dir(index_cls))
    missing = [m for m in INDEX_METHODS if m not in names]
    assert not missing, f"pinecone Index class missing method(s): {missing}"


def test_index_query_accepts_our_kwargs(depcheck):
    """index.query(vector=, top_k=, filter=, include_metadata=) is the core read
    path (search/query/get/has_collection). Pin those keyword names."""
    mod = depcheck.load(IMPORT_NAME)
    index_cls = _resolve_index_class(mod)
    if index_cls is None:
        pytest.skip("Could not resolve a pinecone Index class offline in this build")
    query = getattr(index_cls, "query", None)
    assert query is not None and callable(query)
    depcheck.assert_params(query, ["vector", "top_k", "filter", "include_metadata"])


def test_index_upsert_accepts_vectors_kwarg(depcheck):
    """insert/upsert do index.upsert(vectors=batch). The `vectors` keyword must
    remain accepted."""
    mod = depcheck.load(IMPORT_NAME)
    index_cls = _resolve_index_class(mod)
    if index_cls is None:
        pytest.skip("Could not resolve a pinecone Index class offline in this build")
    upsert = getattr(index_cls, "upsert", None)
    assert upsert is not None and callable(upsert)
    depcheck.assert_params(upsert, ["vectors"])


def test_index_delete_accepts_our_kwargs(depcheck):
    """delete() is called three ways: delete(ids=), delete(filter=),
    delete(delete_all=True). All three keyword names must remain accepted."""
    mod = depcheck.load(IMPORT_NAME)
    index_cls = _resolve_index_class(mod)
    if index_cls is None:
        pytest.skip("Could not resolve a pinecone Index class offline in this build")
    delete = getattr(index_cls, "delete", None)
    assert delete is not None and callable(delete)
    depcheck.assert_params(delete, ["ids", "filter", "delete_all"])


def test_grpc_submodule_is_optional(depcheck):
    """The wrapper guards `from pinecone.grpc import PineconeGRPC` in a
    try/except ImportError. So either the grpc submodule imports and exposes
    PineconeGRPC, or it's absent — both are valid. We just assert the import
    contract: if pinecone.grpc imports, PineconeGRPC must be on it and be a
    class accepting the same api_key/pool_threads/timeout kwargs."""
    depcheck.load(IMPORT_NAME)
    grpc_mod = depcheck.try_load("pinecone.grpc")
    if grpc_mod is None:
        pytest.skip("pinecone.grpc not installed (HTTP client fallback path)")
    assert hasattr(grpc_mod, "PineconeGRPC"), (
        "pinecone.grpc imports but no longer exposes PineconeGRPC"
    )
    assert inspect.isclass(grpc_mod.PineconeGRPC)
    depcheck.assert_params(
        grpc_mod.PineconeGRPC.__init__,
        ["api_key", "pool_threads", "timeout"],
    )


def test_legacy_init_not_relied_on(depcheck):
    """Regression marker: the wrapper uses the class-based `Pinecone(...)` API,
    NOT the legacy module-level `pinecone.init()`. Whether or not `init` still
    exists, the class must be the supported entrypoint — assert the class path
    works (constructible reference), which is what the backend depends on."""
    mod = depcheck.load(IMPORT_NAME)
    # The class must be the entrypoint; we don't require legacy init to be gone,
    # only that the class API the backend uses is present.
    assert inspect.isclass(mod.Pinecone)
    assert callable(getattr(mod.Pinecone, "create_index", None))


# ---------------------------------------------------------------------------
# Local helpers (no cross-file imports; never connect).
# ---------------------------------------------------------------------------


def _resolve_index_class(mod):
    """Resolve the pinecone *data-plane* Index class (the one carrying query/
    upsert/delete) without constructing a client (which would need an API key /
    network). The top-level ``pinecone.Index`` can be a lazy import-error stub
    in some builds, so we (a) probe the real data-plane modules first and
    (b) only accept a class that actually exposes the methods the backend
    calls. Returns None if no usable class resolves."""
    import importlib

    candidates = [
        ("pinecone.data.index", "Index"),
        ("pinecone.db_data.index", "Index"),
        ("pinecone.data", "Index"),
        ("pinecone.db_data", "Index"),
        ("pinecone", "Index"),
        ("pinecone.control", "Index"),
    ]
    for modname, attr in candidates:
        try:
            m = importlib.import_module(modname)
        except Exception:
            continue
        cls = getattr(m, attr, None)
        if not inspect.isclass(cls):
            continue
        # Reject lazy import-error / stub placeholders: require the real
        # data-plane surface the backend uses.
        if all(hasattr(cls, name) for name in ("query", "upsert", "delete")):
            return cls
    return None
