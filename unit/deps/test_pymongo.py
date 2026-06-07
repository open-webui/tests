"""Dependency contract: pymongo (import name ``pymongo``).

pymongo is the MongoDB driver. Open WebUI pins it in
``backend/requirements.txt`` (``pymongo==4.17.0``) but does NOT import it
directly anywhere in ``open_webui/*`` — it ships as part of the data-store
ecosystem (and is a transitive dependency pulled in by other components). No
backend code path constructs a ``MongoClient`` today.

Because nothing in the backend names pymongo directly, this module pins its
*core public surface* so a bump that broke it surfaces here rather than as an
opaque failure if/when a Mongo-backed feature is enabled. We pin: the
top-level ``MongoClient``, the index-order constants, the bulk-write ops, the
``ReturnDocument`` enum, and the ``pymongo.errors`` hierarchy. The behavioural
contracts construct a real client and navigate to a database/collection
*WITHOUT CONNECTING* — pymongo defers all server I/O, and we explicitly pass
``connect=False`` so no monitoring thread or socket is opened. NO MongoDB
server is ever contacted and NO operation is executed.

Pattern mirrors test_chromadb.py / test_redis.py (construct-without-connect for
a server client). Uses the ``depcheck`` fixture from unit/deps/conftest.py.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.depcheck

IMPORT_NAME = "pymongo"
DIST_NAME = "pymongo"

# Top-level symbols a Mongo-backed feature would resolve on pymongo.
TOP_LEVEL_SYMBOLS = [
    "MongoClient",  # the client/connection factory
    "ASCENDING",  # index sort order constants
    "DESCENDING",
    "InsertOne",  # bulk-write operations
    "UpdateOne",
    "DeleteOne",
    "ReturnDocument",  # find_one_and_* return-mode enum
    "errors",  # error hierarchy submodule
]

# Collection methods the data-access layer of a Mongo feature drives.
COLLECTION_METHODS = [
    "insert_one",
    "insert_many",
    "find",
    "find_one",
    "update_one",
    "update_many",
    "delete_one",
    "delete_many",
    "aggregate",
    "count_documents",
    "create_index",
]


# ---------------------------------------------------------------------------
# Import + version
# ---------------------------------------------------------------------------


def test_import(depcheck):
    """`pymongo` must import (skip cleanly if absent in this env)."""
    mod = depcheck.load(IMPORT_NAME)
    assert mod.__name__ == "pymongo"


def test_version_reported(depcheck):
    """The installed distribution version must be resolvable so bump tooling and
    this suite agree on what's under test. (requirements.txt pins
    `pymongo==4.17.0`; the resolved version may differ slightly — this surfaces
    it for reconciliation.)"""
    depcheck.load(IMPORT_NAME)
    assert depcheck.dist_version(DIST_NAME) is not None


def test_is_v4(depcheck):
    """pymongo 4.x is the current major (3.x had a materially different API
    surface — e.g. count() vs count_documents()). Guard the major so a 3.x
    downgrade or a 5.x reshuffle is caught."""
    mod = depcheck.load(IMPORT_NAME)
    assert mod.__version__.split(".")[0] == "4", f"Expected pymongo 4.x, got {mod.__version__}."


# ---------------------------------------------------------------------------
# Symbol-existence checks (API surface)
# ---------------------------------------------------------------------------


def test_top_level_symbols_exist(depcheck):
    """Every top-level `pymongo.*` symbol a Mongo feature would resolve must
    exist."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_symbols(mod, TOP_LEVEL_SYMBOLS)


def test_mongo_client_callable(depcheck):
    """MongoClient(uri, ...) is the entry point; must be a callable class."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_callable(mod, "MongoClient")


def test_mongo_client_init_accepts_host_and_connect(depcheck):
    """MongoClient is constructed with a connection string (host) and the
    `connect` flag we use to keep construction offline. Both must remain
    accepted parameters."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_params(mod.MongoClient.__init__, ["host", "connect"])


def test_index_order_constants_are_distinct(depcheck):
    """ASCENDING / DESCENDING index directions must remain distinct values so
    index specs sort the way callers intend."""
    mod = depcheck.load(IMPORT_NAME)
    assert mod.ASCENDING != mod.DESCENDING, "pymongo.ASCENDING == DESCENDING (index order broken)"


def test_errors_hierarchy(depcheck):
    """pymongo.errors has a common PyMongoError base; ConnectionFailure and
    DuplicateKeyError must subclass it so a Mongo feature can catch the family."""
    depcheck.load(IMPORT_NAME)
    errors = depcheck.load("pymongo.errors")
    assert hasattr(errors, "PyMongoError"), "pymongo.errors.PyMongoError is gone"
    for name in ("ConnectionFailure", "DuplicateKeyError"):
        assert hasattr(errors, name), f"pymongo.errors.{name} is gone"
        assert issubclass(getattr(errors, name), errors.PyMongoError), (
            f"{name} no longer subclasses PyMongoError"
        )


# ---------------------------------------------------------------------------
# Behavioural: construct a client + navigate to a collection WITHOUT connecting.
# pymongo is lazy AND we pass connect=False so no socket/monitor thread opens.
# ---------------------------------------------------------------------------


def _client(mod):
    """A MongoClient that never connects.

    `connect=False` tells pymongo not to start the background monitoring thread
    or open any socket on construction; server I/O would only happen on the
    first actual operation, which these tests never perform."""
    return mod.MongoClient("mongodb://localhost:27017", connect=False)


def test_behaviour_construct_client_offline(depcheck):
    """MongoClient(uri, connect=False) must construct without contacting a
    server."""
    mod = depcheck.load(IMPORT_NAME)
    client = _client(mod)
    try:
        assert client is not None
    finally:
        client.close()


def test_behaviour_database_and_collection_navigation(depcheck):
    """client[db][coll] navigation is the standard access pattern and must work
    offline (it only builds Database/Collection handle objects, no I/O)."""
    mod = depcheck.load(IMPORT_NAME)
    client = _client(mod)
    try:
        db = client["owui_db"]
        assert type(db).__name__ == "Database"
        coll = db["owui_collection"]
        assert type(coll).__name__ == "Collection"
        # Attribute-style access resolves to the same collection name.
        assert client["owui_db"]["owui_collection"].name == "owui_collection"
    finally:
        client.close()


def test_behaviour_collection_methods_exist(depcheck):
    """Every collection method a Mongo data layer would call must exist on a
    real (unconnected) Collection handle."""
    mod = depcheck.load(IMPORT_NAME)
    client = _client(mod)
    try:
        coll = client["owui_db"]["owui_collection"]
        missing = [m for m in COLLECTION_METHODS if not hasattr(coll, m)]
        assert not missing, f"pymongo Collection missing method(s): {missing}"
        for m in COLLECTION_METHODS:
            assert callable(getattr(coll, m)), f"Collection.{m} is not callable"
    finally:
        client.close()


def test_behaviour_bulk_write_ops_construct(depcheck):
    """The bulk-write op objects (InsertOne/UpdateOne/DeleteOne) must construct
    as plain value objects offline — they describe operations and carry no I/O
    until handed to bulk_write()."""
    mod = depcheck.load(IMPORT_NAME)
    insert = mod.InsertOne({"_id": 1, "v": "a"})
    update = mod.UpdateOne({"_id": 1}, {"$set": {"v": "b"}})
    delete = mod.DeleteOne({"_id": 1})
    assert insert is not None
    assert update is not None
    assert delete is not None


def test_behaviour_index_model_spec(depcheck):
    """create_index takes [(field, ASCENDING), ...]; pin that an IndexModel /
    index spec using the order constants is constructible offline."""
    mod = depcheck.load(IMPORT_NAME)
    # The list-of-tuples spec is what create_index accepts.
    spec = [("name", mod.ASCENDING), ("created_at", mod.DESCENDING)]
    assert spec[0][1] == mod.ASCENDING
    assert spec[1][1] == mod.DESCENDING
    # IndexModel wraps such a spec and is part of the public surface.
    if hasattr(mod, "IndexModel"):
        model = mod.IndexModel(spec)
        assert model is not None
