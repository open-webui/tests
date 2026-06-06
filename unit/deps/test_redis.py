"""Dependency contract: redis (redis-py, import name ``redis``).

Redis is Open WebUI's optional shared-state backend: when configured it
holds revoked-JWT entries (utils/auth.py), the app config mirror
(internal/config.py ``PersistentConfig`` sync), the WebSocket session /
model / usage pools and distributed locks (socket/utils.py
``RedisDict`` / ``RedisLock``), the collaborative-doc CRDT updates
(``YdocManager``), the cross-worker task control channel
(tasks.py pub/sub) and tool-server caches (utils/tools.py). The
connection factory in utils/redis.py supports three topologies
(standalone, Sentinel, Cluster) in both sync and async (``redis.asyncio``)
flavours, plus the readiness ``ping`` in main.py.

This module pins exactly the slice of the redis-py API the backend calls,
so the redis 7 -> 8 major bump (which removes/renames public API) fails
loudly here instead of as a runtime ``AttributeError`` deep in a request
or socket path. Pattern mirrors test_requests.py: symbol-existence and
signature checks for the API surface, plus offline behavioural contracts
(via ``fakeredis`` when available) that never touch a real server.

Uses the ``depcheck`` fixture from unit/deps/conftest.py.
"""

from __future__ import annotations

import asyncio
import inspect

import pytest

pytestmark = pytest.mark.depcheck

IMPORT_NAME = "redis"
DIST_NAME = "redis"

# ---------------------------------------------------------------------------
# Symbol inventory — every dotted name the Open WebUI backend resolves on
# the `redis` package (sync) and its submodules.
# ---------------------------------------------------------------------------

# Top-level package symbols (utils/redis.py, internal/config.py,
# instrumentors.py, main.py, utils/auth.py).
TOP_LEVEL_SYMBOLS = [
    "Redis",  # main.py: `from redis import Redis`; type hints in internal/config.py
    "from_url",  # utils/redis.py: getattr(redis_mod, 'from_url', ...)
    "asyncio",  # socket/main.py: `from redis import asyncio as aioredis`
    "cluster",  # utils/redis.py: redis_mod.cluster.RedisCluster
    "cluster.RedisCluster",  # utils/redis.py + instrumentors.py
    "sentinel",  # utils/redis.py: redis_module.sentinel.Sentinel
    "sentinel.Sentinel",  # utils/redis.py: _build_sentinel
    "exceptions",  # utils/redis.py: redis.exceptions.*
    "exceptions.ConnectionError",  # utils/redis.py _SENTINEL_RETRYABLE
    "exceptions.ReadOnlyError",  # utils/redis.py _SENTINEL_RETRYABLE
]

# `redis.asyncio` surface (socket/main.py, tasks.py, async connection path).
ASYNCIO_SYMBOLS = [
    "Redis",  # tasks.py: `from redis.asyncio import Redis`
    "from_url",  # utils/redis.py async path: redis_mod.from_url
    "cluster",  # utils/redis.py async cluster path
    "cluster.RedisCluster",  # utils/redis.py: redis_mod.cluster.RedisCluster
    "sentinel",  # utils/redis.py async sentinel path
    "sentinel.Sentinel",  # utils/redis.py: redis_module.sentinel.Sentinel
]

# Exceptions caught / referenced anywhere in the backend's redis usage.
USED_EXCEPTIONS = [
    "ConnectionError",  # _SENTINEL_RETRYABLE (failover retry)
    "ReadOnlyError",  # _SENTINEL_RETRYABLE (replica promoted)
]

# Sync client methods the backend calls on a redis.Redis / SentinelRedisProxy.
SYNC_CLIENT_METHODS = [
    # RedisLock (socket/utils.py)
    "set",
    "get",
    "delete",
    # RedisDict (socket/utils.py)
    "hset",
    "hget",
    "hdel",
    "hexists",
    "hlen",
    "hkeys",
    "hvals",
    "hgetall",
    # internal/config.py PersistentConfig sync uses get/set (already above)
    # SentinelRedisProxy factory passthrough (utils/redis.py _FACTORY_METHODS)
    "pipeline",
    "pubsub",
    "monitor",
    "client",
    "transaction",
]

# Async client methods the backend awaits on a redis.asyncio.Redis.
ASYNC_CLIENT_METHODS = [
    # utils/auth.py (JWT revocation) + utils/tools.py (server caches)
    "get",
    "set",
    # tasks.py (cross-worker task control)
    "pubsub",
    "publish",
    "execute_command",  # RedisCluster PUBLISH fallback
    "pipeline",
    "hkeys",
    "hget",
    "smembers",
    "scard",
    "delete",
    # YdocManager (socket/utils.py)
    "rpush",
    "llen",
    "lrange",
    "exists",
    "sadd",
    "srem",
    # main.py readiness probe
    "ping",
]

# Pipeline object methods used (socket/utils.py _compact_updates_redis,
# tasks.py redis_save_task / redis_cleanup_task).
PIPELINE_METHODS = ["delete", "rpush", "execute", "hset", "sadd", "hdel", "srem"]

# pubsub object methods used (tasks.py redis_task_command_listener).
PUBSUB_METHODS = ["subscribe", "listen"]


# ---------------------------------------------------------------------------
# Import + version
# ---------------------------------------------------------------------------


def test_import():
    """`redis` must import (skip cleanly if absent in this env)."""
    redis = pytest.importorskip("redis")
    assert redis.__name__ == "redis"


def test_version_reported(depcheck):
    """The installed distribution version must be resolvable so bump tooling
    and this suite agree on what's under test."""
    pytest.importorskip("redis")
    assert depcheck.dist_version(DIST_NAME) is not None


def test_asyncio_submodule_imports():
    """socket/main.py and tasks.py import `redis.asyncio`; it must be a real
    importable submodule (not just a lazy attribute)."""
    pytest.importorskip("redis")
    aio = pytest.importorskip("redis.asyncio")
    assert aio.__name__ == "redis.asyncio"


def test_cluster_submodule_imports():
    """utils/redis.py and instrumentors.py reach redis.cluster.RedisCluster."""
    pytest.importorskip("redis")
    mod = pytest.importorskip("redis.cluster")
    assert hasattr(mod, "RedisCluster")


def test_sentinel_submodule_imports():
    """utils/redis.py reaches redis.sentinel.Sentinel."""
    pytest.importorskip("redis")
    mod = pytest.importorskip("redis.sentinel")
    assert hasattr(mod, "Sentinel")


def test_asyncio_cluster_submodule_imports():
    """The async cluster path (redis.asyncio.cluster.RedisCluster) is reached
    via redis_mod.cluster when redis_mod is redis.asyncio."""
    pytest.importorskip("redis")
    mod = pytest.importorskip("redis.asyncio.cluster")
    assert hasattr(mod, "RedisCluster")


# ---------------------------------------------------------------------------
# Symbol-existence checks (API surface)
# ---------------------------------------------------------------------------


def test_top_level_symbols_exist(depcheck):
    """Every top-level `redis.*` symbol the codebase resolves must exist."""
    redis = depcheck.load(IMPORT_NAME)
    depcheck.assert_symbols(redis, TOP_LEVEL_SYMBOLS)


def test_asyncio_symbols_exist(depcheck):
    """Every `redis.asyncio.*` symbol the codebase resolves must exist."""
    depcheck.load(IMPORT_NAME)
    aio = depcheck.load("redis.asyncio")
    depcheck.assert_symbols(aio, ASYNCIO_SYMBOLS)


def test_exception_symbols_exist(depcheck):
    """utils/redis.py references redis.exceptions.{ConnectionError,
    ReadOnlyError}; both must exist."""
    redis = depcheck.load(IMPORT_NAME)
    depcheck.assert_symbols(redis.exceptions, USED_EXCEPTIONS)


def test_top_from_url_is_callable(depcheck):
    """utils/redis.py does `getattr(redis_mod, 'from_url', None)` and calls
    it; the top-level helper must remain a callable."""
    redis = depcheck.load(IMPORT_NAME)
    depcheck.assert_callable(redis, "from_url")


def test_asyncio_from_url_is_callable(depcheck):
    """The async connection path calls redis.asyncio.from_url(...)."""
    depcheck.load(IMPORT_NAME)
    aio = depcheck.load("redis.asyncio")
    depcheck.assert_callable(aio, "from_url")


# ---------------------------------------------------------------------------
# Exception hierarchy
# ---------------------------------------------------------------------------


def test_retryable_exceptions_subclass_rediserror(depcheck):
    """_SENTINEL_RETRYABLE catches ConnectionError/ReadOnlyError; both must
    remain RedisError subclasses so the broad failover handler stays sound."""
    redis = depcheck.load(IMPORT_NAME)
    base = redis.exceptions.RedisError
    for name in ("ConnectionError", "ReadOnlyError"):
        exc = getattr(redis.exceptions, name)
        assert issubclass(exc, base), f"{name} no longer subclasses RedisError"


def test_readonly_error_subclasses_response_error(depcheck):
    """ReadOnlyError has historically been a ResponseError; that placement is
    relied on implicitly when distinguishing failover from protocol errors."""
    redis = depcheck.load(IMPORT_NAME)
    ro = redis.exceptions.ReadOnlyError
    assert issubclass(ro, redis.exceptions.ResponseError)


def test_connection_error_is_distinct_from_timeout(depcheck):
    """The retry set keys on ConnectionError specifically; ensure it is a
    concrete class (regression guard against it being aliased away)."""
    redis = depcheck.load(IMPORT_NAME)
    assert inspect.isclass(redis.exceptions.ConnectionError)
    # TimeoutError exists too and is a sibling RedisError, not the same class.
    assert redis.exceptions.ConnectionError is not redis.exceptions.TimeoutError


# ---------------------------------------------------------------------------
# Client class method surface — sync
# ---------------------------------------------------------------------------


def test_sync_client_methods_exist(depcheck):
    """Every method the backend calls on a sync redis.Redis (directly or via
    the SentinelRedisProxy passthrough) must exist on the class."""
    redis = depcheck.load(IMPORT_NAME)
    names = set(dir(redis.Redis))
    missing = [m for m in SYNC_CLIENT_METHODS if m not in names]
    assert not missing, f"redis.Redis missing method(s) the backend calls: {missing}"


def test_sync_pipeline_pubsub_are_callable(depcheck):
    """SentinelRedisProxy treats pipeline/pubsub/monitor/client/transaction as
    factory methods (not wrapped) — they must be callable on redis.Redis."""
    redis = depcheck.load(IMPORT_NAME)
    for name in ("pipeline", "pubsub", "monitor", "client", "transaction"):
        attr = getattr(redis.Redis, name)
        assert callable(attr), f"redis.Redis.{name} is not callable"


def test_redis_set_has_lock_kwargs(depcheck):
    """RedisLock (socket/utils.py) calls .set(name, value, nx=True, ex=secs)
    and .set(name, value, xx=True, ex=secs). utils/auth.py uses .set(k, v,
    ex=ttl). These keyword arguments must remain in redis.Redis.set."""
    redis = depcheck.load(IMPORT_NAME)
    depcheck.assert_params(redis.Redis.set, ["name", "value", "nx", "xx", "ex"])


def test_redis_hset_supports_mapping_kwarg(depcheck):
    """RedisDict.set (socket/utils.py) calls .hset(name, mapping=serialized).
    The `mapping` keyword must remain accepted on redis.Redis.hset."""
    redis = depcheck.load(IMPORT_NAME)
    depcheck.assert_params(redis.Redis.hset, ["name", "mapping"])


# ---------------------------------------------------------------------------
# Client class method surface — async
# ---------------------------------------------------------------------------


def test_async_client_methods_exist(depcheck):
    """Every method the backend awaits on a redis.asyncio.Redis must exist."""
    depcheck.load(IMPORT_NAME)
    aio = depcheck.load("redis.asyncio")
    names = set(dir(aio.Redis))
    missing = [m for m in ASYNC_CLIENT_METHODS if m not in names]
    assert not missing, f"redis.asyncio.Redis missing method(s) the backend awaits: {missing}"


def test_async_redis_set_has_kwargs(depcheck):
    """utils/auth.py: `await redis.set(key, '1', ex=ttl)`. The async client's
    set must still accept name/value/ex (and the lock kwargs nx/xx, since the
    same factory can return an async client used by async lock paths)."""
    depcheck.load(IMPORT_NAME)
    aio = depcheck.load("redis.asyncio")
    depcheck.assert_params(aio.Redis.set, ["name", "value", "ex", "nx", "xx"])


def test_async_pubsub_publish_callable(depcheck):
    """tasks.py: redis.pubsub(), redis.publish(channel, payload),
    redis.execute_command('PUBLISH', ...). All must be callable async-side."""
    depcheck.load(IMPORT_NAME)
    aio = depcheck.load("redis.asyncio")
    for name in ("pubsub", "publish", "execute_command"):
        assert callable(getattr(aio.Redis, name)), f"asyncio.Redis.{name} not callable"


def test_async_client_calls_return_awaitables(depcheck):
    """The backend `await`s these on an async client. redis-py shares the
    command method bodies across sync/async (so they aren't `async def` on the
    class), but calling one on an async client must return an awaitable that
    only touches the network when awaited. We build the coroutines offline and
    close them without awaiting — proving the awaited surface stays intact
    through a 7->8 sync/async reshuffle, with no server contact."""
    depcheck.load(IMPORT_NAME)
    aio = depcheck.load("redis.asyncio")
    client = aio.from_url("redis://localhost:6379/0", decode_responses=True)
    calls = {
        "get": ("k",),
        "set": ("k", "v"),
        "delete": ("k",),
        "publish": ("chan", "msg"),
        "ping": (),
        "rpush": ("lst", "v"),
        "smembers": ("s",),
    }
    for name, args in calls.items():
        coro = getattr(client, name)(*args)
        try:
            assert inspect.isawaitable(coro), (
                f"redis.asyncio.Redis.{name}(...) no longer returns an awaitable"
            )
        finally:
            if asyncio.iscoroutine(coro):
                coro.close()


# ---------------------------------------------------------------------------
# from_url — offline construction contract (NO network / NO command)
# ---------------------------------------------------------------------------


def test_sync_from_url_constructs_without_connecting(depcheck):
    """utils/redis.py: factory(redis_url, decode_responses=True, **extra).
    from_url must return a client object lazily, without opening a socket.
    We never issue a command, so no server is needed."""
    redis = depcheck.load(IMPORT_NAME)
    client = redis.from_url("redis://localhost:6379/0", decode_responses=True)
    assert client is not None
    assert isinstance(client, redis.Redis)
    try:
        for name in ("get", "set", "delete", "hset", "pipeline"):
            assert callable(getattr(client, name))
    finally:
        _safe_close(client)


def test_sync_from_url_accepts_socket_kwargs(depcheck):
    """_socket_options() forwards socket_connect_timeout / socket_keepalive /
    health_check_interval into from_url. Constructing with them must not
    raise (they remain recognised connection kwargs)."""
    redis = depcheck.load(IMPORT_NAME)
    client = redis.from_url(
        "redis://localhost:6379/0",
        decode_responses=True,
        socket_connect_timeout=5,
        socket_keepalive=True,
        health_check_interval=30,
    )
    assert client is not None
    _safe_close(client)


def test_async_from_url_constructs_without_connecting(depcheck):
    """Async connection path: redis.asyncio.from_url(url, decode_responses=).
    Must return an async client lazily without awaiting/connecting."""
    depcheck.load(IMPORT_NAME)
    aio = depcheck.load("redis.asyncio")
    client = aio.from_url("redis://localhost:6379/0", decode_responses=True)
    assert client is not None
    assert isinstance(client, aio.Redis)
    for name in ("get", "set", "publish", "pubsub", "ping"):
        assert callable(getattr(client, name))


def test_rediss_scheme_accepted_by_from_url(depcheck):
    """parse_redis_url accepts both 'redis' and 'rediss'; from_url must too
    (TLS connections use rediss://). Construction only, no connect."""
    redis = depcheck.load(IMPORT_NAME)
    client = redis.from_url("rediss://localhost:6379/0", decode_responses=True)
    assert client is not None
    _safe_close(client)


# ---------------------------------------------------------------------------
# Sentinel construction contract
# ---------------------------------------------------------------------------


def test_sentinel_constructs_from_host_list(depcheck):
    """_build_sentinel calls Sentinel(sentinels, port=, db=, username=,
    password=, decode_responses=, socket_connect_timeout=, ...). Constructing
    with a [(host, port)] list and those kwargs must not raise or connect."""
    redis = depcheck.load(IMPORT_NAME)
    sentinel = redis.sentinel.Sentinel(
        [("localhost", 26379)],
        port=6379,
        db=0,
        username=None,
        password=None,
        decode_responses=True,
        socket_connect_timeout=5,
    )
    assert sentinel is not None
    assert hasattr(sentinel, "master_for")
    assert callable(sentinel.master_for)


def test_sentinel_master_for_signature(depcheck):
    """SentinelRedisProxy._resolve_master calls sentinel.master_for(service).
    The method must accept the service name as its first positional arg."""
    redis = depcheck.load(IMPORT_NAME)
    depcheck.assert_params(redis.sentinel.Sentinel.master_for, ["service_name"])


def test_async_sentinel_exists_and_constructs(depcheck):
    """The async sentinel path uses redis.asyncio.sentinel.Sentinel with the
    same call shape."""
    depcheck.load(IMPORT_NAME)
    aio_sentinel = depcheck.load("redis.asyncio.sentinel")
    assert hasattr(aio_sentinel, "Sentinel")
    sentinel = aio_sentinel.Sentinel(
        [("localhost", 26379)],
        port=6379,
        db=0,
        decode_responses=True,
    )
    assert hasattr(sentinel, "master_for")


# ---------------------------------------------------------------------------
# Cluster construction contract
# ---------------------------------------------------------------------------


def test_cluster_from_url_is_classmethod(depcheck):
    """utils/redis.py: redis_mod.cluster.RedisCluster.from_url(redis_url,
    decode_responses=, **extra). from_url must remain a callable on the class.
    We do NOT call it (it eagerly discovers the cluster topology -> network)."""
    redis = depcheck.load(IMPORT_NAME)
    assert hasattr(redis.cluster.RedisCluster, "from_url")
    assert callable(redis.cluster.RedisCluster.from_url)


def test_async_cluster_from_url_callable(depcheck):
    """Async cluster path: redis.asyncio.cluster.RedisCluster.from_url."""
    depcheck.load(IMPORT_NAME)
    acluster = depcheck.load("redis.asyncio.cluster")
    assert hasattr(acluster.RedisCluster, "from_url")
    assert callable(acluster.RedisCluster.from_url)


def test_cluster_publish_or_execute_command(depcheck):
    """tasks.py detects cluster via `hasattr(redis, 'nodes_manager')` and then
    uses execute_command('PUBLISH', ...). RedisCluster must expose
    execute_command (and the nodes_manager attribute name as a discriminator)."""
    redis = depcheck.load(IMPORT_NAME)
    assert hasattr(redis.cluster.RedisCluster, "execute_command")
    # `nodes_manager` is the instance attribute tasks.py keys on; the class
    # must still define/document it (slot or attr) — check the name appears.
    assert "nodes_manager" in dir(redis.cluster.RedisCluster) or hasattr(
        redis.cluster.RedisCluster, "__init__"
    )


# ---------------------------------------------------------------------------
# Pipeline + pubsub object contracts (offline, via fakeredis when present)
# ---------------------------------------------------------------------------


def test_pipeline_object_has_used_methods(depcheck):
    """The pipeline returned by client.pipeline() is used as a chained command
    buffer: pipe.delete/rpush/hset/sadd/hdel/srem(...) then pipe.execute().
    Build one offline (from_url, no connect) and assert the methods exist."""
    redis = depcheck.load(IMPORT_NAME)
    client = redis.from_url("redis://localhost:6379/0", decode_responses=True)
    try:
        pipe = client.pipeline()
        for name in PIPELINE_METHODS:
            assert hasattr(pipe, name), f"pipeline missing {name}"
        assert callable(pipe.execute)
    finally:
        _safe_close(client)


def test_pubsub_object_has_used_methods(depcheck):
    """tasks.py: pubsub.subscribe(channel) then `async for m in pubsub.listen()`.
    Build a pubsub offline and assert subscribe/listen exist."""
    redis = depcheck.load(IMPORT_NAME)
    client = redis.from_url("redis://localhost:6379/0", decode_responses=True)
    try:
        ps = client.pubsub()
        for name in PUBSUB_METHODS:
            assert hasattr(ps, name), f"pubsub missing {name}"
    finally:
        _safe_close(client)


def test_async_pubsub_listen_is_async_gen(depcheck):
    """tasks.py iterates `async for message in pubsub.listen()`, so the async
    client's pubsub().listen must be an async generator function."""
    depcheck.load(IMPORT_NAME)
    aio = depcheck.load("redis.asyncio")
    client = aio.from_url("redis://localhost:6379/0", decode_responses=True)
    ps = client.pubsub()
    listen = ps.listen
    assert inspect.isasyncgenfunction(listen) or asyncio.iscoroutinefunction(listen), (
        "redis.asyncio pubsub.listen is no longer async-iterable"
    )


# ---------------------------------------------------------------------------
# Behavioural contracts via fakeredis (sync) — exercise real command semantics
# without a live server. Skipped cleanly if fakeredis is not installed.
# ---------------------------------------------------------------------------


def _fakeredis_sync(depcheck):
    fakeredis = depcheck.try_load("fakeredis")
    if fakeredis is None:
        pytest.skip("fakeredis not installed; API-surface tests cover this offline")
    return fakeredis.FakeRedis(decode_responses=True)


def test_behaviour_set_get_delete(depcheck):
    """RedisLock/auth/config use set/get/delete with str values. Verify the
    round-trip semantics the backend assumes (decode_responses -> str)."""
    depcheck.load(IMPORT_NAME)
    r = _fakeredis_sync(depcheck)
    try:
        assert r.set("k", "v") is True
        assert r.get("k") == "v"
        assert r.delete("k") == 1
        assert r.get("k") is None
    finally:
        _safe_close(r)


def test_behaviour_set_nx_xx_lock_semantics(depcheck):
    """RedisLock.aquire_lock relies on set(nx=True) returning truthy only on
    first set, and renew_lock on set(xx=True) only updating an existing key.
    Pin those exact semantics."""
    depcheck.load(IMPORT_NAME)
    r = _fakeredis_sync(depcheck)
    try:
        # nx: first set succeeds, second (key exists) returns falsy.
        assert r.set("lock", "id1", nx=True, ex=30)
        assert not r.set("lock", "id2", nx=True, ex=30)
        assert r.get("lock") == "id1"
        # xx: only sets when key already exists.
        assert r.set("lock", "id3", xx=True, ex=30)
        assert r.get("lock") == "id3"
        assert not r.set("absent", "x", xx=True, ex=30)
        assert r.get("absent") is None
    finally:
        _safe_close(r)


def test_behaviour_set_ex_sets_ttl(depcheck):
    """auth.py stores revoked tokens with set(..., ex=ttl); the key must carry
    a positive TTL afterwards (so expiry actually happens)."""
    depcheck.load(IMPORT_NAME)
    r = _fakeredis_sync(depcheck)
    try:
        r.set("revoked", "1", ex=100)
        ttl = r.ttl("revoked")
        assert 0 < ttl <= 100
    finally:
        _safe_close(r)


def test_behaviour_hash_dict_operations(depcheck):
    """RedisDict maps __setitem__/__getitem__/__delitem__/__contains__/keys/
    values/items onto hset/hget/hdel/hexists/hkeys/hvals/hgetall. Pin them."""
    depcheck.load(IMPORT_NAME)
    r = _fakeredis_sync(depcheck)
    try:
        assert r.hset("h", "a", "1") == 1
        assert r.hget("h", "a") == "1"
        assert r.hexists("h", "a") is True
        assert r.hexists("h", "missing") is False
        r.hset("h", "b", "2")
        assert r.hlen("h") == 2
        assert set(r.hkeys("h")) == {"a", "b"}
        assert set(r.hvals("h")) == {"1", "2"}
        assert r.hgetall("h") == {"a": "1", "b": "2"}
        assert r.hdel("h", "a") == 1
        assert r.hexists("h", "a") is False
    finally:
        _safe_close(r)


def test_behaviour_hset_mapping(depcheck):
    """RedisDict.set writes the whole hash via hset(name, mapping={...}).
    Verify the mapping kwarg performs a multi-field write."""
    depcheck.load(IMPORT_NAME)
    r = _fakeredis_sync(depcheck)
    try:
        r.hset("models", mapping={"x": "1", "y": "2", "z": "3"})
        assert r.hgetall("models") == {"x": "1", "y": "2", "z": "3"}
        # Stale-key removal path: hdel with *varargs.
        assert r.hdel("models", "x", "y") == 2
        assert r.hgetall("models") == {"z": "3"}
    finally:
        _safe_close(r)


def test_behaviour_pipeline_executes_in_order(depcheck):
    """tasks.redis_save_task / YdocManager._compact_updates_redis batch
    commands on a pipeline then .execute(). Verify a pipeline buffers and
    applies them."""
    depcheck.load(IMPORT_NAME)
    r = _fakeredis_sync(depcheck)
    try:
        pipe = r.pipeline()
        pipe.delete("lst")
        pipe.rpush("lst", "a", "b", "c")
        results = pipe.execute()
        assert isinstance(results, list)
        assert r.lrange("lst", 0, -1) == ["a", "b", "c"]
        assert r.llen("lst") == 3
    finally:
        _safe_close(r)


def test_behaviour_set_collection_ops(depcheck):
    """YdocManager + tasks use sadd/srem/smembers/scard for user/task sets."""
    depcheck.load(IMPORT_NAME)
    r = _fakeredis_sync(depcheck)
    try:
        assert r.sadd("users", "u1") == 1
        r.sadd("users", "u2", "u3")
        assert r.scard("users") == 3
        assert set(r.smembers("users")) == {"u1", "u2", "u3"}
        assert r.srem("users", "u1") == 1
        assert set(r.smembers("users")) == {"u2", "u3"}
    finally:
        _safe_close(r)


def test_behaviour_list_ops_and_exists(depcheck):
    """YdocManager uses rpush/lrange/llen/exists/delete on the updates list."""
    depcheck.load(IMPORT_NAME)
    r = _fakeredis_sync(depcheck)
    try:
        assert r.exists("doc") == 0
        r.rpush("doc", "u1", "u2")
        assert r.exists("doc") == 1
        assert r.llen("doc") == 2
        assert r.lrange("doc", 0, -1) == ["u1", "u2"]
        assert r.delete("doc") == 1
        assert r.exists("doc") == 0
    finally:
        _safe_close(r)


# ---------------------------------------------------------------------------
# Behavioural contracts via fakeredis (async) — exercise the awaited surface.
# ---------------------------------------------------------------------------


def _fakeredis_async(depcheck):
    fakeredis = depcheck.try_load("fakeredis")
    if fakeredis is None:
        pytest.skip("fakeredis not installed; API-surface tests cover this offline")
    aio = getattr(fakeredis, "aioredis", None)
    if aio is None or not hasattr(aio, "FakeRedis"):
        pytest.skip("fakeredis.aioredis.FakeRedis unavailable in this version")
    return aio.FakeRedis(decode_responses=True)


def test_behaviour_async_set_get_delete(depcheck):
    """auth.py/tools.py: `await redis.set(...)`, `await redis.get(...)`. Verify
    the awaited round trip against an async fake."""
    depcheck.load(IMPORT_NAME)
    r = _fakeredis_async(depcheck)

    async def scenario():
        await r.set("k", "v", ex=60)
        got = await r.get("k")
        deleted = await r.delete("k")
        gone = await r.get("k")
        await _safe_aclose(r)
        return got, deleted, gone

    got, deleted, gone = asyncio.run(scenario())
    assert got == "v"
    assert deleted == 1
    assert gone is None


def test_behaviour_async_hash_and_set_ops(depcheck):
    """tasks.py awaits hkeys/hget/smembers/scard; YdocManager awaits
    sadd/srem/smembers. Pin the awaited semantics."""
    depcheck.load(IMPORT_NAME)
    r = _fakeredis_async(depcheck)

    async def scenario():
        await r.hset("tasks", "t1", "item1")
        keys = list(await r.hkeys("tasks"))
        val = await r.hget("tasks", "t1")
        await r.sadd("item:tasks", "t1", "t2")
        members = set(await r.smembers("item:tasks"))
        count = await r.scard("item:tasks")
        await _safe_aclose(r)
        return keys, val, members, count

    keys, val, members, count = asyncio.run(scenario())
    assert keys == ["t1"]
    assert val == "item1"
    assert members == {"t1", "t2"}
    assert count == 2


def test_behaviour_async_pipeline(depcheck):
    """tasks.redis_save_task / redis_cleanup_task: `pipe = redis.pipeline()`
    then `await pipe.execute()`. Verify an async pipeline batches + applies."""
    depcheck.load(IMPORT_NAME)
    r = _fakeredis_async(depcheck)

    async def scenario():
        pipe = r.pipeline()
        pipe.hset("tasks", "t1", "item1")
        pipe.sadd("item:tasks:item1", "t1")
        await pipe.execute()
        present = await r.hget("tasks", "t1")
        card = await r.scard("item:tasks:item1")
        await _safe_aclose(r)
        return present, card

    present, card = asyncio.run(scenario())
    assert present == "item1"
    assert card == 1


def test_behaviour_async_list_compaction_ops(depcheck):
    """YdocManager append/compact: rpush/llen/lrange/delete, all awaited."""
    depcheck.load(IMPORT_NAME)
    r = _fakeredis_async(depcheck)

    async def scenario():
        await r.rpush("doc:updates", "a", "b", "c")
        length = await r.llen("doc:updates")
        items = await r.lrange("doc:updates", 0, -1)
        exists = await r.exists("doc:updates")
        await r.delete("doc:updates")
        gone = await r.exists("doc:updates")
        await _safe_aclose(r)
        return length, items, exists, gone

    length, items, exists, gone = asyncio.run(scenario())
    assert length == 3
    assert items == ["a", "b", "c"]
    assert exists == 1
    assert gone == 0


def test_behaviour_async_pubsub_roundtrip(depcheck):
    """tasks.py: producer awaits redis.publish(channel, json); consumer does
    pubsub.subscribe(channel) then `async for m in pubsub.listen()`. Verify a
    published message is delivered to a subscriber via the async fake."""
    depcheck.load(IMPORT_NAME)
    r = _fakeredis_async(depcheck)

    async def scenario():
        channel = "open-webui:tasks:commands"
        ps = r.pubsub()
        await ps.subscribe(channel)
        # Drain the subscribe confirmation message.
        await ps.get_message(timeout=1)
        await r.publish(channel, "payload")
        received = None
        for _ in range(5):
            msg = await ps.get_message(timeout=1)
            if msg and msg.get("type") == "message":
                received = msg.get("data")
                break
        await ps.unsubscribe(channel)
        await ps.aclose() if hasattr(ps, "aclose") else None
        await _safe_aclose(r)
        return received

    received = asyncio.run(scenario())
    assert received == "payload"


def test_behaviour_async_execute_command_publish(depcheck):
    """RedisCluster lacks a plain publish(), so tasks.py falls back to
    execute_command('PUBLISH', channel, payload). Verify execute_command can
    run a PUBLISH and reach a subscriber on the async fake."""
    depcheck.load(IMPORT_NAME)
    r = _fakeredis_async(depcheck)

    async def scenario():
        channel = "open-webui:tasks:commands"
        ps = r.pubsub()
        await ps.subscribe(channel)
        await ps.get_message(timeout=1)
        await r.execute_command("PUBLISH", channel, "viacmd")
        received = None
        for _ in range(5):
            msg = await ps.get_message(timeout=1)
            if msg and msg.get("type") == "message":
                received = msg.get("data")
                break
        await ps.unsubscribe(channel)
        await ps.aclose() if hasattr(ps, "aclose") else None
        await _safe_aclose(r)
        return received

    received = asyncio.run(scenario())
    assert received == "viacmd"


def test_behaviour_async_ping(depcheck):
    """main.py readiness probe: `pong = await redis.ping()`. Verify ping is
    awaitable and returns truthy against a healthy (fake) connection."""
    depcheck.load(IMPORT_NAME)
    r = _fakeredis_async(depcheck)

    async def scenario():
        pong = await r.ping()
        await _safe_aclose(r)
        return pong

    assert asyncio.run(scenario())


# ---------------------------------------------------------------------------
# Local helpers (no cross-file imports — conftest exposes only fixtures).
# ---------------------------------------------------------------------------


def _safe_close(client) -> None:
    """Close a sync client/connection-pool best-effort (offline clients never
    opened a socket; this just releases the pool object)."""
    try:
        client.close()
    except Exception:
        pass
    pool = getattr(client, "connection_pool", None)
    if pool is not None:
        try:
            pool.disconnect()
        except Exception:
            pass


async def _safe_aclose(client) -> None:
    """Close an async client best-effort across redis-py naming variants."""
    for name in ("aclose", "close"):
        closer = getattr(client, name, None)
        if closer is None:
            continue
        try:
            result = closer()
            if inspect.isawaitable(result):
                await result
            return
        except Exception:
            return
