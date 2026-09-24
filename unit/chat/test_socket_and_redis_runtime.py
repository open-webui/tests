"""Socket, task and Redis runtime regressions fixed in v0.11.0 that no HTTP request can reach.

The temporary-chat tool gating from the same release is pinned by the integration twin,
integration/chat/test_socket_and_redis_runtime.py. What stays here needs a Redis Sentinel, a
Redis Cluster, a second instance or control over the event loop's clock:

- 🛑 stop the moment it starts (aadab2f): `create_task` minted its own id and `chat_completion`
  stamped `metadata['task_id']` only after scheduling, so the first events an answer emitted
  carried no task id. `create_task(..., task_id=...)` now registers the caller's id.
- 🧹 orphaned sessions (bf35f64, 846ba80): `periodic_session_pool_cleanup` returned for good when
  it lost the lock race, failed a renew or failed a sweep, and `RedisLock` renewed and released
  with plain commands that ignored who held the lock.
- 🧊 Redis cluster connections (fc4906c): the connection cache key omitted `redis_cluster`, so a
  cluster and a single-server request for the same address shared one client.
- 🚏 stop with Redis configured (#27104, issue #26779): a socket timeout killed the listener
  carrying stop requests; `REDIS_SOCKET_TIMEOUT` now defaults to unset.
- 🛟 Redis failover on timeouts (75a8a00, issue #27210): `TimeoutError` was not retryable in
  `SentinelRedisProxy`, and the master was re-resolved on every call.
- 🫥 temporary chats (d484a2a, issue #27432): the status emitter persisted status updates of a
  `temporary:` chat, which has no chats row.

Redis clients are real `redis` objects (they connect lazily) or `create_autospec` stand-ins for
the calls that would reach a server. Every loop is bounded by construction: a patched
`asyncio.sleep` raises `_LoopExit` (a `BaseException` the production `except Exception` cannot
swallow) after a fixed number of awaits, inside `asyncio.wait_for`.

Discriminates: passes on bbfa876af; fails with `task_id` ignored by `create_task`, a cleanup loop
that returns on a lost race or failed renew, `RedisLock` renewing with `set(xx=True)` or
releasing with GET then DEL, `redis_cluster` dropped from the cache key, a fixed socket timeout,
`TimeoutError` removed from the retryable set, and the status emitter keyed on `local:` only.
"""

from __future__ import annotations

import ast
import asyncio
from unittest.mock import AsyncMock, create_autospec, patch

import pytest
import redis
import redis.asyncio
import redis.sentinel

pytestmark = pytest.mark.regression

LOOP_DRIVE_TIMEOUT = 5
SAVED_CHAT_ID = "8e2b5f0c-2c3f-4d1e-9a77-0a1b2c3d4e5f"


class _LoopExit(BaseException):
    """Stops a production `while True` loop after a fixed number of sleeps."""


def _sleep_until(limit: int):
    sleeps = []

    async def sleep(delay=0, *args, **kwargs):
        sleeps.append(delay)
        if len(sleeps) >= limit:
            raise _LoopExit

    return sleep


async def _drive_until_exit(coroutine) -> None:
    with pytest.raises(_LoopExit):
        await asyncio.wait_for(coroutine, timeout=LOOP_DRIVE_TIMEOUT)


@pytest.fixture(scope="session")
def socket_main(owui_module):
    return owui_module("open_webui.socket.main")


@pytest.fixture(scope="session")
def socket_utils(owui_module):
    return owui_module("open_webui.socket.utils")


@pytest.fixture(scope="session")
def tasks_module(owui_module):
    return owui_module("open_webui.tasks")


@pytest.fixture(scope="session")
def redis_utils(owui_module):
    return owui_module("open_webui.utils.redis")


# --- 🛑 aadab2f: the caller's task id is the one registered --------------------------------


@pytest.mark.asyncio
async def test_create_task_registers_the_id_the_caller_stamped(tasks_module):
    async def answer():
        return "done"

    task_id, task = await tasks_module.create_task(
        redis=None, coroutine=answer(), id="chat-1", task_id="stamped-id"
    )

    assert task_id == "stamped-id"
    assert tasks_module.tasks["stamped-id"] is task, "the task was registered under another id"
    assert await asyncio.wait_for(task, timeout=LOOP_DRIVE_TIMEOUT) == "done"


def test_chat_completion_schedules_under_the_id_it_stamped(open_webui_backend):
    main_py = open_webui_backend / "open_webui" / "main.py"
    tree = ast.parse(main_py.read_text(encoding="utf-8"))
    chat_completion = next(
        (
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.AsyncFunctionDef) and node.name == "chat_completion"
        ),
        None,
    )
    assert chat_completion, "main.chat_completion is gone; retarget this audit"
    forwarded = [
        ast.unparse(keyword.value)
        for node in ast.walk(chat_completion)
        if isinstance(node, ast.Call) and ast.unparse(node.func) == "create_task"
        for keyword in node.keywords
        if keyword.arg == "task_id"
    ]
    assert forwarded == ["per_model_metadata['task_id']"], (
        "chat_completion must hand create_task the task id it stamped into the metadata the "
        f"first events carry; it passes {forwarded}"
    )


# --- 🧹 bf35f64: the session cleanup loop keeps contending ---------------------------------


def _cleanup_lock(socket_main, acquire, renew=lambda: True):
    return (
        patch.object(socket_main, "session_aquire_func", acquire),
        patch.object(socket_main, "session_renew_func", renew),
        patch.object(socket_main, "session_release_func", lambda: True),
    )


@pytest.mark.asyncio
async def test_session_cleanup_retries_after_losing_the_lock_race(socket_main):
    attempts = []
    acquire, renew, release = _cleanup_lock(socket_main, lambda: attempts.append(1) or False)

    with acquire, renew, release, patch.object(asyncio, "sleep", _sleep_until(4)):
        await _drive_until_exit(socket_main.periodic_session_pool_cleanup())

    assert len(attempts) >= 4, "one lost acquisition ended session reaping for good"


@pytest.mark.asyncio
async def test_session_cleanup_recontends_after_a_failed_renew(socket_main):
    attempts = []
    renewals = iter([False])

    def acquire():
        attempts.append(1)
        if len(attempts) > 1:
            raise _LoopExit
        return True

    patches = _cleanup_lock(socket_main, acquire, renew=lambda: next(renewals, True))
    with patches[0], patches[1], patches[2], patch.object(socket_main, "SESSION_POOL", {}):
        await _drive_until_exit(socket_main.periodic_session_pool_cleanup())

    assert len(attempts) == 2, "a failed renew returned instead of contending again"


class _UnreachablePool(dict):
    """A session pool whose first sweep fails, as a Redis-backed pool does mid-failover."""

    def __init__(self, entries: dict):
        super().__init__(entries)
        self.failures_left = 1

    def items(self):
        if self.failures_left:
            self.failures_left -= 1
            raise redis.exceptions.ConnectionError("session pool unreachable")
        return super().items()


@pytest.mark.asyncio
async def test_session_cleanup_survives_a_failed_sweep_and_reaps_on_the_next(socket_main):
    pool = _UnreachablePool({"stale": {"id": "u-1", "last_seen_at": 0}})
    patches = _cleanup_lock(socket_main, lambda: True)
    with (
        patches[0],
        patches[1],
        patches[2],
        patch.object(socket_main, "WEBSOCKET_MANAGER", "local"),
        patch.object(socket_main, "SESSION_POOL", pool),
        patch.object(asyncio, "sleep", _sleep_until(3)),
    ):
        await _drive_until_exit(socket_main.periodic_session_pool_cleanup())

    assert pool == {}, "the stale session was never reaped after the failed sweep"


# --- 🧹 846ba80: RedisLock only renews and releases a lock it holds -------------------------


def _redis_holding(lock_name: str, holder: str):
    """A specced sync Redis client holding one key, with EVAL applied atomically."""
    store = {lock_name: holder}
    client = create_autospec(redis.Redis, instance=True)
    client.get.side_effect = store.get
    client.set.side_effect = lambda name, value, nx=False, xx=False, ex=None, **kw: (
        None
        if (nx and name in store) or (xx and name not in store)
        else store.update({name: value}) or True
    )
    client.delete.side_effect = lambda *names: sum(
        store.pop(name, None) is not None for name in names
    )

    def run_script(script, numkeys, *keys_and_args):
        key, owner = keys_and_args[0], keys_and_args[1]
        if store.get(key) != owner:
            return 0
        return 1 if "'expire'" in script else int(store.pop(key, None) is not None)

    client.eval.side_effect = run_script
    return client, store


def _lock_as(socket_utils, client, owner: str):
    with patch.object(socket_utils, "get_redis_connection", return_value=client):
        lock = socket_utils.RedisLock(
            redis_url="redis://127.0.0.1:1/0", lock_name="cleanup-lock", timeout_secs=30
        )
    lock.lock_id = owner
    return lock


def test_renew_does_not_take_over_a_lock_another_instance_holds(socket_utils):
    client, store = _redis_holding("cleanup-lock", "other-instance")

    assert not _lock_as(socket_utils, client, "this-instance").renew_lock()
    assert store["cleanup-lock"] == "other-instance"


def test_release_does_not_delete_a_lock_another_instance_holds(socket_utils):
    client, store = _redis_holding("cleanup-lock", "other-instance")
    client.get.side_effect = lambda name: "this-instance"  # read before a takeover lands

    _lock_as(socket_utils, client, "this-instance").release_lock()

    assert store == {"cleanup-lock": "other-instance"}


def test_the_holder_renews_and_releases_its_own_lock(socket_utils):
    client, store = _redis_holding("cleanup-lock", "this-instance")
    lock = _lock_as(socket_utils, client, "this-instance")

    assert lock.renew_lock()
    lock.release_lock()

    assert store == {}


def test_acquire_is_exclusive(socket_utils):
    client, store = _redis_holding("cleanup-lock", "other-instance")

    assert not _lock_as(socket_utils, client, "this-instance").aquire_lock()
    assert store["cleanup-lock"] == "other-instance"


# --- 🧊 fc4906c and 🚏 #27104: the connection factory ----------------------------------------

REDIS_URL = "redis://127.0.0.1:1/0"


@pytest.fixture
def empty_connection_cache(redis_utils):
    with patch.object(redis_utils, "_CONNECTION_POOL", {}):
        yield


def test_cluster_and_single_server_connections_are_not_shared(redis_utils, empty_connection_cache):
    cluster_client = create_autospec(redis.cluster.RedisCluster, instance=True)
    # a real RedisCluster fills its slot cache on construction, so that call is the boundary
    with patch.object(redis.cluster.RedisCluster, "from_url", return_value=cluster_client):
        cluster = redis_utils.get_redis_connection(REDIS_URL, redis_cluster=True)
    single = redis_utils.get_redis_connection(REDIS_URL, redis_cluster=False)

    assert cluster is cluster_client
    assert isinstance(single, redis.Redis), "the single-server caller got the cluster client"
    assert redis_utils.get_redis_connection(REDIS_URL) is single


def _socket_timeout(redis_utils) -> float | None:
    client = redis_utils.get_redis_connection(REDIS_URL)
    return client.connection_pool.connection_kwargs.get("socket_timeout")


def test_the_default_redis_socket_has_no_read_timeout(redis_utils, empty_connection_cache):
    assert _socket_timeout(redis_utils) is None, (
        "a read deadline on the pubsub socket kills the listener carrying stop requests (#26779)"
    )


def test_a_configured_socket_timeout_is_applied(redis_utils, empty_connection_cache):
    with patch.object(redis_utils, "REDIS_SOCKET_TIMEOUT", 7.5):
        assert _socket_timeout(redis_utils) == 7.5


# --- 🛟 75a8a00 / issue #27210: sentinel failover retries a timed-out master --------------


def _sentinel_with(*masters):
    sentinel = create_autospec(redis.sentinel.Sentinel, instance=True)
    sentinel.master_for.side_effect = list(masters)
    return sentinel


def _async_master(*results):
    master = create_autospec(redis.asyncio.Redis, instance=True)
    master.get = AsyncMock(side_effect=list(results))
    return master


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    [redis.exceptions.TimeoutError("failover"), redis.exceptions.ConnectionError("down")],
    ids=["timeout", "connection-error"],
)
async def test_a_failed_master_is_re_resolved_and_retried(redis_utils, error):
    sentinel = _sentinel_with(_async_master(error), _async_master("value"))
    proxy = redis_utils.SentinelRedisProxy(sentinel, "mymaster", async_mode=True)

    with patch.object(redis_utils, "REDIS_RECONNECT_DELAY", 0):
        assert await proxy.get("key") == "value"

    assert sentinel.master_for.call_count == 2


def test_the_sync_proxy_retries_a_timed_out_master(redis_utils):
    failing = create_autospec(redis.Redis, instance=True)
    failing.get.side_effect = redis.exceptions.TimeoutError("failover")
    healthy = create_autospec(redis.Redis, instance=True)
    healthy.get.return_value = "value"
    proxy = redis_utils.SentinelRedisProxy(
        _sentinel_with(failing, healthy), "mymaster", async_mode=False
    )

    with patch.object(redis_utils, "REDIS_RECONNECT_DELAY", 0):
        assert proxy.get("key") == "value"


@pytest.mark.asyncio
async def test_the_master_is_resolved_once_while_it_answers(redis_utils):
    sentinel = _sentinel_with(_async_master("a", "b"))
    proxy = redis_utils.SentinelRedisProxy(sentinel, "mymaster", async_mode=True)

    assert [await proxy.get("x"), await proxy.get("y")] == ["a", "b"]
    assert sentinel.master_for.call_count == 1


@pytest.mark.asyncio
async def test_a_master_that_stays_down_still_raises(redis_utils):
    masters = [
        _async_master(redis.exceptions.ConnectionError("down"))
        for _ in range(redis_utils.REDIS_SENTINEL_MAX_RETRY_COUNT)
    ]
    proxy = redis_utils.SentinelRedisProxy(_sentinel_with(*masters), "mymaster", async_mode=True)

    with (
        patch.object(redis_utils, "REDIS_RECONNECT_DELAY", 0),
        pytest.raises(redis.exceptions.ConnectionError),
    ):
        await proxy.get("key")


# --- 🫥 d484a2a / issue #27432: a temporary chat's status is not persisted ---------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("chat_id", "writes"),
    [("temporary:abc", 0), ("local:abc", 0), (SAVED_CHAT_ID, 1)],
    ids=["temporary", "legacy-local", "saved"],
)
async def test_status_updates_are_persisted_for_saved_chats_only(socket_main, chat_id, writes):
    chats = create_autospec(type(socket_main.Chats), instance=True)
    with (
        patch.object(socket_main, "Chats", chats),
        patch.object(socket_main.sio, "emit", AsyncMock()),
    ):
        emitter = await socket_main.get_event_emitter(
            {"user_id": "u-1", "chat_id": chat_id, "message_id": "m-1"}
        )
        await emitter({"type": "status", "data": {"description": "working"}})

    assert chats.add_message_status_to_chat_by_id_and_message_id.await_count == writes
