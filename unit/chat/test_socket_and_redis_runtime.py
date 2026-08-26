"""Socket, task and Redis runtime regressions fixed in v0.11.0.

Seven changelog entries share the websocket/Redis runtime: `socket/main.py`,
`socket/utils.py`, `tasks.py`, `utils/redis.py` and the new `utils/chat_id.py`.

- 📻 second tab not joined (d14fddf): `sio.enter_room` sat inside the branch that
  populates `SESSION_POOL` for a *new* session, so a reconnecting tab never joined
  `user:<id>`. Guarded structurally below; see the note on that test, the defect was
  introduced and fixed inside the 0.11.0 cycle so it does not discriminate here.
- 🛑 stop the moment it starts (aadab2f): `create_task` minted its own id and the
  caller stamped `metadata['task_id']` only after scheduling, so the first events an
  answer emitted carried no task id. `create_task(..., task_id=...)` now takes the
  caller's id.
- 🧹 orphaned sessions (bf35f64, 846ba80): `periodic_session_pool_cleanup` returned
  for good when it lost the lock race or failed a renew, and `RedisLock` renewed and
  released with non-atomic commands that ignored who held the lock.
- 🧊 Redis cluster connections (fc4906c): the connection-pool cache key omitted
  `redis_cluster`, so a cluster and a single-server request for the same address
  shared whichever connection was built first.
- 🚏 stop with Redis configured (#27104, issue #26779): a socket timeout killed the
  listener that carries stop requests; `REDIS_SOCKET_TIMEOUT` now defaults to unset.
- 🛟 Redis failover on timeouts (75a8a00, issue #27210): `TimeoutError` was not
  retryable in `SentinelRedisProxy`, and the master was re-resolved on every call.
- 🫥 temporary chats and channels (d484a2a, d2936c8, b45c020, 71c4da8, issue #27432):
  call sites open-coded `startswith(('local:', 'channel:'))` and never learned the new
  `temporary:` prefix, so temporary chats tried to persist and were offered the
  task-list tools. `utils/chat_id.is_saved_chat_id` is now the single answer.

Every loop test is bounded by construction: a patched `asyncio.sleep` raises `_LoopExit`
(a `BaseException`, so production `except Exception` cannot swallow it) after a fixed
number of awaits, and each drive is additionally wrapped in `asyncio.wait_for`.

0.11.1 replaced the single `sleep(SESSION_POOL_TIMEOUT)` between sweeps with renew-sized
chunks that keep the lock alive while waiting, so the wait is asserted as a budget rather
than as one fixed delay.

Discriminates: passes on v0.11.0 and v0.11.1, fails on v0.10.2 (pre-fix the cleanup loop gives up,
the lock renews/releases without an owner check, the connection cache collides,
`socket_timeout` cannot be disabled, sentinel timeouts are fatal, and `chat_id.py`
does not exist so `temporary:` chats persist and get the task tools).
"""

from __future__ import annotations

import ast
import asyncio
import importlib
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
import redis as redis_pkg

pytestmark = pytest.mark.regression

LOOP_DRIVE_TIMEOUT = 5


class _LoopExit(BaseException):
    """Sentinel that stops a production while-True loop after a fixed number of sleeps."""


def _counting_sleep(record: list[float], limit: int):
    """Instant `asyncio.sleep` replacement that aborts the caller after `limit` awaits."""

    async def _sleep(delay=0, *args, **kwargs):
        record.append(delay)
        if len(record) >= limit:
            raise _LoopExit
        return None

    return _sleep


async def _drive(coro):
    """Run a production loop under a hard timeout so a missing exit cannot wedge the run."""
    await asyncio.wait_for(coro, timeout=LOOP_DRIVE_TIMEOUT)


async def _instant_async_sleep(delay=0, *args, **kwargs):
    return None


def _instant_sync_sleep(delay=0, *args, **kwargs):
    return None


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


@pytest.fixture(scope="session")
def tools_module(owui_module):
    return owui_module("open_webui.utils.tools")


@pytest.fixture(scope="session")
def socket_main_tree(open_webui_backend: Path) -> ast.Module:
    source = (open_webui_backend / "open_webui" / "socket" / "main.py").read_text(encoding="utf-8")
    return ast.parse(source)


@pytest.fixture(scope="session")
def app_main_tree(open_webui_backend: Path) -> ast.Module:
    return ast.parse((open_webui_backend / "open_webui" / "main.py").read_text(encoding="utf-8"))


def _chat_id_module():
    """Import `open_webui.utils.chat_id` without skipping: absence is the regression."""
    return importlib.import_module("open_webui.utils.chat_id")


# --- 📻 d14fddf: a reconnecting tab still joins the user's event room ------------------------


def _named_function(tree: ast.Module, name: str):
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    raise AssertionError(f"{name} not found")


def _is_user_room_join(stmt) -> bool:
    if not isinstance(stmt, ast.Expr) or not isinstance(stmt.value, ast.Await):
        return False
    call = stmt.value.value
    if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Attribute):
        return False
    if call.func.attr != "enter_room" or len(call.args) != 2:
        return False
    room = call.args[1]
    return isinstance(room, ast.JoinedStr) and any(
        isinstance(part, ast.Constant) and "user:" in str(part.value) for part in room.values
    )


def test_user_join_enters_the_user_room_unconditionally(socket_main_tree):
    """The join must not hang off the new-session branch, or a second tab gets no events.

    Does not discriminate: v0.10.2 already joins unconditionally, the defect was
    introduced and fixed within the 0.11.0 cycle. Kept as the invariant guard.
    """
    user_join = _named_function(socket_main_tree, "user_join")
    assert any(_is_user_room_join(stmt) for stmt in user_join.body), (
        "user_join must call sio.enter_room(sid, f'user:{...}') at its top level"
    )


# --- 🛑 aadab2f: the task id is minted before the coroutine is scheduled ---------------------


async def _settle_task(tasks_module, task_id: str, item_id: str) -> None:
    """Let the done-callback cleanup run, then drop anything it left behind."""
    for _ in range(3):
        await asyncio.sleep(0)
    tasks_module.tasks.pop(task_id, None)
    remaining = tasks_module.item_tasks.get(item_id) or []
    if task_id in remaining:
        remaining.remove(task_id)
    if not remaining:
        tasks_module.item_tasks.pop(item_id, None)


@pytest.mark.asyncio
async def test_create_task_registers_under_the_caller_supplied_id(tasks_module):
    """Pre-fix create_task always generated its own id, so the caller could not pre-stamp it."""

    async def _noop():
        return "done"

    item_id = "chat-caller-supplied-id"
    task_id, task = await tasks_module.create_task(
        None, _noop(), id=item_id, task_id="fixed-task-id"
    )
    try:
        assert task_id == "fixed-task-id"
        assert tasks_module.tasks["fixed-task-id"] is task
        assert "fixed-task-id" in tasks_module.item_tasks[item_id]
        assert await asyncio.wait_for(task, timeout=LOOP_DRIVE_TIMEOUT) == "done"
    finally:
        task.cancel()
        await _settle_task(tasks_module, "fixed-task-id", item_id)


@pytest.mark.asyncio
async def test_create_task_still_generates_an_id_when_none_is_given(tasks_module):
    """Nearby: the default path is unchanged."""

    async def _noop():
        return "done"

    item_id = "chat-generated-id"
    task_id, task = await tasks_module.create_task(None, _noop(), id=item_id)
    try:
        assert task_id
        assert tasks_module.tasks[task_id] is task
        assert await asyncio.wait_for(task, timeout=LOOP_DRIVE_TIMEOUT) == "done"
    finally:
        task.cancel()
        await _settle_task(tasks_module, task_id, item_id)


def _create_task_calls(func) -> list[ast.Call]:
    return [
        node
        for node in ast.walk(func)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "create_task"
    ]


def test_chat_completion_passes_the_id_it_stamped_into_the_metadata(app_main_tree):
    """Broad: the id in per_model_metadata and the id create_task registers must be the same one."""
    chat_completion = _named_function(app_main_tree, "chat_completion")

    stamped = [
        node
        for node in ast.walk(chat_completion)
        if isinstance(node, ast.Dict)
        and any(isinstance(key, ast.Constant) and key.value == "task_id" for key in node.keys)
    ]
    assert stamped, "chat_completion must mint a task_id into per_model_metadata before scheduling"

    forwarded = [
        keyword
        for call in _create_task_calls(chat_completion)
        for keyword in call.keywords
        if keyword.arg == "task_id"
    ]
    assert forwarded, "chat_completion must hand create_task the id it already stamped"
    for keyword in forwarded:
        assert isinstance(keyword.value, ast.Subscript)
        assert isinstance(keyword.value.value, ast.Name)
        assert keyword.value.value.id == "per_model_metadata"
        assert isinstance(keyword.value.slice, ast.Constant)
        assert keyword.value.slice.value == "task_id"


# --- 🧹 bf35f64: the session cleanup loop keeps contending for the lock ----------------------


class _UndeletablePool(dict):
    """SESSION_POOL whose entry vanishes between the scan and the delete."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.delete_attempts: list[str] = []

    def __delitem__(self, key):
        self.delete_attempts.append(key)
        raise KeyError(key)


@pytest.mark.asyncio
async def test_session_cleanup_retries_a_lost_lock_race(socket_main):
    """Pre-fix one lost acquisition ended session reaping on this instance for good."""
    acquires: list[int] = []
    sleeps: list[float] = []

    def acquire():
        acquires.append(len(acquires))
        return False

    with (
        patch.object(socket_main, "SESSION_POOL", {}),
        patch.object(socket_main, "session_aquire_func", acquire),
        patch.object(socket_main, "session_renew_func", lambda: True),
        patch.object(socket_main, "session_release_func", lambda: True),
        patch.object(asyncio, "sleep", _counting_sleep(sleeps, 4)),
        pytest.raises(_LoopExit),
    ):
        await _drive(socket_main.periodic_session_pool_cleanup())

    assert len(acquires) >= 4


@pytest.mark.asyncio
async def test_session_cleanup_recontends_after_a_failed_renew(socket_main):
    """Pre-fix a failed renew returned, leaving stale sessions to pile up forever."""
    acquires: list[int] = []
    releases: list[int] = []
    renew_results = [False, True]
    sleeps: list[float] = []

    def acquire():
        acquires.append(len(acquires))
        return True

    def renew():
        return renew_results.pop(0) if renew_results else False

    def release():
        releases.append(len(releases))
        return True

    with (
        patch.object(socket_main, "SESSION_POOL", {}),
        patch.object(socket_main, "session_aquire_func", acquire),
        patch.object(socket_main, "session_renew_func", renew),
        patch.object(socket_main, "session_release_func", release),
        patch.object(asyncio, "sleep", _counting_sleep(sleeps, 1)),
        pytest.raises(_LoopExit),
    ):
        await _drive(socket_main.periodic_session_pool_cleanup())

    assert len(acquires) == 2
    assert len(releases) == 2


@pytest.mark.asyncio
async def test_session_cleanup_survives_a_session_reaped_by_another_instance(socket_main):
    """Pre-fix a concurrent delete raised KeyError straight out of the cleanup loop."""
    now = int(time.time())
    pool = _UndeletablePool(
        {"stale": {"id": "u-1", "last_seen_at": now - socket_main.SESSION_POOL_TIMEOUT - 60}}
    )
    sleeps: list[float] = []

    with (
        patch.object(socket_main, "SESSION_POOL", pool),
        patch.object(socket_main, "session_aquire_func", lambda: True),
        patch.object(socket_main, "session_renew_func", lambda: True),
        patch.object(socket_main, "session_release_func", lambda: True),
        patch.object(asyncio, "sleep", _counting_sleep(sleeps, 1)),
        pytest.raises(_LoopExit),
    ):
        await _drive(socket_main.periodic_session_pool_cleanup())

    assert pool.delete_attempts == ["stale"], "the stale session was never reaped"
    assert sleeps, "the KeyError escaped before the loop reached its wait"
    assert all(0 < delay <= socket_main.SESSION_POOL_TIMEOUT for delay in sleeps)


# --- 🧹 846ba80: RedisLock only renews and releases a lock it actually holds -----------------


class FakeLockRedis:
    """In-memory stand-in for the sync redis client RedisLock talks to.

    `stale_reads` models a plain GET that returns a value the key no longer holds,
    which is exactly the window the old read-then-delete release raced in.
    """

    def __init__(self):
        self.store: dict[str, str] = {}
        self.ttl: dict[str, int] = {}
        self.stale_reads: dict[str, str] = {}

    def set(self, name, value, nx=False, xx=False, ex=None):
        exists = name in self.store
        if nx and exists:
            return None
        if xx and not exists:
            return None
        self.store[name] = value
        self.ttl[name] = ex
        return True

    def get(self, name):
        if name in self.stale_reads:
            return self.stale_reads[name]
        return self.store.get(name)

    def delete(self, *names):
        removed = 0
        for name in names:
            if self.store.pop(name, None) is not None:
                self.ttl.pop(name, None)
                removed += 1
        return removed

    def eval(self, script, numkeys, *args):
        key = args[0]
        argv = args[1:]
        if self.store.get(key) != argv[0]:
            return 0
        if "'expire'" in script:
            self.ttl[key] = argv[1]
            return 1
        if "'del'" in script:
            return self.delete(key)
        raise AssertionError(f"unexpected lua script: {script}")


def _lock(socket_utils, fake: FakeLockRedis, lock_id: str = "ours", timeout: int = 30):
    lock = socket_utils.RedisLock(None, "cleanup-lock", timeout)
    lock.redis = fake
    lock.lock_id = lock_id
    return lock


def test_renew_does_not_steal_a_lock_held_by_another_instance(socket_utils):
    """Pre-fix `set(xx=True)` renewed the key whoever owned it, overwriting the holder."""
    fake = FakeLockRedis()
    fake.store["cleanup-lock"] = "other-holder"
    fake.ttl["cleanup-lock"] = 30
    lock = _lock(socket_utils, fake)

    assert not lock.renew_lock()
    assert fake.store["cleanup-lock"] == "other-holder"


def test_release_does_not_delete_a_lock_that_changed_hands(socket_utils):
    """Pre-fix the GET and the DEL were separate, so a takeover between them lost the lock."""
    fake = FakeLockRedis()
    fake.store["cleanup-lock"] = "other-holder"
    fake.stale_reads["cleanup-lock"] = "ours"
    lock = _lock(socket_utils, fake)

    lock.release_lock()

    assert fake.store.get("cleanup-lock") == "other-holder"


def test_renew_refreshes_a_lock_we_do_hold(socket_utils):
    """Nearby: the owner's own renew still works and pushes the expiry out."""
    fake = FakeLockRedis()
    fake.store["cleanup-lock"] = "ours"
    fake.ttl["cleanup-lock"] = 1
    lock = _lock(socket_utils, fake, timeout=45)

    assert lock.renew_lock()
    assert fake.ttl["cleanup-lock"] == 45
    assert fake.store["cleanup-lock"] == "ours"


def test_release_deletes_a_lock_we_do_hold(socket_utils):
    """Nearby: the owner's own release still frees the key."""
    fake = FakeLockRedis()
    fake.store["cleanup-lock"] = "ours"
    lock = _lock(socket_utils, fake)

    lock.release_lock()

    assert "cleanup-lock" not in fake.store


def test_acquire_is_still_exclusive(socket_utils):
    """Nearby: NX acquisition semantics are unchanged."""
    fake = FakeLockRedis()
    first = _lock(socket_utils, fake, lock_id="first")
    second = _lock(socket_utils, fake, lock_id="second")

    assert first.aquire_lock()
    assert not second.aquire_lock()
    assert fake.store["cleanup-lock"] == "first"


# --- 🧊 fc4906c: cluster mode is part of the connection cache key ----------------------------


def test_cluster_and_single_connections_are_not_shared(redis_utils):
    """Pre-fix the cache key omitted redis_cluster, so the second caller got the first's client."""
    url = "redis://cache-key-probe:6379/0"
    cluster_conn = object()
    single_conn = object()

    with (
        patch.object(redis_utils, "_CONNECTION_POOL", {}),
        patch.object(redis_pkg.cluster.RedisCluster, "from_url", lambda *a, **k: cluster_conn),
        patch.object(redis_pkg, "from_url", lambda *a, **k: single_conn),
    ):
        first = redis_utils.get_redis_connection(url, redis_cluster=True)
        second = redis_utils.get_redis_connection(url, redis_cluster=False)

    assert first is cluster_conn
    assert second is single_conn
    assert first is not second


def test_identical_requests_still_share_one_connection(redis_utils):
    """Nearby: caching itself is unchanged for matching arguments."""
    url = "redis://cache-key-probe:6379/0"
    single_conn = object()

    with (
        patch.object(redis_utils, "_CONNECTION_POOL", {}),
        patch.object(redis_pkg, "from_url", lambda *a, **k: single_conn),
    ):
        first = redis_utils.get_redis_connection(url)
        second = redis_utils.get_redis_connection(url)

    assert first is second is single_conn


# --- 🚏 #27104 / issue #26779: the stop listener is not killed by a socket timeout -----------


def test_socket_timeout_is_only_set_when_configured(redis_utils):
    """Pre-fix there was no REDIS_SOCKET_TIMEOUT knob, so the value could not be applied."""
    with patch.object(redis_utils, "REDIS_SOCKET_TIMEOUT", 7.5, create=True):
        opts = redis_utils._socket_options()

    assert opts.get("socket_timeout") == 7.5


def test_default_config_leaves_the_redis_socket_without_a_timeout(redis_utils):
    """Nearby: the shipped default must not put a read deadline on the pubsub listener."""
    assert "socket_timeout" not in redis_utils._socket_options()


def test_connect_timeout_is_untouched(redis_utils):
    """Nearby: the separate connect timeout still comes through."""
    with patch.object(redis_utils, "REDIS_SOCKET_CONNECT_TIMEOUT", 3.0):
        opts = redis_utils._socket_options()

    assert opts["socket_connect_timeout"] == 3.0


# --- 🛟 75a8a00 / issue #27210: sentinel failover retries a timed-out master -----------------


class FakeSentinel:
    """Hands out masters in order and records how often it was asked."""

    def __init__(self, masters):
        self._masters = list(masters)
        self.calls: list[str] = []

    def master_for(self, service_name, *args, **kwargs):
        self.calls.append(service_name)
        index = min(len(self.calls) - 1, len(self._masters) - 1)
        return self._masters[index]


def _flaky_master(error, result="from-new-master"):
    """A master whose first call raises `error` and whose next call succeeds."""
    state = {"calls": 0}

    async def _get(*args, **kwargs):
        state["calls"] += 1
        if state["calls"] == 1:
            raise error
        return result

    return SimpleNamespace(get=_get)


@pytest.mark.asyncio
async def test_a_timed_out_master_is_retried_instead_of_raising(redis_utils):
    """Pre-fix TimeoutError was not retryable, so a failover mid-operation surfaced as an error."""
    master = _flaky_master(redis_pkg.exceptions.TimeoutError("failover"))
    sentinel = FakeSentinel([master])
    proxy = redis_utils.SentinelRedisProxy(sentinel, "mymaster", async_mode=True)

    with patch.object(asyncio, "sleep", _instant_async_sleep):
        result = await proxy.get("key")

    assert result == "from-new-master"
    assert len(sentinel.calls) == 2, "the retry must ask Sentinel for the master again"


def test_sync_mode_also_retries_a_timed_out_master(redis_utils):
    """Pre-fix the sync wrapper shared the same non-retryable tuple."""
    state = {"calls": 0}

    def _get(*args, **kwargs):
        state["calls"] += 1
        if state["calls"] == 1:
            raise redis_pkg.exceptions.TimeoutError("failover")
        return "from-new-master"

    sentinel = FakeSentinel([SimpleNamespace(get=_get)])
    proxy = redis_utils.SentinelRedisProxy(sentinel, "mymaster", async_mode=False)

    with patch.object(time, "sleep", _instant_sync_sleep):
        assert proxy.get("key") == "from-new-master"


@pytest.mark.asyncio
async def test_the_master_is_resolved_once_and_reused(redis_utils):
    """Pre-fix every operation re-asked Sentinel; _clear_master is what resets the memo."""

    async def _get(*args, **kwargs):
        return "ok"

    sentinel = FakeSentinel([SimpleNamespace(get=_get)])
    proxy = redis_utils.SentinelRedisProxy(sentinel, "mymaster", async_mode=True)

    with patch.object(asyncio, "sleep", _instant_async_sleep):
        assert await proxy.get("a") == "ok"
        assert await proxy.get("b") == "ok"

    assert len(sentinel.calls) == 1


@pytest.mark.asyncio
async def test_a_connection_error_still_retries(redis_utils):
    """Nearby: the retryable errors that already worked still do."""
    master = _flaky_master(redis_pkg.exceptions.ConnectionError("down"))
    sentinel = FakeSentinel([master])
    proxy = redis_utils.SentinelRedisProxy(sentinel, "mymaster", async_mode=True)

    with patch.object(asyncio, "sleep", _instant_async_sleep):
        assert await proxy.get("key") == "from-new-master"


@pytest.mark.asyncio
async def test_a_permanently_down_master_still_raises(redis_utils):
    """Nearby: retries are bounded, an unreachable master is not swallowed."""

    async def _get(*args, **kwargs):
        raise redis_pkg.exceptions.ConnectionError("down")

    sentinel = FakeSentinel([SimpleNamespace(get=_get)])
    proxy = redis_utils.SentinelRedisProxy(sentinel, "mymaster", async_mode=True)

    with (
        patch.object(asyncio, "sleep", _instant_async_sleep),
        pytest.raises(redis_pkg.exceptions.ConnectionError),
    ):
        await proxy.get("key")


# --- 🫥 d484a2a / issue #27432: the temporary: prefix is recognised everywhere ---------------


@pytest.mark.parametrize(
    "chat_id",
    ["temporary:abc", "local:abc", "channel:abc", "", None],
)
def test_non_saved_chat_ids_are_not_saved(chat_id):
    """Pre-fix there was no shared helper and `temporary:` was recognised nowhere."""
    assert _chat_id_module().is_saved_chat_id(chat_id) is False


def test_a_real_chat_id_is_saved():
    """Nearby: an ordinary conversation id still persists."""
    assert _chat_id_module().is_saved_chat_id("8e2b5f0c-2c3f-4d1e-9a77-0a1b2c3d4e5f") is True


@pytest.mark.parametrize(
    ("chat_id", "expected"),
    [
        ("temporary:abc", True),
        ("local:abc", True),
        ("channel:abc", False),
        ("8e2b5f0c", False),
        ("", False),
        (None, False),
    ],
)
def test_temporary_chat_ids_are_identified(chat_id, expected):
    assert _chat_id_module().is_temporary_chat_id(chat_id) is expected


@pytest.mark.parametrize(
    ("chat_id", "expected"),
    [
        ("temporary:s1", "s1"),
        ("local:s1", "s1"),
        ("channel:c1", None),
        ("8e2b5f0c", None),
    ],
)
def test_temporary_session_id_is_extracted(chat_id, expected):
    assert _chat_id_module().get_temporary_chat_session_id(chat_id) == expected


def test_all_three_prefixes_are_declared_non_saved():
    chat_id = _chat_id_module()
    assert set(chat_id.NON_SAVED_CHAT_ID_PREFIXES) == {"temporary:", "local:", "channel:"}


def _emitter_sio():
    return SimpleNamespace(emit=AsyncMock(), manager=SimpleNamespace(rooms={"/": {}}))


async def _emit_status(socket_main, chat_id: str):
    """Drive the real event emitter once and report the chat-store calls it made."""
    chats = AsyncMock()
    with (
        patch.object(socket_main, "Chats", chats),
        patch.object(socket_main, "sio", _emitter_sio()),
    ):
        emitter = await socket_main.get_event_emitter(
            {"user_id": "u-1", "chat_id": chat_id, "message_id": "m-1"}
        )
        await emitter({"type": "status", "data": {"description": "working"}})
    return chats


@pytest.mark.asyncio
async def test_a_temporary_chat_status_writes_nothing(socket_main):
    """Pre-fix only `local:` was excluded, so a temporary chat persisted against a missing row."""
    chats = await _emit_status(socket_main, "temporary:abc")
    assert chats.add_message_status_to_chat_by_id_and_message_id.await_count == 0


@pytest.mark.asyncio
async def test_a_legacy_local_chat_status_writes_nothing(socket_main):
    """Nearby: the prefix that was already handled still is."""
    chats = await _emit_status(socket_main, "local:abc")
    assert chats.add_message_status_to_chat_by_id_and_message_id.await_count == 0


@pytest.mark.asyncio
async def test_a_saved_chat_status_is_still_persisted(socket_main):
    """Nearby: a real conversation must keep recording its status updates."""
    chats = await _emit_status(socket_main, "8e2b5f0c-2c3f-4d1e-9a77-0a1b2c3d4e5f")
    assert chats.add_message_status_to_chat_by_id_and_message_id.await_count == 1


_DISABLED_BUILTIN_CATEGORIES = {
    "time": False,
    "knowledge": False,
    "files": False,
    "chats": False,
    "subagents": False,
    "memory": False,
    "web_search": False,
    "image_generation": False,
    "code_interpreter": False,
    "notes": False,
    "channels": False,
    "automations": False,
    "calendar": False,
    "notifications": False,
    "tasks": True,
}


async def _builtin_tool_names(tools_module, chat_id: str) -> set[str]:
    """Drive the real builtin-tool assembly with only the task-list category enabled."""
    request = SimpleNamespace(
        state=SimpleNamespace(internal=False, direct=False),
        app=SimpleNamespace(state=SimpleNamespace()),
    )
    extra_params = {
        "__user__": {"id": "u-1", "role": "user"},
        "__metadata__": {"chat_id": chat_id},
    }
    model = {"info": {"meta": {"builtinTools": dict(_DISABLED_BUILTIN_CATEGORIES)}}}
    config = SimpleNamespace(get_many=AsyncMock(return_value={}), get=AsyncMock(return_value={}))

    with (
        patch.object(tools_module, "Config", config),
        patch.object(tools_module, "has_permission", AsyncMock(return_value=False)),
        patch.object(
            tools_module,
            "Chats",
            SimpleNamespace(get_chat_by_id=AsyncMock(return_value=None)),
            create=True,
        ),
    ):
        tools = await tools_module.get_builtin_tools(request, extra_params, {}, model)
    return set(tools)


@pytest.mark.asyncio
async def test_task_list_tools_are_withheld_from_a_temporary_chat(tools_module):
    """Pre-fix they were always offered, then failed because there is no chats row to store on."""
    names = await _builtin_tool_names(tools_module, "temporary:abc")
    assert "create_tasks" not in names
    assert "update_task" not in names


@pytest.mark.asyncio
async def test_task_list_tools_are_withheld_from_a_channel(tools_module):
    """Pre-fix a channel message got them too."""
    names = await _builtin_tool_names(tools_module, "channel:abc")
    assert "create_tasks" not in names
    assert "update_task" not in names


@pytest.mark.asyncio
async def test_task_list_tools_are_offered_in_a_saved_chat(tools_module):
    """Nearby: the fix must not withhold them from an ordinary conversation."""
    names = await _builtin_tool_names(tools_module, "8e2b5f0c-2c3f-4d1e-9a77-0a1b2c3d4e5f")
    assert {"create_tasks", "update_task"} <= names
