"""Socket/runtime regressions fixed between v0.11.0 and v0.11.1.

Nine changelog entries share one surface: the websocket layer, its two periodic
cleanup loops, the cross-instance task command listener, and the background
tasks the lifespan starts.

- 23 (ce3c175e26): `SocketSessionEventSink` prepended to `EVENT_SINKS` so a role
  change or account deletion from any path cuts the user's live sockets.
- 83 (939bcdb79e, #27762): `periodic_session_pool_cleanup` slept the whole
  `SESSION_POOL_TIMEOUT` without renewing, so the lock lapsed mid-cycle.
- 108+109 (211906d79, PR28053): the three lifespan background coroutines were
  created without keeping a reference, so the loop could collect them mid-run.
- 139 (5586964bb, PR28834): `periodic_usage_pool_cleanup` gave up permanently
  after two failed lock acquisitions, and raised out of the loop on a failed
  renew, stopping usage cleanup cluster-wide.
- 152 (bf3a58dbcd, #28909): `redis_task_command_listener` subscribed once, so a
  cache restart silently killed cross-instance stop-generation.
- 186 (5735123f5, PR28669): `yjs_document_update` cancelled the pending debounced
  save before knowing whether a replacement would be scheduled.
- 190 (a39126c27, PR28311): `get_event_call` caught only builtin `TimeoutError`
  and evicted the still-open session from `SESSION_POOL`.
- 202 (6330350a40, #28777): `RedisDict._last_signature` was per process, so one
  worker's stale fingerprint suppressed every later identical write.

Every loop test is bounded by construction: a stub raises `_LoopExit` (a
`BaseException`, so the production `except Exception` handlers cannot swallow it)
after a fixed number of calls, and each drive is additionally wrapped in
`asyncio.wait_for`.

Discriminates: passes on v0.11.1, fails on v0.11.0 (pre-fix sinks, loops, and
caches behave as described above).
"""

from __future__ import annotations

import ast
import asyncio
import inspect
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
import socketio

pytestmark = pytest.mark.regression

LOOP_DRIVE_TIMEOUT = 5


class _LoopExit(BaseException):
    """Sentinel that stops a production while-True loop after a fixed number of calls."""


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


class FakeRedis:
    """In-memory stand-in for the sync redis client RedisDict talks to."""

    def __init__(self):
        self.hashes: dict[str, dict[str, str]] = {}
        self.strings: dict[str, str] = {}

    def hset(self, name, key=None, value=None, mapping=None):
        target = self.hashes.setdefault(name, {})
        if mapping:
            target.update(mapping)
        if key is not None:
            target[key] = value
        return 1

    def hget(self, name, key):
        return self.hashes.get(name, {}).get(key)

    def hdel(self, name, *keys):
        target = self.hashes.get(name, {})
        return sum(1 for key in keys if target.pop(key, None) is not None)

    def hexists(self, name, key):
        return key in self.hashes.get(name, {})

    def hlen(self, name):
        return len(self.hashes.get(name, {}))

    def hkeys(self, name):
        return list(self.hashes.get(name, {}).keys())

    def hvals(self, name):
        return list(self.hashes.get(name, {}).values())

    def hgetall(self, name):
        return dict(self.hashes.get(name, {}))

    def delete(self, name):
        self.hashes.pop(name, None)
        self.strings.pop(name, None)

    def get(self, name):
        return self.strings.get(name)

    def set(self, name, value):
        self.strings[name] = value


class FakePubSub:
    def __init__(self, fail_subscribe: bool = False):
        self.subscribed: list[str] = []
        self.closed = False
        self.fail_subscribe = fail_subscribe

    async def subscribe(self, channel):
        if self.fail_subscribe:
            raise ConnectionError("shared cache is down")
        self.subscribed.append(channel)

    async def listen(self):
        # A dropped connection ends the stream; the pre-fix listener returned here.
        yield {"type": "subscribe", "data": 1}

    async def aclose(self):
        self.closed = True

    async def close(self):
        self.closed = True


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
def events_module(owui_module):
    return owui_module("open_webui.events")


@pytest.fixture(scope="session")
def main_lifespan_tree(open_webui_backend):
    """Parsed `open_webui/main.py`; importing it would start the real app."""
    source = (open_webui_backend / "open_webui" / "main.py").read_text(encoding="utf-8")
    return ast.parse(source)


def _dotted(node) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _dotted(node.value)
        return f"{base}.{node.attr}" if base else None
    return None


def _make_event(events, name: str, subject: dict | None):
    return events.Event(
        schema="v1",
        id="evt-1",
        event=name,
        resource="user",
        operation="update",
        created_at=0,
        instance_id=None,
        version="test",
        source="api",
        subject=subject,
    )


async def _dispatch_to_sinks(events, event):
    """Mirror publish_event's sink loop without building a request or app."""
    app = SimpleNamespace(state=SimpleNamespace())
    for sink in events.EVENT_SINKS:
        try:
            await sink.handle_event(app, event, request=None)
        except Exception:
            pass


# --- 23: role changes cutting live sessions -------------------------------------------------


@pytest.mark.asyncio
async def test_role_update_disconnects_live_sessions(events_module, socket_main):
    """A user.role_updated event must reach disconnect_user_sessions through the sink list."""
    disconnect = AsyncMock()
    event = _make_event(
        events_module, events_module.EVENTS.USER_ROLE_UPDATED.name, {"type": "user", "id": "u-1"}
    )

    with (
        patch.object(socket_main, "disconnect_user_sessions", disconnect),
        patch.object(events_module.asyncio, "create_task", lambda coro, *a, **kw: coro.close()),
    ):
        await _dispatch_to_sinks(events_module, event)

    disconnect.assert_awaited_once_with("u-1")


@pytest.mark.asyncio
async def test_user_deleted_disconnects_live_sessions(events_module, socket_main):
    disconnect = AsyncMock()
    event = _make_event(
        events_module, events_module.EVENTS.USER_DELETED.name, {"type": "user", "id": "u-2"}
    )

    with (
        patch.object(socket_main, "disconnect_user_sessions", disconnect),
        patch.object(events_module.asyncio, "create_task", lambda coro, *a, **kw: coro.close()),
    ):
        await _dispatch_to_sinks(events_module, event)

    disconnect.assert_awaited_once_with("u-2")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("event_name", "subject"),
    [
        ("user.settings_updated", {"type": "user", "id": "u-3"}),
        ("user.role_updated", {"type": "group", "id": "g-1"}),
        ("user.role_updated", None),
        ("user.role_updated", {"type": "user"}),
    ],
)
async def test_unrelated_events_do_not_disconnect_sessions(
    events_module, socket_main, event_name, subject
):
    """Nearby: only user-subject role/delete events may cut sockets."""
    disconnect = AsyncMock()
    event = _make_event(events_module, event_name, subject)

    with (
        patch.object(socket_main, "disconnect_user_sessions", disconnect),
        patch.object(events_module.asyncio, "create_task", lambda coro, *a, **kw: coro.close()),
    ):
        await _dispatch_to_sinks(events_module, event)

    disconnect.assert_not_awaited()


def test_every_event_sink_exposes_async_handle_event(events_module):
    """Broad: the dispatch loop awaits handle_event on every registered sink."""
    assert events_module.EVENT_SINKS
    for sink in events_module.EVENT_SINKS:
        handler = getattr(sink, "handle_event", None)
        assert inspect.iscoroutinefunction(handler), type(sink).__name__


# --- 83: session cleanup renewing its lock --------------------------------------------------


@pytest.mark.asyncio
async def test_session_cleanup_never_sleeps_past_the_lock_timeout(socket_main):
    """Pre-fix it slept SESSION_POOL_TIMEOUT in one go, so the lock lapsed mid-cycle."""
    sleeps: list[float] = []
    renews: list[bool] = []

    def renew():
        renews.append(True)
        return True

    with (
        patch.object(socket_main, "SESSION_POOL", {}),
        patch.object(socket_main, "session_aquire_func", lambda: True),
        patch.object(socket_main, "session_renew_func", renew),
        patch.object(socket_main, "session_release_func", lambda: True),
        patch.object(asyncio, "sleep", _counting_sleep(sleeps, 6)),
        pytest.raises(_LoopExit),
    ):
        await _drive(socket_main.periodic_session_pool_cleanup())

    assert len(sleeps) == 6
    assert max(sleeps) <= socket_main.WEBSOCKET_REDIS_LOCK_TIMEOUT / 2
    assert len(renews) >= len(sleeps)


@pytest.mark.asyncio
async def test_session_cleanup_reaps_only_stale_sessions(socket_main):
    """Nearby: the reaping itself is unchanged."""
    now = int(time.time())
    pool = {
        "stale": {"id": "u-1", "last_seen_at": now - socket_main.SESSION_POOL_TIMEOUT - 60},
        "fresh": {"id": "u-2", "last_seen_at": now},
    }
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

    assert set(pool) == {"fresh"}


# --- 139: usage cleanup surviving a lock interruption ---------------------------------------


@pytest.mark.asyncio
async def test_usage_cleanup_retries_lock_acquisition_forever(socket_main):
    """Pre-fix it returned for good after two failed acquisitions."""
    acquires: list[int] = []
    sleeps: list[float] = []

    def acquire():
        acquires.append(len(acquires))
        return False

    with (
        patch.object(socket_main, "USAGE_POOL", {}),
        patch.object(socket_main, "aquire_func", acquire),
        patch.object(socket_main, "renew_func", lambda: True),
        patch.object(socket_main, "release_func", lambda: True),
        patch.object(asyncio, "sleep", _counting_sleep(sleeps, 6)),
        pytest.raises(_LoopExit),
    ):
        await _drive(socket_main.periodic_usage_pool_cleanup())

    assert len(acquires) >= 6


@pytest.mark.asyncio
async def test_usage_cleanup_reacquires_after_a_failed_renew(socket_main):
    """Pre-fix a failed renew raised out of the coroutine and cleanup stopped."""
    acquires: list[int] = []
    renew_results = [True, False]
    sleeps: list[float] = []

    def acquire():
        acquires.append(len(acquires))
        if len(acquires) > 1:
            raise _LoopExit
        return True

    def renew():
        return renew_results.pop(0) if renew_results else False

    with (
        patch.object(socket_main, "USAGE_POOL", {}),
        patch.object(socket_main, "aquire_func", acquire),
        patch.object(socket_main, "renew_func", renew),
        patch.object(socket_main, "release_func", lambda: True),
        patch.object(asyncio, "sleep", _counting_sleep(sleeps, 20)),
        pytest.raises(_LoopExit),
    ):
        await _drive(socket_main.periodic_usage_pool_cleanup())

    assert len(acquires) == 2


@pytest.mark.asyncio
async def test_usage_cleanup_expires_only_stale_connections(socket_main):
    """Nearby: the expiry pass itself is unchanged."""
    now = int(time.time())
    pool = {
        "model-idle": {"sid-1": {"updated_at": now - socket_main.TIMEOUT_DURATION - 10}},
        "model-busy": {"sid-2": {"updated_at": now}},
    }
    sleeps: list[float] = []

    with (
        patch.object(socket_main, "USAGE_POOL", pool),
        patch.object(socket_main, "aquire_func", lambda: True),
        patch.object(socket_main, "renew_func", lambda: True),
        patch.object(socket_main, "release_func", lambda: True),
        patch.object(asyncio, "sleep", _counting_sleep(sleeps, 1)),
        pytest.raises(_LoopExit),
    ):
        await _drive(socket_main.periodic_usage_pool_cleanup())

    assert set(pool) == {"model-busy"}


# --- 108 + 109: lifespan keeping references to its background tasks --------------------------


def _lifespan_task_wiring(tree: ast.Module) -> tuple[dict[str, str], set[str]]:
    stored: dict[str, str] = {}
    cancelled: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            call = node.value
            if (
                _dotted(call.func) == "asyncio.create_task"
                and call.args
                and isinstance(call.args[0], ast.Call)
            ):
                coroutine_name = _dotted(call.args[0].func)
                attribute = next(
                    (d for t in node.targets if (d := _dotted(t)) and d.startswith("app.state.")),
                    None,
                )
                if coroutine_name and attribute:
                    stored[coroutine_name] = attribute
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "cancel"
        ):
            owner = _dotted(node.func.value)
            if owner:
                cancelled.add(owner)

    return stored, cancelled


@pytest.mark.parametrize(
    "coroutine_name",
    ["periodic_usage_pool_cleanup", "periodic_session_pool_cleanup", "scheduler_worker_loop"],
)
def test_lifespan_keeps_and_cancels_background_task_handles(main_lifespan_tree, coroutine_name):
    """Pre-fix the handles were dropped, so the loop could collect the tasks mid-run."""
    stored, cancelled = _lifespan_task_wiring(main_lifespan_tree)

    assert coroutine_name in stored, f"{coroutine_name} task handle is not stored on app.state"
    assert stored[coroutine_name] in cancelled, (
        f"{stored[coroutine_name]} is never cancelled on shutdown"
    )


def test_redis_task_command_listener_handle_is_still_stored(main_lifespan_tree):
    """Nearby: the listener already kept its handle before the fix."""
    stored, cancelled = _lifespan_task_wiring(main_lifespan_tree)

    assert stored.get("redis_task_command_listener") == "app.state.redis_task_command_listener"
    assert "app.state.redis_task_command_listener" in cancelled


# --- 152: task command listener reconnecting ------------------------------------------------


@pytest.mark.asyncio
async def test_task_command_listener_resubscribes_after_the_stream_ends(tasks_module):
    """Pre-fix the coroutine returned when the pubsub generator ended, killing stop-generation."""
    pubsubs: list[FakePubSub] = []

    def make_pubsub():
        if len(pubsubs) >= 3:
            raise _LoopExit
        pubsubs.append(FakePubSub())
        return pubsubs[-1]

    app = SimpleNamespace(state=SimpleNamespace(redis=SimpleNamespace(pubsub=make_pubsub)))
    sleeps: list[float] = []

    with (
        patch.object(asyncio, "sleep", _counting_sleep(sleeps, 20)),
        pytest.raises(_LoopExit),
    ):
        await _drive(tasks_module.redis_task_command_listener(app))

    assert len(pubsubs) == 3
    assert all(pubsub.subscribed == [tasks_module.REDIS_PUBSUB_CHANNEL] for pubsub in pubsubs)


@pytest.mark.asyncio
async def test_reconnect_backoff_doubles_while_the_cache_stays_down(tasks_module):
    """Broad: a cache that stays down is retried with a widening, capped delay."""
    assert tasks_module.REDIS_PUBSUB_RECONNECT_INTERVAL == 1.0
    assert tasks_module.REDIS_PUBSUB_MAX_RECONNECT_INTERVAL == 30.0

    pubsubs: list[FakePubSub] = []

    def make_pubsub():
        if len(pubsubs) >= 3:
            raise _LoopExit
        pubsubs.append(FakePubSub(fail_subscribe=True))
        return pubsubs[-1]

    app = SimpleNamespace(state=SimpleNamespace(redis=SimpleNamespace(pubsub=make_pubsub)))
    sleeps: list[float] = []

    with (
        patch.object(asyncio, "sleep", _counting_sleep(sleeps, 20)),
        pytest.raises(_LoopExit),
    ):
        await _drive(tasks_module.redis_task_command_listener(app))

    assert sleeps == [1.0, 2.0, 4.0]
    assert max(sleeps) <= tasks_module.REDIS_PUBSUB_MAX_RECONNECT_INTERVAL


# --- 186: resync updates keeping the pending note save ---------------------------------------


def _yjs_patches(socket_main, pool, stop_item_tasks, create_task):
    return (
        patch.object(socket_main, "get_session_ids_from_room", lambda room: ["sid-1"]),
        patch.object(socket_main, "SESSION_POOL", pool),
        patch.object(socket_main, "YDOC_MANAGER", AsyncMock()),
        patch.object(socket_main.sio, "emit", AsyncMock()),
        patch.object(socket_main, "stop_item_tasks", stop_item_tasks),
        patch.object(socket_main, "create_task", create_task),
    )


async def _run_yjs_update(socket_main, data):
    pool = {"sid-1": {"id": "u-1", "role": "admin"}}
    stop_item_tasks = AsyncMock()
    create_task = AsyncMock(side_effect=lambda redis, coro, *a, **kw: coro.close())

    patches = _yjs_patches(socket_main, pool, stop_item_tasks, create_task)
    with patches[0], patches[1], patches[2] as ydoc, patches[3], patches[4], patches[5]:
        await socket_main.yjs_document_update("sid-1", data)

    ydoc.append_to_updates.assert_awaited_once()
    return stop_item_tasks, create_task


@pytest.mark.asyncio
async def test_resync_update_without_snapshot_keeps_the_pending_save(socket_main):
    """Pre-fix the cancel ran unconditionally, so a resync dropped the debounced save."""
    stop_item_tasks, create_task = await _run_yjs_update(
        socket_main, {"document_id": "doc-1", "update": [1, 2, 3]}
    )

    stop_item_tasks.assert_not_awaited()
    create_task.assert_not_awaited()


@pytest.mark.asyncio
async def test_update_with_snapshot_replaces_the_pending_save(socket_main):
    """Nearby: an update carrying content still cancels then reschedules."""
    stop_item_tasks, create_task = await _run_yjs_update(
        socket_main, {"document_id": "doc-1", "update": [1, 2, 3], "data": {"content": "hello"}}
    )

    stop_item_tasks.assert_awaited_once()
    create_task.assert_awaited_once()


# --- 190: unanswered tool prompts ------------------------------------------------------------


TIMEOUT_REPLY = {"error": "Event call timed out. The browser tab may be inactive or closed."}


async def _call_event_caller(socket_main, pool, error):
    with (
        patch.object(socket_main, "SESSION_POOL", pool),
        patch.object(socket_main.sio, "call", AsyncMock(side_effect=error)),
    ):
        caller = await socket_main.get_event_call(
            {"session_id": "sess-1", "chat_id": "c-1", "message_id": "m-1", "user_id": "u-1"}
        )
        return await caller({"type": "input"})


@pytest.mark.asyncio
async def test_socketio_timeout_is_reported_as_a_timeout(socket_main):
    """Pre-fix only builtin TimeoutError was caught, so the emit's own error escaped."""
    pool = {"sess-1": {"id": "u-1"}}

    result = await _call_event_caller(socket_main, pool, socketio.exceptions.TimeoutError())

    assert result == TIMEOUT_REPLY


@pytest.mark.asyncio
async def test_timeout_does_not_evict_the_open_session(socket_main):
    """Pre-fix the timeout deleted a session whose tab was still connected."""
    pool = {"sess-1": {"id": "u-1"}}

    result = await _call_event_caller(socket_main, pool, TimeoutError())

    assert result == TIMEOUT_REPLY
    assert "sess-1" in pool


@pytest.mark.asyncio
@pytest.mark.parametrize("pool", [{}, {"sess-1": {"id": "someone-else"}}])
async def test_foreign_or_missing_session_is_refused(socket_main, pool):
    """Nearby: session ownership check is unchanged."""
    called = AsyncMock()

    with (
        patch.object(socket_main, "SESSION_POOL", pool),
        patch.object(socket_main.sio, "call", called),
    ):
        caller = await socket_main.get_event_call(
            {"session_id": "sess-1", "chat_id": "c-1", "message_id": "m-1", "user_id": "u-1"}
        )
        result = await caller({"type": "input"})

    assert result == {"error": "Client session disconnected."}
    called.assert_not_awaited()


@pytest.mark.asyncio
async def test_incomplete_request_info_yields_no_caller(socket_main):
    """Nearby: a caller is only built when the routing keys are present."""
    assert await socket_main.get_event_call({"session_id": "sess-1"}) is None


# --- 202: model list signature living in the shared store ------------------------------------


MODELS = {"a": {"id": "a"}, "b": {"id": "b"}}


def _redis_dict(socket_utils, redis, **kwargs):
    with patch.object(socket_utils, "get_redis_connection", return_value=redis):
        return socket_utils.RedisDict("models", "redis://localhost:6379", **kwargs)


def test_repeated_set_repairs_a_diverged_shared_hash(socket_utils):
    """Pre-fix a per-process fingerprint made the repairing write a no-op for that worker."""
    redis = FakeRedis()
    models = _redis_dict(socket_utils, redis)

    models.set(MODELS)
    redis.hdel("models", "b")  # another worker wrote a short list
    models.set(MODELS)

    assert set(models.keys()) == {"a", "b"}


def test_signature_is_stored_in_redis_and_invalidated_by_writes(socket_utils):
    """The fingerprint moved into the shared store and every mutation drops it."""
    redis = FakeRedis()
    models = _redis_dict(socket_utils, redis, cache_set_signature=True)

    models.set(MODELS)
    signature = redis.get("models:signature")
    assert signature

    models["c"] = {"id": "c"}
    assert redis.get("models:signature") is None

    models.set({**MODELS, "c": {"id": "c"}})
    assert redis.get("models:signature")

    del models["c"]
    assert redis.get("models:signature") is None


def test_identical_set_is_skipped_only_while_the_shared_signature_holds(socket_utils):
    """Broad: the write is skipped for a genuinely unchanged list, never for a diverged one."""
    redis = FakeRedis()
    models = _redis_dict(socket_utils, redis, cache_set_signature=True)

    models.set(MODELS)
    redis.hdel("models", "b")
    models.set(MODELS)
    assert set(models.keys()) == {"a"}  # signature still valid, write correctly skipped

    redis.delete("models:signature")
    models.set(MODELS)
    assert set(models.keys()) == {"a", "b"}


def test_redis_dict_basic_operations(socket_utils):
    """Nearby: the mapping surface is unchanged."""
    redis = FakeRedis()
    models = _redis_dict(socket_utils, redis)

    models["a"] = {"id": "a"}
    assert models["a"] == {"id": "a"}
    assert "a" in models
    assert models.get("missing") is None
    assert len(models) == 1

    del models["a"]
    assert "a" not in models
    with pytest.raises(KeyError):
        del models["a"]


def test_empty_mapping_clears_the_hash(socket_utils):
    """Nearby: setting an empty mapping still wipes the stored models."""
    redis = FakeRedis()
    models = _redis_dict(socket_utils, redis)

    models.set(MODELS)
    models.set({})

    assert models.keys() == []
