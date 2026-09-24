"""Socket/runtime regressions fixed between v0.11.0 and v0.11.1 that need a stand-in Redis.

The role-change disconnect and the note resync save are pinned by the integration twin,
integration/chat/test_socket_runtime.py. What stays here needs a shared Redis, a second
instance or control over the event loop's clock:

- 83 (939bcdb79e, #27762): `periodic_session_pool_cleanup` slept the whole
  `SESSION_POOL_TIMEOUT` without renewing, so the cleanup lock lapsed mid-cycle.
- 108+109 (211906d79, PR #28053): the lifespan created its background coroutines without keeping
  a reference, so the event loop could collect them mid-run.
- 139 (5586964bb, PR #28834): `periodic_usage_pool_cleanup` gave up after two failed lock
  acquisitions and raised out of the loop on a failed renew, stopping usage cleanup cluster-wide.
- 152 (bf3a58dbcd, #28909): `redis_task_command_listener` subscribed once, so a cache restart
  silently ended cross-instance stop-generation.
- 190 (a39126c27, PR #28311): `get_event_call` caught only the builtin `TimeoutError` and evicted
  the still-open session from `SESSION_POOL`.
- 202 (6330350a40, #28777): `RedisDict`'s set signature lived in one process, so a worker's stale
  fingerprint skipped the write that would repair a diverged shared hash.

Redis clients are `create_autospec` stand-ins backed by dicts; every loop is bounded by a patched
`asyncio.sleep` that raises `_LoopExit`, a `BaseException` the production `except Exception`
cannot swallow, inside `asyncio.wait_for`.

Discriminates: passes on bbfa876af; fails with the cleanup sleeping a whole timeout between
renews, the usage cleanup returning on a lost race or failed renew, a lifespan task handle
dropped, the listener returning when its stream ends, `get_event_call` catching only the builtin
`TimeoutError` or evicting the session, and the set signature kept in process memory.
"""

from __future__ import annotations

import ast
import asyncio
import time
from unittest.mock import AsyncMock, create_autospec, patch

import pytest
import redis
import redis.asyncio
import redis.asyncio.client
import socketio
from fastapi import FastAPI

pytestmark = pytest.mark.regression

LOOP_DRIVE_TIMEOUT = 5


class _LoopExit(BaseException):
    """Stops a production `while True` loop after a fixed number of calls."""


def _sleep_until(limit: int, sleeps: list[float]):
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


# --- 83: the session cleanup renews its lock while it waits -------------------------------


@pytest.mark.asyncio
async def test_session_cleanup_never_waits_past_the_lock_timeout(socket_main):
    sleeps: list[float] = []
    renewals: list[int] = []
    with (
        patch.object(socket_main, "SESSION_POOL", {}),
        patch.object(socket_main, "session_aquire_func", lambda: True),
        patch.object(socket_main, "session_renew_func", lambda: renewals.append(1) or True),
        patch.object(socket_main, "session_release_func", lambda: True),
        patch.object(asyncio, "sleep", _sleep_until(6, sleeps)),
    ):
        await _drive_until_exit(socket_main.periodic_session_pool_cleanup())

    waits = [delay for delay in sleeps if delay > 0]  # sleep(0) only yields between batches
    assert waits and max(waits) <= socket_main.WEBSOCKET_REDIS_LOCK_TIMEOUT / 2, (
        f"the cleanup slept {max(waits, default=0)}s between renews, so the lock lapsed (#27762)"
    )
    assert len(renewals) >= len(waits)


# --- 139: usage cleanup survives losing its lock -----------------------------------------


def _usage_lock(socket_main, acquire, renew):
    return (
        patch.object(socket_main, "aquire_func", acquire),
        patch.object(socket_main, "renew_func", renew),
        patch.object(socket_main, "release_func", lambda: True),
    )


@pytest.mark.asyncio
async def test_usage_cleanup_keeps_contending_for_the_lock(socket_main):
    attempts: list[int] = []
    acquire, renew, release = _usage_lock(socket_main, lambda: attempts.append(1) or False, None)
    with acquire, renew, release, patch.object(asyncio, "sleep", _sleep_until(6, [])):
        await _drive_until_exit(socket_main.periodic_usage_pool_cleanup())

    assert len(attempts) >= 6, "usage cleanup gave up after losing the lock race twice"


@pytest.mark.asyncio
async def test_usage_cleanup_reacquires_after_a_failed_renew(socket_main):
    attempts: list[int] = []
    renewals = iter([True, False])

    def acquire():
        attempts.append(1)
        if len(attempts) > 1:
            raise _LoopExit
        return True

    acquire_patch, renew_patch, release_patch = _usage_lock(
        socket_main, acquire, lambda: next(renewals, False)
    )
    with (
        acquire_patch,
        renew_patch,
        release_patch,
        patch.object(socket_main, "USAGE_POOL", {}),
        patch.object(asyncio, "sleep", _sleep_until(20, [])),
    ):
        await _drive_until_exit(socket_main.periodic_usage_pool_cleanup())

    assert len(attempts) == 2, "a failed renew ended usage cleanup instead of contending again"


@pytest.mark.asyncio
async def test_usage_cleanup_expires_only_idle_connections(socket_main):
    now = int(time.time())
    pool = {
        "model-idle": {"sid-1": {"updated_at": now - socket_main.TIMEOUT_DURATION - 10}},
        "model-busy": {"sid-2": {"updated_at": now}},
    }
    acquire, renew, release = _usage_lock(socket_main, lambda: True, lambda: True)
    with (
        acquire,
        renew,
        release,
        patch.object(socket_main, "USAGE_POOL", pool),
        patch.object(asyncio, "sleep", _sleep_until(1, [])),
    ):
        await _drive_until_exit(socket_main.periodic_usage_pool_cleanup())

    assert set(pool) == {"model-busy"}


# --- 108 + 109: the lifespan keeps and cancels its background task handles ----------------


def _stored_and_cancelled_tasks(main_tree: ast.Module) -> tuple[dict[str, str], set[str]]:
    """`{coroutine: app.state attribute}` for stored `asyncio.create_task` handles, and the
    handles `.cancel()` is called on."""
    stored: dict[str, str] = {}
    cancelled: set[str] = set()
    for node in ast.walk(main_tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            call = node.value
            if ast.unparse(call.func) == "asyncio.create_task" and call.args:
                targets = [ast.unparse(target) for target in node.targets]
                handles = [target for target in targets if target.startswith("app.state.")]
                if handles and isinstance(call.args[0], ast.Call):
                    stored[ast.unparse(call.args[0].func)] = handles[0]
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == "cancel":
                cancelled.add(ast.unparse(node.func.value))
    return stored, cancelled


@pytest.mark.parametrize(
    "coroutine",
    [
        "periodic_usage_pool_cleanup",
        "periodic_session_pool_cleanup",
        "scheduler_worker_loop",
        "redis_task_command_listener",
    ],
)
def test_the_lifespan_keeps_and_cancels_each_background_task(open_webui_backend, coroutine):
    main_py = open_webui_backend / "open_webui" / "main.py"
    stored, cancelled = _stored_and_cancelled_tasks(ast.parse(main_py.read_text("utf-8")))

    assert coroutine in stored, (
        f"the {coroutine} task handle is not kept on app.state, so the loop may collect it"
    )
    assert stored[coroutine] in cancelled, f"{stored[coroutine]} is never cancelled on shutdown"


# --- 152: the task command listener resubscribes ---------------------------------------


async def _stream_that_ends():
    yield {"type": "subscribe", "data": 1}


def _redis_whose_pubsubs(fail_subscribe: bool, limit: int):
    """A specced async client handing out `limit` pubsubs, then stopping the loop."""
    pubsubs: list = []

    def open_pubsub():
        if len(pubsubs) >= limit:
            raise _LoopExit
        pubsub = create_autospec(redis.asyncio.client.PubSub, instance=True)
        pubsub.listen.side_effect = _stream_that_ends
        if fail_subscribe:
            pubsub.subscribe.side_effect = redis.exceptions.ConnectionError("cache is down")
        pubsubs.append(pubsub)
        return pubsub

    client = create_autospec(redis.asyncio.Redis, instance=True)
    client.pubsub.side_effect = open_pubsub
    return client, pubsubs


def _listener_app(client) -> FastAPI:
    app = FastAPI()
    app.state.redis = client
    return app


@pytest.mark.asyncio
async def test_the_task_command_listener_resubscribes_after_its_stream_ends(tasks_module):
    client, pubsubs = _redis_whose_pubsubs(fail_subscribe=False, limit=3)
    with patch.object(asyncio, "sleep", _sleep_until(20, [])):
        await _drive_until_exit(tasks_module.redis_task_command_listener(_listener_app(client)))

    assert len(pubsubs) == 3, "the listener returned when the stream ended (#28909)"
    for pubsub in pubsubs:
        pubsub.subscribe.assert_awaited_once_with(tasks_module.REDIS_PUBSUB_CHANNEL)


@pytest.mark.asyncio
async def test_reconnects_back_off_while_the_cache_stays_down(tasks_module):
    client, _ = _redis_whose_pubsubs(fail_subscribe=True, limit=3)
    sleeps: list[float] = []
    with patch.object(asyncio, "sleep", _sleep_until(20, sleeps)):
        await _drive_until_exit(tasks_module.redis_task_command_listener(_listener_app(client)))

    first = tasks_module.REDIS_PUBSUB_RECONNECT_INTERVAL
    assert sleeps == [first, first * 2, first * 4]
    assert max(sleeps) <= tasks_module.REDIS_PUBSUB_MAX_RECONNECT_INTERVAL


# --- 190: an unanswered event call ---------------------------------------------------------

TIMEOUT_REPLY = {"error": "Event call timed out. The browser tab may be inactive or closed."}
REQUEST = {"session_id": "sess-1", "chat_id": "c-1", "message_id": "m-1", "user_id": "u-1"}


async def _call_the_browser(socket_main, pool: dict, answer) -> dict:
    with (
        patch.object(socket_main, "SESSION_POOL", pool),
        patch.object(socket_main.sio, "call", AsyncMock(side_effect=answer)),
    ):
        caller = await socket_main.get_event_call(REQUEST)
        return await caller({"type": "input"})


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "timeout", [socketio.exceptions.TimeoutError(), TimeoutError()], ids=["socketio", "builtin"]
)
async def test_a_timed_out_call_is_reported_and_keeps_the_session(socket_main, timeout):
    pool = {"sess-1": {"id": "u-1"}}

    assert await _call_the_browser(socket_main, pool, timeout) == TIMEOUT_REPLY
    assert "sess-1" in pool, "the timeout evicted a session whose tab is still connected"


@pytest.mark.asyncio
@pytest.mark.parametrize("pool", [{}, {"sess-1": {"id": "someone-else"}}], ids=["gone", "foreign"])
async def test_a_foreign_or_missing_session_is_never_called(socket_main, pool):
    reply = await _call_the_browser(socket_main, pool, AssertionError("the browser was called"))

    assert reply == {"error": "Client session disconnected."}


# --- 202: the set signature lives in the shared store ------------------------------------

MODELS = {"a": {"id": "a"}, "b": {"id": "b"}}


def _shared_redis():
    """A specced sync client over one in-memory store, as every worker would share it."""
    hashes: dict[str, dict] = {}
    strings: dict[str, str] = {}
    client = create_autospec(redis.Redis, instance=True)
    client.hset.side_effect = lambda name, key=None, value=None, mapping=None, **kw: (
        hashes.setdefault(name, {}).update({**(mapping or {}), **({key: value} if key else {})})
    )
    client.hget.side_effect = lambda name, key: hashes.get(name, {}).get(key)
    client.hdel.side_effect = lambda name, *keys: sum(
        hashes.get(name, {}).pop(key, None) is not None for key in keys
    )
    client.hkeys.side_effect = lambda name: list(hashes.get(name, {}))
    client.hexists.side_effect = lambda name, key: key in hashes.get(name, {})
    client.get.side_effect = strings.get
    client.set.side_effect = lambda name, value, **kw: strings.update({name: value})
    client.delete.side_effect = lambda *names: sum(
        (hashes.pop(name, None) or strings.pop(name, None)) is not None for name in names
    )
    return client


def _worker_models(socket_utils, client):
    with patch.object(socket_utils, "get_redis_connection", return_value=client):
        return socket_utils.RedisDict(
            name="models", redis_url="redis://127.0.0.1:1/0", cache_set_signature=True
        )


def test_a_repeated_set_repairs_a_hash_another_worker_shortened(socket_utils):
    shared = _shared_redis()
    this_worker = _worker_models(socket_utils, shared)
    other_worker = _worker_models(socket_utils, shared)

    this_worker.set(MODELS)
    other_worker.set({"a": {"id": "a"}})  # a short list from another worker
    this_worker.set(MODELS)

    assert set(this_worker.keys()) == {"a", "b"}, (
        "a fingerprint kept in this worker's memory skipped the repairing write (#28777)"
    )


def test_an_unchanged_set_is_skipped_while_the_shared_signature_holds(socket_utils):
    shared = _shared_redis()
    models = _worker_models(socket_utils, shared)
    models.set(MODELS)
    shared.hset.reset_mock()

    models.set(MODELS)

    shared.hset.assert_not_called()


def test_every_write_drops_the_shared_signature(socket_utils):
    shared = _shared_redis()
    models = _worker_models(socket_utils, shared)

    models.set(MODELS)
    assert shared.get("models:signature")
    models["c"] = {"id": "c"}
    assert shared.get("models:signature") is None
    models.set({**MODELS, "c": {"id": "c"}})
    del models["c"]
    assert shared.get("models:signature") is None
