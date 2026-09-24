"""Websocket-layer regressions fixed in v0.11.2 that need a Redis Cluster or a large session pool.

- a5ea8b0b8 (PR #29165, issue #19840): `redis_task_command_listener` called `redis.pubsub()`
  straight away. A `RedisCluster` client cannot route a subscribe until `initialize()` has filled
  its slot cache, so the listener never subscribed and a stop sent from another instance was
  dropped while the reply ran to the end.
- 89716ea88 (PR #28180): the server used python-socketio's default `Packet`, which walks every
  outgoing payload for `bytes` Open WebUI never sends. `JSONOnlyPacket` turns binary events off
  and hands inbound client attachments back as int lists, the form the Yjs handlers store.
- d7674c517 (PR #28835): `periodic_session_pool_cleanup` read the whole pool with one `keys()`
  and then did a `get` and a `del` per session, holding the event loop for the entire sweep. It
  now walks bounded HSCAN batches, deletes each batch in one call and yields between batches.
  `get_user_ids_from_room` reads this worker's own Socket.IO sessions instead of the pool.

Redis clients are `create_autospec` stand-ins, the session pool is a real `RedisDict` on one,
and every loop is bounded by construction: a patched `asyncio.sleep` or the cluster stand-in
raises `_LoopExit` (a `BaseException` the production `except Exception` cannot swallow) after a
fixed number of calls, inside `asyncio.wait_for`.

Discriminates: passes on bbfa876af; fails with the `initialize()` call removed from the
listener, the server built on the default `Packet`, the reaper deleting session by session or
reading the whole pool at once, and room members looked up in the session pool.
"""

from __future__ import annotations

import asyncio
import json
import time
from unittest.mock import create_autospec, patch

import pytest
import redis
import redis.asyncio.client
import redis.asyncio.cluster
import socketio.packet
from fastapi import FastAPI

pytestmark = pytest.mark.regression

LOOP_DRIVE_TIMEOUT = 5
BATCH_SIZE = 2


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


# --- a5ea8b0b8: stop-generation across instances on a Redis Cluster ----------------------


def _cluster_carrying(messages: list[dict], subscriptions: int):
    """A specced RedisCluster that cannot route a subscribe before `initialize()`."""
    cluster = create_autospec(redis.asyncio.cluster.RedisCluster, instance=True)
    state = {"initialized": False, "pubsubs": []}

    async def initialize():
        state["initialized"] = True

    async def stream():
        for message in messages:
            yield message

    def open_pubsub():
        if len(state["pubsubs"]) >= subscriptions:
            raise _LoopExit
        if not state["initialized"]:
            raise redis.exceptions.RedisClusterException("no slot cache yet")
        pubsub = create_autospec(redis.asyncio.client.PubSub, instance=True)
        pubsub.listen.side_effect = stream
        state["pubsubs"].append(pubsub)
        return pubsub

    cluster.initialize.side_effect = initialize
    cluster.pubsub.side_effect = open_pubsub
    return cluster, state["pubsubs"]


async def _listen_on(tasks_module, cluster) -> None:
    app = FastAPI()
    app.state.redis = cluster
    with patch.object(asyncio, "sleep", _sleep_until(20, [])):
        await _drive_until_exit(tasks_module.redis_task_command_listener(app))


def _command(action: str, task_id: str) -> dict:
    return {"type": "message", "data": json.dumps({"action": action, "task_id": task_id})}


@pytest.mark.asyncio
async def test_a_stop_sent_through_the_cluster_cancels_the_local_task(tasks_module):
    running = asyncio.get_running_loop().create_task(asyncio.Event().wait())
    cluster, pubsubs = _cluster_carrying([_command("stop", "task-1")], subscriptions=1)

    with patch.dict(tasks_module.tasks, {"task-1": running}):
        await _listen_on(tasks_module, cluster)
        await asyncio.wait([running], timeout=0.5)

    assert running.cancelled(), "the listener never subscribed on the cluster client (#19840)"
    pubsubs[0].subscribe.assert_awaited_once_with(tasks_module.REDIS_PUBSUB_CHANNEL)


@pytest.mark.asyncio
async def test_every_reconnect_initializes_the_cluster_again(tasks_module):
    cluster, pubsubs = _cluster_carrying([], subscriptions=3)

    await _listen_on(tasks_module, cluster)

    assert len(pubsubs) == 3
    assert cluster.initialize.await_count == cluster.pubsub.call_count, (
        "a reconnect subscribed without refilling the cluster's slot cache first"
    )


@pytest.mark.asyncio
async def test_only_a_stop_command_cancels_a_task(tasks_module):
    running = asyncio.get_running_loop().create_task(asyncio.Event().wait())
    messages = [
        {"type": "subscribe", "data": 1},
        _command("ping", "task-1"),
        {"type": "message", "data": b"not json"},
    ]
    cluster, _ = _cluster_carrying(messages, subscriptions=1)

    with patch.dict(tasks_module.tasks, {"task-1": running}):
        await _listen_on(tasks_module, cluster)
        await asyncio.wait([running], timeout=0.2)

    assert not running.done()
    running.cancel()


# --- 89716ea88: no binary scan on outgoing payloads --------------------------------------


def test_a_payload_holding_bytes_stays_a_plain_event(socket_main):
    packet = socket_main.sio.packet_class(
        socketio.packet.EVENT, data=["chat-events", {"blob": b"\x00\x01"}]
    )

    assert packet.packet_type == socketio.packet.EVENT, (
        "the server packet walked the payload for bytes and promoted it to a binary event"
    )
    assert packet.attachments == []


def test_client_attachments_come_back_as_int_lists(socket_main):
    reconstructed = socket_main.sio.packet_class.reconstruct_binary(
        {"document_id": "doc-1", "update": {"_placeholder": True, "num": 0}}, [b"\x01\x02\x03"]
    )

    assert reconstructed == {"document_id": "doc-1", "update": [1, 2, 3]}


def test_an_ordinary_event_round_trips(socket_main):
    payload = ["chat-events", {"chat_id": "c-1", "data": {"type": "message", "content": "hi"}}]
    packet_class = socket_main.sio.packet_class

    encoded = packet_class(socketio.packet.EVENT, data=payload, namespace="/").encode()
    decoded = packet_class(encoded_packet=encoded)

    assert decoded.packet_type == socketio.packet.EVENT
    assert decoded.data == payload


# --- d7674c517: a bounded, non-blocking session reaper -----------------------------------


def _pool_over_redis(socket_utils, sessions: dict):
    """A real RedisDict on a specced client whose HSCAN pages `BATCH_SIZE` fields at a time."""
    fields = {sid: json.dumps(entry) for sid, entry in sessions.items()}
    scan_order = list(fields)  # HSCAN visits every field present for the whole scan
    client = create_autospec(redis.Redis, instance=True)

    def hscan(name, cursor=0, match=None, count=None, **kwargs):
        page = [sid for sid in scan_order[cursor : cursor + BATCH_SIZE] if sid in fields]
        next_cursor = cursor + BATCH_SIZE if cursor + BATCH_SIZE < len(scan_order) else 0
        return next_cursor, {sid: fields[sid] for sid in page}

    client.hscan.side_effect = hscan
    client.hdel.side_effect = lambda name, *keys: sum(fields.pop(k, None) is not None for k in keys)
    with patch.object(socket_utils, "get_redis_connection", return_value=client):
        pool = socket_utils.RedisDict(name="session-pool", redis_url="redis://127.0.0.1:1/0")
    return pool, client, fields


def _sessions(socket_main) -> dict:
    stale = int(time.time()) - socket_main.SESSION_POOL_TIMEOUT - 60
    sessions = {f"stale-{index}": {"id": "u", "last_seen_at": stale} for index in range(5)}
    return {**sessions, "live": {"id": "u", "last_seen_at": int(time.time())}}


async def _reap_once(socket_main, pool, sleeps: list[float]) -> None:
    renewals = iter([True])

    def renew():
        if next(renewals, None) is None:
            raise _LoopExit  # the second renew is after the sweep
        return True

    with (
        patch.object(socket_main, "WEBSOCKET_MANAGER", "redis"),
        patch.object(socket_main, "SESSION_POOL", pool),
        patch.object(socket_main, "session_aquire_func", lambda: True),
        patch.object(socket_main, "session_renew_func", renew),
        patch.object(socket_main, "session_release_func", lambda: True),
        patch.object(asyncio, "sleep", _sleep_until(50, sleeps)),
    ):
        await _drive_until_exit(socket_main.periodic_session_pool_cleanup())


@pytest.mark.asyncio
async def test_the_reaper_deletes_batch_by_batch_and_yields_between(socket_main, socket_utils):
    pool, client, remaining = _pool_over_redis(socket_utils, _sessions(socket_main))
    sleeps: list[float] = []

    await _reap_once(socket_main, pool, sleeps)

    assert set(remaining) == {"live"}
    batches = client.hscan.call_count
    assert client.hdel.call_count <= batches, "stale sessions were deleted one call each"
    assert sleeps[:batches] == [0] * batches, "the sweep held the event loop between batches"
    client.hkeys.assert_not_called()
    client.hgetall.assert_not_called()


async def _reap_once_local(socket_main, pool: dict) -> None:
    with (
        patch.object(socket_main, "SESSION_POOL", pool),
        patch.object(socket_main, "session_aquire_func", lambda: True),
        patch.object(socket_main, "session_renew_func", lambda: True),
        patch.object(socket_main, "session_release_func", lambda: True),
        patch.object(asyncio, "sleep", _sleep_until(2, [])),
    ):
        await _drive_until_exit(socket_main.periodic_session_pool_cleanup())


@pytest.mark.asyncio
async def test_the_reaper_keeps_sessions_that_are_still_heartbeating(socket_main):
    now = int(time.time())
    timeout = socket_main.SESSION_POOL_TIMEOUT
    pool = {
        "stale": {"id": "u-1", "last_seen_at": now - timeout - 60},
        "borderline": {"id": "u-2", "last_seen_at": now - timeout},
        "fresh": {"id": "u-3", "last_seen_at": now},
    }

    with patch.object(socket_main, "WEBSOCKET_MANAGER", "local"):
        await _reap_once_local(socket_main, pool)

    assert set(pool) == {"borderline", "fresh"}


@pytest.mark.asyncio
async def test_room_members_come_from_this_workers_sessions(socket_main, socket_utils):
    pool, client, _ = _pool_over_redis(socket_utils, {})
    sessions = {"s-1": {"user": {"id": "u-1"}}, "s-2": {"user": {"id": "u-2"}}}

    async def get_session(sid, namespace=None):
        return sessions[sid]

    with (
        patch.object(socket_main, "SESSION_POOL", pool),
        patch.object(socket_main, "get_session_ids_from_room", lambda room: list(sessions)),
        patch.object(socket_main.sio, "get_session", get_session),
    ):
        user_ids = await socket_main.get_user_ids_from_room("channel:c-1")

    assert set(user_ids) == {"u-1", "u-2"}
    assert client.method_calls == [], "a room fan-out made a session pool round trip per member"
