"""Websocket-layer regressions fixed in v0.11.2.

Three changes to `socket/main.py` and `tasks.py` that only show up once an
instance is behind a Redis Cluster or is carrying a large session pool.

- a5ea8b0b8 (PR #29165, issue #19840): `redis_task_command_listener` called
  `redis.pubsub()` straight away. A `RedisCluster` client cannot route a
  subscribe until `initialize()` has filled its slot cache, so the listener
  raised, retried, and never subscribed. Stop-generation from another instance
  was therefore silently dropped and the reply ran to the end.
- 89716ea88 (PR #28180): the server used python-socketio's default `Packet`,
  whose `__init__` walks every outgoing payload looking for `bytes` that Open
  WebUI never emits. `JSONOnlyPacket` sets `uses_binary_events = False`, which
  skips the walk, and normalizes inbound client attachments to int lists.
- d7674c517 (PR #28835): `periodic_session_pool_cleanup` materialized the whole
  pool with `keys()` and then did one `get` plus one `del` per session, holding
  the event loop for the entire sweep. It now walks bounded HSCAN batches,
  deletes each batch in one call, and yields between batches. The same commit
  made `get_user_ids_from_room` read this worker's local Socket.IO sessions
  instead of one pool round trip per member.

Both loop tests are bounded by construction: a patched `asyncio.sleep` raises
`_LoopExit` (a `BaseException`, so the production `except Exception` handlers
cannot swallow it) after a fixed number of awaits, the cluster fake raises the
same sentinel after a fixed number of `pubsub()` attempts, and every drive is
additionally wrapped in `asyncio.wait_for`.

Discriminates: passes on v0.11.3, fails on v0.11.1 (pre-fix the listener never
subscribes on a cluster client, the default packet class scans every payload,
and the reaper plus the room fan-out do one blocking round trip per session).
"""

from __future__ import annotations

import asyncio
import inspect
import time
from contextlib import suppress
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
import socketio
import socketio.packet

pytestmark = pytest.mark.regression

LOOP_DRIVE_TIMEOUT = 5
TASK_SETTLE_TIMEOUT = 0.5


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


async def _pending():
    await asyncio.Event().wait()


async def _discard(task: asyncio.Task):
    task.cancel()
    with suppress(asyncio.CancelledError):
        await task


@pytest.fixture(scope="session")
def socket_main(owui_module):
    return owui_module("open_webui.socket.main")


@pytest.fixture(scope="session")
def tasks_module(owui_module):
    return owui_module("open_webui.tasks")


# --- a5ea8b0b8: stop-generation across instances on a Redis Cluster --------------------------


class FakeClusterPubSub:
    def __init__(self, messages):
        self.messages = messages
        self.subscribed: list[str] = []
        self.closed = False

    async def subscribe(self, channel):
        self.subscribed.append(channel)

    async def listen(self):
        for message in self.messages:
            yield message

    async def aclose(self):
        self.closed = True


class FakeClusterRedis:
    """RedisCluster shape: pubsub cannot be routed until initialize() has filled the slot cache."""

    def __init__(self, messages, max_attempts):
        self.messages = messages
        self.max_attempts = max_attempts
        self.attempts = 0
        self.initialized = False
        self.pubsubs: list[FakeClusterPubSub] = []

    async def initialize(self):
        self.initialized = True

    def pubsub(self):
        self.attempts += 1
        if self.attempts > self.max_attempts:
            raise _LoopExit
        if not self.initialized:
            raise RuntimeError("Redis Cluster has no slot cache yet")
        self.pubsubs.append(FakeClusterPubSub(self.messages))
        return self.pubsubs[-1]


def _stop_message(tasks_module, task_id: str) -> dict:
    return {
        "type": "message",
        "data": tasks_module.JSONCodec.dumps({"action": "stop", "task_id": task_id}),
    }


async def _run_cluster_listener(tasks_module, messages, max_attempts, task_registry):
    redis = FakeClusterRedis(messages, max_attempts)
    app = SimpleNamespace(state=SimpleNamespace(redis=redis))
    sleeps: list[float] = []

    with (
        patch.object(tasks_module, "tasks", task_registry),
        patch.object(asyncio, "sleep", _counting_sleep(sleeps, 20)),
        pytest.raises(_LoopExit),
    ):
        await _drive(tasks_module.redis_task_command_listener(app))

    return redis


@pytest.mark.asyncio
async def test_cluster_stop_command_cancels_the_local_task(tasks_module):
    """Pre-fix pubsub() raised on a cluster client, so the stop never reached this instance."""
    task = asyncio.get_running_loop().create_task(_pending())
    await asyncio.sleep(0)

    redis = await _run_cluster_listener(
        tasks_module, [_stop_message(tasks_module, "task-1")], 1, {"task-1": task}
    )

    await asyncio.wait([task], timeout=TASK_SETTLE_TIMEOUT)
    stop_reached = task.cancelled()
    if not task.done():
        await _discard(task)

    assert stop_reached
    assert [pubsub.subscribed for pubsub in redis.pubsubs] == [
        [tasks_module.REDIS_PUBSUB_CHANNEL]
    ]
    assert redis.initialized


@pytest.mark.asyncio
async def test_cluster_listener_reinitializes_on_every_reconnect(tasks_module):
    """Broad: a failover ends the stream, and the fresh subscribe needs the slot cache again."""
    redis = await _run_cluster_listener(tasks_module, [], 3, {})

    assert [pubsub.subscribed for pubsub in redis.pubsubs] == [
        [tasks_module.REDIS_PUBSUB_CHANNEL]
    ] * 3
    assert all(pubsub.closed for pubsub in redis.pubsubs)


@pytest.mark.asyncio
async def test_cluster_listener_ignores_non_stop_commands(tasks_module):
    """Nearby: only a stop action may cancel a running task."""
    task = asyncio.get_running_loop().create_task(_pending())
    await asyncio.sleep(0)

    messages = [
        {"type": "subscribe", "data": 1},
        {"type": "message", "data": tasks_module.JSONCodec.dumps({"action": "ping"})},
        {"type": "message", "data": b"not json"},
    ]
    await _run_cluster_listener(tasks_module, messages, 1, {"task-1": task})

    await asyncio.wait([task], timeout=TASK_SETTLE_TIMEOUT)
    assert not task.cancelled()
    await _discard(task)


@pytest.mark.asyncio
async def test_single_instance_stop_cancels_the_local_task(tasks_module):
    """Nearby: stopping without Redis configured still cancels in-process."""
    task = asyncio.get_running_loop().create_task(_pending())
    await asyncio.sleep(0)

    with patch.object(tasks_module, "tasks", {"task-1": task}):
        result = await tasks_module.stop_task(None, "task-1")

    assert result["status"] is True
    assert task.cancelled()


@pytest.mark.asyncio
async def test_single_instance_stop_reports_an_unknown_task(tasks_module):
    """Nearby: an id that no worker owns is reported, not cancelled blindly."""
    with patch.object(tasks_module, "tasks", {}):
        result = await tasks_module.stop_task(None, "missing")

    assert result["status"] is False


# --- 89716ea88: no binary scan on outgoing socket.io payloads --------------------------------


EVENT_PAYLOAD = [
    "chat-events",
    {"chat_id": "c-1", "message_id": "m-1", "data": {"type": "message", "content": "hello"}},
]


def test_server_packet_class_declares_no_binary_events(socket_main):
    """Pre-fix the default Packet was used, and it advertises binary support."""
    assert socket_main.sio.packet_class.uses_binary_events is False


@pytest.mark.asyncio
async def test_normal_emit_payload_is_never_scanned_for_binary(socket_main):
    """Pre-fix every emit walked the whole payload looking for bytes that are never sent."""
    scanned: list[object] = []

    def recording_scan(cls, data):
        scanned.append(data)
        return False

    manager = socket_main.sio.manager
    sid = await manager.connect("eio-scan-1", "/")
    try:
        with (
            patch.object(socketio.packet.Packet, "data_is_binary", classmethod(recording_scan)),
            patch.object(socket_main.sio, "_send_eio_packet", AsyncMock()) as send_eio_packet,
        ):
            await socket_main.sio.emit("chat-events", EVENT_PAYLOAD[1], to=sid)
    finally:
        await manager.disconnect(sid, "/")

    assert scanned == []
    frames = [call.args[1].data for call in send_eio_packet.await_args_list]
    assert len(frames) == 1
    assert isinstance(frames[0], str)


def test_payload_holding_bytes_stays_a_plain_event(socket_main):
    """Broad: the packet class never promotes an event to a binary event."""
    packet = socket_main.sio.packet_class(
        socketio.packet.EVENT, data=["chat-events", {"blob": b"\x00\x01"}]
    )

    assert packet.packet_type == socketio.packet.EVENT
    assert packet.attachments == []


def test_client_attachments_reconstruct_as_int_lists(socket_main):
    """Yjs handlers store updates as int lists; pre-fix reconstruction handed back raw bytes."""
    reconstructed = socket_main.sio.packet_class.reconstruct_binary(
        {"document_id": "doc-1", "update": {"_placeholder": True, "num": 0}}, [b"\x01\x02\x03"]
    )

    assert reconstructed == {"document_id": "doc-1", "update": [1, 2, 3]}


def test_ordinary_event_round_trips_through_encode_and_decode(socket_main):
    """Nearby: a JSON payload still encodes to one frame and decodes back unchanged."""
    packet_class = socket_main.sio.packet_class
    encoded = packet_class(socketio.packet.EVENT, data=EVENT_PAYLOAD, namespace="/").encode()

    assert isinstance(encoded, str)
    decoded = packet_class(encoded_packet=encoded)

    assert decoded.packet_type == socketio.packet.EVENT
    assert decoded.data == EVENT_PAYLOAD


def test_ack_and_connect_packets_are_unchanged(socket_main):
    """Nearby: the non-event packet types keep their handling."""
    packet_class = socket_main.sio.packet_class

    ack = packet_class(socketio.packet.ACK, data=["ok"], id=7)
    assert ack.packet_type == socketio.packet.ACK
    assert ack.id == 7

    connect = packet_class(socketio.packet.CONNECT, data={"sid": "s-1"}, namespace="/")
    assert connect.packet_type == socketio.packet.CONNECT


# --- d7674c517: bounded non-blocking session pool reaper -------------------------------------


BATCH_SIZE = 8
STALE_BATCHES = 2


class FakeSessionPool:
    """Redis-backed SESSION_POOL stand-in that records every blocking round trip."""

    def __init__(self, entries: dict, batch_size: int = BATCH_SIZE):
        self.entries = dict(entries)
        self.batch_size = batch_size
        self.round_trips: list[str] = []

    def scan_batches(self):
        items = list(self.entries.items())
        for start in range(0, len(items), self.batch_size):
            self.round_trips.append("hscan")
            yield items[start : start + self.batch_size]

    def delete_many(self, *keys):
        self.round_trips.append(f"hdel:{len(keys)}")
        for key in keys:
            self.entries.pop(key, None)

    def keys(self):
        self.round_trips.append("hkeys")
        return list(self.entries)

    def items(self):
        self.round_trips.append("hgetall")
        return list(self.entries.items())

    def get(self, key, default=None):
        self.round_trips.append("hget")
        return self.entries.get(key, default)

    def pop(self, key, default=None):
        self.round_trips.append("hdel:1")
        return self.entries.pop(key, default)

    def __delitem__(self, key):
        self.round_trips.append("hdel:1")
        del self.entries[key]


def _session_pool_entries(socket_main, now: int):
    entries = {}
    for index in range(BATCH_SIZE * STALE_BATCHES):
        entries[f"stale-{index}"] = {
            "id": f"u-{index}",
            "last_seen_at": now - socket_main.SESSION_POOL_TIMEOUT - 60,
        }
    for index in range(BATCH_SIZE):
        entries[f"fresh-{index}"] = {"id": f"u-live-{index}", "last_seen_at": now}
    return entries


async def _drive_reaper(socket_main, pool, sleep_limit, manager="redis"):
    sleeps: list[float] = []

    with (
        patch.object(socket_main, "WEBSOCKET_MANAGER", manager),
        patch.object(socket_main, "SESSION_POOL", pool),
        patch.object(socket_main, "session_aquire_func", lambda: True),
        patch.object(socket_main, "session_renew_func", lambda: True),
        patch.object(socket_main, "session_release_func", lambda: True),
        patch.object(asyncio, "sleep", _counting_sleep(sleeps, sleep_limit)),
        pytest.raises(_LoopExit),
    ):
        await _drive(socket_main.periodic_session_pool_cleanup())

    return sleeps


@pytest.mark.asyncio
async def test_reaper_yields_to_the_event_loop_between_batches(socket_main):
    """Pre-fix the whole pool was swept without a single yield, blocking every other socket."""
    pool = FakeSessionPool(_session_pool_entries(socket_main, int(time.time())))
    batches = STALE_BATCHES + 1

    sleeps = await _drive_reaper(socket_main, pool, batches + 1)

    assert sleeps[:batches] == [0] * batches


@pytest.mark.asyncio
async def test_reaper_round_trips_scale_with_batches_not_sessions(socket_main):
    """Pre-fix each session cost a get and a del; the sweep is now one scan plus one delete."""
    pool = FakeSessionPool(_session_pool_entries(socket_main, int(time.time())))
    batches = STALE_BATCHES + 1

    await _drive_reaper(socket_main, pool, batches + 1)

    assert len(pool.round_trips) <= 2 * batches
    assert set(pool.entries) == {f"fresh-{index}" for index in range(BATCH_SIZE)}


@pytest.mark.asyncio
async def test_reaper_deletes_each_expired_batch_in_one_call(socket_main):
    """Broad: no per-session delete survives, whatever the pool size."""
    pool = FakeSessionPool(_session_pool_entries(socket_main, int(time.time())))

    await _drive_reaper(socket_main, pool, STALE_BATCHES + 2)

    assert [trip for trip in pool.round_trips if trip.startswith("hdel")] == [
        f"hdel:{BATCH_SIZE}"
    ] * STALE_BATCHES


@pytest.mark.asyncio
async def test_room_user_ids_come_from_local_sockets_not_the_session_pool(socket_main):
    """Pre-fix every room fan-out cost one blocking pool round trip per member session."""
    pool = FakeSessionPool({"s-1": {"id": "u-1"}, "s-2": {"id": "u-2"}})
    sessions = {"s-1": {"user": {"id": "u-1"}}, "s-2": {"user": {"id": "u-2"}}}

    async def fake_get_session(sid, namespace=None):
        return sessions[sid]

    with (
        patch.object(socket_main, "SESSION_POOL", pool),
        patch.object(socket_main, "get_session_ids_from_room", lambda room: ["s-1", "s-2"]),
        patch.object(socket_main.sio, "get_session", fake_get_session),
    ):
        user_ids = socket_main.get_user_ids_from_room("room-1")
        if inspect.isawaitable(user_ids):
            user_ids = await user_ids

    assert set(user_ids) == {"u-1", "u-2"}
    assert pool.round_trips == []


@pytest.mark.asyncio
async def test_empty_session_pool_reaps_cleanly(socket_main):
    """Nearby: an idle instance sweeps an empty pool without touching it."""
    pool: dict = {}

    await _drive_reaper(socket_main, pool, 2, manager="memory")

    assert pool == {}


@pytest.mark.asyncio
async def test_reaper_keeps_sessions_that_are_still_heartbeating(socket_main):
    """Nearby: only entries past SESSION_POOL_TIMEOUT are reaped."""
    now = int(time.time())
    pool = {
        "stale": {"id": "u-1", "last_seen_at": now - socket_main.SESSION_POOL_TIMEOUT - 60},
        "fresh": {"id": "u-2", "last_seen_at": now},
        "borderline": {"id": "u-3", "last_seen_at": now - socket_main.SESSION_POOL_TIMEOUT},
    }

    await _drive_reaper(socket_main, pool, 2, manager="memory")

    assert set(pool) == {"fresh", "borderline"}
