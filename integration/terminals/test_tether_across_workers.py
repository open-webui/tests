"""Journey: a Tether device serves its owner whichever worker the owner's request reaches.

With several workers, or several instances behind a load balancer, the device holds its one
connection to one of them while the owner's pages, calls and chats land on any. Redis carries
the rest: the device is on the owner's terminal list on every worker, a call made on another
worker travels to the device and back with bodies larger than the relay window, a shell opened
there relays both ways and ends with the device's close code, a device that hangs up in the
middle of such a call interrupts it with `tether_interrupted`, the owner's pages on another
worker are told when the device comes and goes, and a reply that finished on another worker is
announced to the device. A device that connects again on another worker takes the place of its
earlier connection, which is closed. A device that states it works on two calls at once, with
one on each worker, refuses a third made on the other worker at once with "Tether has too many
active operations." instead of leaving it to time out.

`/api/config` tells the Tether app whether the server can take it (`features.tether`): with one
worker, and with several workers on Redis. A server without WebSocket support has the flag off.
Several workers without Redis is no supported setup and has no test.

Two instances on one database and one real Redis stand for two workers; the device is the fake
of `harness/tether_device.py`.

Discriminates: on dev c15f28649, each mutation below, made in a backend copy, turns the named
tests red. A worker that never forwards a call to the worker holding the device, and a worker
that only knows the devices connected to itself, each fail the five two-worker tests; the
`X-User-Id` a caller sends passed on to the device fails the call test; a
connection replaced on another worker and left open fails the reconnect test; a device that
hangs up without word to the other worker, and no `terminals:changed` at all, each fail the
hang-up test; a notification for a device on another worker dropped fails the notification
test; `features.tether` on whatever the WebSocket support fails the flag test. On dev 485103cde,
a call to a busy device dropped without word by the worker holding it fails the busy test.
"""

from __future__ import annotations

import dataclasses
import hashlib
import random
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Iterator

import httpx
import pytest

from harness import backends
from harness import upstream as reply
from harness.actors import Actor, admin_of, create_user
from harness.backends import wait_until
from harness.chat import ask
from harness.instance import LaunchedInstance
from harness.listener import json_answer
from harness.socket_client import SocketSession, connected
from harness.terminal_server import close_of, terminal_session
from harness.tether_device import (
    CHUNK,
    DeviceRequest,
    DeviceSocket,
)

pytestmark = [
    pytest.mark.journey,
    pytest.mark.api,
    pytest.mark.requires_source,
    pytest.mark.slow,
]

INTERRUPTED = {"code": "tether_interrupted", "outcome": "unknown", "retryable": False}
CWD = {"cwd": "/home/user"}


@pytest.fixture(scope="module")
def shared_redis() -> Iterator[backends.RedisProcess]:
    server = backends.RedisProcess()
    server.start()
    yield server
    server.close()


@pytest.fixture(scope="module")
def workers(shared_redis, instance_with) -> tuple[LaunchedInstance, LaunchedInstance]:
    """Two instances on one database and one Redis: the one devices connect to and another."""
    shared = {
        "REDIS_URL": shared_redis.url,
        "WEBSOCKET_MANAGER": "redis",
        "UVICORN_WORKERS": "2",
    }
    near = instance_with({**shared, "WEBUI_NAME": "worker-a"})
    far = instance_with({**shared, "WEBUI_NAME": "worker-b", "DATABASE_URL": near.database_url})
    # an instance listens on Redis for socket events only once a socket has connected
    for instance in (near, far):
        with connected(admin_of(instance)):
            pass
    return near, far


def _on(account: Actor, instance: LaunchedInstance) -> Actor:
    """The same account on another instance: they share the secret key and the database."""
    return dataclasses.replace(account, base_url=instance.base_url)


def _devices(actor: Actor) -> dict[str, str]:
    with actor.client() as client:
        listed = client.get("/api/v1/terminals/")
    assert listed.status_code == 200, listed.text
    return {
        entry["id"]: entry["name"] for entry in listed.json() if entry["id"].startswith("tether:")
    }


def _changes(tab: SocketSession) -> int:
    return sum(
        1
        for event in list(tab.events)
        if (event.get("data") or {}).get("type") == "terminals:changed"
    )


def test_a_call_made_on_another_worker_reaches_the_device_and_comes_back_whole(
    workers, connect_device
):
    near, far = workers
    owner, stranger = create_user(near), create_user(near)
    tag = uuid.uuid4().hex[:8]
    device = connect_device(owner, on=near, name=f"Workstation {tag}", window=CHUNK)
    source = random.Random(11)
    upload, download = source.randbytes(5 * CHUNK + 123), source.randbytes(16 * CHUNK + 321)

    def uploaded(request: DeviceRequest):
        return json_answer({"sha256": hashlib.sha256(request.body).hexdigest()})

    device.route("POST", "/files/upload", uploaded)
    device.route(
        "GET", "/files/view", (200, {"Content-Type": "application/octet-stream"}, download)
    )
    base = f"/api/v1/terminals/{device.terminal_id}"

    with _on(owner, far).client() as client:
        stored = client.post(
            f"{base}/files/upload",
            params={"directory": f"/home/user/{tag}"},
            content=upload,
            headers={"Content-Type": "application/octet-stream", "X-User-Id": stranger.id},
        )
        fetched = client.get(f"{base}/files/view")
    with _on(stranger, far).client() as client:
        refused = client.get(f"{base}/files/view")

    listed = {"near": _devices(owner), "far": _devices(_on(owner, far))}
    assert listed == {
        "near": {device.terminal_id: f"Workstation {tag}"},
        "far": {device.terminal_id: f"Workstation {tag}"},
    }, f"the device is listed for its owner as {listed}"
    assert stored.status_code == 200, f"the upload answered {stored.status_code} {stored.text}"
    assert stored.json() == {"sha256": hashlib.sha256(upload).hexdigest()}, (
        f"the device got {len(device.requests_to('/files/upload')[0].body)} of {len(upload)} bytes"
    )
    [received] = device.requests_to("/files/upload")
    assert received.path == f"/files/upload?directory=%2Fhome%2Fuser%2F{tag}", received.path
    assert received.headers["x-user-id"] == owner.id, (
        f"the device was told the caller is {received.headers['x-user-id']}"
    )
    assert fetched.status_code == 200 and fetched.content == download, (
        f"the download arrived as {fetched.status_code} with "
        f"{len(fetched.content)} of {len(download)} bytes"
    )
    assert refused.status_code == 404, (
        f"another account's call on the other worker answered {refused.status_code}"
    )
    assert _devices(_on(stranger, far)) == {}, "a stranger sees the device on the other worker"


def test_a_shell_opened_on_another_worker_relays_both_ways_and_ends_with_the_devices_code(
    workers, connect_device
):
    near, far = workers
    owner = create_user(near)
    tag = uuid.uuid4().hex[:8]
    device = connect_device(owner, on=near)

    def shell(socket: DeviceSocket) -> None:
        socket.receive()
        socket.send(f"prompt {tag} $ ")
        socket.send(b"\x1b[0m\x00\xff")
        typed = socket.receive()
        socket.send(f"ran {typed}")
        socket.close(4000)

    device.script_sockets(shell, "/api/terminals/*")
    with terminal_session(far.base_url, owner.token, device.terminal_id) as session:
        prompt = session.recv(timeout=10)
        raw = session.recv(timeout=10)
        session.send(f"whoami {tag}")
        answered = session.recv(timeout=10)
        closed = close_of(session, timeout=10)

    assert (prompt, raw, answered) == (
        f"prompt {tag} $ ",
        b"\x1b[0m\x00\xff",
        f"ran whoami {tag}",
    ), f"the device's output arrived as {prompt!r}, {raw!r}, {answered!r}"
    [socket] = device.sockets
    assert socket.messages[1:] == [f"whoami {tag}"], f"the device was sent {socket.messages[1:]}"
    assert socket.headers["x-user-id"] == owner.id, socket.headers
    assert closed is not None and closed[0] == 4000, (
        f"the device closed the shell with 4000, the page's session ended with {closed}"
    )


def test_a_device_that_hangs_up_interrupts_a_call_made_on_another_worker_and_tells_the_pages_there(
    workers, connect_device
):
    near, far = workers
    owner = create_user(near)
    distant = _on(owner, far)
    with connected(distant) as tab:
        device = connect_device(owner, on=near)
        device.route("GET", "/files/cwd", json_answer(CWD))
        device.route("POST", "/execute", lambda _request: device.drop())
        assert wait_until(lambda: _changes(tab) >= 1, 10), (
            "the owner's page on the other worker was not told that the device connected"
        )
        with distant.client() as client:
            before = client.get(f"/api/v1/terminals/{device.terminal_id}/files/cwd")
            interrupted = client.post(
                f"/api/v1/terminals/{device.terminal_id}/execute", json={"command": "make"}
            )
            assert wait_until(lambda: _devices(distant) == {}, 10), (
                f"the other worker still lists the device that hung up: {_devices(distant)}"
            )
            after = client.get(f"/api/v1/terminals/{device.terminal_id}/files/cwd")
        assert wait_until(lambda: _changes(tab) >= 2, 10), (
            "the owner's page on the other worker was not told that the device went"
        )

    assert (before.status_code, before.json()) == (200, CWD), before.text
    assert [entry.json() for entry in device.requests_to("/execute")] == [{"command": "make"}]
    assert interrupted.status_code == 502, f"{interrupted.status_code} {interrupted.text}"
    answer = interrupted.json()
    assert {key: value for key, value in answer.items() if key != "error"} == INTERRUPTED, answer
    assert "do not automatically repeat it" in answer["error"], answer
    assert after.status_code == 404, f"a gone device answered {after.status_code} {after.text}"


def test_a_device_that_connects_again_on_another_worker_takes_the_place_of_its_earlier_connection(
    workers, connect_device
):
    near, far = workers
    owner = create_user(near)
    connection_id, tag = str(uuid.uuid4()), uuid.uuid4().hex[:8]
    earlier = connect_device(owner, on=near, connection_id=connection_id, name=f"Earlier {tag}")
    earlier.route("GET", "/files/cwd", json_answer({"cwd": "/earlier"}))
    call = f"/api/v1/terminals/tether:{connection_id}/files/cwd"
    with owner.client() as client:
        first = client.get(call)

    later = connect_device(owner, on=far, connection_id=connection_id, name=f"Later {tag}")
    later.route("GET", "/files/cwd", json_answer({"cwd": "/later"}))
    with owner.client() as client, _on(owner, far).client() as distant:
        answers = [client.get(call), distant.get(call)]
    # the worker holding the earlier connection looks every 15 s whether it is still the device's
    closed = earlier.wait_closed(25)

    assert (first.status_code, first.json()) == (200, {"cwd": "/earlier"}), first.text
    assert [(answer.status_code, answer.json()) for answer in answers] == [
        (200, {"cwd": "/later"})
    ] * 2, f"after the device connected again its calls answered {[a.text for a in answers]}"
    assert len(earlier.requests_to("/files/cwd")) == 1, "the replaced connection was still called"
    listed = {"near": _devices(owner), "far": _devices(_on(owner, far))}
    assert listed == {
        "near": {later.terminal_id: f"Later {tag}"},
        "far": {later.terminal_id: f"Later {tag}"},
    }, f"after the device connected again it is listed as {listed}"
    assert closed == (1000, ""), f"the earlier connection was closed with {closed}"
    assert later.closed is None, f"the later connection was closed too: {later.closed}"


def test_a_reply_finished_on_another_worker_is_announced_to_the_device(workers, connect_device):
    near, far = workers
    owner = create_user(near)
    tag = uuid.uuid4().hex[:8]
    device = connect_device(owner, on=near, notifications=True)
    prompt = f"is it done {tag}"
    near.upstream.queue(reply.text(f"it is done {tag}", match=reply.answering(prompt)))

    with _on(owner, far).client() as client:
        allowed = client.post(
            "/api/v1/users/user/settings/update", json={"ui": {"notificationEnabled": True}}
        )
        assert allowed.status_code == 200, f"switching notifications on failed: {allowed.text}"
        turn, _ = ask(client, prompt)

    assert wait_until(lambda: device.notifications, 10), (
        "the device heard nothing of the reply that finished on the other worker"
    )
    [notification] = device.notifications
    assert (notification["user_id"], notification["body"], notification["path"]) == (
        owner.id,
        f"it is done {tag}",
        f"/c/{turn.chat_id}",
    ), f"the device was notified with {notification}"


def _offers_tether(instance: LaunchedInstance) -> bool:
    config = httpx.get(f"{instance.base_url}/api/config", timeout=30.0)
    assert config.status_code == 200, config.text
    features = config.json()["features"]
    assert features["auth_sessions"] is True, f"no app sessions are offered: {features}"
    return features["tether"]


def test_the_server_offers_tether_with_one_worker_or_with_redis_but_not_without_websockets(
    instance, workers, instance_with
):
    near, _ = workers
    without_sockets = instance_with({"ENABLE_WEBSOCKET_SUPPORT": "false"})

    offered = {
        "one worker": _offers_tether(instance),
        "several workers on Redis": _offers_tether(near),
        "no WebSocket support": _offers_tether(without_sockets),
    }
    assert offered == {
        "one worker": True,
        "several workers on Redis": True,
        "no WebSocket support": False,
    }, f"/api/config offers Tether as {offered}"


def _cwd(actor: Actor, device_terminal: str) -> httpx.Response | None:
    """The device's working folder as `actor` asks for it; None when no answer came in 15 s."""
    with actor.client() as client:
        client.timeout = 15
        try:
            return client.get(f"/api/v1/terminals/{device_terminal}/files/cwd")
        except httpx.TimeoutException:
            return None


def test_a_busy_device_refuses_a_call_made_on_another_worker_at_once(workers, connect_device):
    near, far = workers
    owner = create_user(near)
    distant = _on(owner, far)
    device = connect_device(owner, on=near, hello={"max_concurrent_streams": 2})
    assert device.ready and device.ready["max_concurrent_streams"] == 2, (
        f"the device asked to work on 2 calls at once and was granted {device.ready}"
    )
    finish = threading.Event()

    def slow(_request: DeviceRequest):
        finish.wait(60)
        return json_answer(CWD)

    device.route("GET", "/files/cwd", slow)
    with ThreadPoolExecutor(2) as callers:
        try:
            # one call on each worker, so neither worker alone sees the device busy
            running = [
                callers.submit(_cwd, owner, device.terminal_id),
                callers.submit(_cwd, distant, device.terminal_id),
            ]
            assert wait_until(lambda: len(device.requests_to("/files/cwd")) == 2, 20), (
                f"only {len(device.requests_to('/files/cwd'))} of 2 calls reached the device"
            )
            one_more = _cwd(distant, device.terminal_id)
            reached = len(device.requests_to("/files/cwd"))
        finally:
            finish.set()
        finished = [call.result(timeout=30) for call in running]

    assert one_more is not None, "the third call at once on the other worker was never answered"
    assert one_more.status_code >= 400, f"{one_more.status_code} {one_more.text}"
    assert "Tether has too many active operations." in one_more.json()["error"], one_more.text
    assert reached == 2, "the refused call reached the device"
    assert [(call.status_code, call.json()) for call in finished if call] == [(200, CWD)] * 2, (
        f"the two calls the device worked on answered {finished}"
    )
