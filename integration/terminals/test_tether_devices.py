"""Journey: a person's own device, connected through Tether, serves as their terminal.

Tether connects out to Open WebUI with the person's session and is listed as the terminal
`tether:<connection id>`. Every call to that terminal travels down the device's one connection
and the answer comes back up it: the method, path, query and body arrive as the page sent
them, with the owner's id as `X-User-Id`, whatever the caller put in that header, and the
status, content type and body come back unchanged. Nobody else reaches the device. A body
larger than the relay window crosses whole in both directions, since each side returns credit
as it reads. A shell on the device relays both ways and ends with the code the device closes
it with.

When the device hangs up in the middle of a call, the caller gets 502 with the code
`tether_interrupted`, an unknown outcome and no retry, and the device is off the terminal list.
An answer the device breaks off (an error frame, a frame missing from the sequence) interrupts
that call the same way and leaves the device serving the next one.

The device is the fake of `harness/tether_device.py` (`connect_device`).

Discriminates: on dev c15f28649, in a backend copy: relaying every call to `/` fails all six
(the device gets nothing on the path it serves); listing every account's devices to every
account fails the call test (the stranger sees the device); not returning credit for what was
read of an answer fails the large-body test (the download stops at the window); answering an
interrupted relay like any other proxy error fails the three interruption cases (no
`tether_interrupted` code); ending a device's shell with 1000 whatever the device said fails
the shell test.
"""

from __future__ import annotations

import hashlib
import json
import random
import uuid

import pytest

from harness.backends import wait_until
from harness.listener import json_answer
from harness.terminal_server import close_of, terminal_session
from harness.tether_device import CHUNK, DeviceRequest, DeviceSocket

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source]

INTERRUPTED = {"code": "tether_interrupted", "outcome": "unknown", "retryable": False}


def _listed_terminals(actor) -> dict[str, str]:
    with actor.client() as client:
        listed = client.get("/api/v1/terminals/")
    assert listed.status_code == 200, listed.text
    return {entry["id"]: entry["name"] for entry in listed.json()}


def _without_message(answer) -> dict:
    return {key: value for key, value in answer.json().items() if key != "error"}


def test_a_call_to_a_devices_terminal_reaches_the_device_as_its_owner_and_is_answered(
    make_user, connect_device
):
    owner, stranger = make_user(), make_user()
    tag = uuid.uuid4().hex[:8]
    device = connect_device(owner, name=f"Workstation {tag}")
    assert device.ready and device.ready["type"] == "ready", f"the hello got {device.ready}"

    def written(request: DeviceRequest):
        return json_answer({"wrote": request.json()["content"]}, status=201)

    device.route("POST", "/files/write", written)
    call = f"/api/v1/terminals/{device.terminal_id}/files/write"
    with owner.client() as client:
        answered = client.post(
            call,
            params={"path": f"notes/{tag}.txt", "overwrite": "true"},
            headers={"X-User-Id": stranger.id},
            json={"content": f"hello {tag}"},
        )
    with stranger.client() as client:
        refused = client.post(call, json={"content": "from someone else"})

    assert _listed_terminals(owner).get(device.terminal_id) == f"Workstation {tag}"
    assert (answered.status_code, answered.json()) == (201, {"wrote": f"hello {tag}"}), (
        f"the device's answer came back as {answered.status_code} {answered.text}"
    )
    assert answered.headers["content-type"] == "application/json"
    [received] = device.received
    assert (received.method, received.path, received.service) == (
        "POST",
        f"/files/write?path=notes%2F{tag}.txt&overwrite=true",
        "open_terminal",
    ), f"the device was called with {received.method} {received.path} for {received.service}"
    assert received.json() == {"content": f"hello {tag}"}, received.body
    assert received.headers["x-user-id"] == owner.id, (
        f"the device was told the caller is {received.headers['x-user-id']}"
    )
    assert refused.status_code == 404, (
        f"another account's call to the device answered {refused.status_code} {refused.text}"
    )
    assert device.terminal_id not in _listed_terminals(stranger), "a stranger sees the device"


def test_a_body_larger_than_the_relay_window_crosses_whole_both_ways(make_user, connect_device):
    owner = make_user()
    device = connect_device(owner, window=CHUNK)
    assert device.ready and device.ready["window"] == CHUNK, f"the hello got {device.ready}"
    source = random.Random(7)
    upload, download = source.randbytes(5 * CHUNK + 123), source.randbytes(16 * CHUNK + 321)

    def uploaded(request: DeviceRequest):
        return json_answer({"sha256": hashlib.sha256(request.body).hexdigest()})

    device.route("POST", "/files/upload", uploaded)
    device.route(
        "GET", "/files/view", (200, {"Content-Type": "application/octet-stream"}, download)
    )
    with owner.client() as client:
        client.timeout = 40
        stored = client.post(
            f"/api/v1/terminals/{device.terminal_id}/files/upload",
            content=upload,
            headers={"Content-Type": "application/octet-stream"},
        )
        fetched = client.get(f"/api/v1/terminals/{device.terminal_id}/files/view")

    assert stored.status_code == 200, f"the upload answered {stored.status_code} {stored.text}"
    assert stored.json() == {"sha256": hashlib.sha256(upload).hexdigest()}, (
        f"the device got {len(device.requests_to('/files/upload')[0].body)} of {len(upload)} bytes"
    )
    assert fetched.status_code == 200, fetched.text
    assert fetched.content == download, (
        f"the download arrived as {len(fetched.content)} of {len(download)} bytes"
    )


def test_a_device_that_hangs_up_mid_call_answers_interrupted_and_leaves_the_list(
    make_user, connect_device
):
    owner = make_user()
    device = connect_device(owner)
    device.route("GET", "/files/cwd", json_answer({"cwd": "/home/user"}))
    device.route("POST", "/execute", lambda _request: device.drop())
    with owner.client() as client:
        before = client.get(f"/api/v1/terminals/{device.terminal_id}/files/cwd")
        interrupted = client.post(
            f"/api/v1/terminals/{device.terminal_id}/execute", json={"command": "make"}
        )
        assert wait_until(lambda: device.terminal_id not in _listed_terminals(owner), 10), (
            "the device stayed on its owner's terminal list after it hung up"
        )
        after = client.get(f"/api/v1/terminals/{device.terminal_id}/files/cwd")

    assert (before.status_code, before.json()) == (200, {"cwd": "/home/user"}), before.text
    assert [entry.json() for entry in device.requests_to("/execute")] == [{"command": "make"}]
    assert interrupted.status_code == 502, f"{interrupted.status_code} {interrupted.text}"
    assert _without_message(interrupted) == INTERRUPTED, interrupted.text
    assert "do not automatically repeat it" in interrupted.json()["error"], interrupted.text
    assert after.status_code == 404, f"a gone device answered {after.status_code} {after.text}"


def _error_frame(request: DeviceRequest) -> None:
    request.stream.send("error", message="the disk went away")


def _skipped_frame(request: DeviceRequest) -> None:
    request.stream.send("http-response", status=200, headers={"Content-Type": "text/plain"})
    request.stream.sequence += 1
    request.stream.send("data", b"half an answer")
    request.stream.send("end")


@pytest.mark.parametrize(
    "broken,reason",
    [
        pytest.param(_error_frame, "the disk went away", id="error-frame"),
        pytest.param(_skipped_frame, "Missing relay packet", id="skipped-frame"),
    ],
)
def test_an_answer_the_device_breaks_off_interrupts_that_call_only(
    make_user, connect_device, broken, reason
):
    owner = make_user()
    device = connect_device(owner)
    device.route("GET", "/files/cwd", json_answer({"cwd": "/home/user"}))
    device.route("POST", "/execute", broken)
    with owner.client() as client:
        interrupted = client.post(
            f"/api/v1/terminals/{device.terminal_id}/execute", json={"command": "make"}
        )
        after = client.get(f"/api/v1/terminals/{device.terminal_id}/files/cwd")

    assert len(device.requests_to("/execute")) == 1, "the call never reached the device"
    assert interrupted.status_code == 502, f"{interrupted.status_code} {interrupted.text}"
    assert _without_message(interrupted) == INTERRUPTED, interrupted.text
    assert f"Tether connection interrupted: {reason}." in interrupted.json()["error"], (
        interrupted.text
    )
    assert (after.status_code, after.json()) == (200, {"cwd": "/home/user"}), (
        f"the call after the broken one answered {after.status_code} {after.text}"
    )
    assert device.closed is None, f"the device's connection was closed: {device.closed}"


def test_a_shell_on_a_device_relays_both_ways_and_ends_with_the_devices_code(
    make_user, connect_device
):
    owner = make_user()
    tag = uuid.uuid4().hex[:8]
    device = connect_device(owner)

    def shell(socket: DeviceSocket) -> None:
        socket.receive()
        socket.send(f"prompt {tag} $ ")
        socket.send(b"\x1b[0m\x00\xff")
        typed = socket.receive()
        socket.send(f"ran {typed}")
        socket.close(4000)

    device.script_sockets(shell, "/api/terminals/*")
    with terminal_session(owner.base_url, owner.token, device.terminal_id) as session:
        prompt = session.recv(timeout=10)
        raw = session.recv(timeout=10)
        session.send(f"whoami {tag}")
        answered = session.recv(timeout=10)
        closed = close_of(session, timeout=10)

    [socket] = device.sockets
    signed_in, typed = socket.messages
    assert (prompt, raw, answered) == (
        f"prompt {tag} $ ",
        b"\x1b[0m\x00\xff",
        f"ran whoami {tag}",
    ), f"the device's output arrived as {prompt!r}, {raw!r}, {answered!r}"
    assert typed == f"whoami {tag}", f"the device was sent {typed!r} for what the person typed"
    assert json.loads(signed_in)["type"] == "auth", f"the shell's first message was {signed_in}"
    assert socket.path.startswith("/api/terminals/") and f"user_id={owner.id}" in socket.path, (
        f"the shell was opened on {socket.path}"
    )
    assert socket.headers["x-user-id"] == owner.id, socket.headers
    assert closed is not None and closed[0] == 4000, (
        f"the device closed the shell with 4000, the page's session ended with {closed}"
    )
