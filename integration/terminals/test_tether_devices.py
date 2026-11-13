"""Journey: a person's own device, connected through Tether, serves as their terminal.

Tether connects out to Open WebUI with the person's session and is listed as the terminal
`tether:<connection id>`, with no address and with chat uploads going to its files. Every call
to that terminal travels down the device's one connection and the answer comes back up it: the
method, path, query and body arrive as the page sent them, with the owner's id as `X-User-Id`,
whatever the caller put in that header, and the status, content type and body come back
unchanged. Nobody else reaches the device. A body larger than the relay window crosses whole in
both directions, since each side returns credit as it reads. A shell on the device relays both
ways, is signed in to without the person's token and ends with the code the device closes it
with.

Connecting. The hello names a version (1 or 2), a connection id, a name of 1 to 100 characters,
a window of at least 64 KiB and the device's services. A hello that breaks one of these rules
is closed with 4001 and lists nothing; the window is capped at 4 MiB, the name is trimmed, and
version 1 offers the terminal without saying so. A device that connects again under its
connection id takes the place of its earlier connection, which is closed with 1000; the same id
of another account is a device of its own. A connected device renames itself and changes its
services, which takes its terminal off the list and back, and hands in the renewed token of its
session. A token of another session, of another account or of a plain sign-in ends the
connection, as does a message the server cannot read, while a frame of no call is ignored. The
owner's open pages, and nobody else's, are told each time a device comes, changes or goes
(`terminals:changed`).

In a chat. The model gets the device's tools, its system prompt, the AGENTS.md of its home and
its skills, and its call runs on the device in the owner's name with the chat's id. A picture
the model draws is copied into the device's working folder. A device lost while the model was
thinking tells the model why (offline, terminal withdrawn, session revoked) and one that hangs
up during the call tells it not to repeat the call. A question before the turn that the device
never answers is given up on and the device is told to stop.

When the device hangs up in the middle of a call, the caller gets 502 with the code
`tether_interrupted`, an unknown outcome and no retry, and the device is off the terminal list.
An answer the device breaks off (an error frame, a frame missing from the sequence, data past
the window) interrupts that call the same way and leaves the device serving the next one. A
device works on 16 calls at once and the next is refused. An answer the page gets in one piece
may be 16 MiB; a download may be larger. An open shell ends with 1011 when its device hangs up,
and no shell opens while the device is offline (1013, "Tether is offline. Reconnecting.").

Own connections. A person adds their own OpenAI-compatible provider, OpenAPI tool server or
terminal in the Tether app; the hello lists each with its kind, name and the service its calls
travel under, and only a connection whose service the device offers counts. The provider's
models join the owner's model list, tagged "<device> · <connection>", and a chat with one is
sent down the device under that service with no credential of the owner's; nobody else sees or
reaches the model, and it leaves the list when the device withdraws the connection. The tool
server is listed among the owner's tools as `server:tether:<connection id>:<service>` and the
model's call to it reaches the device. The terminal is listed as
`tether:<connection id>:<service>` under "<device> · <connection>".

Reconnects. A read (a model list) cut off because the device dropped before answering is
repeated once the device has connected again, and a command (a POST) is never repeated. The
page's own reads through the terminal proxy are not repeated on dev 485103cde: a body is always
streamed to the device, so the read counts as one that cannot be sent twice. That test stays
red (issue draft "a terminal read cut off by a Tether reconnect fails").

Notifications. A reply that finished and a channel's new message are announced to the person's
devices that asked for notifications, once the person switched Browser Notifications on, and
never to the author of the message.

The device is the fake of `harness/tether_device.py` (`connect_device`). Several workers are in
test_tether_across_workers.py, the browser in e2e/terminals/test_tether_devices.py.

Discriminates: on dev c15f28649, each mutation below, made in a backend copy, turns the named
tests red.
The first tests: relaying every call to `/` fails every test that calls a device; listing every
account's devices to every account fails the call test (the stranger sees the device); the
`X-User-Id` a caller sends passed on to the device, the relay address in the list, and no word
on chat uploads, each fail the call test; not returning credit for what was read of an answer
fails the large-body test; answering an interrupted relay like any other proxy error fails the
interruption cases; a window that is not enforced fails the past-the-window case; ending a
device's shell with 1000 whatever the device said, or handing the device the person's token,
fails the shell test.
The hello: any first message taken as a hello, the window, name, connection id or services
taken unchecked each fail their cases of the refused-hello test; a window not capped, a name not
trimmed and no terminal for version 1 each fail the within-the-rules test.
Reconnect and updates: the earlier connection left open, or the same id of any account closed,
fails the reconnect test; a services update that ignores the name, ignores the services or
takes any name fails the rename test; no `terminals:changed` on connect, on a change or on
disconnect, and one sent to every page, each fail the pages test; an auth update that takes any
token, or none, fails the renewal cases; a frame of no call that closes the device, and an
unreadable frame that is skipped, each fail the frame test.
The chat: a device's tools looked up among the admin's terminals fails the chat test and the
five after it; a call relayed without the session check or the service check fails its lost
case, a failed call that tells the model only that it failed fails all three; an interrupted
call reported like any failed one fails the hang-up test; no cancel sent, and no time limit on
a call, each fail the unanswered-question test; a file Open WebUI uploads sent without its
form's content type fails the picture test.
Limits and shells: no limit of 16 fails the sixteen-calls test; no limit on a buffered answer,
and one of 8 MiB, each fail the 16 MiB test; an offline device closed like an unknown terminal
(4004), and an interrupted shell closed without its reason, each fail the offline-shell test.
Notifications: every device notified, and the person's setting ignored, each fail the
finished-reply test; the author notified too, and nobody notified, each fail the channel test.
Own connections and reconnects, on dev 485103cde: the user's Tether connections never looked up
fails the model, tool server and terminal tests and the reconnect test; a read never repeated
fails the reconnect test; the page's file read passes once the terminal proxy sends a body-less
read without a streamed body.
"""

from __future__ import annotations

import base64
import hashlib
import json
import random
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from email.parser import BytesParser
from email.policy import HTTP

import httpx
import pytest

from harness import upstream as reply
from harness.backends import wait_until
from harness.channel_chat import serve_openai_images
from harness.channel_quotes import enable_channels, group_channel, post_message
from harness.chat import ask, send_message, wait_for_reply
from harness.image_engines import IMAGES_CONFIG, PNG_BASE64, save_image_settings
from harness.listener import json_answer
from harness.mfa import new_address
from harness.openapi_server import openapi_spec
from harness.socket_client import SocketSession, connected
from harness.terminal_server import close_of, terminal_session
from harness.tether_device import (
    CHUNK,
    DEFAULT_WINDOW,
    DeviceRequest,
    DeviceSocket,
    TetherDevice,
    own_connection,
    tether_device,
    tether_session,
)
from harness.upstream import conversation

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source]

INTERRUPTED = {"code": "tether_interrupted", "outcome": "unknown", "retryable": False}
SESSIONS = "/api/v1/auths/sessions"
# the largest window the server grants (2 MiB before c67d774d7) and the largest answer it buffers
LARGEST_WINDOW = 4 * 1024 * 1024
LARGEST_BUFFERED = 16 * 1024 * 1024
HOME = "/home/user"
CWD = {"cwd": f"{HOME}/work", "home": HOME}


def _terminal_entries(actor) -> list[dict]:
    with actor.client() as client:
        listed = client.get("/api/v1/terminals/")
    assert listed.status_code == 200, listed.text
    return listed.json()


def _listed_terminals(actor) -> dict[str, str]:
    return {entry["id"]: entry["name"] for entry in _terminal_entries(actor)}


def _devices(actor) -> dict[str, str]:
    """The devices in the account's terminal list, by terminal id, with their names."""
    return {
        terminal_id: name
        for terminal_id, name in _listed_terminals(actor).items()
        if terminal_id.startswith("tether:")
    }


def _cwd_of(actor, device: TetherDevice) -> httpx.Response:
    with actor.client() as client:
        return client.get(f"/api/v1/terminals/{device.terminal_id}/files/cwd")


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

    [entry] = [entry for entry in _terminal_entries(owner) if entry["id"] == device.terminal_id]
    assert entry == {
        "id": f"tether:{device.connection_id}",
        "url": "",
        "name": f"Workstation {tag}",
        "template_id": None,
        "scope": None,
        "config": {"chat_uploads": "filesystem"},
    }, f"the owner's terminal list shows the device as {entry}"
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


def _past_the_window(request: DeviceRequest) -> None:
    request.stream.send("http-response", status=200, headers={"Content-Type": "text/plain"})
    # one frame over the whole window; credit returned between frames would hide a series of them
    request.stream.send("data", b"x" * (CHUNK + 1024), wait_for_credit=False)
    request.stream.send("end")


@pytest.mark.parametrize(
    "broken,reason",
    [
        pytest.param(_error_frame, "the disk went away", id="error-frame"),
        pytest.param(_skipped_frame, "Missing relay packet", id="skipped-frame"),
        pytest.param(_past_the_window, "Receive window exceeded", id="past-the-window"),
    ],
)
def test_an_answer_the_device_breaks_off_interrupts_that_call_only(
    make_user, connect_device, broken, reason
):
    owner = make_user()
    device = connect_device(owner, window=CHUNK)
    device.route("GET", "/files/cwd", json_answer({"cwd": "/home/user"}))
    device.route("POST", "/execute", broken)
    with owner.client() as client:
        interrupted = client.post(
            f"/api/v1/terminals/{device.terminal_id}/execute", json={"command": "make"}
        )
        after = client.get(f"/api/v1/terminals/{device.terminal_id}/files/cwd")

    assert len(device.requests_to("/execute")) == 1, "the call never reached the device"
    assert interrupted.status_code == 502, (
        f"the broken answer came back as {interrupted.status_code} with "
        f"{len(interrupted.content)} bytes: {interrupted.text[:200]}"
    )
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
    assert json.loads(signed_in) == {"type": "auth", "token": "", "chat_id": ""}, (
        f"the shell's first message to the device was {signed_in}"
    )
    assert socket.path.startswith("/api/terminals/") and f"user_id={owner.id}" in socket.path, (
        f"the shell was opened on {socket.path}"
    )
    assert socket.headers["x-user-id"] == owner.id, socket.headers
    assert closed is not None and closed[0] == 4000, (
        f"the device closed the shell with 4000, the page's session ended with {closed}"
    )


REFUSED_HELLOS = {
    "another-message-first": {"type": "services"},
    "version-3": {"version": 3},
    "window-under-64-kib": {"window": CHUNK - 1},
    "window-as-text": {"window": str(DEFAULT_WINDOW)},
    "no-name": {"name": None},
    "blank-name": {"name": "   "},
    "name-of-101-characters": {"name": "n" * 101},
    "no-connection-id": {"connection_id": None},
    "connection-id-that-is-no-uuid": {"connection_id": "workstation"},
    "services-as-text": {"services": "open_terminal"},
    "service-name-with-a-space": {"services": ["Open Terminal"]},
    "17-services": {"services": [f"service_{number}" for number in range(17)]},
    "version-2-without-services": {"services": None},
}


@pytest.mark.parametrize("hello", REFUSED_HELLOS.values(), ids=list(REFUSED_HELLOS))
def test_a_hello_that_breaks_a_rule_is_refused_and_lists_no_device(instance, make_user, hello):
    owner = make_user()
    token = tether_session(instance, owner)["token"]
    name = f"Workstation {uuid.uuid4().hex[:8]}"

    with tether_device(instance.base_url, token, hello=hello) as refused:
        closed = refused.wait_closed(10)
        listed_meanwhile = _devices(owner)
    with tether_device(instance.base_url, token, name=name) as accepted:
        listed = _devices(owner)

    assert refused.ready is None, f"the server answered the hello with {refused.ready}"
    assert closed is not None and closed[0] == 4001, f"the refused device was closed with {closed}"
    assert listed_meanwhile == {}, f"a refused device is on the list: {listed_meanwhile}"
    assert listed == {accepted.terminal_id: name}, (
        f"the same session with a hello that keeps the rules lists {listed}"
    )


def test_a_hello_within_the_rules_is_taken_with_its_window_capped_and_its_name_trimmed(
    instance, make_user
):
    owner = make_user()
    token = tether_session(instance, owner)["token"]
    tag = uuid.uuid4().hex[:8]
    longest = f"{tag} ".ljust(100, "n")

    def device(**options):
        return tether_device(instance.base_url, token, **options)

    with (
        device(name=f"  Padded {tag}  ", window=64 * 1024 * 1024) as wide,
        device(name=longest, hello={"window": None}) as long_named,
        device(name=f"Old {tag}", hello={"version": 1, "services": None}) as old,
        device(name=f"Twice {tag}", services=["open_terminal", "open_terminal", "notes"]) as twice,
        device(name=f"Idle {tag}", services=[]) as idle,
    ):
        listed = _devices(owner)

    assert wide.ready and wide.ready["window"] == LARGEST_WINDOW, (
        f"a hello asking for a 64 MiB window got {wide.ready}"
    )
    assert long_named.ready and long_named.ready["window"] == DEFAULT_WINDOW, (
        f"a hello naming no window got {long_named.ready}"
    )
    assert idle.ready and idle.ready["type"] == "ready", (
        f"a device offering no service got {idle.ready}"
    )
    assert listed == {
        wide.terminal_id: f"Padded {tag}",
        long_named.terminal_id: longest,
        old.terminal_id: f"Old {tag}",
        twice.terminal_id: f"Twice {tag}",
    }, f"the owner's list shows {listed}"


def test_a_device_that_connects_again_takes_the_place_of_its_earlier_connection(
    make_user, connect_device
):
    owner, neighbour = make_user(), make_user()
    connection_id, tag = str(uuid.uuid4()), uuid.uuid4().hex[:8]
    earlier = connect_device(owner, connection_id=connection_id, name=f"Earlier {tag}")
    earlier.route("GET", "/files/cwd", json_answer({"cwd": "/earlier"}))
    first = _cwd_of(owner, earlier)

    later = connect_device(owner, connection_id=connection_id, name=f"Later {tag}")
    later.route("GET", "/files/cwd", json_answer({"cwd": "/later"}))
    closed = earlier.wait_closed(10)
    theirs = connect_device(neighbour, connection_id=connection_id, name=f"Neighbour {tag}")
    theirs.route("GET", "/files/cwd", json_answer({"cwd": "/neighbour"}))
    second = _cwd_of(owner, later)
    of_neighbour = _cwd_of(neighbour, theirs)

    assert (first.status_code, first.json()) == (200, {"cwd": "/earlier"}), first.text
    assert closed == (1000, ""), f"the earlier connection was closed with {closed}"
    assert _devices(owner) == {later.terminal_id: f"Later {tag}"}, (
        f"after the device connected again its owner's list shows {_devices(owner)}"
    )
    assert (second.status_code, second.json()) == (200, {"cwd": "/later"}), (
        f"the call after the device connected again answered {second.status_code} {second.text}"
    )
    assert len(earlier.requests_to("/files/cwd")) == 1, "the replaced connection was still called"
    assert later.closed is None, (
        f"another account's device with the same id closed the owner's: {later.closed}"
    )
    assert _devices(neighbour) == {theirs.terminal_id: f"Neighbour {tag}"}, _devices(neighbour)
    assert (of_neighbour.status_code, of_neighbour.json()) == (200, {"cwd": "/neighbour"}), (
        f"the neighbour's call answered {of_neighbour.status_code} {of_neighbour.text}"
    )


def test_a_device_renames_itself_and_takes_its_terminal_off_the_list_and_back(
    make_user, connect_device
):
    owner = make_user()
    tag = uuid.uuid4().hex[:8]
    device = connect_device(owner, name=f"Laptop {tag}")
    device.route("GET", "/files/cwd", json_answer(CWD))
    listed_as = {device.terminal_id: f"Desk {tag}"}

    device.announce(["open_terminal"], name=f"  Desk {tag} ")
    assert wait_until(lambda: _devices(owner) == listed_as, 10), (
        f"after the device took a new name its owner's list shows {_devices(owner)}"
    )
    device.announce(["notes"])
    assert wait_until(lambda: _devices(owner) == {}, 10), (
        f"a device that no longer offers its terminal is still listed: {_devices(owner)}"
    )
    withdrawn = _cwd_of(owner, device)
    device.announce(["notes", "open_terminal"])
    assert wait_until(lambda: _devices(owner) == listed_as, 10), (
        f"after the device offered its terminal again its owner's list shows {_devices(owner)}"
    )
    offered = _cwd_of(owner, device)
    device.announce(["open_terminal"], name="")
    closed = device.wait_closed(10)

    assert withdrawn.status_code == 404, (
        f"a call to a withdrawn terminal answered {withdrawn.status_code} {withdrawn.text}"
    )
    assert len(device.requests_to("/files/cwd")) == 1, "the withdrawn terminal was called"
    assert (offered.status_code, offered.json()) == (200, CWD), offered.text
    assert closed is not None and closed[0] == 4001, (
        f"a device that took an empty name was closed with {closed}"
    )
    assert wait_until(lambda: _devices(owner) == {}, 10), (
        f"the closed device is still listed: {_devices(owner)}"
    )


def _changes(tab: SocketSession) -> int:
    """How often the tab was told to reload its terminals."""
    return sum(
        1
        for event in list(tab.events)
        if (event.get("data") or {}).get("type") == "terminals:changed"
    )


def _told(tab: SocketSession, times: int) -> bool:
    return wait_until(lambda: _changes(tab) >= times, 10)


def test_the_owners_open_pages_are_told_when_a_device_comes_changes_or_goes(
    make_user, connect_device
):
    owner, bystander = make_user(), make_user()
    with connected(owner) as tab, connected(bystander) as other_tab:
        device = connect_device(owner)
        assert _told(tab, 1), "the owner's page was not told that a device connected"
        device.announce(["open_terminal"], name=f"Renamed {uuid.uuid4().hex[:8]}")
        assert _told(tab, 2), "the owner's page was not told that the device changed"
        device.drop()
        assert _told(tab, 3), "the owner's page was not told that the device went"
        connect_device(bystander)
        assert _told(other_tab, 1), "the bystander's page was not told of the bystander's device"

        assert _changes(other_tab) == 1, (
            f"the bystander's page was told {_changes(other_tab)} times to reload its "
            "terminals, once for its own device and never for the owner's"
        )
        assert _changes(tab) == 3, (
            f"the owner's page was told {_changes(tab)} times, not 3: another account's device "
            "reached it or a change was announced twice"
        )


def _renewed(instance, session: dict) -> dict:
    """The session's next pair of tokens, as Tether gets before its access token runs out."""
    renewed = httpx.post(
        f"{instance.base_url}{SESSIONS}/refresh",
        json={"refresh_token": session["refresh_token"]},
        headers={"X-Forwarded-For": new_address()},
        timeout=60.0,
    )
    assert renewed.status_code == 200, f"renewing the device's session failed: {renewed.text}"
    return renewed.json()


@pytest.mark.parametrize("token_of", ["another-session", "another-account", "a-plain-sign-in"])
def test_a_device_stays_connected_with_a_renewed_token_of_its_own_session_only(
    instance, make_user, connect_device, token_of
):
    owner, other = make_user(), make_user()
    tag = uuid.uuid4().hex[:8]
    device = connect_device(owner, name=f"Laptop {tag}")
    device.route("GET", "/files/cwd", json_answer(CWD))

    device.reauthenticate(_renewed(instance, device.session)["token"])
    # read after the token, in order: its effect shows that the token was taken
    device.announce(["open_terminal"], name=f"Renewed {tag}")
    assert wait_until(lambda: _devices(owner) == {device.terminal_id: f"Renewed {tag}"}, 10), (
        f"after its session was renewed the device is listed as {_devices(owner)}, "
        f"its connection: {device.closed}"
    )
    served = _cwd_of(owner, device)
    foreign = {
        "another-session": lambda: tether_session(instance, owner)["token"],
        "another-account": lambda: tether_session(instance, other)["token"],
        "a-plain-sign-in": lambda: owner.token,
    }[token_of]()
    device.reauthenticate(foreign)
    closed = device.wait_closed(10)

    assert (served.status_code, served.json()) == (200, CWD), (
        f"a call after the renewal answered {served.status_code} {served.text}"
    )
    assert closed is not None and closed[0] == 4001, (
        f"a device handing in a token of {token_of} was closed with {closed}"
    )
    assert wait_until(lambda: _devices(owner) == {}, 10), (
        f"the closed device is still listed: {_devices(owner)}"
    )


def test_a_frame_of_no_call_is_ignored_and_a_message_the_server_cannot_read_ends_the_connection(
    make_user, connect_device
):
    owner = make_user()
    garbled, chatty = connect_device(owner), connect_device(owner)
    garbled.route("GET", "/files/cwd", json_answer(CWD))

    garbled.send_frame({"type": "data", "stream": str(uuid.uuid4()), "sequence": 0}, b"stray")
    served = _cwd_of(owner, garbled)
    still_open = garbled.closed
    garbled.socket.send(b"\x00")
    chatty.send_json({"type": "greeting"})
    closed = {"garbled": garbled.wait_closed(10), "chatty": chatty.wait_closed(10)}

    assert (served.status_code, served.json()) == (200, CWD), (
        f"after a frame of no call the device's next call answered {served.status_code}"
    )
    assert still_open is None, f"a frame of no call closed the device: {still_open}"
    assert {name: how and how[0] for name, how in closed.items()} == {
        "garbled": 4001,
        "chatty": 4001,
    }, f"a frame too short to read and a message of an unknown kind were closed with {closed}"
    assert wait_until(lambda: _devices(owner) == {}, 10), (
        f"the closed devices are still listed: {_devices(owner)}"
    )


def _terminal_spec() -> dict:
    command = {"type": "object", "properties": {"command": {"type": "string"}}}
    return {
        "openapi": "3.0.0",
        "info": {"title": "terminal", "version": "1"},
        "paths": {
            "/execute": {
                "post": {
                    "operationId": "run_command",
                    "requestBody": {"content": {"application/json": {"schema": command}}},
                    "responses": {"200": {"description": "ok"}},
                }
            }
        },
    }


def _serve_terminal(device: TetherDevice, tag: str) -> None:
    """What an Open Terminal answers before and during a chat turn, marked with `tag`."""

    def ran(request: DeviceRequest):
        return json_answer({"output": f"ran {request.json()['command']} on the device"})

    skill = {
        "id": f"terminal:ship-{tag}",
        "name": f"ship-{tag}",
        "description": "Ship the app to production",
        "location": f"{HOME}/.skills/ship-{tag}/SKILL.md",
    }
    device.route("GET", "/openapi.json", json_answer(_terminal_spec()))
    device.route("GET", "/api/config", json_answer({"features": {"system": True}}))
    device.route("GET", "/system", json_answer({"prompt": f"You work on the device {tag}."}))
    device.route("GET", "/files/cwd", json_answer(CWD))
    device.route("GET", "/files/read", json_answer({"content": f"Answer in rhyme {tag}."}))
    device.route("GET", "/skills", json_answer([skill]))
    device.route("POST", "/execute", ran)


def _requests_for(upstream, prompt: str) -> list[dict]:
    return [body for body in upstream.chat_requests() if reply.answering(prompt)(body)]


def _tool_results(body: dict) -> list:
    return [json.loads(entry["content"]) for entry in conversation(body) if entry["role"] == "tool"]


def test_a_chat_on_a_device_gives_the_model_its_tools_instructions_and_skills_and_runs_the_call(
    make_user, connect_device, upstream
):
    owner = make_user()
    tag = uuid.uuid4().hex[:8]
    device = connect_device(owner)
    _serve_terminal(device, tag)
    prompt = f"build the docs {tag}"
    upstream.queue(
        reply.tool_call("run_command", {"command": f"make {tag}"}, match=reply.answering(prompt)),
        reply.text(f"built {tag}", match=reply.answering(prompt)),
    )

    with owner.client() as client:
        turn, message = ask(client, prompt, terminal_id=device.terminal_id)

    assert message["content"] == f"built {tag}", f"the reply ended as {message}"
    asked, told = _requests_for(upstream, prompt)
    sent = [(entry["role"], entry["content"]) for entry in asked["messages"]]
    tools = {tool["function"]["name"]: tool["function"] for tool in asked["tools"]}
    assert ("system", f"You work on the device {tag}.") in sent, (
        f"the device's system prompt did not reach the model: {sent}"
    )
    assert ("user", f"# AGENTS.md\n\nAnswer in rhyme {tag}.") in sent, (
        f"the AGENTS.md of the device's home did not reach the model: {sent}"
    )
    assert any(f"<id>terminal:ship-{tag}</id>" in str(content) for _, content in sent), (
        f"the device's skill was not listed for the model: {sent}"
    )
    assert "run_command" in tools, f"the model was offered {sorted(tools)}"
    assert tools["run_command"]["description"].endswith(
        f"The current working directory is: {HOME}/work"
    ), tools["run_command"]["description"]
    assert _tool_results(told) == [{"output": f"ran make {tag} on the device"}], (
        f"the model was told {_tool_results(told)} of its command"
    )

    [executed] = device.requests_to("/execute")
    assert executed.json() == {"command": f"make {tag}"}, executed.body
    assert executed.headers["x-session-id"] == turn.chat_id, executed.headers
    [read] = device.requests_to("/files/read")
    assert read.path == "/files/read?path=%2Fhome%2Fuser%2FAGENTS.md", read.path
    fetched = {entry.path.split("?")[0] for entry in device.received}
    assert fetched == {
        "/openapi.json",
        "/api/config",
        "/system",
        "/files/cwd",
        "/files/read",
        "/skills",
        "/execute",
    }, f"the turn asked the device for {sorted(fetched)}"
    for entry in device.received:
        assert entry.headers.get("x-user-id") == owner.id, (
            f"{entry.method} {entry.path} named the caller as {entry.headers.get('x-user-id')}"
        )
        assert "authorization" not in entry.headers and owner.token not in str(entry.headers), (
            f"{entry.method} {entry.path} carried a credential: {entry.headers}"
        )


def _go_offline(device: TetherDevice, owner, client: httpx.Client) -> None:
    device.drop()
    assert wait_until(lambda: _devices(owner) == {}, 10), "the device stayed listed"


def _withdraw_the_terminal(device: TetherDevice, owner, client: httpx.Client) -> None:
    device.announce(["notes"])
    assert wait_until(lambda: _devices(owner) == {}, 10), "the terminal stayed listed"


def _revoke_the_session(device: TetherDevice, owner, client: httpx.Client) -> None:
    revoked = client.delete(
        f"{SESSIONS}/{device.session['session_id']}", headers={"X-Forwarded-For": new_address()}
    )
    assert revoked.status_code == 200, f"revoking the device's session failed: {revoked.text}"


@pytest.mark.parametrize(
    "lose,told",
    [
        pytest.param(_go_offline, "503: Tether is offline.", id="offline"),
        pytest.param(
            _withdraw_the_terminal, "503: Tether service is unavailable.", id="terminal-withdrawn"
        ),
        pytest.param(
            _revoke_the_session, "403: Tether session is no longer available.", id="session-revoked"
        ),
    ],
)
def test_the_model_is_told_why_when_the_device_is_lost_before_its_call(
    make_user, connect_device, upstream, lose, told
):
    owner = make_user()
    tag = uuid.uuid4().hex[:8]
    device = connect_device(owner)
    _serve_terminal(device, tag)
    prompt = f"build the docs {tag}"
    thinking = threading.Event()
    upstream.queue(
        reply.tool_call(
            "run_command", {"command": "make"}, match=reply.answering(prompt), hold=thinking
        ),
        reply.text(f"could not build {tag}", match=reply.answering(prompt)),
    )

    with owner.client() as client:
        turn = send_message(client, prompt, terminal_id=device.terminal_id)
        try:
            assert wait_until(lambda: _requests_for(upstream, prompt), 30), (
                "the turn never reached the model"
            )
            lose(device, owner, client)
        finally:
            thinking.set()
        message = wait_for_reply(client, turn)

    asked, answered = _requests_for(upstream, prompt)
    offered = {tool["function"]["name"] for tool in asked["tools"]}
    assert "run_command" in offered, f"the device's tool was never offered: {sorted(offered)}"
    assert _tool_results(answered) == [{"error": told}], (
        f"the model was told {_tool_results(answered)} of its call"
    )
    assert device.requests_to("/execute") == [], "the call still reached the device"
    assert message["content"] == f"could not build {tag}", f"the reply ended as {message}"


def test_a_device_that_hangs_up_during_the_models_call_tells_the_model_not_to_repeat_it(
    make_user, connect_device, upstream
):
    owner = make_user()
    tag = uuid.uuid4().hex[:8]
    device = connect_device(owner)
    _serve_terminal(device, tag)
    device.route("POST", "/execute", lambda _request: device.drop())
    prompt = f"deploy {tag}"
    upstream.queue(
        reply.tool_call("run_command", {"command": "deploy"}, match=reply.answering(prompt)),
        reply.text(f"check the deployment {tag}", match=reply.answering(prompt)),
    )

    with owner.client() as client:
        _, message = ask(client, prompt, terminal_id=device.terminal_id)

    _, answered = _requests_for(upstream, prompt)
    [result] = _tool_results(answered)
    assert [entry.json() for entry in device.requests_to("/execute")] == [{"command": "deploy"}]
    assert {key: value for key, value in result.items() if key != "error"} == INTERRUPTED, (
        f"the model was told {result} of the interrupted call"
    )
    assert "do not automatically repeat it" in result["error"], result
    assert message["content"] == f"check the deployment {tag}", f"the reply ended as {message}"


def test_a_device_that_never_answers_a_question_before_the_turn_is_given_up_on_and_told_to_stop(
    make_user, connect_device, upstream
):
    owner = make_user()
    tag = uuid.uuid4().hex[:8]
    device = connect_device(owner)
    _serve_terminal(device, tag)

    def silent(request: DeviceRequest) -> None:
        request.stream.cancelled.wait(30)

    device.route("GET", "/api/config", silent)
    prompt = f"build the docs {tag}"
    upstream.queue(
        reply.tool_call("run_command", {"command": f"make {tag}"}, match=reply.answering(prompt)),
        reply.text(f"built {tag}", match=reply.answering(prompt)),
    )

    with owner.client() as client:
        _, message = ask(client, prompt, terminal_id=device.terminal_id)

    asked, told = _requests_for(upstream, prompt)
    [unanswered] = device.requests_to("/api/config")
    assert message["content"] == f"built {tag}", f"the reply ended as {message}"
    assert unanswered.stream.cancelled.is_set(), (
        "the device was never told to stop working on the question it left unanswered"
    )
    assert "system" not in {entry["role"] for entry in asked["messages"]}, (
        f"a system prompt reached the model without the device's answer: {asked['messages']}"
    )
    assert _tool_results(told) == [{"output": f"ran make {tag} on the device"}], (
        f"after the unanswered question the model's command came back as {_tool_results(told)}"
    )


def test_a_device_works_on_sixteen_calls_at_once_and_refuses_the_next(make_user, connect_device):
    owner = make_user()
    device = connect_device(owner)
    finish = threading.Event()

    def slow(_request: DeviceRequest):
        finish.wait(60)
        return json_answer(CWD)

    device.route("GET", "/files/cwd", slow)
    device.route("GET", "/files/list", json_answer({"entries": []}))
    listing = f"/api/v1/terminals/{device.terminal_id}/files/list"

    with ThreadPoolExecutor(16) as callers, owner.client() as client:
        try:
            running = [callers.submit(_cwd_of, owner, device) for _ in range(16)]
            assert wait_until(lambda: len(device.requests_to("/files/cwd")) == 16, 20), (
                f"only {len(device.requests_to('/files/cwd'))} of 16 calls reached the device"
            )
            one_more = client.get(listing)
            reached = len(device.requests_to("/files/list"))
        finally:
            finish.set()
        finished = [call.result(timeout=30) for call in running]
        afterwards = client.get(listing)

    assert one_more.status_code >= 400 and (
        "Tether has too many active operations." in one_more.json()["error"]
    ), f"the 17th call at once answered {one_more.status_code} {one_more.text}"
    assert reached == 0, "the refused call reached the device"
    assert [(call.status_code, call.json()) for call in finished] == [(200, CWD)] * 16, (
        f"the 16 calls answered {[call.status_code for call in finished]}"
    )
    assert (afterwards.status_code, afterwards.json()) == (200, {"entries": []}), (
        f"once the 16 calls had ended the next answered {afterwards.status_code} {afterwards.text}"
    )


def test_an_answer_over_16_mib_arrives_as_a_download_and_is_refused_as_a_page_answer(
    make_user, connect_device
):
    owner = make_user()
    device = connect_device(owner)
    at_the_limit = b"x" * LARGEST_BUFFERED
    over_the_limit = at_the_limit + b"x"
    text, download = {"Content-Type": "text/plain"}, {"Content-Type": "application/octet-stream"}
    device.route("GET", "/files/read", (200, text, at_the_limit))
    device.route("GET", "/files/display", (200, text, over_the_limit))
    device.route("GET", "/files/view", (200, download, over_the_limit))
    base = f"/api/v1/terminals/{device.terminal_id}"

    with owner.client() as client:
        fits = client.get(f"{base}/files/read")
        too_large = client.get(f"{base}/files/display")
        streamed = client.get(f"{base}/files/view")
        afterwards = client.get(f"{base}/files/read")

    assert fits.status_code == 200 and fits.content == at_the_limit, (
        f"a text answer of 16 MiB arrived as {fits.status_code} with {len(fits.content)} bytes"
    )
    assert too_large.status_code >= 400 and (
        "Response too large to buffer; use streaming." in too_large.json()["error"]
    ), f"a text answer over 16 MiB answered {too_large.status_code} {too_large.text[:200]}"
    assert streamed.status_code == 200 and streamed.content == over_the_limit, (
        f"a download over 16 MiB arrived as {streamed.status_code} "
        f"with {len(streamed.content)} bytes"
    )
    assert afterwards.status_code == 200, (
        f"the call after the refused answer got {afterwards.status_code}; "
        f"the device's connection: {device.closed}"
    )


def test_a_shell_ends_when_its_device_hangs_up_and_none_opens_while_the_device_is_offline(
    make_user, connect_device
):
    owner, stranger = make_user(), make_user()
    device = connect_device(owner)

    with terminal_session(stranger.base_url, stranger.token, device.terminal_id) as session:
        of_stranger = close_of(session, timeout=10)
    with terminal_session(owner.base_url, owner.token, device.terminal_id) as session:
        session.recv(timeout=10)
        session.send("whoami")
        echoed = session.recv(timeout=10)
        device.drop()
        interrupted = close_of(session, timeout=10)
    assert wait_until(lambda: _devices(owner) == {}, 10), "the device stayed on the list"
    with terminal_session(owner.base_url, owner.token, device.terminal_id) as session:
        offline = close_of(session, timeout=10)

    assert echoed == "whoami", f"the shell on the connected device answered {echoed!r}"
    assert len(device.sockets) == 1, f"{len(device.sockets)} shells were opened on the device"
    assert interrupted == (1011, "Terminal connection interrupted. Reconnecting."), (
        f"the open shell of a device that hung up ended with {interrupted}"
    )
    assert offline == (1013, "Tether is offline. Reconnecting."), (
        f"a shell on a device that is offline ended with {offline}"
    )
    assert of_stranger == (1013, "Tether is offline. Reconnecting."), (
        f"another account's shell on the device ended with {of_stranger}"
    )


def _set_notifications(client: httpx.Client, enabled: bool) -> None:
    """Browser Notifications in the account's Settings > Notifications."""
    saved = client.post(
        "/api/v1/users/user/settings/update", json={"ui": {"notificationEnabled": enabled}}
    )
    assert saved.status_code == 200, f"saving the notification setting failed: {saved.text}"


def test_a_finished_reply_is_announced_to_the_devices_that_asked_once_the_person_allows_it(
    make_user, connect_device, upstream
):
    owner = make_user()
    tag = uuid.uuid4().hex[:8]
    asking = connect_device(owner, notifications=True)
    quiet = connect_device(owner)
    unannounced, announced = f"first question {tag}", f"second question {tag}"
    upstream.queue(
        reply.text(f"first answer {tag}", match=reply.answering(unannounced)),
        reply.text(f"second answer {tag}", match=reply.answering(announced)),
    )

    with owner.client() as client:
        _set_notifications(client, False)
        ask(client, unannounced)
        _set_notifications(client, True)
        turn, _ = ask(client, announced)

    assert wait_until(lambda: asking.notifications, 10), (
        "a device that asked for notifications heard nothing of the finished reply"
    )
    [notification] = asking.notifications
    assert {key: value for key, value in notification.items() if key not in ("id", "title")} == {
        "user_id": owner.id,
        "body": f"second answer {tag}",
        "path": f"/c/{turn.chat_id}",
    }, f"the device was notified with {notification}"
    assert notification["id"] and notification["title"].endswith(" / Open WebUI"), notification
    assert quiet.notifications == [], (
        f"a device that asked for no notifications got {quiet.notifications}"
    )


def test_a_channel_message_is_announced_to_the_devices_of_the_other_members(
    admin, make_user, connect_device, preserve
):
    author, member, outsider = make_user(), make_user(), make_user()
    tag = uuid.uuid4().hex[:8]
    preserve("admin_config")
    enable_channels(admin)
    channel_id = group_channel(author, member)
    devices = {
        who: connect_device(account, notifications=True)
        for who, account in (("author", author), ("member", member), ("outsider", outsider))
    }
    for account in (author, member, outsider):
        with account.client() as client:
            _set_notifications(client, True)

    message_id = post_message(author, channel_id, f"lunch at noon {tag}")

    assert wait_until(lambda: devices["member"].notifications, 10), (
        "the device of a member heard nothing of the channel's new message"
    )
    [notification] = devices["member"].notifications
    assert {key: value for key, value in notification.items() if key != "title"} == {
        "id": f"channel:{channel_id}:{message_id}",
        "user_id": member.id,
        "body": f"lunch at noon {tag}",
        "path": f"/channels/{channel_id}",
    }, f"the member's device was notified with {notification}"
    assert notification["title"].endswith(" / Open WebUI"), notification
    assert devices["author"].notifications == [], (
        f"the author's own device was notified: {devices['author'].notifications}"
    )
    assert devices["outsider"].notifications == [], (
        f"the device of someone outside the channel got {devices['outsider'].notifications}"
    )


def _uploaded_file(request: DeviceRequest) -> tuple[str | None, bytes]:
    """The file name and bytes of a multipart upload the device received."""
    head = f"Content-Type: {request.headers['content-type']}\r\n\r\n".encode()
    form = BytesParser(policy=HTTP).parsebytes(head + request.body)
    [part] = [
        part
        for part in form.iter_parts()
        if part.get_param("name", header="content-disposition") == "file"
    ]
    return part.get_filename(), part.get_payload(decode=True)


def test_a_picture_the_model_draws_is_copied_into_the_devices_working_folder(
    admin, make_user, connect_device, upstream, listener, preserve
):
    owner = make_user()
    tag = uuid.uuid4().hex[:8]
    preserve(IMAGES_CONFIG)
    with admin.client() as client:
        save_image_settings(client, **serve_openai_images(listener))
    device = connect_device(owner)
    _serve_terminal(device, tag)

    def saved(request: DeviceRequest):
        name, content = _uploaded_file(request)
        return json_answer({"path": f"{HOME}/work/{name}", "size": len(content)})

    device.route("POST", "/files/upload", saved)
    prompt = f"draw a lighthouse {tag}"
    upstream.queue(
        reply.tool_call(
            "generate_image", {"prompt": "a lighthouse"}, match=reply.answering(prompt)
        ),
        reply.text(f"drawn {tag}", match=reply.answering(prompt)),
    )

    with owner.client() as client:
        _, message = ask(
            client, prompt, terminal_id=device.terminal_id, features={"image_generation": True}
        )

    _, told = _requests_for(upstream, prompt)
    [result] = _tool_results(told)
    [picture] = result.get("images") or [None]
    assert picture, f"the model was told {result} of its picture"
    [upload] = device.requests_to("/files/upload")
    name, content = _uploaded_file(upload)
    assert upload.path == "/files/upload?directory=%2Fhome%2Fuser%2Fwork", upload.path
    assert name == f"{picture['id']}.png", f"the copy on the device is named {name}"
    assert content == base64.b64decode(PNG_BASE64), (
        f"the device got {len(content)} bytes that are not the picture"
    )
    assert upload.headers["x-user-id"] == owner.id, upload.headers
    assert picture.get("terminal_path") == f"{HOME}/work/{picture['id']}.png", (
        f"the model was not told where the picture is on the device: {picture}"
    )
    assert message["content"] == f"drawn {tag}", f"the reply ended as {message}"


def _streamed_completion(text: str) -> list[bytes]:
    """A chat completion as an OpenAI-compatible provider streams it."""

    def chunk(delta: dict, finish_reason: str | None) -> bytes:
        event = {
            "id": "chatcmpl-device",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "device-model",
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }
        return f"data: {json.dumps(event)}\n\n".encode()

    return [
        chunk({"role": "assistant", "content": ""}, None),
        chunk({"content": text}, None),
        chunk({}, "stop"),
        b"data: [DONE]\n\n",
    ]


def _listed_models(actor) -> dict[str, dict]:
    with actor.client() as client:
        listed = client.get("/api/models", params={"refresh": "true"})
    assert listed.status_code == 200, listed.text
    return {model["id"]: model for model in listed.json()["data"]}


def test_a_model_of_a_connection_added_in_tether_is_its_owners_alone_and_answers_through_the_device(
    make_user, connect_device
):
    owner, stranger = make_user(), make_user()
    tag = uuid.uuid4().hex[:8]
    own_model, unoffered_model = f"home-model-{tag}", f"unoffered-model-{tag}"
    device = connect_device(
        owner,
        name=f"Laptop {tag}",
        services=["open_terminal", "home_llm"],
        connections=[
            own_connection("openai", "home_llm", "Home LLM"),
            # declared, but its service is not among those the device offers
            own_connection("openai", "spare_llm", "Spare LLM"),
        ],
    )

    def models(request: DeviceRequest):
        model_id = own_model if request.service == "home_llm" else unoffered_model
        return json_answer({"object": "list", "data": [{"id": model_id, "object": "model"}]})

    def completion(request: DeviceRequest):
        return 200, {"Content-Type": "text/event-stream"}, _streamed_completion(f"hi from {tag}")

    device.route("GET", "/models", models)
    device.route("POST", "/chat/completions", completion)

    listed = _listed_models(owner)
    assert own_model in listed, f"the owner's model list lacks the device's model: {sorted(listed)}"
    assert unoffered_model not in listed, "a connection whose service is not offered was listed"
    assert {"name": f"Laptop {tag} · Home LLM"} in listed[own_model]["tags"], (
        f"the device's model is tagged {listed[own_model]['tags']}"
    )
    assert own_model not in _listed_models(stranger), "a stranger sees the owner's own model"

    prompt = f"hello laptop {tag}"
    with owner.client() as client:
        _, message = ask(client, prompt, model=own_model)
    with stranger.client() as client:
        refused = client.post(
            "/api/chat/completions",
            json={"model": own_model, "messages": [{"role": "user", "content": "me too"}]},
        )

    assert message["content"] == f"hi from {tag}", f"the reply ended as {message}"
    [sent] = device.requests_to("/chat/completions")
    assert sent.service == "home_llm", f"the chat went to the device's {sent.service!r}"
    assert sent.json()["model"] == own_model, sent.json()
    assert prompt in json.dumps(sent.json()["messages"]), sent.json()["messages"]
    assert owner.token not in json.dumps(sent.headers), (
        f"the owner's credential reached the device's provider: {sent.headers}"
    )
    assert "x-user-id" not in sent.headers, sent.headers
    assert refused.status_code >= 400, (
        f"another account's chat with the owner's model answered {refused.status_code}"
    )
    assert len(device.requests_to("/chat/completions")) == 1, (
        "the stranger's chat reached the device"
    )

    device.announce(["open_terminal"])
    assert wait_until(lambda: own_model not in _listed_models(owner), 10), (
        "the model stayed listed after the device withdrew its connection"
    )


def test_a_tool_server_added_in_tether_is_listed_for_its_owner_and_serves_the_models_call(
    make_user, connect_device, upstream
):
    owner, stranger = make_user(), make_user()
    tag = uuid.uuid4().hex[:8]
    device = connect_device(
        owner,
        name=f"Laptop {tag}",
        services=["home_tools"],
        connections=[own_connection("openapi", "home_tools", "Weather")],
    )
    device.route("GET", "/openapi.json", json_answer(openapi_spec(f"Weather {tag}", ["forecast"])))
    device.route("GET", "/forecast", json_answer({"sky": f"sunny {tag}"}))
    tool_id = f"server:tether:{device.connection_id}:home_tools"

    with owner.client() as client:
        tools = client.get("/api/v1/tools/")
    with stranger.client() as client:
        of_stranger = client.get("/api/v1/tools/")
    assert tools.status_code == 200 and of_stranger.status_code == 200, tools.text
    names = {tool["id"]: tool["name"] for tool in tools.json()}
    assert names.get(tool_id) == f"Laptop {tag} · Weather", (
        f"the owner's tools list the device's server as {names.get(tool_id)!r}: {names}"
    )
    assert tool_id not in {tool["id"] for tool in of_stranger.json()}, (
        "a stranger sees the owner's tool server"
    )

    prompt = f"what is the weather {tag}"
    upstream.queue(
        reply.tool_call("forecast", {}, match=reply.answering(prompt)),
        reply.text(f"sunny it is {tag}", match=reply.answering(prompt)),
    )
    with owner.client() as client:
        _, message = ask(client, prompt, tool_ids=[tool_id])

    asked, told = _requests_for(upstream, prompt)
    offered = {tool["function"]["name"] for tool in asked.get("tools", [])}
    assert "forecast" in offered, f"the model was offered {sorted(offered)}"
    [called] = device.requests_to("/forecast")
    assert called.service == "home_tools", f"the call went to the device's {called.service!r}"
    assert owner.token not in json.dumps(called.headers), called.headers
    assert _tool_results(told) == [{"sky": f"sunny {tag}"}], (
        f"the model was told {_tool_results(told)} of its call"
    )
    assert message["content"] == f"sunny it is {tag}", f"the reply ended as {message}"


def test_a_terminal_added_in_tether_is_listed_under_its_own_name_and_serves_its_owner_alone(
    make_user, connect_device
):
    owner, stranger = make_user(), make_user()
    tag = uuid.uuid4().hex[:8]
    device = connect_device(
        owner,
        name=f"Laptop {tag}",
        services=["lab_box"],
        connections=[own_connection("terminal", "lab_box", "Lab")],
    )
    device.route("GET", "/files/cwd", json_answer(CWD))
    terminal_id = f"tether:{device.connection_id}:lab_box"

    with owner.client() as client:
        answered = client.get(f"/api/v1/terminals/{terminal_id}/files/cwd")
    with stranger.client() as client:
        refused = client.get(f"/api/v1/terminals/{terminal_id}/files/cwd")

    assert _devices(owner) == {terminal_id: f"Laptop {tag} · Lab"}, (
        f"the owner's terminal list shows the device as {_devices(owner)}"
    )
    assert (answered.status_code, answered.json()) == (200, CWD), answered.text
    [received] = device.requests_to("/files/cwd")
    assert received.service == "lab_box", f"the call went to the device's {received.service!r}"
    assert refused.status_code == 404, (
        f"another account's call to the terminal answered {refused.status_code} {refused.text}"
    )
    assert _devices(stranger) == {}, "a stranger sees the owner's terminal"


def _cut_off_then_reconnect(owner, connect_device, method: str, path: str, call, serve):
    """Run `call` while the device hangs up on its `method path`, connect the device again and
    return the call's answer and the new connection, which `serve` sets up."""
    connection_id = str(uuid.uuid4())
    dropping = connect_device(owner, connection_id=connection_id, **serve.get("options", {}))

    def hang_up(_request: DeviceRequest) -> None:
        dropping.drop()

    dropping.route(method, path, hang_up)
    with ThreadPoolExecutor(1) as caller:
        pending = caller.submit(call, connection_id)
        assert dropping.wait_closed(20) is not None, "the device never hung up"
        later = connect_device(owner, connection_id=connection_id, **serve.get("options", {}))
        for (route_method, route_path), answer in serve["routes"].items():
            later.route(route_method, route_path, answer)
        return pending.result(timeout=40), later


def test_a_read_cut_off_by_a_reconnect_is_repeated_on_the_new_connection_and_a_command_is_not(
    make_user, connect_device
):
    owner = make_user()
    tag = uuid.uuid4().hex[:8]
    own_model = f"home-model-{tag}"
    llm = {
        "options": {
            "services": ["home_llm"],
            "connections": [own_connection("openai", "home_llm", "Home LLM")],
        },
        "routes": {("GET", "/models"): json_answer({"data": [{"id": own_model}]})},
    }
    terminal = {"routes": {("POST", "/execute"): json_answer({"output": "ran it again"})}}

    def run(connection_id: str) -> httpx.Response:
        with owner.client() as client:
            client.timeout = 40
            return client.post(
                f"/api/v1/terminals/tether:{connection_id}/execute", json={"command": "make"}
            )

    listed, read_again = _cut_off_then_reconnect(
        owner, connect_device, "GET", "/models", lambda _id: _listed_models(owner), llm
    )
    ran, run_again = _cut_off_then_reconnect(
        owner, connect_device, "POST", "/execute", run, terminal
    )

    assert own_model in listed, (
        f"a model list cut off by a reconnect lacks the device's model: {sorted(listed)}"
    )
    assert len(read_again.requests_to("/models")) >= 1, "the list was never read from the device"
    assert ran.status_code == 502, f"{ran.status_code} {ran.text}"
    assert _without_message(ran) == INTERRUPTED, ran.text
    assert run_again.requests_to("/execute") == [], (
        "a command cut off by a reconnect was run again on the new connection"
    )


def test_a_file_read_from_the_page_cut_off_by_a_reconnect_is_repeated(make_user, connect_device):
    # Red on dev 485103cde: the page's calls reach the device with a streamed body even when they
    # have none, so a read is never repeated (issue draft "a terminal read cut off by a Tether
    # reconnect fails instead of being repeated").
    owner = make_user()

    def read(connection_id: str) -> httpx.Response:
        with owner.client() as client:
            client.timeout = 40
            return client.get(f"/api/v1/terminals/tether:{connection_id}/files/cwd")

    answered, read_again = _cut_off_then_reconnect(
        owner,
        connect_device,
        "GET",
        "/files/cwd",
        read,
        {"routes": {("GET", "/files/cwd"): json_answer(CWD)}},
    )

    assert (answered.status_code, answered.json()) == (200, CWD), (
        f"a read cut off by a reconnect answered {answered.status_code} {answered.text}"
    )
    assert len(read_again.requests_to("/files/cwd")) == 1, "the read never reached the new device"
