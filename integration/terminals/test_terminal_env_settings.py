"""Journey: the terminal settings an operator can only give in the environment.

Three of them. `TERMINAL_SERVER_CONNECTIONS` names the terminal connections of a new instance,
which are connected (and their tool lists fetched) while it boots. `TERMINAL_PROXY_HEADERS`
adds response headers to every answer of the terminal proxy, ahead of the terminal's own.
`AIOHTTP_CLIENT_TIMEOUT_TERMINAL` is how long a chat waits for its terminal: unset it takes
`AIOHTTP_CLIENT_TIMEOUT_TOOL_SERVER_DATA`, a value that is no number means 60 seconds, empty
and `0` mean no limit. Past the limit the message fails with
"Terminal '<name>' timed out. It may still be starting. Please try again."

Every instance here is booted with `AIOHTTP_CLIENT_TIMEOUT_TOOL_SERVER_DATA=1`, so a terminal
timeout that falls back to it shows after one second. The terminal is the fake one, whose
skill list, the first thing a chat turn asks it for, a test can hold.

Red on dev c15f28649 and on v0.13.3, a product bug that is not filed yet:
`test_a_header_the_admin_names_in_another_case_than_the_terminal_still_wins`. A header of
`TERMINAL_PROXY_HEADERS` whose name is written in another case than the terminal's is sent
beside the terminal's own, where the docs say the admin's takes precedence.

Discriminates: every other test passes on dev c15f28649. In backend copies of it, one edit each:
* `terminal_server.connections` defaulting to an empty list: the first-boot test and every
  test that uses the named terminal fail;
* the start-up call of `set_terminal_servers` dropped: the first-boot test and the test with
  a terminal that is down fail (no tool list was fetched while booting);
* `TERMINAL_PROXY_HEADERS` no longer merged into the answer: the three cases of the header
  test fail, and the test of the other case on its first check;
* the fallback around its JSON parsing dropped: the instance with unreadable headers does not
  boot;
* a failed tool list raising out of the fetch (`return_exceptions=False`): the test with a
  terminal that is down fails (the working terminal's chat gets no tools);
* `AIOHTTP_CLIENT_TIMEOUT_TERMINAL` defaulting to 60 without the tool server value: the
  one-second test fails (the message is still waiting when the test gives up on it);
* a non-number falling back to the tool server value in place of 60, and no limit at all for
  it: the non-number half of the three-second test, then of the minute test, fails;
* empty read as unset, and empty or `0` read as 60: the empty half of the three-second test,
  then the empty and `0` halves of the minute test, fail.
"""

from __future__ import annotations

import json
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Iterator

import pytest

from harness import upstream as reply
from harness.backends import wait_until
from harness.chat import ask
from harness.instance import LaunchedInstance
from harness.listener import json_answer
from harness.terminal_server import FakeTerminalServer, serving_terminal

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source, pytest.mark.slow]

DEAD_URL = "http://127.0.0.1:9"
TOOL_SERVER_TIMEOUT = {"AIOHTTP_CLIENT_TIMEOUT_TOOL_SERVER_DATA": "1"}
ADMIN_HEADERS = {
    "Content-Security-Policy": "sandbox allow-scripts; default-src 'none'",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
}
# named in lower case; red on dev, see the finding in the module docstring
LOWER_CASE_HEADER = {"x-harbour-zone": "admin"}
STILL_STARTING = "timed out. It may still be starting. Please try again."
SPEC = {
    "openapi": "3.0.0",
    "info": {"title": "Terminal", "version": "1"},
    "paths": {
        "/execute": {
            "post": {"operationId": "run_command", "responses": {"200": {"description": "ok"}}}
        }
    },
}


@dataclass
class HeldTerminal:
    """The fake terminal; while `held` is set its skill list waits for `release`."""

    server: FakeTerminalServer
    held: threading.Event = field(default_factory=threading.Event)
    release: threading.Event = field(default_factory=threading.Event)

    def skills(self, _request):
        if self.held.is_set():
            self.release.wait(120)
        return json_answer([])

    def connection(self, name: str, **fields) -> dict:
        return {
            "id": f"terminal-{uuid.uuid4().hex[:8]}",
            "name": name,
            "enabled": True,
            "url": self.server.base_url,
            "path": "/openapi.json",
            "key": "",
            "auth_type": "none",
            **fields,
        }


@pytest.fixture(scope="module")
def terminal() -> Iterator[HeldTerminal]:
    with serving_terminal() as server:
        held = HeldTerminal(server)
        server.route("GET", "/openapi.json", json_answer(SPEC))
        server.route("GET", "/skills", held.skills)
        server.route("GET", "/files/cwd", json_answer({"cwd": "/home/user", "home": "/home/user"}))
        server.route(
            "GET",
            "/files/list",
            (
                200,
                {
                    "Content-Type": "application/json",
                    "X-Frame-Options": "SAMEORIGIN",
                    "Cache-Control": "max-age=60",
                },
                b'{"entries": []}',
            ),
        )
        server.route(
            "GET",
            "/files/download",
            (
                200,
                {"Content-Type": "application/octet-stream", "X-Frame-Options": "SAMEORIGIN"},
                b"archive",
            ),
        )
        server.route(
            "GET",
            "/files/zone",
            (200, {"Content-Type": "application/json", "X-Harbour-Zone": "terminal"}, b"{}"),
        )
        yield held


@pytest.fixture(autouse=True)
def _answering_again(terminal) -> Iterator[None]:
    terminal.release.clear()
    yield
    terminal.held.clear()
    terminal.release.set()


@dataclass
class Booted:
    instance: LaunchedInstance
    connections: list[dict]
    # the tool list fetches the terminal got before the instance was asked anything
    fetched_while_booting: int

    @property
    def terminal_id(self) -> str:
        return self.connections[-1]["id"]


def _boot(instance_with, terminal: HeldTerminal, connections: list[dict], **env) -> Booted:
    fetched_before = len(terminal.server.requests_to("/openapi.json"))
    launched = instance_with(
        {"TERMINAL_SERVER_CONNECTIONS": json.dumps(connections), **TOOL_SERVER_TIMEOUT, **env}
    )
    fetched = len(terminal.server.requests_to("/openapi.json")) - fetched_before
    return Booted(launched, connections, fetched)


@pytest.fixture(scope="module")
def configured(instance_with, terminal) -> Booted:
    """An instance whose environment names the terminal and the admin's headers."""
    connection = terminal.connection("Harbour terminal")
    headers = {**ADMIN_HEADERS, **LOWER_CASE_HEADER}
    return _boot(instance_with, terminal, [connection], TERMINAL_PROXY_HEADERS=json.dumps(headers))


@pytest.fixture(scope="module")
def misconfigured(instance_with, terminal) -> Booted:
    """An instance with unreadable headers, a timeout that is no number and one terminal down."""
    down = terminal.connection("Down terminal", url=DEAD_URL)
    working = terminal.connection("Working terminal")
    return _boot(
        instance_with,
        terminal,
        [down, working],
        TERMINAL_PROXY_HEADERS='{"X-Frame-Options": "DENY"',
        AIOHTTP_CLIENT_TIMEOUT_TERMINAL="sixty",
    )


@pytest.fixture(scope="module")
def unlimited(instance_with, terminal) -> dict[str, Booted]:
    """Two instances that give a terminal no time limit: by an empty value and by `0`."""
    return {
        value: _boot(
            instance_with,
            terminal,
            [terminal.connection("Patient terminal")],
            AIOHTTP_CLIENT_TIMEOUT_TERMINAL=value,
        )
        for value in ("", "0")
    }


@dataclass
class Asked:
    message: dict
    # what the provider was sent for the message
    requests: list[dict]

    @property
    def error(self) -> str | None:
        return (self.message.get("error") or {}).get("content")

    @property
    def offered(self) -> list[str]:
        tools = self.requests[0].get("tools", []) if self.requests else []
        return [tool["function"]["name"] for tool in tools]


def _ask_with_terminal(booted: Booted, timeout: float = 60.0) -> Asked:
    """The admin sends a message with the instance's terminal picked."""
    prompt = f"where am I {uuid.uuid4().hex[:8]}"
    provider = booted.instance.upstream
    provider.queue(reply.text("In your home folder.", match=reply.answering(prompt)))
    with booted.instance.client() as client:
        _, message = ask(client, prompt, timeout=timeout, terminal_id=booted.terminal_id)
    requests = [body for body in provider.chat_requests() if reply.answering(prompt)(body)]
    return Asked(message, requests)


def _menu(booted: Booted) -> dict[str, str]:
    with booted.instance.client() as client:
        listed = client.get("/api/v1/terminals/")
    assert listed.status_code == 200, listed.text
    return {entry["id"]: entry["name"] for entry in listed.json()}


def test_a_terminal_named_in_the_environment_is_connected_from_the_first_boot(configured):
    [connection] = configured.connections
    with configured.instance.client() as client:
        saved = client.get("/api/v1/configs/terminal_servers")
        features = client.get("/api/config").json()["features"]
        folder = client.get(f"/api/v1/terminals/{connection['id']}/files/cwd")
    asked = _ask_with_terminal(configured)

    assert saved.status_code == 200, saved.text
    assert saved.json() == {"TERMINAL_SERVER_CONNECTIONS": [connection]}, (
        f"Admin Settings do not show the connection of the environment: {saved.text}"
    )
    assert _menu(configured) == {connection["id"]: "Harbour terminal"}
    assert features["enable_terminal_servers"] is True, features
    assert folder.status_code == 200, f"the terminal was not reached: {folder.text}"
    assert configured.fetched_while_booting == 1, (
        f"the terminal's tool list was fetched {configured.fetched_while_booting} times while "
        "the instance booted"
    )
    assert asked.error is None and "run_command" in asked.offered, (
        f"a chat on the terminal of the environment was offered {asked.offered}: {asked.message}"
    )


@pytest.mark.parametrize(
    "path, kept",
    [
        ("files/list", {"content-type": "application/json", "cache-control": "max-age=60"}),
        ("files/download", {"content-type": "application/octet-stream"}),
        ("files/missing", {}),
    ],
    ids=["buffered", "streamed", "not found"],
)
def test_the_admins_headers_are_on_every_answer_of_the_terminal_and_win_over_its_own(
    configured, path, kept
):
    with configured.instance.client() as client:
        answered = client.get(f"/api/v1/terminals/{configured.terminal_id}/{path}")
        own_route = client.get("/api/v1/terminals/")

    expected = {name.lower(): [value] for name, value in ADMIN_HEADERS.items()}
    sent = {name: answered.headers.get_list(name) for name in expected}
    assert sent == expected, (
        f"the answer of {path} ({answered.status_code}) carries {sent}; the admin set {expected} "
        "and the terminal sent X-Frame-Options: SAMEORIGIN on the two it has"
    )
    still_there = {name: answered.headers.get(name) for name in kept}
    assert still_there == kept, f"the terminal's other headers were lost: {dict(answered.headers)}"
    assert not any(name in own_route.headers for name in expected), (
        f"the terminal headers are on Open WebUI's own answer: {dict(own_route.headers)}"
    )


def test_a_header_the_admin_names_in_another_case_than_the_terminal_still_wins(configured):
    with configured.instance.client() as client:
        added = client.get(f"/api/v1/terminals/{configured.terminal_id}/files/list")
        answered = client.get(f"/api/v1/terminals/{configured.terminal_id}/files/zone")

    assert added.headers.get_list("x-harbour-zone") == ["admin"], (
        f"the admin's header is not on an answer without one: {dict(added.headers)}"
    )
    assert answered.headers.get_list("x-harbour-zone") == ["admin"], (
        "the admin set x-harbour-zone: admin and the terminal sent X-Harbour-Zone: terminal; "
        f"the answer carries {answered.headers.get_list('x-harbour-zone')}"
    )


def test_headers_that_are_no_json_add_nothing(misconfigured):
    with misconfigured.instance.client() as client:
        answered = client.get(f"/api/v1/terminals/{misconfigured.terminal_id}/files/list")

    assert answered.status_code == 200, answered.text
    assert answered.headers.get_list("x-frame-options") == ["SAMEORIGIN"], (
        "with TERMINAL_PROXY_HEADERS unreadable the terminal's own header did not arrive as "
        f"sent: {answered.headers.get_list('x-frame-options')}"
    )


def test_a_terminal_that_is_down_at_boot_leaves_the_instance_and_the_other_terminal_working(
    misconfigured,
):
    down, working = misconfigured.connections
    with misconfigured.instance.client() as client:
        unreachable = client.get(f"/api/v1/terminals/{down['id']}/files/cwd")
    asked = _ask_with_terminal(misconfigured)

    assert _menu(misconfigured) == {
        down["id"]: "Down terminal",
        working["id"]: "Working terminal",
    }
    assert misconfigured.fetched_while_booting == 1, (
        f"the working terminal's tool list was fetched {misconfigured.fetched_while_booting} "
        "times while the instance booted beside a terminal that is down"
    )
    assert unreachable.status_code == 502, f"{unreachable.status_code} {unreachable.text}"
    assert asked.error is None and "run_command" in asked.offered, (
        f"a chat on the working terminal was offered {asked.offered}: {asked.message}"
    )


def test_without_a_terminal_timeout_a_chat_waits_as_long_as_for_a_tool_server(configured, terminal):
    terminal.held.set()
    too_slow = _ask_with_terminal(configured)
    terminal.held.clear()
    terminal.release.set()
    in_time = _ask_with_terminal(configured)

    assert too_slow.error == f"Terminal 'Harbour terminal' {STILL_STARTING}", (
        "with AIOHTTP_CLIENT_TIMEOUT_TOOL_SERVER_DATA=1 and no terminal timeout, a message "
        f"whose terminal did not answer ended as {too_slow.message}"
    )
    assert too_slow.requests == [], "the model was asked without the chat's terminal"
    assert in_time.error is None and in_time.message["content"] == "In your home folder.", (
        f"the same terminal answering at once: {in_time.message}"
    )


def _held_for_three_seconds(terminal: HeldTerminal, booted: Booted) -> Asked:
    """A message whose terminal answers after three seconds."""
    terminal.held.set()
    timer = threading.Timer(3, terminal.release.set)
    timer.start()
    try:
        return _ask_with_terminal(booted)
    finally:
        timer.cancel()
        terminal.release.set()


@pytest.mark.parametrize("value", ["sixty", ""], ids=["no number", "empty"])
def test_a_terminal_timeout_that_is_empty_or_no_number_does_not_fall_back_to_the_tool_servers(
    misconfigured, unlimited, terminal, value
):
    booted = misconfigured if value == "sixty" else unlimited[value]

    asked = _held_for_three_seconds(terminal, booted)

    assert asked.error is None and asked.message["content"] == "In your home folder.", (
        f"with AIOHTTP_CLIENT_TIMEOUT_TERMINAL={value!r} a terminal that took three seconds "
        f"was given up on after the tool server timeout of one: {asked.message}"
    )


def test_a_terminal_that_takes_over_a_minute_is_waited_for_only_without_a_time_limit(
    misconfigured, unlimited, terminal
):
    terminal.held.set()
    patient = {f"AIOHTTP_CLIENT_TIMEOUT_TERMINAL={value!r}": on for value, on in unlimited.items()}
    asked_before = len(terminal.server.requests_to("/skills"))

    def askers() -> set[str]:
        held = terminal.server.requests_to("/skills")[asked_before:]
        return {request.headers["x-user-id"] for request in held}

    with ThreadPoolExecutor(max_workers=3) as pool:
        try:
            waiting = {
                label: pool.submit(_ask_with_terminal, on, 100.0) for label, on in patient.items()
            }
            # the minute of the two has begun before the minute of the third does
            both_wait = wait_until(lambda: len(askers()) == 2, timeout=30)
            sixty = pool.submit(_ask_with_terminal, misconfigured, 100.0)
            given_up = wait_until(sixty.done, timeout=80)
        finally:
            terminal.release.set()
        answers = {label: asked.result() for label, asked in waiting.items()}

    assert both_wait, f"the terminal was not asked by both instances without a limit: {askers()}"
    assert given_up, "a timeout that is no number never gave up on the terminal"
    assert sixty.result().error == f"Terminal 'Working terminal' {STILL_STARTING}", (
        f"a timeout that is no number ended the message as {sixty.result().message}"
    )
    contents = {label: (asked.error, asked.message["content"]) for label, asked in answers.items()}
    assert contents == {label: (None, "In your home folder.") for label in patient}, (
        "the terminal answered after more than a minute, and without a time limit the messages "
        f"ended as {contents}"
    )
