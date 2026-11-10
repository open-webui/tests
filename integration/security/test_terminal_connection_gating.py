"""A terminal connection serves nobody while disabled and only the admin while ungranted.

Two fixes in 0.11.0. `753798923`: switching a terminal connection off only hid it from the list;
anyone who knew its id kept using it through the HTTP proxy and the interactive WebSocket. The
fix refuses a disabled connection on every entry point. `867006acc` (PR #27581, issues #27580
and #27064): with BYPASS_ADMIN_ACCESS_CONTROL off, a connection without access grants, which is
every connection right after it is added, was refused to everyone including the admin who
created it; the fix makes it admin-only. Each entry point is driven against a fake terminal
server that records whether anything reached it.

Since the terminal rework on dev (6f2fc7bc7) one lookup serves every entry point, and a terminal
that is switched off or not granted to the caller answers exactly as an id that does not exist:
404 "Terminal server '<id>' not found" on the proxy, close 4004 on the WebSocket and "not found"
in a chat. Before it they answered 403 and 4003 with "disabled" or "Access denied".

A chat with a terminal selected is the fourth entry point. Without Redis every chat rebuilds the
terminal list from the saved connections and leaves disabled ones out, so the gate only shows on
an instance backed by Redis whose cached list still names the terminal: the admin switches it
off through a config import, which saves the connection without rebuilding the cache.

Discriminates: passes on dev bbfa876af; with the proxy and WebSocket `enabled` checks of
753798923 removed the disabled-terminal proxy and session tests fail (the fake terminal gets the
request and the shell; the list filtered before the fix), with the `enabled` check removed from
`get_terminal_tools` the chat on the disabled terminal is offered its tools, and with 867006acc
reverted the admin tests on the no-bypass instance fail on every entry point (403, a 4003 close,
not listed). Retargeted for 6f2fc7bc7: passes on dev c15f28649; in a backend copy whose
`find_terminal_connection` and `get_terminal_tools` drop their `enabled` checks the
disabled-terminal proxy, session, close and chat tests fail (the fake terminal gets the request
and the shell, the chat is offered `run_command`), and with their access checks dropped the user
rows of the empty-grants tests and the grant-of-their-own test fail (the user reaches the
terminal).

The closed-before-sign-in test pins PR open-webui/open-webui#33863 (fix ac45fc0fc, issue #33850):
a terminal WebSocket the browser closed before sending its sign-in made the server try to close
it again and log an error traceback. Passes on dev e8c0ad85e, fails on a backend copy with
ac45fc0fc reverted (the traceback is in the log).
"""

from __future__ import annotations

import uuid
from typing import Callable, Iterator

import pytest
from websockets.exceptions import ConnectionClosed
from websockets.sync.client import connect

from harness import upstream as reply
from harness.actors import Actor, admin_of, create_user
from harness.chat import ask
from harness.listener import json_answer
from harness.terminal_server import (
    TERMINAL_SERVERS_CONFIG,
    FakeTerminalServer,
    close_of,
    configure_terminals,
    read_grant,
    serving_terminal,
    terminal_session,
)
from integration.stateful_redis import StatefulRedis

pytestmark = [pytest.mark.regression, pytest.mark.api, pytest.mark.requires_source]


@pytest.fixture
def terminal() -> Iterator[FakeTerminalServer]:
    with serving_terminal() as server:
        server.route("GET", "/probe", json_answer({"ok": True}))
        yield server


UNKNOWN_ID = "terminal-that-was-never-added"


def _granted_to(actor: Actor) -> dict:
    return {"access_grants": [read_grant(actor.id)]}


def _listed(caller: Actor, connection: dict, terminal: FakeTerminalServer) -> bool:
    with caller.client() as client:
        listed = client.get("/api/v1/terminals/")
    listed.raise_for_status()
    return any(entry["id"] == connection["id"] for entry in listed.json())


def _proxied(caller: Actor, connection: dict, terminal: FakeTerminalServer) -> bool:
    with caller.client() as client:
        response = client.get(f"/api/v1/terminals/{connection['id']}/probe")
        unknown = client.get(f"/api/v1/terminals/{UNKNOWN_ID}/probe")
    assert unknown.status_code == 404, unknown.text
    reached = bool(terminal.requests_to("/probe"))
    if reached:
        assert response.status_code == 200, response.text
    else:
        refusal = unknown.text.replace(UNKNOWN_ID, connection["id"])
        assert (response.status_code, response.text) == (404, refusal), (
            f"a refused terminal answers {response.status_code} {response.text}, which tells it "
            f"apart from one that does not exist ({unknown.text})"
        )
    return reached


def _session_opened(caller: Actor, connection: dict, terminal: FakeTerminalServer) -> bool:
    with terminal_session(caller.base_url, caller.token, connection["id"]) as session:
        try:
            session.send("echo hello")
            echoed = session.recv(timeout=10)
        except ConnectionClosed:
            echoed = None
    opened = echoed == "echo hello"
    assert bool(terminal.sessions) == opened
    return opened


ENTRY_POINTS: dict[str, Callable[[Actor, dict, FakeTerminalServer], bool]] = {
    "list": _listed,
    "http_proxy": _proxied,
    "websocket": _session_opened,
}


def _save(admin: Actor, terminal: FakeTerminalServer, *connections: dict) -> None:
    with admin.client() as client:
        configure_terminals(client, *connections)
    terminal.clear()


# The shared instance, where BYPASS_ADMIN_ACCESS_CONTROL keeps its default (on).


@pytest.fixture
def save(admin, preserve, terminal) -> Callable[..., None]:
    """`save(*connections)` makes them the shared instance's terminal connections."""
    preserve(TERMINAL_SERVERS_CONFIG)
    return lambda *connections: _save(admin, terminal, *connections)


@pytest.fixture
def member(make_user) -> Actor:
    return make_user()


@pytest.mark.parametrize("entry_point", sorted(ENTRY_POINTS))
def test_no_entry_point_serves_a_disabled_terminal(save, member, terminal, entry_point):
    disabled = terminal.connection(enabled=False, config=_granted_to(member))
    save(disabled)
    assert not ENTRY_POINTS[entry_point](member, disabled, terminal), (
        f"{entry_point} still serves a switched-off terminal to a granted user"
    )


def test_a_disabled_terminal_closes_the_session_like_one_that_does_not_exist(
    save, member, terminal
):
    disabled = terminal.connection(enabled=False, config=_granted_to(member))
    save(disabled)
    closes = {}
    for label, terminal_id in [("disabled", disabled["id"]), ("unknown", UNKNOWN_ID)]:
        with terminal_session(member.base_url, member.token, terminal_id) as session:
            closes[label] = close_of(session, timeout=10)
    assert closes["disabled"] == closes["unknown"] == (4004, "Terminal server not found"), closes
    assert not terminal.sessions, "the switched-off terminal was asked for a shell"


def test_a_session_closed_before_it_signs_in_logs_no_error(instance, save, member, terminal):
    """#33850: a panel closed right after it opened left a traceback in the server log."""
    connection = terminal.connection(config=_granted_to(member))
    save(connection)
    log_offset = instance.log_size()
    url = member.base_url.replace("http://", "ws://", 1) + (
        f"/api/v1/terminals/{connection['id']}/api/terminals/{uuid.uuid4().hex}"
    )
    with connect(url, proxy=None, open_timeout=10, close_timeout=2):
        pass

    # a real sign-in failure still closes the way it did
    with terminal_session(member.base_url, "not-a-token", connection["id"]) as session:
        assert close_of(session, timeout=10) == (4001, "Invalid token")
    log = instance.log_since(log_offset)
    assert "Traceback" not in log, f"a session gone before its sign-in logged an error:\n{log}"
    assert not terminal.sessions, "a session that never signed in reached the terminal"


@pytest.mark.parametrize("entry_point", sorted(ENTRY_POINTS))
def test_an_enabled_terminal_is_served_on_every_entry_point(save, member, terminal, entry_point):
    enabled = terminal.connection(config=_granted_to(member))
    save(enabled)
    assert ENTRY_POINTS[entry_point](member, enabled, terminal)


@pytest.mark.parametrize("entry_point", sorted(ENTRY_POINTS))
def test_a_terminal_saved_without_the_enabled_flag_stays_in_service(
    admin, preserve, member, terminal, entry_point
):
    """Connections saved before the flag existed, as a config import brings them back."""
    preserve(TERMINAL_SERVERS_CONFIG)
    legacy = terminal.connection(config=_granted_to(member))
    del legacy["enabled"]
    with admin.client() as client:
        imported = client.post(
            "/api/v1/configs/import", json={"config": {"terminal_server.connections": [legacy]}}
        )
    assert imported.status_code == 200, imported.text
    assert ENTRY_POINTS[entry_point](member, legacy, terminal)


def test_the_admin_bypass_reaches_a_terminal_granted_to_someone_else(save, admin, terminal):
    connection = terminal.connection(config={"access_grants": [read_grant("someone-else")]})
    save(connection)
    assert _proxied(admin, connection, terminal)


# An instance with BYPASS_ADMIN_ACCESS_CONTROL off.


@pytest.fixture
def strict_instance(instance_with):
    return instance_with({"BYPASS_ADMIN_ACCESS_CONTROL": "false"})


@pytest.fixture
def strict_admin(strict_instance) -> Iterator[Actor]:
    """The no-bypass instance's admin; its terminal connections are restored afterwards."""
    admin = admin_of(strict_instance)
    with admin.client() as client:
        snapshot = client.get(TERMINAL_SERVERS_CONFIG[0])
        snapshot.raise_for_status()
        yield admin
        restored = client.post(TERMINAL_SERVERS_CONFIG[1], json=snapshot.json())
        assert restored.status_code == 200, restored.text


@pytest.fixture
def strict_member(strict_instance) -> Actor:
    return create_user(strict_instance)


def test_the_admin_reaches_a_terminal_without_grants(strict_admin, terminal):
    ungranted = terminal.connection(config={"access_grants": []})
    _save(strict_admin, terminal, ungranted)
    assert _proxied(strict_admin, ungranted, terminal), (
        "a connection with no grants yet was refused to the admin who added it, so nobody "
        "could open it to grant access (#27581)"
    )


@pytest.mark.parametrize("entry_point", sorted(ENTRY_POINTS))
@pytest.mark.parametrize(
    "grants_config",
    [
        pytest.param(None, id="config_none"),
        pytest.param({}, id="no_grants_key"),
        pytest.param({"access_grants": []}, id="empty_grants"),
        pytest.param({"access_grants": None}, id="grants_none"),
    ],
)
def test_every_empty_grants_shape_is_admin_only(
    strict_admin, strict_member, terminal, grants_config, entry_point
):
    ungranted = terminal.connection(config=grants_config)
    _save(strict_admin, terminal, ungranted)
    reaches = ENTRY_POINTS[entry_point]
    assert reaches(strict_admin, ungranted, terminal), f"{entry_point} refused the admin (#27580)"
    terminal.clear()
    assert not reaches(strict_member, ungranted, terminal), f"{entry_point} served a user"


def test_a_user_needs_a_grant_of_their_own(strict_admin, strict_member, terminal):
    granted_elsewhere = terminal.connection(config={"access_grants": [read_grant("someone")]})
    granted_to_member = terminal.connection(config=_granted_to(strict_member))
    _save(strict_admin, terminal, granted_elsewhere, granted_to_member)
    assert not _proxied(strict_member, granted_elsewhere, terminal)
    assert _proxied(strict_member, granted_to_member, terminal)


# A chat on a terminal, on an instance whose terminal list is cached in Redis.

RUN_COMMAND_SPEC = {
    "openapi": "3.0.0",
    "info": {"title": "Terminal", "version": "1"},
    "paths": {
        "/execute": {
            "post": {"operationId": "run_command", "responses": {"200": {"description": "ok"}}}
        }
    },
}


@pytest.fixture(scope="module")
def redis():
    store = StatefulRedis()
    yield store
    store.close()


@pytest.fixture
def cached_terminal(instance_with, redis, preserve):
    """(instance, member, terminal, connection): a terminal granted to the member and cached."""
    launched = instance_with({"REDIS_URL": redis.url})
    preserve(TERMINAL_SERVERS_CONFIG, on=launched)
    member = create_user(launched)
    with serving_terminal() as terminal:
        terminal.route("GET", "/openapi.json", json_answer(RUN_COMMAND_SPEC))
        connection = terminal.connection(config=_granted_to(member))
        _save(admin_of(launched), terminal, connection)
        yield launched, member, terminal, connection


def _switch_off_by_import(launched, connection: dict) -> None:
    with launched.client() as client:
        imported = client.post(
            "/api/v1/configs/import",
            json={"config": {"terminal_server.connections": [{**connection, "enabled": False}]}},
        )
    assert imported.status_code == 200, imported.text


def _chat_on_terminal(launched, member: Actor, connection: dict) -> tuple[dict, list[str]]:
    """The stored reply and the tools the model was offered; none if it was never asked."""
    launched.upstream.queue(reply.text("ready"))
    with member.client() as client:
        _, message = ask(client, "list my files", terminal_id=connection["id"])
    requests = launched.upstream.chat_requests()
    offered = [
        tool["function"]["name"] for request in requests for tool in request.get("tools", [])
    ]
    return message, offered


@pytest.mark.slow
def test_a_chat_gets_no_tools_from_a_disabled_terminal_still_cached(cached_terminal):
    launched, member, terminal, connection = cached_terminal
    _switch_off_by_import(launched, connection)

    message, offered = _chat_on_terminal(launched, member, connection)

    assert "run_command" not in offered, "a switched-off terminal still hands its tools to a chat"
    assert terminal.received == [], "the chat reached the switched-off terminal"
    error = (message.get("error") or {}).get("content", "")
    assert "Terminal unavailable" in error and "not found" in error, message


@pytest.mark.slow
def test_a_chat_gets_the_tools_of_an_enabled_cached_terminal(cached_terminal):
    launched, member, terminal, connection = cached_terminal

    message, offered = _chat_on_terminal(launched, member, connection)

    assert message["content"] == "ready", message
    assert "run_command" in offered
    assert terminal.requests_to("/openapi.json") == [], "the cached list was not used"
