"""A terminal connection serves nobody while disabled and only the admin while ungranted.

Two fixes in 0.11.0. `753798923`: switching a terminal connection off only hid it from the list;
anyone who knew its id kept using it through the HTTP proxy and the interactive WebSocket. The
fix refuses a disabled connection on every entry point. `867006acc` (PR #27581, issues #27580
and #27064): with BYPASS_ADMIN_ACCESS_CONTROL off, a connection without access grants, which is
every connection right after it is added, was refused to everyone including the admin who
created it; the fix makes it admin-only. Each entry point is driven against a fake terminal
server that records whether anything reached it. Twin of
unit/security/test_terminal_connection_gating.py.

Discriminates: passes on dev bbfa876af; with the proxy and WebSocket `enabled` checks of
753798923 removed the disabled-terminal proxy and session tests fail (the fake terminal gets the
request and the shell; the list filtered before the fix), and with 867006acc reverted the admin
tests on the no-bypass instance fail on every entry point (403, a 4003 close, not listed).
"""

from __future__ import annotations

from typing import Callable, Iterator

import pytest
from websockets.exceptions import ConnectionClosed

from harness.actors import Actor, admin_of, create_user
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

pytestmark = [pytest.mark.regression, pytest.mark.api, pytest.mark.requires_source]


@pytest.fixture
def terminal() -> Iterator[FakeTerminalServer]:
    with serving_terminal() as server:
        server.route("GET", "/probe", json_answer({"ok": True}))
        yield server


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
    reached = bool(terminal.requests_to("/probe"))
    assert response.status_code == (200 if reached else 403), response.text
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


def test_a_disabled_terminal_closes_the_session_saying_so(save, member, terminal):
    disabled = terminal.connection(enabled=False, config=_granted_to(member))
    save(disabled)
    with terminal_session(member.base_url, member.token, disabled["id"]) as session:
        code, reason = close_of(session, timeout=10) or (None, "")
    assert code == 4003 and "disabled" in reason.lower(), (code, reason)


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
