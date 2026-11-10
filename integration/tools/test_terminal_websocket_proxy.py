"""Journey: a shell session through Open WebUI carries what the terminal sends and how it ends.

The page opens a terminal's shell on `/api/v1/terminals/<id>/api/terminals/<session>` and signs
in with its first message. Open WebUI opens the same session on the terminal and relays both
ways: text and binary output arrive as the terminal sent them, in order, and what the person
types reaches the terminal. When the terminal closes the session, the page's session closes with
the terminal's code. A close without a code and a terminal that hangs up cannot be passed on as
they are (1005 and 1006 are never sent), so the page gets 1011 and reconnects. A terminal that
refuses the session outright closes the page's session with 4001 for 401, 4003 for 403, 4004
for 404 and 1011 for anything else.

The terminal is the fake of `harness/terminal_server.py` with a scripted session
(`script_sessions`, `refuse_sessions`).

Discriminates: on dev c15f28649, in a backend copy: relaying only text output fails the relay
test (the binary message never arrives); closing the page's session with 1011 whatever the
terminal's code fails the relay test and the 1000, 4000 and 4004 cases; passing the terminal's
code on unmapped fails the two cases without a code; reading every refused session as 1011
fails the 401, 403 and 404 cases.
"""

from __future__ import annotations

import uuid

import pytest

from harness.listener import json_answer
from harness.terminal_server import (
    TERMINAL_SERVERS_CONFIG,
    TerminalSession,
    close_of,
    configure_terminals,
    read_grant,
    serving_terminal,
    terminal_session,
)

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source]

KEY = "sk-shell-relay"
BINARY_OUTPUT = b"\x1b[32mgreen\x1b[0m\x00\xff"


@pytest.fixture
def terminal():
    with serving_terminal() as server:
        server.route("GET", "/openapi.json", json_answer({"openapi": "3.1.0", "paths": {}}))
        yield server


@pytest.fixture
def person_with_terminal(admin, make_user, preserve, terminal):
    """A person and the id of the fake terminal the admin shared with them."""
    preserve(TERMINAL_SERVERS_CONFIG)
    person = make_user()
    connection = terminal.connection(
        key=KEY, auth_type="bearer", config={"access_grants": [read_grant(person.id)]}
    )
    with admin.client() as client:
        configure_terminals(client, connection)
    return person, connection["id"]


def test_the_shell_gets_what_the_terminal_prints_and_the_terminal_what_is_typed(
    person_with_terminal, terminal
):
    person, terminal_id = person_with_terminal
    tag = uuid.uuid4().hex[:8]

    def shell(session: TerminalSession) -> None:
        session.authenticated()
        session.send(f"prompt {tag} $ ")
        session.send(BINARY_OUTPUT)
        typed = session.receive()
        session.send(f"ran {typed}")
        session.close(4000, "the shell exited")

    terminal.script_sessions(shell, "/api/terminals/*")
    with terminal_session(person.base_url, person.token, terminal_id) as session:
        prompt = session.recv(timeout=10)
        coloured = session.recv(timeout=10)
        session.send(f"whoami {tag}")
        answered = session.recv(timeout=10)
        closed = close_of(session, timeout=10)

    [upstream] = terminal.sessions
    assert upstream.failure is None, f"the scripted shell failed: {upstream.failure!r}"
    assert (prompt, coloured) == (f"prompt {tag} $ ", BINARY_OUTPUT), (
        f"the terminal's output arrived as {prompt!r} then {coloured!r}"
    )
    assert upstream.messages == [f"whoami {tag}"], (
        f"the terminal was sent {upstream.messages} for what the person typed"
    )
    assert answered == f"ran whoami {tag}", f"the shell's answer arrived as {answered!r}"
    assert closed is not None and closed[0] == 4000, (
        f"the terminal closed the session with 4000, the page's session ended with {closed}"
    )
    assert upstream.auth["token"] == KEY, f"the terminal was signed in to with {upstream.auth}"


def _closing_with(code: int | None):
    def shell(session: TerminalSession) -> None:
        session.authenticated()
        session.send("bye")
        session.close(code)

    return shell


def _hanging_up(session: TerminalSession) -> None:
    session.authenticated()
    session.send("bye")
    session.drop()


@pytest.mark.parametrize(
    "shell,expected",
    [
        pytest.param(_closing_with(1000), 1000, id="normal-close"),
        pytest.param(_closing_with(4000), 4000, id="terminal-code-4000"),
        pytest.param(_closing_with(4004), 4004, id="terminal-code-4004"),
        pytest.param(_closing_with(None), 1011, id="close-without-a-code"),
        pytest.param(_hanging_up, 1011, id="hangs-up"),
    ],
)
def test_the_session_ends_with_the_terminals_close_code_or_1011_without_one(
    person_with_terminal, terminal, shell, expected
):
    person, terminal_id = person_with_terminal
    terminal.script_sessions(shell)

    with terminal_session(person.base_url, person.token, terminal_id) as session:
        last_output = session.recv(timeout=10)
        closed = close_of(session, timeout=10)

    [upstream] = terminal.sessions
    assert upstream.failure is None, f"the scripted shell failed: {upstream.failure!r}"
    assert last_output == "bye", f"the output before the close arrived as {last_output!r}"
    assert closed is not None, "the page's session stayed open after the terminal ended it"
    assert closed[0] == expected, f"the page's session ended with {closed}, not {expected}"


@pytest.mark.parametrize(
    "status,expected",
    [(401, 4001), (403, 4003), (404, 4004), (500, 1011)],
    ids=["unauthorized", "forbidden", "not-found", "server-error"],
)
def test_a_session_the_terminal_refuses_ends_with_the_code_for_its_status(
    person_with_terminal, terminal, status, expected
):
    person, terminal_id = person_with_terminal
    terminal.refuse_sessions(status, "/api/terminals/*")
    terminal.clear()

    with terminal_session(person.base_url, person.token, terminal_id) as session:
        closed = close_of(session, timeout=10)

    tries = [entry for entry in terminal.received if entry.path.startswith("/api/terminals/")]
    assert len(tries) == 1, f"the terminal was asked for the session {len(tries)} times"
    assert tries[0].headers["x-user-id"] == person.id, tries[0].headers
    assert not terminal.sessions, "a refused session was opened on the terminal"
    assert closed is not None, f"the page's session stayed open after the terminal's {status}"
    assert closed[0] == expected, f"a {status} ended the page's session with {closed}"
