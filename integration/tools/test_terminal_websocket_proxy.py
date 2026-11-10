"""Journey: a shell session through Open WebUI carries what the terminal sends and how it ends.

The page opens a terminal's shell on `/api/v1/terminals/<id>/api/terminals/<session>` and signs
in with its first message. Open WebUI opens the same session on the terminal and relays both
ways: text and binary output arrive as the terminal sent them, in order, and what the person
types reaches the terminal. When the terminal closes the session, the page's session closes with
the terminal's code. A close without a code and a terminal that hangs up cannot be passed on as
they are (1005 and 1006 are never sent), so the page gets 1011 and reconnects. A terminal that
refuses the session outright closes the page's session with 4001 for 401, 4003 for 403, 4004
for 404 and 1011 for anything else.

Every other stream of a terminal goes the same way under its own path: live file events,
the process list, one process's output, the desktop's screen and its sound. The terminal is
asked in the person's name (`user_id` in the address, `X-User-Id`) and for the chat the stream
belongs to (`X-Session-Id`), whatever the page put in its own address; only the output stream
of a process passes `offset` and `tail` on. A shell's session id reaches the terminal as one
path segment, so an id with `?` or `#` in it cannot add to what the terminal is asked. The
terminal is signed in to with the connection's own key (Bearer) or with no token at all, never
with the person's token, and gets the rest of the page's sign-in message as it was (the resume
cursor of a shell).

A stream that does not sign in with valid JSON within ten seconds is closed with 4001 before
the terminal hears of it. A stream for a saved chat the person cannot write to is refused with
4003 and the reason, as is every stream while tool servers are switched off and one on a
connection without an address; a terminal that is down ends it with 1011. An open stream is
checked again every ten seconds: once its chat is no longer shared with the person it ends
with 4003. A shell, and only a shell, is reported to the admin's event functions when it opens
and when it closes.

However a page leaves (a close, a connection cut without one, a reset, a killed browser, gone
right after signing in), Open WebUI closes its own connection to the terminal: the terminal
holds exactly the streams of the pages still connected, however many came and went. That goes
for a Browser tab's stream as well, which is relayed under the token Open WebUI issued for it.

The terminal is the fake of `harness/terminal_server.py` with a scripted session
(`script_sessions`, `refuse_sessions`).

Discriminates: on dev c15f28649, in a backend copy: relaying only text output fails the relay
test (the binary message never arrives); closing the page's session with 1011 whatever the
terminal's code fails the relay test and the 1000, 4000 and 4004 cases; passing the terminal's
code on unmapped fails the two cases without a code; reading every refused session as 1011
fails the 401, 403 and 404 cases. For the tests of every stream, the sign-in, the refusals,
the recheck, the reports and the pages that leave, each in a backend copy of its own: not
passing `offset` and `tail` on fails the process output case; taking `user_id` from the page's
address fails every stream case; the session id put into the terminal's address unencoded
fails the crafted id case; an empty `X-Session-Id` fails the stream, sign-in and chat refusal
tests; dropping the page's binary or its text messages fails every stream case; sending the
page's sign-in message on unchanged, or only its type and token, fails the four sign-in cases;
waiting 60 s for the sign-in fails the silent case and treating a sign-in that is not JSON
like a bad token the other; closing a refused resolution like an unknown terminal, or not
checking the chat, fails the chat refusal and the unshared chat tests; starting no recheck
fails the unshared chat test; trying a connection without an address fails that case and
closing a failed connection with 1000 the terminal-down case; ignoring the tool servers switch
in both places that check it fails the switched-off test; reporting no opened shell, no closed
shell, or every stream fails the report test; ending the relay only when both directions have
ended (what Terminals 0.3.1 did) fails all five ways of leaving and the stream cases, and the
same in the Browser tab relay fails the four ways of leaving after the greeting.
"""

from __future__ import annotations

import json
import time
import uuid
from contextlib import ExitStack
from typing import Callable
from urllib.parse import parse_qs, quote, unquote, urlsplit

import pytest
from websockets.sync.client import ClientConnection
from websockets.sync.client import connect as connect_socket

from harness.actors import create_user
from harness.chat_history import seed_chat
from harness.instance import free_port
from harness.listener import json_answer
from harness.plugins import installed_function
from harness.terminal_server import (
    TERMINAL_SERVERS_CONFIG,
    BarePage,
    TerminalSession,
    close_of,
    configure_terminals,
    pages_of_another_process,
    read_grant,
    serving_terminal,
    stream_url,
    terminal_session,
    terminal_socket,
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
def share_terminal(admin, preserve, terminal) -> Callable[..., str]:
    """`share_terminal(*people, **fields)`: the admin saves the fake terminal as the only
    terminal connection, shared with those people; returns its id."""
    preserve(TERMINAL_SERVERS_CONFIG)

    def share(*people, **fields) -> str:
        grants = [read_grant(person.id) for person in people]
        connection = terminal.connection(**{"config": {"access_grants": grants}, **fields})
        with admin.client() as client:
            configure_terminals(client, connection)
        return connection["id"]

    return share


@pytest.fixture
def person_with_terminal(make_user, share_terminal):
    """A person and the id of the fake terminal the admin shared with them."""
    person = make_user()
    return person, share_terminal(person, key=KEY, auth_type="bearer")


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


# every stream, and what the terminal is asked

KEYSTROKES = b"ls -la\r\x03"
RECHECK_BOUND_SECONDS = 13
RELEASE_BOUND_SECONDS = 5

STREAMS = {
    "shell": "api/terminals/shell-{tag}",
    "shell-with-a-crafted-id": "api/terminals/"
    + quote("s-{tag}?user_id=someone-else&tail=1#x y", safe="{}"),
    "file-events": "events",
    "process-list": "execute",
    "process-output": "execute/run-{tag}/status",
    "desktop-screen": "desktop/stream",
    "desktop-sound": "desktop/audio",
}


def _answering_back(greeting: str):
    """A stream that greets with a text and a binary message, then sends back what it gets."""

    def stream(session: TerminalSession) -> None:
        session.authenticated()
        session.send(greeting)
        session.send(BINARY_OUTPUT)
        while True:
            session.send(session.receive(timeout=120))

    return stream


def _saved_chat(person, about: str) -> str:
    with person.client() as client:
        chat_id, _ = seed_chat(client, [{"role": "user", "content": about}])
    return chat_id


def _is_live(page_side: ClientConnection) -> bool:
    question = f"still there {uuid.uuid4().hex[:8]}?"
    page_side.send(question)
    return page_side.recv(timeout=10) == question


def _eventually(read, timeout: float = 10):
    """The first truthy `read()` within the timeout, else its last value."""
    deadline = time.monotonic() + timeout
    value = read()
    while not value and time.monotonic() < deadline:
        time.sleep(0.1)
        value = read()
    return value


@pytest.mark.parametrize("stream", sorted(STREAMS))
def test_every_stream_of_a_terminal_is_relayed_and_asked_for_in_the_persons_name(
    share_terminal, make_user, terminal, stream
):
    person = make_user()
    terminal_id = share_terminal(person)
    tag = uuid.uuid4().hex[:8]
    chat_id = _saved_chat(person, f"a chat about {tag}")
    path = STREAMS[stream].format(tag=tag)
    terminal.script_sessions(_answering_back(f"hello {tag}"))

    with terminal_socket(
        person.base_url,
        person.token,
        terminal_id,
        f"{path}?offset=5&tail=7&user_id=someone-else",
        chat_id=chat_id,
    ) as page_side:
        greeting = page_side.recv(timeout=10)
        coloured = page_side.recv(timeout=10)
        page_side.send(f"typed {tag}")
        page_side.send(KEYSTROKES)
        sent_back = [page_side.recv(timeout=10), page_side.recv(timeout=10)]

    assert len(terminal.sessions) == 1, (
        f"the terminal got {[session.path for session in terminal.sessions]} for one {stream}"
    )
    [upstream] = terminal.sessions
    asked = urlsplit(upstream.path)
    expected_query = {"user_id": [person.id]}
    if stream == "process-output":
        expected_query |= {"offset": ["5"], "tail": ["7"]}
    assert unquote(asked.path) == unquote(f"/{path}"), (
        f"the {stream} stream was opened on the terminal as {upstream.path}"
    )
    assert "?" not in asked.path and "#" not in asked.path, (
        f"the session id split the address the terminal was asked for: {upstream.path}"
    )
    assert parse_qs(asked.query) == expected_query, (
        f"the terminal was asked for the {stream} stream with {asked.query!r}"
    )
    assert (upstream.headers.get("x-user-id"), upstream.headers.get("x-session-id")) == (
        person.id,
        chat_id,
    ), f"the terminal was told person and chat as {upstream.headers}"
    assert (greeting, coloured) == (f"hello {tag}", BINARY_OUTPUT), (
        f"the terminal's output arrived as {greeting!r} then {coloured!r}"
    )
    assert upstream.messages == [f"typed {tag}", KEYSTROKES], (
        f"the terminal got {upstream.messages} for a typed line and raw keystrokes"
    )
    assert sent_back == [f"typed {tag}", KEYSTROKES], f"the answers arrived as {sent_back}"
    assert upstream.ended.wait(RELEASE_BOUND_SECONDS), (
        f"the {stream} stream stayed open on the terminal after the page had closed it"
    )


@pytest.mark.parametrize(
    "auth_type,expected_token",
    [("bearer", KEY), ("none", ""), ("session", ""), ("system_oauth", "")],
)
def test_the_terminal_is_signed_in_to_with_its_own_key_and_never_the_persons_token(
    share_terminal, make_user, terminal, auth_type, expected_token
):
    person = make_user()
    terminal_id = share_terminal(person, key=f"  {KEY} \n", auth_type=auth_type)
    tag = uuid.uuid4().hex[:8]
    sign_in = {"chat_id": f"temporary:{tag}", "resume": True, "cursor": 7, "flow_control": True}
    terminal.clear()

    with terminal_socket(
        person.base_url, person.token, terminal_id, f"api/terminals/shell-{tag}", **sign_in
    ) as page_side:
        live = _is_live(page_side)

    assert live, "the shell did not answer"
    [upstream] = terminal.sessions
    assert upstream.auth == {"type": "auth", **sign_in, "token": expected_token}, (
        f"a terminal with {auth_type} authentication was signed in to with {upstream.auth}"
    )
    sent = json.dumps([upstream.path, upstream.headers, upstream.auth])
    assert person.token not in sent, f"the person's own token was sent to the terminal: {sent}"
    assert upstream.headers.get("x-session-id") == sign_in["chat_id"], upstream.headers


# refusals


@pytest.mark.parametrize(
    "first_message", ["{not json", None], ids=["not-json", "nothing-for-ten-seconds"]
)
def test_a_stream_that_does_not_sign_in_is_closed_before_the_terminal_hears_of_it(
    person_with_terminal, terminal, first_message
):
    person, terminal_id = person_with_terminal
    terminal.clear()
    url = person.base_url.replace("http://", "ws://", 1) + (
        f"/api/v1/terminals/{terminal_id}/api/terminals/{uuid.uuid4().hex}"
    )

    with connect_socket(url, proxy=None, open_timeout=10, close_timeout=2) as page_side:
        if first_message is not None:
            page_side.send(first_message)
        closed = close_of(page_side, timeout=15)
    with terminal_session(person.base_url, person.token, terminal_id) as page_side:
        signed_in = _is_live(page_side)

    assert closed == (4001, "Auth timeout or invalid payload"), (
        f"a stream that sent {first_message!r} for its sign-in ended with {closed}"
    )
    assert signed_in, "the same shell did not open with a proper sign-in"
    assert len(terminal.sessions) == 1, (
        f"the terminal got {len(terminal.sessions)} sessions, one of them never signed in"
    )


def test_a_stream_for_a_chat_the_person_cannot_write_to_is_refused_saying_so(
    share_terminal, make_user, terminal
):
    owner, stranger = make_user(), make_user()
    terminal_id = share_terminal(owner, stranger)
    chat_id = _saved_chat(owner, "the owner's chat")
    terminal.clear()

    with terminal_socket(
        owner.base_url, owner.token, terminal_id, "events", chat_id=chat_id
    ) as page_side:
        owner_live = _is_live(page_side)
    with terminal_socket(
        stranger.base_url, stranger.token, terminal_id, "events", chat_id=chat_id
    ) as page_side:
        refused = close_of(page_side, timeout=10)
    with terminal_socket(stranger.base_url, stranger.token, terminal_id, "events") as page_side:
        stranger_live = _is_live(page_side)

    assert owner_live, "the chat's owner got no stream for their own chat"
    assert stranger_live, "the other person got no stream of the shared terminal at all"
    assert refused == (4003, "Chat access denied"), (
        f"a stream for another person's chat ended with {refused}"
    )
    asked_as = [
        (session.headers["x-user-id"], session.headers["x-session-id"])
        for session in terminal.sessions
    ]
    assert asked_as == [(owner.id, chat_id), (stranger.id, "")], (
        f"the terminal was asked for streams as {asked_as}"
    )


@pytest.mark.parametrize(
    "address,expected",
    [
        pytest.param(None, (1011, ""), id="terminal-down"),
        pytest.param("", (4003, "Terminal server URL not configured"), id="no-address"),
    ],
)
def test_a_session_on_a_terminal_that_cannot_be_reached_ends_with_the_code_for_it(
    share_terminal, make_user, address, expected
):
    person = make_user()
    url = f"http://127.0.0.1:{free_port()}" if address is None else address
    terminal_id = share_terminal(person, url=url)

    with terminal_session(person.base_url, person.token, terminal_id) as page_side:
        closed = close_of(page_side, timeout=10)

    assert closed == expected, f"a session on a terminal at {url!r} ended with {closed}"


@pytest.mark.slow
def test_no_stream_opens_while_tool_servers_are_switched_off(
    instance_with, preserve, terminal, person_with_terminal
):
    switched_off = instance_with({"ENABLE_TOOL_SERVERS": "false"})
    person, terminal_id = person_with_terminal
    preserve(TERMINAL_SERVERS_CONFIG, on=switched_off)
    member = create_user(switched_off)
    connection = terminal.connection(
        id=terminal_id, config={"access_grants": [read_grant(member.id)]}
    )
    with switched_off.client() as client:
        configure_terminals(client, connection)
    terminal.clear()

    closes = {}
    for path in ("api/terminals/shell", "events", "execute"):
        with terminal_socket(switched_off.base_url, member.token, terminal_id, path) as page_side:
            closes[path] = close_of(page_side, timeout=10)
    with terminal_session(person.base_url, person.token, terminal_id) as page_side:
        live_where_they_are_on = _is_live(page_side)

    assert live_where_they_are_on, "the same terminal gave no shell with tool servers on"
    assert set(closes.values()) == {(4003, "Tool servers are disabled")}, (
        f"with tool servers switched off the streams ended with {closes}"
    )
    assert len(terminal.sessions) == 1, (
        f"the terminal got {len(terminal.sessions)} streams, some from the instance whose tool "
        "servers are off"
    )


def test_an_open_shell_ends_once_its_chat_is_no_longer_shared_with_the_person(
    share_terminal, make_user, terminal
):
    owner, guest = make_user(), make_user()
    terminal_id = share_terminal(owner, guest)
    chat_id = _saved_chat(owner, "a chat the owner lets a guest continue")
    with owner.client() as client:
        shared = client.post(f"/api/v1/chats/{chat_id}/share", json={"share_mode": None})
        assert shared.status_code == 200, shared.text
        granted = client.post(
            f"/api/v1/chats/shared/{chat_id}/access/update",
            json={"access_grants": [read_grant(guest.id)], "share_mode": "continue"},
        )
        assert granted.status_code == 200, granted.text
    terminal.clear()

    with ExitStack() as stack:
        # opened first, the owner's shell is checked first, so it has passed once the guest's ends
        owners, guests = (
            stack.enter_context(
                terminal_socket(
                    person.base_url,
                    person.token,
                    terminal_id,
                    f"api/terminals/{uuid.uuid4().hex}",
                    chat_id=chat_id,
                )
            )
            for person in (owner, guest)
        )
        assert _is_live(owners) and _is_live(guests), "the chat's two shells did not both open"

        with owner.client() as client:
            unshared = client.delete(f"/api/v1/chats/{chat_id}/share")
        assert unshared.status_code == 200, unshared.text
        closed = close_of(guests, timeout=RECHECK_BOUND_SECONDS)

        assert closed == (4003, "Chat access denied"), (
            f"the guest's shell on a chat no longer shared with them ended with {closed}"
        )
        assert _is_live(owners), "the owner's shell was closed with the guest's"
        [guests_shell] = [
            shell for shell in terminal.sessions if shell.headers["x-user-id"] == guest.id
        ]
        assert guests_shell.ended.wait(RELEASE_BOUND_SECONDS), (
            "the guest's shell stayed open on the terminal after Open WebUI had ended it"
        )


# audit events

FORWARD_SESSION_EVENTS = """
import json
import urllib.request


class Event:
    def event(self, event):
        if event["event"].startswith("terminal.session."):
            body = json.dumps(event).encode()
            headers = {{"Content-Type": "application/json"}}
            request = urllib.request.Request("{url}", data=body, headers=headers)
            urllib.request.urlopen(request, timeout=10).close()
"""


def test_a_shell_is_reported_when_it_opens_and_when_it_closes_and_no_other_stream_is(
    admin, person_with_terminal, listener
):
    person, terminal_id = person_with_terminal
    session_id = f"shell-{uuid.uuid4().hex[:8]}"
    listener.route("POST", "/events", json_answer({}))

    def reported(name: str) -> list[tuple]:
        events = [request.json() for request in listener.requests_to("/events")]
        return [
            (event["subject"], event["actor"]["id"])
            for event in events
            if event["event"] == name and event["data"] == {"server_id": terminal_id}
        ]

    source = FORWARD_SESSION_EVENTS.format(url=f"{listener.base_url}/events")
    with installed_function(admin, source):
        with terminal_socket(person.base_url, person.token, terminal_id, "events") as page_side:
            file_events_live = _is_live(page_side)
        with terminal_socket(
            person.base_url, person.token, terminal_id, f"api/terminals/{session_id}"
        ) as page_side:
            shell_live = _is_live(page_side)
            opened = _eventually(lambda: reported("terminal.session.opened"))
            closed_while_open = reported("terminal.session.closed")
        closed = _eventually(lambda: reported("terminal.session.closed"))

    the_shell = ({"type": "terminal.session", "id": session_id}, person.id)
    assert file_events_live and shell_live, "the two streams did not both open"
    assert opened == [the_shell], f"the opened shell was reported as {opened}"
    assert closed_while_open == [], f"a shell still open was reported closed: {closed_while_open}"
    assert closed == [the_shell], f"the closed shell was reported as {closed}"


# what a page that leaves lets go of

COMINGS_AND_GOINGS = 24
LEAVING = {
    "closes": BarePage.close,
    "is-cut-off": BarePage.cut,
    "is-reset": BarePage.reset,
    "leaves-right-after-signing-in": BarePage.close,
    "is-killed": None,
}


def _held_by_the_terminal(terminal) -> list[str]:
    """The pages whose streams are open on the terminal right now, by the name they signed in as."""
    with terminal.lock:
        sessions = list(terminal.sessions)
    return sorted(
        (session.auth or {}).get("page", "not signed in")
        for session in sessions
        if not session.ended.is_set()
    )


def _browser_tab_stream(person, terminal, terminal_id: str) -> str:
    """The path of a Browser tab's stream, under the token Open WebUI issues for the tab."""
    tag = uuid.uuid4().hex[:8]
    tab = {"id": f"tab-{tag}", "token": f"capability-{tag}"}
    terminal.route("POST", "/browser/sessions", json_answer(tab))
    with person.client() as client:
        opened = client.post(f"/api/v1/terminals/{terminal_id}/browser/sessions", json={})
    assert opened.status_code == 200, f"opening a Browser tab failed: {opened.text}"
    return f"browser/view/{tab['id']}/{opened.json()['token']}/ws"


@pytest.mark.parametrize("leaving", sorted(LEAVING))
def test_the_terminal_holds_only_the_streams_of_the_pages_still_connected(
    person_with_terminal, terminal, leaving
):
    """A higher limit is a stopgap: the count must return to baseline."""
    person, terminal_id = person_with_terminal
    greeting = _answering_back("open")

    def stream(session: TerminalSession) -> None:
        # a page gone before its stream opened finds a terminal with nothing to say to it
        gone_at_once = leaving == "leaves-right-after-signing-in"
        if gone_at_once and session.authenticated().get("page", "").startswith("left-"):
            session.receive(timeout=120)
        greeting(session)

    terminal.script_sessions(stream)
    paths = [path.format(tag=uuid.uuid4().hex[:8]) for path in STREAMS.values()]
    # a Browser tab's stream is relayed on its own, and its first message goes to the terminal
    paths.append(_browser_tab_stream(person, terminal, terminal_id))
    terminal.clear()

    def url(number: int) -> str:
        return stream_url(person.base_url, terminal_id, paths[number % len(paths)])

    def sign_in(name: str) -> dict:
        return {"type": "auth", "token": person.token, "page": name}

    def page(name: str, number: int) -> BarePage:
        opened = BarePage(url(number), sign_in(name))
        # both halves of the greeting, so the terminal has nothing left to send when it leaves
        greeting = (opened.receive(), opened.receive())
        assert greeting == ("open", BINARY_OUTPUT), f"{name} got {greeting} on {url(number)}"
        return opened

    staying = {name: page(name, number) for number, name in enumerate(["stays-0", "stays-1"])}
    try:
        if leaving == "is-killed":
            pages = [
                (url(number), sign_in(f"left-{number}")) for number in range(COMINGS_AND_GOINGS)
            ]
            with pages_of_another_process(pages, messages=2) as browser:
                held_meanwhile = _held_by_the_terminal(terminal)
                browser.kill()
            assert len(held_meanwhile) == COMINGS_AND_GOINGS + len(staying), (
                f"before the other browser was killed the terminal held {held_meanwhile}"
            )
        elif leaving == "leaves-right-after-signing-in":
            for number in range(COMINGS_AND_GOINGS):
                BarePage(url(number), sign_in(f"left-{number}")).close()
        else:
            for number in range(COMINGS_AND_GOINGS):
                LEAVING[leaving](page(f"left-{number}", number))
        staying["stays-late"] = page("stays-late", 2)

        _eventually(
            lambda: _held_by_the_terminal(terminal) == sorted(staying),
            timeout=RELEASE_BOUND_SECONDS,
        )
        # a stream opened for a page that had already gone would show up only now
        time.sleep(1)
        held = _held_by_the_terminal(terminal)
        answers = {}
        for name, page_side in staying.items():
            page_side.send(f"still there, {name}?")
            answers[name] = page_side.receive()
    finally:
        for page_side in staying.values():
            page_side.cut()

    came_and_went = len(terminal.sessions) - len(staying)
    browser_tabs = [
        session for session in terminal.sessions if session.path.startswith("/browser/view/")
    ]
    if leaving != "leaves-right-after-signing-in":
        assert came_and_went == COMINGS_AND_GOINGS, (
            f"{came_and_went} of the {COMINGS_AND_GOINGS} streams that came and went reached "
            "the terminal"
        )
        assert len(browser_tabs) == COMINGS_AND_GOINGS // len(paths), (
            f"{len(browser_tabs)} Browser tab streams reached the terminal"
        )
    assert held == sorted(staying), (
        f"{COMINGS_AND_GOINGS} pages came and went ({leaving}), {sorted(staying)} are still "
        f"connected, and the terminal holds the streams of {held}"
    )
    assert answers == {name: f"still there, {name}?" for name in staying}, (
        f"the pages that stayed got {answers} from their streams"
    )
