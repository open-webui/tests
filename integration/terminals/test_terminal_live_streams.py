"""Journey: a real shell, live file events and command output through Open WebUI's stream proxy.

The fake terminal of `integration/tools/test_terminal_websocket_proxy.py` pins what the proxy
passes on. Here the other end is real: a plain Open Terminal, and a terminal of a Terminals
orchestrator, where the stream goes through Terminals to the person's own machine. A person
opens a shell, runs commands in it, resizes the window and the shell follows, leaves and comes
back to the same shell at the place they left it (the resume cursor of the sign-in message
reaches the terminal). Live file events arrive for a folder they watch, the list of running
commands follows a command to its end, and a command's output stream gives its lines, from an
offset or only its tail when asked.

An Open Terminal shell takes one page at a time. So a shell is free for its next page only once
Open WebUI has let go of the connection it held for the last one: however that page left (a
close, a cut connection, a reset, a killed browser), the next page gets the shell within
seconds, round after round.

On a Terminals orchestrator a stream is refused with 4003 and the reason where the person may
not use the terminal: the connection itself picked in place of one of its terminals, a terminal
not shared with them, a terminal with a machine per chat without a saved chat. A shell lands in
the machine of its person, or of its chat on such a terminal. An open shell ends with 4003
within the ten seconds of the access recheck once the terminal is no longer shared with its
person, while the shell of a person who kept access goes on.

Terminals 0.4.1 closes the page's stream with 1000 whatever code Open Terminal ended it with,
so nothing here asserts an Open Terminal close code on an orchestrator terminal.

Discriminates: on dev c15f28649 with Terminals 0.4.1 and Open Terminal 0.17.7, in a backend
copy of its own each: ending the relay only when both directions have ended (what Terminals
0.3.1 did) fails the shell test and all eight cases of the next page (the shell stays taken);
dropping the page's binary messages, or sending only type and token of the sign-in message on,
fails the shell test on both terminals (nothing is typed; the output comes without positions);
dropping the page's text messages fails the shell test (the window size) and the file events
test (the folder is never watched); not passing `offset` and `tail` on fails the file events
test; an empty `X-Session-Id` and leaving the template and context headers out fail the
machine test, the latter the refusal test too; closing a refused resolution like an unknown
terminal fails the refusal and the unshared terminal tests; starting no recheck fails the
unshared terminal test.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Callable, Iterator

import pytest
from websockets.exceptions import ConnectionClosed

from harness.actors import Actor
from harness.chat_history import seed_chat
from harness.terminal_server import (
    TERMINAL_SERVERS_CONFIG,
    BarePage,
    configure_terminals,
    pages_of_another_process,
    read_grant,
    stream_url,
    terminal_socket,
)
from harness.terminals import create_template, execute, output_of, share_template

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source]

RELEASE_BOUND_SECONDS = 5
RECHECK_BOUND_SECONDS = 13
ROUNDS = 5


class Shell:
    """A page's side of a resumed shell: what it shows so far and how far its output was read."""

    def __init__(self, page_side) -> None:
        self.page_side = page_side
        self.screen = b""
        self.announced: list[dict] = []
        self.cursor: int | None = None
        own_receive = getattr(page_side, "receive", None)
        self._receive = own_receive or (lambda timeout: page_side.recv(timeout=timeout))

    def type(self, line: str) -> None:
        self.page_side.send(line.encode() + b"\n")

    def resize(self, columns: int, rows: int) -> None:
        self.page_side.send(json.dumps({"type": "resize", "cols": columns, "rows": rows}))

    def shows(self, text: str, timeout: float = 15) -> bool:
        """Read output until the screen holds `text`; False when it does not come in time."""
        deadline = time.monotonic() + timeout
        while text.encode() not in self.screen:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            try:
                message = self._receive(remaining)
            except TimeoutError:
                return False
            if isinstance(message, bytes):
                self.screen += message
                continue
            announced = json.loads(message)
            if announced.get("type") == "output":
                self.announced.append(announced)
                self.cursor = announced["end"]
        return True


@pytest.fixture(params=["open-terminal", "orchestrator-terminal"])
def a_terminal(request, admin, make_user, preserve) -> tuple[Actor, str]:
    """A person and a real terminal shared with them: a plain Open Terminal, or a terminal of a
    Terminals orchestrator with a machine per person."""
    person = make_user()
    if request.param == "orchestrator-terminal":
        return person, request.getfixturevalue("shared_terminal")(person)
    open_terminal = request.getfixturevalue("open_terminal")
    preserve(TERMINAL_SERVERS_CONFIG)
    connection = open_terminal.connection(config={"access_grants": [read_grant(person.id)]})
    with admin.client() as client:
        configure_terminals(client, connection)
    return person, connection["id"]


@pytest.fixture
def new_shell() -> Iterator[Callable[..., str]]:
    """`new_shell(person, terminal_id, chat_id="")` starts a shell the way a new shell tab does
    and returns its id; the shells are ended after the test."""
    started: list[tuple[Actor, str, str, dict]] = []

    def start(person: Actor, terminal_id: str, chat_id: str = "") -> str:
        headers = {"X-Session-Id": chat_id} if chat_id else {}
        with person.client() as client:
            created = client.post(f"/api/v1/terminals/{terminal_id}/api/terminals", headers=headers)
        assert created.status_code == 200, f"starting a shell failed: {created.text}"
        started.append((person, terminal_id, created.json()["id"], headers))
        return created.json()["id"]

    yield start
    for person, terminal_id, shell_id, headers in started:
        with person.client() as client:
            client.delete(
                f"/api/v1/terminals/{terminal_id}/api/terminals/{shell_id}", headers=headers
            )


def _attached(
    person: Actor, terminal_id: str, shell_id: str, cursor: int = 0, chat_id: str = ""
) -> Shell:
    """A page attached to the shell, reading its output from `cursor`.

    A shell takes one page at a time and refuses a second, so right after another page left
    this tries again until the shell is free, and fails where it is not within the bound.
    """
    url = stream_url(person.base_url, terminal_id, f"api/terminals/{shell_id}")
    sign_in = {"type": "auth", "token": person.token, "resume": True, "cursor": cursor}
    deadline = time.monotonic() + RELEASE_BOUND_SECONDS
    while True:
        shell = Shell(BarePage(url, {**sign_in, "chat_id": chat_id}))
        mark = f"here-{uuid.uuid4().hex[:8]}"
        try:
            # the shell prints a typed comment back and runs nothing
            shell.type(f"# {mark}")
            if shell.shows(mark):
                return shell
            refused = "it printed nothing back"
        except ConnectionError as error:
            refused = str(error)
        shell.page_side.cut()
        assert time.monotonic() < deadline, (
            f"the shell took no page {RELEASE_BOUND_SECONDS}s after its last one had left "
            f"({refused}): Open WebUI still holds the connection it had opened for that page"
        )
        time.sleep(0.1)


def test_a_shell_runs_commands_follows_the_window_and_is_picked_up_where_it_was_left(
    a_terminal, new_shell
):
    person, terminal_id = a_terminal
    shell_id = new_shell(person, terminal_id)
    tag = uuid.uuid4().hex[:8]

    shell = _attached(person, terminal_id, shell_id)
    shell.type(f"KEPT=kept-{tag}; echo first-$((40+2))-{tag}")
    ran_first = shell.shows(f"first-42-{tag}")
    left_at = shell.cursor
    shell.resize(132, 43)
    shell.type(f"echo window-$(stty size | tr ' ' x)-{tag}")
    resized = shell.shows(f"window-43x132-{tag}")
    shell.page_side.close()

    back = _attached(person, terminal_id, shell_id, cursor=left_at)
    back.type("echo back-$KEPT")
    same_shell = back.shows(f"back-kept-{tag}")
    back.page_side.close()
    from_the_start = _attached(person, terminal_id, shell_id)
    from_the_start.page_side.close()

    assert ran_first, f"the first command's output never arrived: {shell.screen!r}"
    assert resized, f"the shell did not take the window's new size: {shell.screen!r}"
    assert same_shell, f"the page that came back got another shell: {back.screen!r}"
    assert left_at is not None, "the shell's output came without its positions"
    assert [chunk["start"] for chunk in back.announced[:1]] == [left_at], (
        f"the page came back at output position {left_at} and was sent the output of "
        f"{back.announced[:1]}"
    )
    assert f"window-43x132-{tag}".encode() in back.screen, (
        f"the output printed after position {left_at} was not replayed: {back.screen!r}"
    )
    assert f"first-42-{tag}".encode() not in back.screen, (
        f"the output from before position {left_at} was replayed as well: {back.screen!r}"
    )
    assert f"first-42-{tag}".encode() in from_the_start.screen, (
        f"a page reading the shell from its start did not get its first output: "
        f"{from_the_start.screen!r}"
    )


LEAVING = {
    "closes": BarePage.close,
    "is-cut-off": BarePage.cut,
    "is-reset": BarePage.reset,
    "is-killed": None,
}


@pytest.mark.parametrize("leaving", sorted(LEAVING))
def test_a_shell_is_free_for_its_next_page_however_the_last_one_left(
    a_terminal, new_shell, leaving
):
    """A higher limit is a stopgap: the count must return to baseline."""
    person, terminal_id = a_terminal
    shell_id = new_shell(person, terminal_id)
    tag = uuid.uuid4().hex[:8]
    url = stream_url(person.base_url, terminal_id, f"api/terminals/{shell_id}")
    sign_in = {"type": "auth", "token": person.token, "resume": True, "cursor": 0}

    first = _attached(person, terminal_id, shell_id)
    first.type(f"KEPT=kept-{tag}")
    first.page_side.close()
    answers = []
    for number in range(ROUNDS):
        shell = _attached(person, terminal_id, shell_id)
        shell.type(f"echo round-{number}-$KEPT")
        answers.append(shell.shows(f"round-{number}-kept-{tag}"))
        if leaving != "is-killed":
            LEAVING[leaving](shell.page_side)
            continue
        shell.page_side.close()
        with pages_of_another_process([(url, sign_in)], RELEASE_BOUND_SECONDS) as browser:
            browser.kill()
    last = _attached(person, terminal_id, shell_id)
    last.type("echo last-$KEPT")
    still_the_same_shell = last.shows(f"last-kept-{tag}")
    last.page_side.close()

    assert answers == [True] * ROUNDS, f"the shell answered its pages as {answers}"
    assert still_the_same_shell, f"after {ROUNDS} pages the shell had lost its state"


def test_file_changes_and_command_output_arrive_live(a_terminal):
    person, terminal_id = a_terminal
    tag = uuid.uuid4().hex[:8]
    command = f"for n in 1 2 3; do echo line-$n-{tag}; sleep 0.3; done"
    with person.client() as client:
        folder = client.get(f"/api/v1/terminals/{terminal_id}/files/cwd")
        assert folder.status_code == 200, folder.text

        with terminal_socket(person.base_url, person.token, terminal_id, "events") as events:
            events.send(json.dumps({"type": "subscribe", "path": folder.json()["cwd"]}))
            watching = json.loads(events.recv(timeout=15))
            output_of(execute(client, terminal_id, f"touch made-{tag}"))
            changed = json.loads(events.recv(timeout=15))

        started = client.post(f"/api/v1/terminals/{terminal_id}/execute", json={"command": command})
        assert started.status_code == 200, started.text
        process_id = started.json()["id"]

    statuses = []
    with terminal_socket(person.base_url, person.token, terminal_id, "execute") as processes:
        deadline = time.monotonic() + 15
        while statuses[-1:] != ["done"] and time.monotonic() < deadline:
            listed = json.loads(processes.recv(timeout=15))
            statuses += [entry["status"] for entry in listed if entry["id"] == process_id]

    def output(query: str = "") -> tuple[str, dict]:
        frames = []
        path = f"execute/{process_id}/status{query}"
        with terminal_socket(person.base_url, person.token, terminal_id, path) as stream:
            frames.append(json.loads(stream.recv(timeout=15)))
            while frames[-1]["status"] == "running":
                frames.append(json.loads(stream.recv(timeout=15)))
        printed = "".join(entry["data"] for frame in frames for entry in frame["output"])
        return printed.replace("\r", ""), frames[-1]

    whole, last_frame = output()
    from_the_second, _ = output("?offset=1")
    only_the_last, _ = output("?tail=1")

    assert (watching, changed) == ({"type": "fs:ready"}, {"type": "fs:change"}), (
        f"watching the folder gave {watching}, then {changed} once a file was made in it"
    )
    assert statuses and statuses[-1] == "done", (
        f"the list of running commands followed the command as {statuses}"
    )
    assert whole == f"line-1-{tag}\nline-2-{tag}\nline-3-{tag}\n", (
        f"the command's output stream gave {whole!r}"
    )
    assert (last_frame["status"], last_frame["exit_code"]) == ("done", 0), last_frame
    assert from_the_second == f"line-2-{tag}\nline-3-{tag}\n", (
        f"the output from the second entry on came as {from_the_second!r}"
    )
    assert only_the_last == f"line-3-{tag}\n", f"the output's tail came as {only_the_last!r}"


# a terminal of a Terminals orchestrator


def _saved_chat(person: Actor, about: str) -> str:
    with person.client() as client:
        chat_id, _ = seed_chat(client, [{"role": "user", "content": about}])
    return chat_id


def _watching(person: Actor, terminal_id: str, **sign_in) -> tuple | dict:
    """What a live file events stream answers to watching the home folder: the terminal's
    `fs:ready`, or the close code and reason when the stream is refused."""
    with terminal_socket(person.base_url, person.token, terminal_id, "events", **sign_in) as events:
        try:
            events.send(json.dumps({"type": "subscribe", "path": "/home/user"}))
            return json.loads(events.recv(timeout=30))
        except ConnectionClosed as closed:
            return (closed.rcvd.code, closed.rcvd.reason) if closed.rcvd else (1006, "")


def test_a_stream_on_a_terminal_the_person_may_not_use_is_refused_saying_why(
    admin, make_user, connect_terminals, terminals
):
    member, outsider = make_user(), make_user()
    connection = connect_terminals(read_grant(member.id), read_grant(outsider.id))
    with admin.client() as client:
        per_person = create_template(client, connection)["terminal_id"]
        per_chat = create_template(client, connection, scope="chat")["terminal_id"]
        share_template(client, per_person, read_grant(member.id))
        share_template(client, per_chat, read_grant(member.id))
    chat_id = _saved_chat(member, "a chat with a machine of its own")

    answers = {
        "shared with them": _watching(member, per_person),
        "their saved chat": _watching(member, per_chat, chat_id=chat_id),
        "the connection itself": _watching(member, connection["id"]),
        "not shared with them": _watching(outsider, per_person),
        "no saved chat": _watching(member, per_chat),
    }

    assert answers == {
        "shared with them": {"type": "fs:ready"},
        "their saved chat": {"type": "fs:ready"},
        "the connection itself": (4003, "Select a terminal template"),
        "not shared with them": (4003, "Terminal access denied"),
        "no saved chat": (4003, "This terminal requires a saved chat"),
    }, f"the live file events of the orchestrator's terminals answered {answers}"
    machines = sorted(machine["context_id"] for machine in terminals.instances())
    assert machines == sorted([f"user:{member.id}", f"chat:{chat_id}"]), (
        f"the refused streams started machines: {machines}"
    )


def test_a_shell_runs_in_the_machine_of_its_person_or_of_its_chat(
    admin, make_user, connect_terminals, terminals, new_shell
):
    person = make_user()
    connection = connect_terminals(read_grant(person.id))
    with admin.client() as client:
        per_person = create_template(client, connection)["terminal_id"]
        per_chat = create_template(client, connection, scope="chat")["terminal_id"]
        share_template(client, per_person, read_grant(person.id))
        share_template(client, per_chat, read_grant(person.id))
    chat_id = _saved_chat(person, "a chat with a machine of its own")
    tag = uuid.uuid4().hex[:8]

    wrote = {}
    for name, terminal_id, chat in [("person", per_person, ""), ("chat", per_chat, chat_id)]:
        shell_id = new_shell(person, terminal_id, chat)
        shell = _attached(person, terminal_id, shell_id, chat_id=chat)
        shell.type(f"echo from-the-{name}-shell > ~/note-{tag}; echo wrote-$((1+1))-{name}")
        wrote[name] = shell.shows(f"wrote-2-{name}")
        shell.page_side.close()

    assert wrote == {"person": True, "chat": True}, f"the two shells ran their command: {wrote}"
    held = terminals.holding(f"note-{tag}")
    assert held == {
        f"user:{person.id}": "from-the-person-shell\n",
        f"chat:{chat_id}": "from-the-chat-shell\n",
    }, f"the files the two shells wrote are in the machines {held}"


def test_an_open_shell_ends_once_its_terminal_is_no_longer_shared_with_the_person(
    admin, make_user, shared_terminal, new_shell
):
    kept, dropped = make_user(), make_user()
    terminal_id = shared_terminal(kept, dropped)
    shells = {}
    try:
        # attached first, the kept shell is checked first, so it has passed once the other ends
        for name, person in [("kept", kept), ("dropped", dropped)]:
            shells[name] = _attached(person, terminal_id, new_shell(person, terminal_id))

        with admin.client() as client:
            share_template(client, terminal_id, read_grant(kept.id))
        deadline = time.monotonic() + RECHECK_BOUND_SECONDS
        ended = False
        while not ended and time.monotonic() < deadline:
            try:
                shells["dropped"].page_side.receive(timeout=deadline - time.monotonic())
            except TimeoutError:
                break
            except ConnectionError:
                ended = True
        closed = shells["dropped"].page_side.closed_with
        shells["kept"].type("echo still-$((6*7))")
        kept_runs = shells["kept"].shows("still-42")
    finally:
        for shell in shells.values():
            shell.page_side.cut()

    assert ended, (
        f"{RECHECK_BOUND_SECONDS}s after the terminal was unshared the person's shell was "
        "still open"
    )
    assert closed == (4003, "Terminal access denied"), (
        f"the shell on the terminal no longer shared with its person ended with {closed}"
    )
    assert kept_runs, "the shell of the person who kept the terminal stopped with the other"
