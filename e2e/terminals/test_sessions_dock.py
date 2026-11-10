"""Journey: the Sessions dock under a terminal's file browser.

Under the file browser the commands the model started in the chat are listed: a strip labelled
Sessions with the number of running ones, hidden while there are none. Opened, each command is a
tab that says in its tooltip whether it runs or with which code it ended, and shows the command
line and its output, read only. A finished command can be dismissed and stays gone although
the terminal still lists it; a running one cannot.

On a real Open Terminal the scripted model starts one command that waits and one that ends
with code 4: both are listed with their state and output, typing into the output reaches
nothing, the waiting command ends once it is fed through the terminal, and both are dismissed.

On a fake terminal the test speaks for the terminal's streams. The dock asks for the list and
the output in the chat's name, reads output on from where it has read to after a broken stream
and says "Live updates unavailable" meanwhile; the arrow keys and Home move between the
commands, and the one in front gets its output. Where the list's stream never delivers anything
the dock asks over HTTP every second instead, and says when earlier output was left out. A
stream the terminal closes with a 4xxx code is not opened again.

Discriminates: green on the dev c15f28649 build; in frontend copies of it, three builds:

- A: a dismissed command that comes back with the next list fails the live test; no fallback
  to asking fails the every second test; a stream closed with 4003 that is opened again fails
  the refused test.
- B: output shown without its command line fails the real terminal test (and the two fake
  tests that read the command line).
- C: left out output not mentioned fails the every second test; a broken stream that is not
  opened again fails the refused test; streams that sign in without the chat fail the live test
  and the real terminal test (no command is listed).
- D: arrow keys that do not move fail the live test; finished commands counted as running fail
  the real terminal test.
- E: output read from the start every time fails the live test and the every second test; a
  broken stream that is not mentioned fails the refused test.
"""

from __future__ import annotations

import json
import re
import time
import uuid
from urllib.parse import parse_qs, urlparse

import pytest
from playwright.sync_api import Locator, Page, expect

from harness import upstream as reply
from harness.listener import json_answer
from harness.terminal_server import (
    TERMINAL_SERVERS_CONFIG,
    FakeTerminalServer,
    TerminalSession,
    configure_terminals,
    read_grant,
    serving_terminal,
)
from utils.cached_chat import ask, calling, choose_terminal, open_files_tab, pick_terminal
from utils.chat_ui import expect_reply, send
from utils.shell_tab import READABLE_SHELL, expect_on_screen, screen_of

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

HOME = "/home/user/"
COMMAND = "make chart"
PROCESS = "chart-1"
NO_LIVE_UPDATES = "Live updates unavailable"
SPEC = {
    "openapi": "3.0.0",
    "info": {"title": "Terminal", "version": "1"},
    "paths": {
        "/execute": {
            "post": {"operationId": "run_command", "responses": {"200": {"description": "ok"}}}
        }
    },
}


@pytest.fixture
def terminal(admin, preserve):
    """A fake terminal that serves an empty home folder; its connection is saved by the test."""
    preserve(TERMINAL_SERVERS_CONFIG)
    with serving_terminal() as server:
        server.route("GET", "/openapi.json", json_answer(SPEC))
        server.route("GET", "/api/config", json_answer({"features": {"terminal": True}}))
        server.route("GET", "/files/cwd", json_answer({"cwd": HOME, "home": HOME}))
        server.route("POST", "/files/cwd", json_answer({"cwd": HOME}))
        server.route("GET", "/files/list", json_answer({"entries": [], "writable": True}))
        yield server


def _connect(admin, terminal: FakeTerminalServer, person) -> str:
    name = f"Chart room {uuid.uuid4().hex[:6]}"
    connection = terminal.connection(name=name, config={"access_grants": [read_grant(person.id)]})
    with admin.client() as client:
        configure_terminals(client, connection)
    return name


def _wait_until(condition, seconds: float = 15.0) -> None:
    """Wait for a state the page or the terminal reaches on its own; the caller asserts it."""
    deadline = time.monotonic() + seconds
    while not condition() and time.monotonic() < deadline:
        time.sleep(0.05)


def _process(status: str = "running", exit_code: int | None = None, **fields) -> dict:
    listed = {"id": PROCESS, "command": COMMAND, "status": status, "exit_code": exit_code}
    return {**listed, **fields}


def _output(text: str, next_offset: int, **process) -> dict:
    return {
        **_process(**process),
        "output": [{"type": "output", "data": text}],
        "next_offset": next_offset,
        "truncated": False,
    }


def _stream(terminal: FakeTerminalServer, path: str, number: int = 1) -> TerminalSession:
    """The `number`th stream (from 1) a page opened on the terminal's `path`, signed in."""

    def opened() -> list[TerminalSession]:
        return [session for session in terminal.sessions if urlparse(session.path).path == path]

    _wait_until(lambda: len(opened()) >= number)
    assert len(opened()) >= number, (
        f"the page opened {len(opened())} streams on {path}, expected {number}; the terminal "
        f"saw {[session.path for session in terminal.sessions]}"
    )
    session = opened()[number - 1]
    session.authenticated()
    return session


def _asked_for(path: str) -> dict[str, str]:
    """The query of a request or stream path, each parameter once."""
    return {name: values[0] for name, values in parse_qs(urlparse(path).query).items()}


def _saved_chat_with_files(page: Page, upstream, terminal: str) -> str:
    """Pick the terminal, ask one question and open the file browser; returns the chat's id."""
    choose_terminal(page, terminal)
    question = f"plan the harbour walk {uuid.uuid4().hex[:6]}"
    upstream.queue(reply.text("Start at the lighthouse.", match=reply.answering(question)))
    send(page, question)
    expect_reply(page, "Start at the lighthouse.")
    expect(page).to_have_url(re.compile(r"/c/"))
    open_files_tab(page)
    return page.url.rsplit("/", 1)[-1]


def _dock(page: Page) -> Locator:
    # the dock has no role or name of its own
    return page.locator(".terminal-dock")


def _collapsed(page: Page) -> Locator:
    """The bar of the collapsed dock: Sessions and how many of them run."""
    return _dock(page).get_by_role("button", name=re.compile(r"^Sessions"))


def _session_tabs(page: Page) -> Locator:
    return _dock(page).get_by_role("tablist", name="Sessions").get_by_role("tab")


def _states(page: Page) -> list[str]:
    """Each listed command with its state, as its tab's tooltip reads, sorted."""
    return sorted(tab.get_attribute("title") or "" for tab in _session_tabs(page).all())


def _expect_states(page: Page, *expected: str) -> None:
    _wait_until(lambda: _states(page) == sorted(expected))
    assert _states(page) == sorted(expected), f"the dock lists {_states(page)}"


def test_the_dock_follows_the_terminals_live_list_and_output_of_the_chats_commands(
    terminal, admin, make_user, page_for, upstream
):
    person = make_user()
    name = _connect(admin, terminal, person)
    terminal.script_sessions(lambda session: None, "/execute")
    terminal.script_sessions(lambda session: None, "/execute/*")
    page = page_for(person, init_script=READABLE_SHELL)
    chat_id = _saved_chat_with_files(page, upstream, name)

    listing = _stream(terminal, "/execute")
    assert listing.auth.get("chat_id") == chat_id, (
        f"the dock of chat {chat_id} asked for the commands of {listing.auth}"
    )
    expect(page.get_by_role("region", name="File browser")).to_be_visible()
    listing.send(json.dumps([]))
    expect(_collapsed(page)).to_be_hidden()

    listing.send(json.dumps([_process()]))
    expect(_collapsed(page)).to_have_text(re.compile(r"^\s*Sessions\s*1\s*$"))
    _collapsed(page).click()
    _expect_states(page, f"{COMMAND} (Running)")
    expect(_dock(page).get_by_role("button", name=f"Dismiss {COMMAND}")).to_have_count(0)

    output = _stream(terminal, f"/execute/{PROCESS}/status")
    asked = _asked_for(output.path)
    assert (asked.get("offset"), asked.get("tail")) == ("0", "1000"), (
        f"the dock asked for the command's output with {output.path}"
    )
    assert output.auth.get("chat_id") == chat_id, output.auth
    output.send(json.dumps(_output("plotting depth\r\n", next_offset=1)))
    expect_on_screen(_dock(page), f"$ {COMMAND}")
    expect_on_screen(_dock(page), "plotting depth")
    expect(_dock(page).get_by_text(NO_LIVE_UPDATES)).to_have_count(0)

    # the output's stream breaks and is picked up after what was read
    output.close(1011)
    resumed = _stream(terminal, f"/execute/{PROCESS}/status", 2)
    assert _asked_for(resumed.path)["offset"] == "1", (
        f"after one piece of output the dock asked again with {resumed.path}"
    )
    expect(_dock(page).get_by_text(NO_LIVE_UPDATES)).to_be_visible()
    resumed.send(json.dumps(_output("plotted\r\n", next_offset=2, status="done", exit_code=3)))
    expect_on_screen(_dock(page), "plotted")
    expect(_dock(page).get_by_text(NO_LIVE_UPDATES)).to_have_count(0)
    _expect_states(page, f"{COMMAND} (Exit 3)")
    shown = screen_of(_dock(page))
    assert shown.count(f"$ {COMMAND}") == 1 and shown.count("plotting depth") == 1, (
        f"after the stream was picked up again the output reads {shown!r}"
    )

    second = _process(id="tide-2", command="print tides")
    listing.send(json.dumps([_process("done", 3), second]))
    _expect_states(page, f"{COMMAND} (Exit 3)", "print tides (Running)")
    # the arrow keys move between the commands, and the one in front streams its output
    first, last = _session_tabs(page).all()
    first.focus()
    page.keyboard.press("ArrowRight")
    expect(last).to_have_attribute("aria-selected", "true")
    expect(last).to_be_focused()
    following = _stream(terminal, "/execute/tide-2/status")
    following.send(json.dumps(_output("high 06:20\r\n", 1, id="tide-2", command="print tides")))
    expect_on_screen(_dock(page), "$ print tides")
    expect_on_screen(_dock(page), "high 06:20")
    page.keyboard.press("Home")
    expect(first).to_have_attribute("aria-selected", "true")
    expect_on_screen(_dock(page), "plotted")
    _dock(page).get_by_role("button", name=f"Dismiss {COMMAND}").click()
    _expect_states(page, "print tides (Running)")
    # the terminal still lists the dismissed command
    listing.send(json.dumps([_process("done", 3), {**second, "status": "done", "exit_code": 0}]))
    _expect_states(page, "print tides (Exit 0)")
    _dock(page).get_by_role("button", name="Dismiss print tides").click()
    expect(_dock(page)).to_be_hidden()


def test_without_live_updates_the_dock_asks_the_terminal_every_second(
    terminal, admin, make_user, page_for
):
    person = make_user()
    name = _connect(admin, terminal, person)
    # what the terminal holds of the command: its pieces of output and its state
    held = [(["plotting depth\r\n"], _process())]

    def read(request):
        pieces, process = held[-1]
        offset = int(_asked_for(request.path).get("offset", 0))
        output = [{"type": "output", "data": piece} for piece in pieces[offset:]]
        truncated = process["status"] != "running"
        answer = {**process, "output": output, "next_offset": len(pieces), "truncated": truncated}
        return json_answer(answer)

    terminal.refuse_sessions(404, "/execute")
    terminal.route("GET", "/execute", lambda request: json_answer([held[-1][1]]))
    terminal.route("GET", f"/execute/{PROCESS}/status", read)
    page = page_for(person, init_script=READABLE_SHELL)
    pick_terminal(page, name)

    expect(_collapsed(page)).to_have_text(re.compile(r"^\s*Sessions\s*1\s*$"))
    asked = len(terminal.requests_to("/execute"))
    _wait_until(lambda: len(terminal.requests_to("/execute")) >= asked + 2)
    assert len(terminal.requests_to("/execute")) >= asked + 2, (
        "the dock stopped asking the terminal for its commands"
    )
    _collapsed(page).click()
    _expect_states(page, f"{COMMAND} (Running)")
    expect_on_screen(_dock(page), f"$ {COMMAND}")
    expect_on_screen(_dock(page), "plotting depth")
    first_read = terminal.requests_to(f"/execute/{PROCESS}/status")[0]
    assert _asked_for(first_read.path) == {"wait": "0", "offset": "0", "tail": "1000"}, (
        f"the dock read the command's output with {first_read.path}"
    )

    held.append((["plotting depth\r\n", "plotted\r\n"], _process("done", 0)))
    expect_on_screen(_dock(page), "plotted")
    reads = [request.path for request in terminal.requests_to(f"/execute/{PROCESS}/status")]
    assert {_asked_for(read)["offset"] for read in reads} == {"0", "1"}, (
        f"with one piece of output read the dock read on with {reads}"
    )
    _expect_states(page, f"{COMMAND} (Exit 0)")
    shown = screen_of(_dock(page))
    assert "[Earlier output omitted]" in shown and shown.count("plotting depth") == 1, (
        f"with output left out the dock reads {shown!r}"
    )
    expect(_dock(page).get_by_text(NO_LIVE_UPDATES)).to_have_count(0)


def test_a_dock_refused_the_terminals_live_list_says_so_and_does_not_ask_again(
    terminal, admin, make_user, page_for
):
    person = make_user()
    name = _connect(admin, terminal, person)
    terminal.script_sessions(lambda session: None, "/execute")
    terminal.route("GET", "/execute", json_answer([_process()]))
    page = page_for(person, init_script=READABLE_SHELL)
    pick_terminal(page, name)
    listing = _stream(terminal, "/execute")
    listing.send(json.dumps([_process()]))
    _collapsed(page).click()
    _expect_states(page, f"{COMMAND} (Running)")

    # a broken stream is opened again
    listing.close(1011)
    expect(_dock(page).get_by_text(NO_LIVE_UPDATES)).to_be_visible()
    again = _stream(terminal, "/execute", 2)
    again.send(json.dumps([_process()]))
    expect(_dock(page).get_by_text(NO_LIVE_UPDATES)).to_have_count(0)

    again.close(4003)
    expect(_dock(page).get_by_text(NO_LIVE_UPDATES)).to_be_visible()
    _expect_states(page, f"{COMMAND} (Running)")
    # nothing more happens: no third stream and no request for the list
    time.sleep(3)
    opened = [
        session.path for session in terminal.sessions if urlparse(session.path).path == "/execute"
    ]
    assert len(opened) == 2 and not terminal.requests_to("/execute"), (
        f"after the refusal the dock opened {opened} and asked "
        f"{len(terminal.requests_to('/execute'))} times for the list"
    )


def _commands(person, connection: dict, chat_id: str) -> dict[str, dict]:
    """The commands the terminal lists for the chat, by command line."""
    with person.client() as client:
        listed = client.get(
            f"/api/v1/terminals/{connection['id']}/execute", headers={"X-Session-Id": chat_id}
        )
    assert listed.status_code == 200, listed.text
    return {process["command"]: process for process in listed.json()}


def test_the_commands_the_model_runs_are_listed_with_their_output_and_state_until_dismissed(
    open_terminal, admin, make_user, preserve, page_for, upstream
):
    preserve(TERMINAL_SERVERS_CONFIG)
    person = make_user()
    connection = open_terminal.connection(
        name=f"Harbour {uuid.uuid4().hex[:6]}",
        config={"access_grants": [read_grant(person.id)]},
    )
    with admin.client() as client:
        configure_terminals(client, connection)
    page = page_for(person, init_script=READABLE_SHELL)
    pick_terminal(page, connection["name"])
    page.keyboard.press("Control+Backslash")
    expect(page.get_by_role("region", name="File browser")).to_be_visible()
    expect(_collapsed(page)).to_be_hidden()

    tag = uuid.uuid4().hex[:8]
    waiting = f"printf 'chart %s\\n' open-{tag}; cat"
    finished = f"printf 'chart %s\\n' done-{tag}; exit 4"
    ask(
        page,
        upstream,
        f"start the chart jobs {tag}",
        calling("run_command", {"command": waiting}),
        calling("run_command", {"command": finished, "wait": 10}),
        reply.text("Both are started."),
    )
    chat_id = page.url.rsplit("/", 1)[-1]
    expect(_collapsed(page)).to_have_text(re.compile(r"^\s*Sessions\s*1\s*$"))
    _collapsed(page).click()
    _expect_states(page, f"{waiting} (Running)", f"{finished} (Exit 4)")
    expect(_dock(page).get_by_role("button", name=f"Dismiss {waiting}")).to_have_count(0)

    _session_tabs(page).filter(has_text=f"open-{tag}").click()
    expect_on_screen(_dock(page), f"$ {waiting}")
    expect_on_screen(_dock(page), f"chart open-{tag}")
    # the output is only shown: what is typed over it goes nowhere
    typed = _dock(page).get_by_role("textbox", name="Terminal input").locator("visible=true")
    typed.press_sequentially(f"typed-{tag}")
    typed.press("Enter")
    process = _commands(person, connection, chat_id)[waiting]
    with person.client() as client:
        fed = client.post(
            f"/api/v1/terminals/{connection['id']}/execute/{process['id']}/input",
            headers={"X-Session-Id": chat_id},
            json={"input": f"fed-{tag}\\n\\x04"},
        )
    assert fed.status_code == 200, fed.text
    expect_on_screen(_dock(page), f"fed-{tag}")
    _expect_states(page, f"{waiting} (Exit 0)", f"{finished} (Exit 4)")
    assert f"typed-{tag}" not in screen_of(_dock(page)), (
        f"what was typed into the dock reached the command: {screen_of(_dock(page))!r}"
    )

    _session_tabs(page).filter(has_text=f"done-{tag}").click()
    expect_on_screen(_dock(page), f"chart done-{tag}")
    _dock(page).get_by_role("button", name=f"Dismiss {finished}").click()
    _expect_states(page, f"{waiting} (Exit 0)")
    _dock(page).get_by_role("button", name=f"Dismiss {waiting}").click()
    expect(_dock(page)).to_be_hidden()
    assert set(_commands(person, connection, chat_id)) == {waiting, finished}, (
        "dismissing a command from the dock removed it from the terminal"
    )
