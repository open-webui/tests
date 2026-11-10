"""Journey: the shell of a terminal in a tab of the chat's workspace.

New Terminal in the New tab menu starts a shell on the picked terminal and connects the tab to
it. Most tests run against a fake terminal whose shells say only what the test has them say
(`script_sessions`), so each reads what the page sent and decides what comes back and how the
connection ends:

- The shell signs in for its chat, asks to resume from where it has read to and for flow
  control, reports its size when it connects and when the window changes, and says whether it
  is in front. Output is drawn, binary or text, and what is typed reaches the terminal.
- Output the terminal announces is acknowledged once flow control is confirmed. A connection
  that breaks shows "Connection interrupted. Reconnecting to the same terminal…", and the shell
  comes back on the same session from the last output it drew; each failed try waits twice as
  long as the one before.
- A shell closed with 4001 or 4003 says "Terminal access denied" and offers Retry; one closed
  with 1000 or 4004, or one the terminal no longer knows, says "Session no longer available"
  and offers Start new terminal; a shell the terminal refuses to start shows the terminal's
  reason.
- A shell opened before the first message is moved into the chat when it is saved: the message
  is held until the terminal confirms, a refusal shows the terminal's reason in an alert with
  Retry, and the shell is stored with the chat's layout.
- Closing the tab asks "Stop this terminal and close the tab?", Cancel keeps the shell, Stop
  terminal ends it on the terminal; a shell the terminal has forgotten closes quietly and one it
  cannot stop gets its tab back with the terminal's reason.
- After a reload, and in a second browser, the tab connects to the same session again.

Three tests run on real terminals, each on an Open Terminal the admin connected and on a
terminal of a Terminals connection: a reload brings back the same shell with its variables and
its earlier output, two shells of one chat are separate and closing one ends only that one, and
a shell that is open in a second window waits there and takes over once the first is closed.
On a Terminals terminal with a machine per chat or per model, a shell opened in a chat writes
its file into that chat's machine or into the machine of the chat's model.

`test_a_shell_open_in_a_second_window_waits_and_takes_over_once_the_first_is_closed` is red for
a terminal on Terminals 0.4.1: the second window says "Session no longer available" (finding
"issue: A shell that is open in another window shows "Session no longer available" on a
terminal behind Terminals"). It passes with a Terminals image that passes the terminal's close
code on.

Not here: a new chat started beside a shell of a saved chat cannot send its first message
(finding "issue: [dev] The first message of a new chat is not sent while a shell tab from the
previous chat is open", red in `e2e/tools/test_terminal_tabs_and_git.py`).

Discriminates: green on the dev c15f28649 build; in frontend copies of it, three builds:

- A: a resized shell that tells the terminal nothing fails the sign-in test; output never
  acknowledged fails the acknowledge test; every reconnect after one second fails the waiting
  test; 4003 and 1000 treated as a lost connection fail the access and the ended test; a shell
  never moved into the chat fails the first message test; Cancel that stops the shell fails the
  closing test; a tab that stays closed fails the cannot stop test.
- B: no word on whether the shell is in front fails the sign-in test; an output cursor that
  never moves fails the acknowledge test; a Retry and a Start new terminal that do nothing fail
  the access, the ended and the refused start test; a first message that does not wait for the
  terminal fails the first message test (and the refused move test); a closed tab that leaves
  its shell running fails the closing test, the two shells test and the cannot stop test; a
  shell not stored with the chat fails the reload tests, fake and real, and the second window
  test.
- C: a sign-in without the chat fails the sign-in test, the per chat machine test and, on the
  real terminals, the reload and second window tests (the shell is refused); no reconnecting
  notice fails the acknowledge test; no lookup before connecting fails the ended test at the
  forgotten shell (and the waiting test, which counts the lookups); a refusal to move the shell
  that is not heard fails the refused move test; a 404 on closing treated as a failure fails
  the closing test at the forgotten shell; a failed start without its reason fails the refused
  start test.
- D: a sign-in without the chat's model fails the per model machine test; text the terminal
  sends that is not drawn fails the sign-in test; truncated history that is not mentioned
  fails the acknowledge test.
- E: a sign-in that does not ask for flow control fails the sign-in test; a shell that cannot
  be stopped and does not say why fails the cannot stop test.
"""

from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import dataclass, field

import pytest
from playwright.sync_api import Locator, Page, expect

from harness import upstream as reply
from harness.actors import Actor
from harness.listener import json_answer
from harness.terminal_server import (
    TERMINAL_SERVERS_CONFIG,
    FakeTerminalServer,
    TerminalRequest,
    TerminalSession,
    configure_terminals,
    read_grant,
    serving_terminal,
)
from harness.upstream import MOCK_MODEL_ID
from utils.cached_chat import choose_terminal
from utils.chat_ui import expect_reply, send
from utils.shell_tab import (
    READABLE_SHELL,
    RECONNECTING,
    expect_on_screen,
    open_shell,
    screen_of,
    shell_notice,
    shell_panel,
    shell_tab,
    type_into,
)

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

SHELLS = "/api/terminals"
SPEC = {
    "openapi": "3.0.0",
    "info": {"title": "Terminal", "version": "1"},
    "paths": {
        "/execute": {
            "post": {"operationId": "run_command", "responses": {"200": {"description": "ok"}}}
        }
    },
}
ANSWER = "Start at the lighthouse."


@dataclass
class ScriptedShells:
    """A fake terminal whose shells say only what the test has them say."""

    terminal: FakeTerminalServer
    person: Actor
    name: str
    started: list[str] = field(default_factory=list)
    stopped: list[str] = field(default_factory=list)

    def start(self, request: TerminalRequest):
        shell_id = f"shell-{len(self.started) + 1}"
        self.started.append(shell_id)
        self.terminal.route("GET", f"{SHELLS}/{shell_id}", json_answer({"id": shell_id}))
        self.terminal.route("DELETE", f"{SHELLS}/{shell_id}", self.stop)
        return json_answer({"id": shell_id})

    def stop(self, request: TerminalRequest):
        self.stopped.append(request.path.split("?")[0].rsplit("/", 1)[-1])
        return json_answer({"status": "deleted"})

    def connection(self, number: int, of: str = "shell-1") -> TerminalSession:
        """The `number`th connection (from 1) a page opened to the shell `of`, signed in."""
        path = f"{SHELLS}/{of}"

        def opened() -> list[TerminalSession]:
            return [
                session for session in self.terminal.sessions if session.path.split("?")[0] == path
            ]

        _wait_until(lambda: len(opened()) >= number)
        assert len(opened()) >= number, (
            f"the page opened {len(opened())} connections to {of}, expected {number}; "
            f"the terminal saw {[session.path for session in self.terminal.sessions]}"
        )
        session = opened()[number - 1]
        session.authenticated()
        return session


@pytest.fixture
def shells(admin, make_user, preserve):
    preserve(TERMINAL_SERVERS_CONFIG)
    person = make_user()
    with serving_terminal() as terminal:
        name = f"Quay {uuid.uuid4().hex[:6]}"
        scripted = ScriptedShells(terminal, person, name)
        # a chat turn with the terminal picked reads its tools and its folder
        terminal.route("GET", "/openapi.json", json_answer(SPEC))
        terminal.route("GET", "/files/cwd", json_answer({"cwd": "/"}))
        terminal.route("POST", "/files/cwd", json_answer({"cwd": "/"}))
        terminal.route("GET", "/api/config", json_answer({"features": {"terminal": True}}))
        terminal.route("POST", SHELLS, scripted.start)
        # a shell stays silent until the test speaks for it
        terminal.script_sessions(lambda session: None, f"{SHELLS}/*")
        connection = terminal.connection(
            name=name, config={"access_grants": [read_grant(person.id)]}
        )
        with admin.client() as client:
            configure_terminals(client, connection)
        yield scripted


def _wait_until(condition, seconds: float = 15.0) -> None:
    """Wait for a state the page or the terminal reaches on its own; the caller asserts it."""
    deadline = time.monotonic() + seconds
    while not condition() and time.monotonic() < deadline:
        time.sleep(0.05)


def _said(session: TerminalSession, kind: str) -> list[dict]:
    """The control messages of that type the page sent on the connection, in order."""
    texts = [message for message in list(session.messages) if isinstance(message, str)]
    return [parsed for parsed in map(json.loads, texts) if parsed.get("type") == kind]


def _typed(session: TerminalSession) -> bytes:
    return b"".join(message for message in list(session.messages) if isinstance(message, bytes))


def _prints(session: TerminalSession, text: str, start: int, truncated: bool = False) -> int:
    """The terminal prints `text` as output that starts at `start`; returns where it ends."""
    end = start + len(text.encode())
    announced = {"type": "output", "start": start, "end": end, "truncated": truncated}
    session.send(json.dumps(announced))
    session.send(text.encode())
    return end


def _saved_chat(page: Page, upstream, terminal: str) -> str:
    """Pick the terminal and ask one question; returns the saved chat's id."""
    choose_terminal(page, terminal)
    question = f"plan the harbour walk {uuid.uuid4().hex[:6]}"
    upstream.queue(reply.text(ANSWER, match=reply.answering(question)))
    send(page, question)
    expect_reply(page, ANSWER)
    expect(page).to_have_url(re.compile(r"/c/"))
    return page.url.rsplit("/", 1)[-1]


def test_a_shell_signs_in_for_its_chat_reports_its_size_and_draws_what_the_terminal_prints(
    shells, page_for, upstream
):
    page = page_for(shells.person, init_script=READABLE_SHELL)
    chat_id = _saved_chat(page, upstream, shells.name)
    shell = open_shell(page)

    session = shells.connection(1)
    assert shells.started == ["shell-1"], f"one New Terminal started {shells.started}"
    sign_in = {key: session.auth.get(key) for key in ("chat_id", "resume", "cursor")}
    assert sign_in == {"chat_id": chat_id, "resume": True, "cursor": 0}, (
        f"the shell of chat {chat_id} signed in with {session.auth}"
    )
    assert session.auth.get("flow_control") is True, session.auth

    _wait_until(lambda: _said(session, "resize") and _said(session, "activity"))
    assert _said(session, "activity") == [{"type": "activity", "active": True}], session.messages
    first_size = _said(session, "resize")[0]
    assert first_size["cols"] > 20 and first_size["rows"] > 5, (
        f"the shell reported the size {first_size}"
    )

    session.send(b"harbour $ ")
    expect_on_screen(shell, "harbour $ ")
    session.send("tide tables loaded")
    expect_on_screen(shell, "harbour $ tide tables loaded")
    type_into(shell, "ls -l")
    _wait_until(lambda: _typed(session) == b"ls -l")
    assert _typed(session) == b"ls -l", f"typing ls -l reached the terminal as {session.messages}"

    page.set_viewport_size({"width": 1100, "height": 600})
    _wait_until(lambda: _said(session, "resize")[-1] != first_size)
    last_size = _said(session, "resize")[-1]
    assert last_size["cols"] < first_size["cols"] and last_size["rows"] < first_size["rows"], (
        f"in a smaller window the shell went from {first_size} to {last_size}"
    )

    # another tab in front of the shell
    page.get_by_role("tablist", name="Tabs").get_by_role("tab").first.click()
    _wait_until(lambda: len(_said(session, "activity")) == 2)
    shell_tab(page).click()
    _wait_until(lambda: len(_said(session, "activity")) == 3)
    assert [message["active"] for message in _said(session, "activity")] == [True, False, True], (
        f"hiding and showing the shell was reported as {_said(session, 'activity')}"
    )


def test_output_is_acknowledged_and_an_interrupted_shell_resumes_from_where_it_was(
    shells, page_for
):
    page = page_for(shells.person, init_script=READABLE_SHELL)
    choose_terminal(page, shells.name)
    shell = open_shell(page)
    session = shells.connection(1)
    session.send(json.dumps({"type": "flow_control"}))
    read_to = _prints(session, "first tide in\r\n", start=0)
    expect_on_screen(shell, "first tide in")
    _wait_until(lambda: _said(session, "ack"))
    assert _said(session, "ack") == [{"type": "ack", "cursor": read_to}], session.messages
    expect(shell_notice(shell)).to_have_count(0)

    session.drop()
    expect(shell_notice(shell)).to_have_text(RECONNECTING)
    resumed = shells.connection(2)
    assert (resumed.auth.get("resume"), resumed.auth.get("cursor")) == (True, read_to), (
        f"after {read_to} bytes the shell came back with {resumed.auth}"
    )
    expect(shell_notice(shell)).to_have_count(0)
    assert shells.started == ["shell-1"], f"reconnecting started {shells.started}"

    # the terminal no longer has what came between
    read_to = _prints(resumed, "third tide in\r\n", start=read_to + 400, truncated=True)
    expect_on_screen(shell, "third tide in")
    shown = screen_of(shell)
    assert "first tide in" in shown and "[Earlier output is no longer available]" in shown, (
        f"after the reconnect the screen reads {shown!r}"
    )
    # this connection never said it wants acknowledgements
    type_into(shell, "q")
    _wait_until(lambda: _typed(resumed) == b"q")
    assert _typed(resumed) == b"q" and not _said(resumed, "ack"), resumed.messages

    resumed.close(1011)
    again = shells.connection(3)
    assert again.auth.get("cursor") == read_to, (
        f"after output up to {read_to} the shell came back with {again.auth}"
    )


def test_a_shell_that_cannot_reconnect_waits_longer_before_each_try(shells, page_for):
    page = page_for(shells.person, init_script=READABLE_SHELL)
    choose_terminal(page, shells.name)
    shell = open_shell(page)
    session = shells.connection(1)
    tries: list[float] = []

    def looked_up(request: TerminalRequest):
        tries.append(time.monotonic())
        return json_answer({"id": "shell-1"})

    # each try looks the shell up before it connects
    shells.terminal.route("GET", f"{SHELLS}/shell-1", looked_up)
    shells.terminal.refuse_sessions(500, f"{SHELLS}/*")
    session.drop()
    _wait_until(lambda: len(tries) >= 3)
    assert len(tries) >= 3, f"the shell tried to reconnect {len(tries)} times in 15 s"
    waits = [round(later - earlier, 1) for earlier, later in zip(tries, tries[1:])]
    assert waits[0] >= 1.8 and waits[1] >= 3.6, (
        f"the waits between the tries were {waits} s, expected them to double from 1 s"
    )
    expect(shell_notice(shell)).to_have_text(RECONNECTING)


def test_a_shell_the_person_may_no_longer_use_says_so_and_retry_connects_it_again(shells, page_for):
    page = page_for(shells.person, init_script=READABLE_SHELL)
    choose_terminal(page, shells.name)
    shell = open_shell(page)
    for number, code in enumerate((4003, 4001), start=1):
        session = shells.connection(number)
        session.send(f"watch {number} $ ".encode())
        expect_on_screen(shell, f"watch {number} $ ")
        session.close(code)
        expect(shell_notice(shell)).to_contain_text("Terminal access denied")
        expect(shell.get_by_role("button")).to_have_text(["Retry"])
        shell.get_by_role("button", name="Retry").click()
    session = shells.connection(3)
    session.send(b"watch 3 $ ")
    expect_on_screen(shell, "watch 3 $ ")
    expect(shell_notice(shell)).to_have_count(0)
    assert shells.started == ["shell-1"], f"Retry started {shells.started}"


def test_a_shell_that_has_ended_says_so_and_start_new_terminal_opens_another(shells, page_for):
    page = page_for(shells.person, init_script=READABLE_SHELL)
    choose_terminal(page, shells.name)
    shell = open_shell(page)
    for number, code in enumerate((1000, 4004), start=1):
        session = shells.connection(1, of=f"shell-{number}")
        session.send(f"berth {number} $ ".encode())
        expect_on_screen(shell, f"berth {number} $ ")
        session.close(code)
        expect(shell_notice(shell)).to_contain_text("Session no longer available")
        expect(shell.get_by_role("button")).to_have_text(["Start new terminal"])
        shell.get_by_role("button", name="Start new terminal").click()

    session = shells.connection(1, of="shell-3")
    session.send(b"berth 3 $ ")
    expect_on_screen(shell, "berth 3 $ ")
    assert shells.started == ["shell-1", "shell-2", "shell-3"], shells.started

    # the connection breaks and the terminal no longer knows the shell
    shells.terminal.route(
        "GET", f"{SHELLS}/shell-3", json_answer({"error": "Session not found"}, status=404)
    )
    session.drop()
    expect(shell_notice(shell)).to_contain_text("Session no longer available")
    expect(shell.get_by_role("button")).to_have_text(["Start new terminal"])


def _asked(upstream, question: str) -> bool:
    return any(map(reply.answering(question), upstream.chat_requests()))


def _stored_tabs(person: Actor, chat_id: str) -> list[dict]:
    """The tabs the server keeps for the person's view of the chat."""
    with person.client() as client:
        response = client.get(f"/api/v1/chats/{chat_id}/view")
    assert response.status_code == 200, response.text
    state = response.json()["state"]
    return state["panels"]["tabs"] if state else []


def _stored_shells(person: Actor, chat_id: str) -> list[str | None]:
    return [
        tab.get("sessionId") for tab in _stored_tabs(person, chat_id) if tab["kind"] == "terminal"
    ]


def test_a_shell_opened_before_the_first_message_moves_into_the_chat_once_it_is_saved(
    shells, page_for, upstream
):
    page = page_for(shells.person, init_script=READABLE_SHELL)
    choose_terminal(page, shells.name)
    shell = open_shell(page)
    session = shells.connection(1)
    assert session.auth.get("chat_id") == "", f"a shell without a chat signed in {session.auth}"
    session.send(b"mooring $ ")
    expect_on_screen(shell, "mooring $ ")

    page.get_by_role("tab", name="New Chat", exact=True).click()
    question = f"where do the ferries leave {uuid.uuid4().hex[:6]}"
    upstream.queue(reply.text("From the east pier.", match=reply.answering(question)))
    send(page, question)
    _wait_until(lambda: _said(session, "chat"))
    moves = _said(session, "chat")
    assert len(moves) == 1 and moves[0].get("chat_id"), (
        f"sending the first message told the shell {session.messages}"
    )
    chat_id = moves[0]["chat_id"]
    # the message waits for the terminal's word that the shell has moved: nothing happens yet
    _wait_until(lambda: _asked(upstream, question), seconds=2)
    assert not _asked(upstream, question), "the model was asked before the shell had moved"
    session.send(json.dumps({"type": "chat", "chat_id": chat_id}))
    expect_reply(page, "From the east pier.")
    expect(page).to_have_url(re.compile(rf"/c/{chat_id}$"))
    expect(page.get_by_role("alert")).to_have_count(0)

    _wait_until(lambda: _stored_shells(shells.person, chat_id) == ["shell-1"])
    assert _stored_shells(shells.person, chat_id) == ["shell-1"], (
        f"the chat's stored tabs are {_stored_tabs(shells.person, chat_id)}"
    )
    # from now on the shell signs in for its chat
    session.close(1011)
    resumed = shells.connection(2)
    assert resumed.auth.get("chat_id") == chat_id, (
        f"the shell of chat {chat_id} came back with {resumed.auth}"
    )
    assert _said(session, "chat") == moves, f"the shell was moved twice: {session.messages}"


def test_a_shell_the_terminal_does_not_move_holds_the_first_message_back_and_says_why(
    shells, page_for, upstream
):
    page = page_for(shells.person, init_script=READABLE_SHELL)
    choose_terminal(page, shells.name)
    open_shell(page)
    session = shells.connection(1)
    page.get_by_role("tab", name="New Chat", exact=True).click()
    question = f"where do the ferries leave {uuid.uuid4().hex[:6]}"
    upstream.queue(reply.text("From the east pier.", match=reply.answering(question)))
    send(page, question)
    _wait_until(lambda: _said(session, "chat"))
    assert _said(session, "chat"), f"sending the first message told the shell {session.messages}"
    chat_id = _said(session, "chat")[0]["chat_id"]
    session.send(json.dumps({"type": "chat", "error": "The shell is busy elsewhere"}))
    # the reason shows with or without "Could not attach sessions to this chat.", by timing
    alert = page.get_by_role("alert").filter(has_text="The shell is busy elsewhere")
    expect(alert).to_be_visible()
    assert not _asked(upstream, question), "the message was sent although the shell did not move"

    alert.get_by_role("button", name="Retry").click()
    _wait_until(lambda: len(_said(session, "chat")) == 2)
    assert [move.get("chat_id") for move in _said(session, "chat")] == [chat_id, chat_id], (
        f"Retry told the shell {session.messages}"
    )
    session.send(json.dumps({"type": "chat", "chat_id": chat_id}))
    expect(alert).to_have_count(0)
    _wait_until(lambda: _stored_shells(shells.person, chat_id) == ["shell-1"])
    assert _stored_shells(shells.person, chat_id) == ["shell-1"], (
        f"the chat's stored tabs are {_stored_tabs(shells.person, chat_id)}"
    )


def _stop_dialog(page: Page):
    return page.get_by_role("dialog", name="Stop terminal?")


def test_closing_a_shell_asks_first_and_then_stops_it_on_the_terminal(shells, page_for, upstream):
    page = page_for(shells.person, init_script=READABLE_SHELL)
    chat_id = _saved_chat(page, upstream, shells.name)
    shell = open_shell(page)
    session = shells.connection(1)
    session.send(b"slipway $ ")
    expect_on_screen(shell, "slipway $ ")

    page.get_by_role("button", name="Close Terminal 1", exact=True).click()
    dialog = _stop_dialog(page)
    expect(dialog).to_contain_text("Stop this terminal and close the tab?")
    dialog.get_by_role("button", name="Cancel").click()
    expect(dialog).to_have_count(0)
    expect(shell_tab(page)).to_be_visible()
    type_into(shell, "w")
    _wait_until(lambda: _typed(session) == b"w")
    assert _typed(session) == b"w" and not shells.stopped and not session.ended.is_set(), (
        f"after Cancel the terminal has stopped {shells.stopped}, the shell got {session.messages}"
    )

    page.get_by_role("button", name="Close Terminal 1", exact=True).click()
    _stop_dialog(page).get_by_role("button", name="Stop terminal").click()
    expect(shell_tab(page)).to_have_count(0)
    _wait_until(lambda: shells.stopped and session.ended.is_set())
    assert shells.stopped == ["shell-1"], f"closing the tab stopped {shells.stopped}"
    assert session.ended.is_set(), "the closed tab kept its connection to the shell"
    stop = shells.terminal.requests_to(f"{SHELLS}/shell-1")[-1]
    assert (stop.method, stop.headers.get("x-session-id")) == ("DELETE", chat_id), (
        f"the shell of chat {chat_id} was stopped with {stop.method} for the chat "
        f"{stop.headers.get('x-session-id')!r}"
    )
    _wait_until(lambda: _stored_shells(shells.person, chat_id) == [])
    assert _stored_shells(shells.person, chat_id) == [], _stored_tabs(shells.person, chat_id)

    # a shell the terminal has already forgotten closes without a word
    gone = open_shell(page)
    shells.connection(1, of="shell-2")
    shells.terminal.route(
        "DELETE", f"{SHELLS}/shell-2", json_answer({"error": "Session not found"}, status=404)
    )
    page.get_by_role("button", name="Close Terminal 1", exact=True).click()
    _stop_dialog(page).get_by_role("button", name="Stop terminal").click()
    expect(gone).to_have_count(0)
    expect(shell_tab(page)).to_have_count(0)
    expect(page.get_by_text("Session not found")).to_have_count(0)


def test_a_shell_the_terminal_cannot_stop_keeps_its_tab_and_says_why(shells, page_for):
    page = page_for(shells.person, init_script=READABLE_SHELL)
    choose_terminal(page, shells.name)
    shell = open_shell(page)
    session = shells.connection(1)
    shells.terminal.route(
        "DELETE", f"{SHELLS}/shell-1", json_answer({"detail": "The shell is locked"}, status=500)
    )
    page.get_by_role("button", name="Close Terminal 1", exact=True).click()
    _stop_dialog(page).get_by_role("button", name="Stop terminal").click()
    expect(page.get_by_text("The shell is locked")).to_be_visible()
    expect(shell_tab(page)).to_be_visible()
    # the tab that came back connects to the same shell again
    again = shells.connection(2)
    again.send(b"still here $ ")
    shell_tab(page).click()
    expect_on_screen(shell, "still here $ ")
    assert shells.started == ["shell-1"], shells.started
    assert session.ended.is_set(), "the first connection stayed open beside the new one"


def test_a_reloaded_chat_reconnects_its_shell_to_the_same_session_also_on_another_device(
    shells, page_for, upstream, config
):
    page = page_for(shells.person, init_script=READABLE_SHELL)
    chat_id = _saved_chat(page, upstream, shells.name)
    shell = open_shell(page)
    session = shells.connection(1)
    read_to = _prints(session, "low water 04:10\r\n", start=0)
    expect_on_screen(shell, "low water 04:10")
    _wait_until(lambda: _stored_shells(shells.person, chat_id) == ["shell-1"])
    assert _stored_shells(shells.person, chat_id) == ["shell-1"], (
        f"the chat's stored tabs are {_stored_tabs(shells.person, chat_id)}"
    )

    page.reload()
    # a reload is a cold load
    expect(shell_tab(page)).to_be_visible(timeout=config.navigation_timeout)
    restored = shells.connection(2)
    sign_in = {key: restored.auth.get(key) for key in ("chat_id", "resume", "cursor")}
    assert sign_in == {"chat_id": chat_id, "resume": True, "cursor": 0}, (
        f"after the reload the shell of chat {chat_id} signed in with {restored.auth}"
    )
    assert read_to > 0 and session.ended.is_set(), "the reloaded page kept its old connection"
    # the terminal plays back what the shell printed so far
    _prints(restored, "low water 04:10\r\n", start=0)
    shell_tab(page).click()
    expect_on_screen(shell_panel(page), "low water 04:10")

    elsewhere = page_for(shells.person, path=f"/c/{chat_id}", init_script=READABLE_SHELL)
    expect(shell_tab(elsewhere)).to_be_visible()
    other = shells.connection(3)
    assert other.auth.get("chat_id") == chat_id, other.auth
    assert shells.started == ["shell-1"], f"reopening the chat started {shells.started}"


@pytest.fixture(params=["an Open Terminal", "a terminal on Terminals"])
def harbour(request, make_user) -> tuple[Actor, str, str]:
    """A fresh account with a real terminal it may use, as the admin's own connection to an Open
    Terminal and as a terminal on a Terminals connection: the account, the terminal's name and
    its id."""
    person = make_user()
    name = f"Harbour {uuid.uuid4().hex[:6]}"
    if request.param == "a terminal on Terminals":
        return person, name, request.getfixturevalue("shared_terminal")(person, name=name)
    request.getfixturevalue("preserve")(TERMINAL_SERVERS_CONFIG)
    connection = request.getfixturevalue("open_terminal").connection(
        name=name, config={"access_grants": [read_grant(person.id)]}
    )
    with request.getfixturevalue("admin").client() as client:
        configure_terminals(client, connection)
    return person, name, connection["id"]


def _running_shells(person: Actor, terminal_id: str, chat_id: str = "") -> list[str]:
    """The shells the terminal runs for the person, as the page would list them."""
    headers = {"X-Session-Id": chat_id} if chat_id else {}
    with person.client() as client:
        listed = client.get(f"/api/v1/terminals/{terminal_id}{SHELLS}", headers=headers)
    assert listed.status_code == 200, listed.text
    return [shell["id"] for shell in listed.json()]


def _enter(shell: Locator, line: str) -> None:
    """Type a line into a shell that has shown its prompt, and press Enter."""
    _wait_until(lambda: screen_of(shell).strip())
    assert screen_of(shell).strip(), (
        f"the shell shows no prompt; its tab reads {shell.inner_text()!r}"
    )
    type_into(shell, line)
    shell.get_by_role("textbox", name="Terminal input").press("Enter")


def _shell_with_a_variable(page: Page, upstream, terminal: str, tag: str) -> tuple[str, Locator]:
    """A saved chat with a shell that holds `tide` and has printed it; the chat's id and the
    shell's panel."""
    chat_id = _saved_chat(page, upstream, terminal)
    shell = open_shell(page)
    _enter(shell, f"tide={tag}")
    _enter(shell, "echo before-$tide")
    expect_on_screen(shell, f"before-{tag}")
    return chat_id, shell


def test_a_shell_of_a_real_terminal_is_the_same_shell_after_a_reload(
    harbour, page_for, upstream, config
):
    person, name, terminal_id = harbour
    page = page_for(person, init_script=READABLE_SHELL)
    tag = uuid.uuid4().hex[:8]
    chat_id, shell = _shell_with_a_variable(page, upstream, name, tag)
    running = _running_shells(person, terminal_id, chat_id)
    assert len(running) == 1, f"one New Terminal left the terminal running {running}"
    _wait_until(lambda: _stored_shells(person, chat_id) == running)
    assert _stored_shells(person, chat_id) == running, _stored_tabs(person, chat_id)

    page.reload()
    # a reload is a cold load
    expect(shell_tab(page)).to_be_visible(timeout=config.navigation_timeout)
    shell_tab(page).click()
    shell = shell_panel(page)
    # the terminal plays back what the shell printed before
    expect_on_screen(shell, f"before-{tag}")
    _enter(shell, "echo after-$tide")
    expect_on_screen(shell, f"after-{tag}")
    assert _running_shells(person, terminal_id, chat_id) == running, (
        f"after the reload the terminal runs {_running_shells(person, terminal_id, chat_id)}, "
        f"before it ran {running}"
    )


def test_a_shell_open_in_a_second_window_waits_and_takes_over_once_the_first_is_closed(
    harbour, page_for, upstream
):
    """Red on a terminal on Terminals 0.4.1, finding "issue: A shell that is open in another
    window shows "Session no longer available" on a terminal behind Terminals"."""
    person, name, terminal_id = harbour
    page = page_for(person, init_script=READABLE_SHELL)
    tag = uuid.uuid4().hex[:8]
    chat_id, shell = _shell_with_a_variable(page, upstream, name, tag)
    _wait_until(lambda: len(_stored_shells(person, chat_id)) == 1)
    assert len(_stored_shells(person, chat_id)) == 1, _stored_tabs(person, chat_id)

    elsewhere = page_for(person, path=f"/c/{chat_id}", init_script=READABLE_SHELL)
    shell_tab(elsewhere).click()
    waiting = shell_panel(elsewhere)
    expect(shell_notice(waiting)).to_have_text(RECONNECTING)
    # the shell stays with the window that has it
    _enter(shell, "echo both-$tide")
    expect_on_screen(shell, f"both-{tag}")

    page.close()
    expect_on_screen(waiting, f"both-{tag}")
    expect(shell_notice(waiting)).to_have_count(0)
    _enter(waiting, "echo after-$tide")
    expect_on_screen(waiting, f"after-{tag}")
    running = _running_shells(person, terminal_id, chat_id)
    assert len(running) == 1, f"two windows on one shell left the terminal running {running}"


def test_two_shells_of_a_chat_are_shells_of_their_own_and_closing_one_leaves_the_other(
    harbour, page_for
):
    person, name, terminal_id = harbour
    page = page_for(person, init_script=READABLE_SHELL)
    choose_terminal(page, name)
    first = open_shell(page, 1)
    tag = uuid.uuid4().hex[:8]
    _enter(first, f"berth=north-{tag}")
    second = open_shell(page, 2)
    _enter(second, "echo second-${berth:-unset}")
    expect_on_screen(second, "second-unset")
    shell_tab(page, 1).click()
    _enter(first, "echo first-$berth")
    expect_on_screen(first, f"first-north-{tag}")
    assert len(_running_shells(person, terminal_id)) == 2, _running_shells(person, terminal_id)

    page.get_by_role("button", name="Close Terminal 2", exact=True).click()
    _stop_dialog(page).get_by_role("button", name="Stop terminal").click()
    expect(shell_tab(page, 2)).to_have_count(0)
    _wait_until(lambda: len(_running_shells(person, terminal_id)) == 1)
    assert len(_running_shells(person, terminal_id)) == 1, (
        f"closing one of two shells left {_running_shells(person, terminal_id)} running"
    )
    _enter(first, "echo still-$berth")
    expect_on_screen(first, f"still-north-{tag}")
    # the freed number is given out again
    expect(open_shell(page, 2)).to_be_visible()
    assert len(_running_shells(person, terminal_id)) == 2, _running_shells(person, terminal_id)


@pytest.mark.parametrize("scope", ["chat", "model"])
def test_a_shell_runs_in_the_machine_of_its_chat_or_of_the_chats_model(
    scope, make_user, shared_terminal, terminals, page_for, upstream
):
    person = make_user()
    name = f"Harbour {uuid.uuid4().hex[:6]}"
    shared_terminal(person, name=name, scope=scope)
    page = page_for(person, init_script=READABLE_SHELL)
    chat_id = _saved_chat(page, upstream, name)
    shell = open_shell(page)
    note = f"berth-{uuid.uuid4().hex[:8]}.txt"
    _enter(shell, f"echo moored > ~/{note}; echo written-$?")
    expect_on_screen(shell, "written-0")
    machine = f"chat:{chat_id}" if scope == "chat" else f"model:{MOCK_MODEL_ID}"
    assert terminals.holding(note) == {machine: "moored\n"}, (
        f"the shell of chat {chat_id} wrote its file into {terminals.holding(note)}, with the "
        f"machines {[machine['context_id'] for machine in terminals.instances()]}"
    )


def test_a_shell_the_terminal_refuses_to_start_says_why_and_starts_on_the_next_try(
    shells, page_for
):
    page = page_for(shells.person, init_script=READABLE_SHELL)
    choose_terminal(page, shells.name)
    refusal = "Maximum number of terminal sessions (2) reached"
    shells.terminal.route("POST", SHELLS, json_answer({"error": refusal}, status=429))
    shell = open_shell(page)
    expect(shell_notice(shell)).to_contain_text(refusal)
    expect(shell.get_by_role("button")).to_have_text(["Start new terminal"])
    assert not shells.terminal.sessions, (
        f"without a shell the page connected to {[s.path for s in shells.terminal.sessions]}"
    )

    shells.terminal.route("POST", SHELLS, shells.start)
    shell.get_by_role("button", name="Start new terminal").click()
    session = shells.connection(1)
    session.send(b"room again $ ")
    expect_on_screen(shell, "room again $ ")
    expect(shell_notice(shell)).to_have_count(0)
