"""Regression: two settings of an admin's terminal connection that shape what the model is offered.

Commit `a20b622ba` added them to the Add/Edit Terminal Connection dialog, saved in the
connection's `config`. Working Directory Context (`working_directory_context`, on by default)
switched off stops Open WebUI from asking the terminal for its current directory before a turn,
so the `run_command` description no longer ends with "The current working directory is: ..."
and stays the same whichever folder the terminal sits in. The terminal's own system prompt and
the AGENTS.md in its home still reach the model. User Shell Tools (`user_shell_tools`, `auto`
by default) set to `always` keeps the terminal's `read_user_terminal` and
`send_user_terminal_input` offered while the user's tab has no shell open; with `auto` they are
offered only while it has one. An automation is never offered them; that exclusion has no
test here, since an automation run on dev 1c010b438 never reaches the model
(open-webui/open-webui#32066).

Whether the user has a shell open is the terminal's answer: since 47a9e7d48 Open WebUI asks the
terminal for the shell of this user and chat before the turn (Open Terminal answers 404 when there
is none), where it used to ask the user's browser tab.

Discriminates: passes on dev 1c010b438; with a20b622ba's backend hunks reverted in a backend copy
the working-directory-off test and both `always` tests fail (the description names the folder; the
shell tools are dropped). The default, `auto`, open-shell and AGENTS.md tests pass on both. On dev
d22c4449c the `auto` tests turn red when the terminal's answer is ignored in a backend copy.

A connection saved with nothing but its address and key, as an API client or an import may
send it, is switched on, asked for `/openapi.json` and sent its key as bearer, and a field the
form does not know is kept. On dev c15f28649 that test fails in backend copies where the
form's default for Auth is `none` (no key reaches the terminal) and where the form ignores
unknown fields (the extra field is gone); with the address check dropped from Verify
Connection it fails on the answer for an empty address.

Every save of the connections is reported as the event `config.terminal_servers.updated`
with the admin who saved and the number of connections, and a save that is refused reports
nothing. `policy` and `lifecycle`, the two entries a connection of the policy model before
templates could be sent with, are not stored; everything else sent is. On dev c15f28649 the
event test fails in a backend copy that saves without publishing the event, and the policy
test in one that stores a connection as it was sent.

What the terminal fails to say never holds a turn up. A terminal that cannot name its folder
leaves `run_command` without the folder line. Its prompt is read for the person at each turn and,
when that read fails or is empty, taken as it was read when the connection was saved; a terminal
that does not announce a prompt (`features.system` in `/api/config`) is not asked for one. A
shell check that fails, answers something else than a shell's output or takes more than five
seconds hides the shell tools.

Discriminates (dev c15f28649, each in a backend copy): the folder line added whatever the
terminal answered turns the folder test red (and the working-directory-off test); the saved
prompt never used, the two fallback cases of the prompt test, and the saved prompt always used,
its first case; the `features.system` check dropped, the first two cases of the announce test,
and the status of `/api/config` ignored, its third; any answer to the shell check taken as an
open shell, the failed-check test (and the `auto` tests); the five second limit on the check
removed, the slow test; the note on `read_user_terminal` dropped, the `always` test.
"""

from __future__ import annotations

import threading
import time
import uuid
from contextlib import contextmanager
from typing import Iterator

import pytest

from harness import upstream as reply
from harness.chat import ask
from harness.listener import Answer, json_answer
from harness.plugins import installed_function
from harness.socket_client import connected
from harness.terminal_server import (
    TERMINAL_SERVERS_CONFIG,
    FakeTerminalServer,
    configure_terminals,
    read_grant,
    serving_terminal,
)

pytestmark = [pytest.mark.regression, pytest.mark.api, pytest.mark.requires_source]

WORKDIR = "/srv/build/project"
HOME = "/home/ada"
AGENTS_MD = "Run the linter before every commit."
TERMINAL_PROMPT = "You are working inside the build box."
CWD_LINE = "The current working directory is:"
SHELL_TOOLS = {"read_user_terminal", "send_user_terminal_input"}
# Open Terminal reads the user's shell here and answers 404 when that user and chat have none open
USER_SHELL_PATH = "/api/terminals/user/output"
NO_SHELL = json_answer({"detail": "No connected user shell"}, status=404)


def _operation(operation_id: str, description: str) -> dict:
    return {
        "operationId": operation_id,
        "description": description,
        "responses": {"200": {"description": "ok"}},
    }


SPEC = {
    "openapi": "3.0.0",
    "info": {"title": "terminal", "version": "1"},
    "paths": {
        "/execute": {"post": _operation("run_command", "Run a shell command.")},
        USER_SHELL_PATH: {"get": _operation("read_user_terminal", "Read the user's shell.")},
        "/api/terminals/user/input": {
            "post": _operation("send_user_terminal_input", "Type into the user's shell.")
        },
    },
}


@pytest.fixture
def terminal(preserve) -> Iterator[FakeTerminalServer]:
    preserve(TERMINAL_SERVERS_CONFIG)
    with serving_terminal() as server:
        server.route("GET", "/openapi.json", json_answer(SPEC))
        server.route("GET", "/files/cwd", json_answer({"cwd": WORKDIR, "home": HOME}))
        server.route("GET", "/files/read", json_answer({"content": AGENTS_MD}))
        server.route("GET", "/api/config", json_answer({"features": {"system": True}}))
        server.route("GET", "/system", json_answer({"prompt": TERMINAL_PROMPT}))
        server.route("GET", USER_SHELL_PATH, NO_SHELL)
        yield server


@pytest.fixture
def member(make_user):
    return make_user()


def _connect(admin, server: FakeTerminalServer, member, **settings) -> str:
    """Save the terminal as the admin's dialog does, shared with `member`; returns its id."""
    connection = server.connection(config={"access_grants": [read_grant(member.id)], **settings})
    with admin.client() as client:
        configure_terminals(client, connection)
    server.clear()
    return connection["id"]


@contextmanager
def _tab(actor) -> Iterator[str]:
    """A signed-in tab of `actor`; yields its socket id."""
    with connected(actor) as session:
        yield session.client.get_sid(namespace="/")


def _open_shell(server: FakeTerminalServer, actor) -> None:
    """The terminal reports a shell `actor` opened in a chat; anyone else still has none."""

    def answer(request) -> tuple:
        if request.headers.get("x-user-id") == actor.id and request.headers.get("x-session-id"):
            return json_answer({"output": "$ "})
        return NO_SHELL

    server.route("GET", USER_SHELL_PATH, answer)


def _chat_request(actor, upstream, **options) -> dict:
    """One turn from a tab of `actor`; returns the request the model was sent."""
    prompt = f"check the build {uuid.uuid4().hex[:6]}"
    upstream.queue(reply.text("done", match=reply.answering(prompt)))
    with _tab(actor) as session_id, actor.client() as client:
        _, message = ask(client, prompt, session_id=session_id, **options)
    assert message["content"] == "done", message
    return next(
        request
        for request in reversed(upstream.chat_requests())
        if prompt in str(reply.conversation(request)[-1].get("content"))
    )


def _offered(request: dict) -> dict[str, str]:
    return {
        tool["function"]["name"]: tool["function"].get("description", "")
        for tool in request.get("tools") or []
    }


def _texts(request: dict) -> list[str]:
    return [str(entry.get("content")) for entry in request["messages"]]


def test_without_working_directory_context_run_command_names_no_folder(
    admin, terminal, member, upstream
):
    terminal_id = _connect(admin, terminal, member, working_directory_context=False)

    request = _chat_request(member, upstream, terminal_id=terminal_id)

    description = _offered(request)["run_command"]
    assert CWD_LINE not in description and WORKDIR not in description, (
        f"Working Directory Context is off, yet run_command reads {description!r}"
    )
    # the one left is the AGENTS.md lookup, which reads the home folder
    asked = terminal.requests_to("/files/cwd")
    assert len(asked) == 1, f"the terminal was asked for its folder {len(asked)} times"


def test_without_working_directory_context_the_terminal_prompt_and_agents_md_still_arrive(
    admin, terminal, member, upstream
):
    terminal_id = _connect(admin, terminal, member, working_directory_context=False)

    request = _chat_request(member, upstream, terminal_id=terminal_id)

    texts = _texts(request)
    assert any(TERMINAL_PROMPT in text for text in texts), "the terminal's system prompt is gone"
    assert f"# AGENTS.md\n\n{AGENTS_MD}" in texts, "the AGENTS.md in the home is gone"


@pytest.mark.parametrize("setting", [{}, {"working_directory_context": True}], ids=["unset", "on"])
def test_with_working_directory_context_run_command_names_the_folder(
    admin, terminal, member, upstream, setting
):
    terminal_id = _connect(admin, terminal, member, **setting)

    request = _chat_request(member, upstream, terminal_id=terminal_id)

    description = _offered(request)["run_command"]
    assert description.endswith(f"\n\n{CWD_LINE} {WORKDIR}"), description
    assert f"# AGENTS.md\n\n{AGENTS_MD}" in _texts(request)


def test_user_shell_tools_always_offers_them_with_no_shell_open(admin, terminal, member, upstream):
    terminal_id = _connect(admin, terminal, member, user_shell_tools="always")

    offered = _offered(_chat_request(member, upstream, terminal_id=terminal_id))

    assert SHELL_TOOLS <= offered.keys(), (
        f"set to Always Include, yet with the shell closed the model got {sorted(offered)}"
    )
    assert "run_command" in offered
    reading = offered["read_user_terminal"]
    assert reading.startswith("Read the user's shell.") and "Terminal tab" in reading, (
        f"the model is not told that read_user_terminal reads the Terminal tab: {reading!r}"
    )


@pytest.mark.parametrize("setting", [{}, {"user_shell_tools": "auto"}], ids=["unset", "auto"])
def test_user_shell_tools_auto_follows_the_open_shell(admin, terminal, member, upstream, setting):
    terminal_id = _connect(admin, terminal, member, **setting)

    closed = _offered(_chat_request(member, upstream, terminal_id=terminal_id))
    _open_shell(terminal, member)
    opened = _offered(_chat_request(member, upstream, terminal_id=terminal_id))

    assert not SHELL_TOOLS & closed.keys(), f"offered with the shell closed: {sorted(closed)}"
    assert "run_command" in closed
    assert SHELL_TOOLS <= opened.keys(), f"not offered with the shell open: {sorted(opened)}"
    checks = terminal.requests_to(USER_SHELL_PATH)
    assert len(checks) == 2, f"expected one shell check per turn, got {len(checks)}"
    with member.client() as client:
        for check in checks:
            chat = client.get(f"/api/v1/chats/{check.headers.get('x-session-id')}")
            assert chat.status_code == 200, (
                f"the shell check named no chat of the user: {check.headers}"
            )


def test_a_connection_saved_with_only_its_address_and_key_is_on_and_sends_the_key_as_bearer(
    admin, terminal
):
    connection_id = f"terminal-{uuid.uuid4().hex[:8]}"
    key = f"key-{uuid.uuid4().hex[:8]}"
    sent = {"id": connection_id, "url": terminal.base_url, "key": key, "region": "north"}
    with admin.client() as client:
        saved = client.post(
            TERMINAL_SERVERS_CONFIG[1], json={"TERMINAL_SERVER_CONNECTIONS": [sent]}
        )
        stored = client.get(TERMINAL_SERVERS_CONFIG[0]).json()["TERMINAL_SERVER_CONNECTIONS"]
        menu = [entry["id"] for entry in client.get("/api/v1/terminals/").json()]
        folder = client.get(f"/api/v1/terminals/{connection_id}/files/cwd")
        without_address = client.post(
            TERMINAL_SERVERS_CONFIG[1],
            json={"TERMINAL_SERVER_CONNECTIONS": [{"id": connection_id, "key": key}]},
        )
        unverified = client.post(
            "/api/v1/configs/terminal_servers/verify", json={"url": "", "key": key}
        )
        still_stored = client.get(TERMINAL_SERVERS_CONFIG[0]).json()["TERMINAL_SERVER_CONNECTIONS"]

    assert saved.status_code == 200, saved.text
    expected = {
        **sent,
        "enabled": True,
        "path": "/openapi.json",
        "auth_type": "bearer",
        "forward_cookies": False,
    }
    assert len(stored) == 1, f"one connection was saved, {len(stored)} are stored: {stored}"
    assert {name: stored[0].get(name) for name in expected} == expected, (
        f"the connection was not stored as sent, with the form's defaults: {stored}"
    )
    assert menu == [connection_id], f"the terminal menu shows {menu}"
    assert folder.status_code == 200, folder.text
    credentials = {
        path: [request.headers.get("authorization") for request in terminal.requests_to(path)]
        for path in ("/openapi.json", "/files/cwd")
    }
    assert credentials == {"/openapi.json": [f"Bearer {key}"], "/files/cwd": [f"Bearer {key}"]}, (
        f"the terminal was asked for its tool list and its folder with {credentials}"
    )
    assert without_address.status_code == 422, (
        f"a connection without an address was saved: {without_address.status_code}"
    )
    assert (unverified.status_code, unverified.json().get("detail")) == (
        400,
        "Terminal server URL is required",
    ), f"Verify Connection without an address answered {unverified.status_code} {unverified.text}"
    assert still_stored == stored, f"a refused save changed the connections: {still_stored}"


FORWARD_CONNECTION_EVENTS = """
import json
import urllib.request


class Event:
    def event(self, event):
        if event["event"] == "config.terminal_servers.updated":
            body = json.dumps(event).encode()
            headers = {{"Content-Type": "application/json"}}
            request = urllib.request.Request("{url}", data=body, headers=headers)
            urllib.request.urlopen(request, timeout=10).close()
"""


def _eventually(read, timeout: float = 10):
    """The first truthy `read()` within the timeout, else its last value."""
    deadline = time.monotonic() + timeout
    value = read()
    while not value and time.monotonic() < deadline:
        time.sleep(0.1)
        value = read()
    return value


def test_a_save_of_the_connections_is_reported_with_who_saved_and_how_many(
    admin, make_user, terminal, listener
):
    saver, member = make_user(role="admin"), make_user()
    listener.route("POST", "/events", json_answer({}))
    connections = [terminal.connection(), terminal.connection()]

    def reported() -> list[tuple]:
        events = [request.json() for request in listener.requests_to("/events")]
        return [(event["actor"]["id"], event["subject"], event["data"]) for event in events]

    source = FORWARD_CONNECTION_EVENTS.format(url=f"{listener.base_url}/events")
    with installed_function(admin, source):
        with member.client() as client:
            refused = client.post(
                TERMINAL_SERVERS_CONFIG[1], json={"TERMINAL_SERVER_CONNECTIONS": connections}
            )
        with saver.client() as client:
            configure_terminals(client, *connections)
        saves = _eventually(reported)
        with saver.client() as client:
            configure_terminals(client)
        both_saves = _eventually(lambda: reported() if len(reported()) > 1 else None)

    assert refused.status_code == 401, f"a user's save answered {refused.status_code}"
    subject = {"type": "config", "id": "terminal_server.connections"}
    assert saves == [(saver.id, subject, {"count": 2})], (
        f"one save of two connections by an admin, after a refused one, was reported as {saves}"
    )
    assert both_saves == [(saver.id, subject, {"count": 2}), (saver.id, subject, {"count": 0})], (
        f"a second save, of no connection, left the reports at {both_saves or reported()}"
    )


def test_a_connection_sent_with_the_policy_of_the_old_model_is_stored_without_it(admin, terminal):
    connection = terminal.connection(
        policy_id="default",
        policy={"image": "registry.example.com/terminal:1"},
        lifecycle={"idle_timeout_minutes": 30},
    )
    with admin.client() as client:
        saved = client.post(
            TERMINAL_SERVERS_CONFIG[1], json={"TERMINAL_SERVER_CONNECTIONS": [connection]}
        )
        stored = client.get(TERMINAL_SERVERS_CONFIG[0]).json()["TERMINAL_SERVER_CONNECTIONS"]

    assert saved.status_code == 200, saved.text
    # the kind is stored empty until Verify Connection or the dialog's Save has found it
    kept = {"server_type": None} | {
        name: value for name, value in connection.items() if name not in ("policy", "lifecycle")
    }
    assert stored == [kept], f"the connection was stored as {stored}"
    assert saved.json()["TERMINAL_SERVER_CONNECTIONS"] == [kept], (
        f"the save answered with {saved.text}"
    )


@pytest.mark.parametrize(
    "answer",
    [json_answer({"detail": "no session"}, status=500), json_answer({"home": HOME})],
    ids=["500", "no-folder-named"],
)
def test_a_terminal_that_cannot_name_its_folder_still_serves_the_turn(
    admin, terminal, member, upstream, answer
):
    terminal_id = _connect(admin, terminal, member, working_directory_context=True)
    terminal.route("GET", "/files/cwd", answer)

    request = _chat_request(member, upstream, terminal_id=terminal_id)

    description = _offered(request)["run_command"]
    assert description == "Run a shell command.", (
        f"the terminal named no folder, yet run_command reads {description!r}"
    )
    assert terminal.requests_to("/files/cwd"), "the terminal was not asked for its folder"


PERSONS_PROMPT = "You work for this person only."


def _prompt_per_caller(server: FakeTerminalServer, saved: Answer, live: Answer) -> None:
    """The terminal's prompt as read when the connection is saved, and as read for a person."""
    server.route(
        "GET", "/system", lambda request: live if "x-user-id" in request.headers else saved
    )


@pytest.mark.parametrize(
    ("live", "expected"),
    [
        (
            json_answer({"prompt": "You work for this person only."}),
            "You work for this person only.",
        ),
        (json_answer({"detail": "starting"}, status=503), TERMINAL_PROMPT),
        (json_answer({"prompt": ""}), TERMINAL_PROMPT),
    ],
    ids=["read-for-the-person", "refused-for-the-person", "empty-for-the-person"],
)
def test_the_terminal_prompt_is_read_for_the_person_and_else_as_saved_with_the_connection(
    admin, terminal, member, upstream, live, expected
):
    _prompt_per_caller(terminal, json_answer({"prompt": TERMINAL_PROMPT}), live)
    terminal_id = _connect(admin, terminal, member)

    request = _chat_request(member, upstream, terminal_id=terminal_id)

    system = "\n".join(
        str(entry["content"]) for entry in request["messages"] if entry["role"] == "system"
    )
    assert expected in system, f"the model's system prompt lacks {expected!r}: {system!r}"
    if expected != TERMINAL_PROMPT:
        assert TERMINAL_PROMPT not in system, "the prompt saved with the connection came as well"
    asked_as_the_person = [
        entry.headers["x-user-id"]
        for entry in terminal.requests_to("/system")
        if "x-user-id" in entry.headers
    ]
    assert asked_as_the_person == [member.id], asked_as_the_person


@pytest.mark.parametrize(
    "config",
    [
        json_answer({"features": {"system": False}}),
        json_answer({"features": {}}),
        json_answer({"features": {"system": True}}, status=500),
    ],
    ids=["feature-off", "feature-absent", "config-refused"],
)
def test_a_terminal_that_does_not_announce_a_prompt_is_not_asked_for_one(
    admin, terminal, member, upstream, config
):
    terminal_id = _connect(admin, terminal, member)
    with_prompt = _chat_request(member, upstream, terminal_id=terminal_id)
    assert any(TERMINAL_PROMPT in text for text in _texts(with_prompt)), (
        "the terminal's prompt did not arrive while it announced one"
    )
    terminal.route("GET", "/api/config", config)
    # saved again, so the prompt kept with the connection is read anew
    _save_again(admin)
    terminal.clear()

    request = _chat_request(member, upstream, terminal_id=terminal_id)

    assert not any(TERMINAL_PROMPT in text for text in _texts(request)), (
        "the model got the prompt of a terminal that announces none"
    )
    assert terminal.requests_to("/api/config"), "the terminal's features were never read"
    assert not terminal.requests_to("/system"), "the terminal was asked for a prompt all the same"


def _save_again(admin) -> None:
    with admin.client() as client:
        saved = client.get(TERMINAL_SERVERS_CONFIG[0]).json()
        configure_terminals(client, *saved["TERMINAL_SERVER_CONNECTIONS"])


@pytest.mark.parametrize(
    "answer",
    [
        json_answer({"detail": "shell crashed"}, status=500),
        json_answer({"output": None}),
        json_answer(["$ "]),
    ],
    ids=["500", "no-output", "not-a-shell-state"],
)
def test_a_shell_check_the_terminal_fails_hides_the_shell_tools_and_the_turn_goes_on(
    admin, terminal, member, upstream, answer
):
    terminal_id = _connect(admin, terminal, member, user_shell_tools="auto")
    terminal.route("GET", USER_SHELL_PATH, answer)

    failed = _offered(_chat_request(member, upstream, terminal_id=terminal_id))
    _open_shell(terminal, member)
    opened = _offered(_chat_request(member, upstream, terminal_id=terminal_id))

    assert "run_command" in failed and not SHELL_TOOLS & failed.keys(), (
        f"with a failed shell check the model got {sorted(failed)}"
    )
    assert SHELL_TOOLS <= opened.keys(), f"not offered with the shell open: {sorted(opened)}"


@pytest.mark.slow
def test_a_shell_check_the_terminal_never_answers_does_not_hold_the_turn(
    admin, terminal, member, upstream
):
    terminal_id = _connect(admin, terminal, member, user_shell_tools="auto")
    answered = threading.Event()

    def after_the_turn(_request):
        answered.wait(30)
        return json_answer({"output": "$ "})

    terminal.route("GET", USER_SHELL_PATH, after_the_turn)
    try:
        offered = _offered(_chat_request(member, upstream, terminal_id=terminal_id))
    finally:
        answered.set()

    assert "run_command" in offered and not SHELL_TOOLS & offered.keys(), sorted(offered)
