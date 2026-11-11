"""Journey: what a terminal an admin connected gets from the model's tool calls, and what the
model gets back when the terminal refuses, is gone or has no usable tool list.

A chat sent with a terminal picked offers the model every operation of the terminal's OpenAPI
document. Each call the model makes goes out with the credential the connection's Auth setting
asks for (Bearer: the connection's key, Session: the person's own session, None and System OAuth
without an OAuth sign-in: no credential), names the person (`X-User-Id`) and the chat
(`X-Session-Id`), and carries the browser's cookies only with Forward cookies on. An answer of
400 or above, a terminal that went away and one slower than `AIOHTTP_CLIENT_TIMEOUT_TOOL_SERVER`
all reach the model as the tool's result with an `error` in it, and the reply goes on. The error
of a timed out call is empty on dev, which leaves the model without a reason (red test, draft
issue named in it). A terminal whose tool list cannot be read (the document answers an error,
has no `paths`, or has no operation) fails the reply with "Terminal unavailable: ..." before the
model is asked; a YAML document is read like a JSON one. A `terminal_id` that is none of the
person's terminals, and any terminal on a model whose Terminal capability is off, is passed
over: the model answers without terminal tools.

Discriminates: passes on dev c15f28649. In a backend copy, each of these turned the named test
red and left the others green: the bearer key sent for every Auth setting (credential test, none
and system_oauth and session cases); `X-Session-Id` not set (credential test, all cases); the
cookies passed whatever Forward cookies says (cookie test, off case); an answer of 400 or above
handed on as a normal result (refusal test, all cases); the error of a failed call neither
returned by the call nor caught around it (gone, slow and refusal tests: the reply ends in an
error; with only the first undone the model is still handed the error); a document without
`paths` accepted, a terminal without a tool list taken as one with none, and the "has no
available tools" check dropped (one case each of the unusable tool list test); YAML no longer
read (YAML test, both cases); the check that the terminal is one of the person's dropped
(unknown terminal test: the reply fails); the Terminal capability ignored (capability test). The
timed-out test is red on dev c15f28649 and on v0.13.3 (the error is empty) and passes with the
timeout's name used as the error text.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from contextlib import ExitStack
from typing import Iterator

import pytest

from harness import upstream as reply
from harness.actors import admin_of, create_user
from harness.chat import send_message, wait_for_reply
from harness.listener import json_answer
from harness.model_presets import preset_with_capabilities
from harness.terminal_server import (
    TERMINAL_SERVERS_CONFIG,
    FakeTerminalServer,
    configure_terminals,
    read_grant,
    serving_terminal,
)

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source]

CONNECTION_KEY = "terminal-key-5f3a"
HOME = "/home/user"
AGENTS_MD = "Keep every change small."


def _run_command() -> dict:
    body = {
        "type": "object",
        "properties": {"command": {"type": "string"}},
        "required": ["command"],
    }
    return {
        "operationId": "run_command",
        "description": "Run a shell command.",
        "requestBody": {"required": True, "content": {"application/json": {"schema": body}}},
        "responses": {"200": {"description": "ok"}},
    }


SPEC = {
    "openapi": "3.0.0",
    "info": {"title": "terminal", "version": "1"},
    "paths": {"/execute": {"post": _run_command()}},
}
SPEC_AS_YAML = """\
openapi: 3.0.0
info:
  title: terminal
  version: "1"
paths:
  /execute:
    post:
      operationId: run_command
      description: Run a shell command.
      requestBody:
        required: true
        content:
          application/json:
            schema:
              type: object
              properties:
                command:
                  type: string
              required: [command]
      responses:
        "200":
          description: ok
"""


NO_PATHS = json_answer({"openapi": "3.0.0", "info": {"title": "terminal", "version": "1"}})


def _serve(server: FakeTerminalServer) -> None:
    server.route("GET", "/openapi.json", json_answer(SPEC))
    server.route("GET", "/files/cwd", json_answer({"cwd": HOME, "home": HOME}))
    server.route("GET", "/files/read", json_answer({"content": AGENTS_MD}))
    server.route("POST", "/execute", json_answer({"output": "ran"}))


@pytest.fixture
def terminal(preserve) -> Iterator[FakeTerminalServer]:
    preserve(TERMINAL_SERVERS_CONFIG)
    with serving_terminal() as server:
        _serve(server)
        yield server


@pytest.fixture
def member(make_user):
    return make_user()


def _connect(admin, server: FakeTerminalServer, member, **fields) -> str:
    """Save the terminal as the admin's dialog does, shared with `member`; returns its id."""
    connection = server.connection(config={"access_grants": [read_grant(member.id)]}, **fields)
    with admin.client() as client:
        configure_terminals(client, connection)
    server.clear()
    return connection["id"]


def _prompt() -> str:
    return f"tidy the build folder {uuid.uuid4().hex[:8]}"


def _requests_answering(upstream, prompt: str) -> list[dict]:
    answering = reply.answering(prompt)
    return [request for request in upstream.chat_requests() if answering(request)]


def _offered(upstream, prompt: str) -> set[str]:
    request = _requests_answering(upstream, prompt)[0]
    return {tool["function"]["name"] for tool in request.get("tools") or []}


def _plain_turn(actor, upstream, **options) -> str:
    """One message the model answers without a call; returns its prompt."""
    prompt = _prompt()
    upstream.queue(reply.text("done", match=reply.answering(prompt)))
    with actor.client() as client:
        message = wait_for_reply(client, send_message(client, prompt, **options))
    assert message["content"] == "done", f"the reply did not arrive: {message}"
    return prompt


def _command_turn(actor, upstream, terminal_id: str, cookies: dict | None = None):
    """The model runs one command and then answers; returns the turn, the reply and the result."""
    prompt = _prompt()
    answering = reply.answering(prompt)
    upstream.queue(
        reply.tool_call("run_command", {"command": "ls"}, match=answering),
        reply.text("the folder is tidy", match=answering),
    )
    with actor.client() as client:
        for name, value in (cookies or {}).items():
            client.cookies.set(name, value)
        turn = send_message(client, prompt, terminal_id=terminal_id)
        message = wait_for_reply(client, turn)
    return turn, message, _command_result(upstream, prompt)


def _command_result(upstream, prompt: str) -> dict | str | None:
    """What the model was handed as the command's result, None when it never was."""
    sent_back = _requests_answering(upstream, prompt)[-1]["messages"]
    results = [entry["content"] for entry in sent_back if entry["role"] == "tool"]
    if not results:
        return None
    try:
        return json.loads(results[-1])
    except ValueError:
        return results[-1]


def _went_on(message: dict, result) -> None:
    assert message.get("content") == "the folder is tidy" and not message.get("error"), (
        f"the reply did not go on after the failed command: {message}"
    )
    assert isinstance(result, dict) and "error" in result, (
        f"the model was not told that the command failed: {result!r}"
    )


@pytest.mark.parametrize("auth_type", ["bearer", "none", "session", "system_oauth"])
def test_a_command_carries_the_credential_of_the_auth_setting_and_names_person_and_chat(
    admin, terminal, member, upstream, auth_type
):
    terminal_id = _connect(admin, terminal, member, auth_type=auth_type, key=CONNECTION_KEY)
    # a password sign-in has no OAuth session, so System OAuth has no token to send
    credential = {"bearer": f"Bearer {CONNECTION_KEY}", "session": f"Bearer {member.token}"}

    turn, message, result = _command_turn(member, upstream, terminal_id)

    assert result == {"output": "ran"}, f"the command's output did not reach the model: {result}"
    assert message["content"] == "the folder is tidy", message
    [command] = terminal.requests_to("/execute")
    assert json.loads(command.body) == {"command": "ls"}
    assert command.headers.get("authorization") == credential.get(auth_type), (
        f"a terminal with Auth {auth_type} was sent {command.headers.get('authorization')!r}"
    )
    named = {name: command.headers.get(name) for name in ("x-user-id", "x-session-id")}
    assert named == {"x-user-id": member.id, "x-session-id": turn.chat_id}, (
        f"the command did not name its person and chat: {named}"
    )


@pytest.mark.parametrize("forward_cookies", [True, False], ids=["on", "off"])
def test_a_command_carries_the_browsers_cookies_only_with_forward_cookies(
    admin, terminal, member, upstream, forward_cookies
):
    terminal_id = _connect(admin, terminal, member, forward_cookies=forward_cookies)

    _command_turn(member, upstream, terminal_id, cookies={"sso_hint": "from-the-browser"})

    [command] = terminal.requests_to("/execute")
    expected = {"sso_hint": "from-the-browser"} if forward_cookies else {}
    assert command.cookies == expected, (
        f"with Forward cookies {'on' if forward_cookies else 'off'} the terminal got "
        f"{command.cookies}"
    )


@pytest.mark.parametrize("status", [401, 403, 500])
def test_a_command_the_terminal_refuses_is_handed_to_the_model_as_an_error(
    admin, terminal, member, upstream, status
):
    terminal_id = _connect(admin, terminal, member)
    refusal = f"refused {uuid.uuid4().hex[:6]}"
    terminal.route("POST", "/execute", json_answer({"detail": refusal}, status=status))

    _, message, result = _command_turn(member, upstream, terminal_id)

    _went_on(message, result)
    assert f"HTTP error {status}" in result["error"] and refusal in result["error"], (
        f"the model was not told what the terminal answered: {result}"
    )


def test_a_command_to_a_terminal_that_went_away_is_handed_to_the_model_as_an_error(
    admin, preserve, member, upstream
):
    preserve(TERMINAL_SERVERS_CONFIG)
    prompt = _prompt()
    answering = reply.answering(prompt)
    release = threading.Event()
    upstream.queue(
        reply.tool_call("run_command", {"command": "ls"}, match=answering, hold=release),
        reply.text("the folder is tidy", match=answering),
    )
    with ExitStack() as running, member.client() as client:
        server = running.enter_context(serving_terminal())
        _serve(server)
        terminal_id = _connect(admin, server, member)
        turn = send_message(client, prompt, terminal_id=terminal_id)
        _wait_until(lambda: _requests_answering(upstream, prompt), "the model was never asked")
        # the model has its tools; the terminal stops before the model's call arrives
        running.close()
        release.set()
        message = wait_for_reply(client, turn)

    assert "run_command" in _offered(upstream, prompt)
    _went_on(message, _command_result(upstream, prompt))


def _wait_until(condition, failure: str, timeout: float = 30.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if found := condition():
            return found
        time.sleep(0.05)
    raise AssertionError(failure)


def _slow_command_turn(instance_with, preserve) -> tuple[dict, dict, FakeTerminalServer]:
    """A command the terminal answers only after the one second the instance allows a tool."""
    slow = instance_with({"AIOHTTP_CLIENT_TIMEOUT_TOOL_SERVER": "1"})
    preserve(TERMINAL_SERVERS_CONFIG, on=slow)
    person = create_user(slow)
    finished = threading.Event()

    def never_in_time(_request):
        finished.wait(20)
        return json_answer({"output": "ran"})

    with serving_terminal() as server:
        _serve(server)
        server.route("POST", "/execute", never_in_time)
        terminal_id = _connect(admin_of(slow), server, person)
        try:
            _, message, result = _command_turn(person, slow.upstream, terminal_id)
        finally:
            finished.set()
    return message, result, server


@pytest.mark.slow
def test_a_command_slower_than_the_tool_timeout_is_handed_to_the_model_as_an_error(
    instance_with, preserve
):
    message, result, server = _slow_command_turn(instance_with, preserve)

    _went_on(message, result)
    assert len(server.requests_to("/execute")) == 1, "the slow command was sent again"


@pytest.mark.slow
def test_a_command_slower_than_the_tool_timeout_tells_the_model_that_it_timed_out(
    instance_with, preserve
):
    """Red on dev c15f28649 and on v0.13.3: the model is handed an empty error
    (open-webui/open-webui#34096)."""
    _, result, _ = _slow_command_turn(instance_with, preserve)

    assert isinstance(result, dict) and str(result.get("error") or "").strip(), (
        f"the model was handed {result!r} for a command that timed out: nothing says what failed"
    )


@pytest.mark.parametrize(
    ("document", "reason"),
    [
        (json_answer({"detail": "the spec is gone"}, status=500), "is unavailable"),
        (NO_PATHS, "is unavailable"),
        (json_answer({**SPEC, "paths": {}}), "has no available tools"),
    ],
    ids=["document-answers-500", "document-without-paths", "document-without-operations"],
)
def test_a_terminal_without_a_usable_tool_list_fails_the_message_and_says_why(
    admin, terminal, member, upstream, document, reason
):
    terminal.route("GET", "/openapi.json", document)
    terminal_id = _connect(admin, terminal, member)
    prompt = _prompt()

    with member.client() as client:
        message = wait_for_reply(client, send_message(client, prompt, terminal_id=terminal_id))

    failure = (message.get("error") or {}).get("content", "")
    assert failure.startswith("Terminal unavailable:") and terminal_id in failure, (
        f"the message did not fail on its terminal: {message}"
    )
    assert reason in failure, f"the message does not say what is wrong: {failure}"
    assert not _requests_answering(upstream, prompt), "the model was asked all the same"

    # the same terminal with a tool list, saved again as the admin would
    terminal.route("GET", "/openapi.json", json_answer(SPEC))
    with admin.client() as client:
        saved = client.get(TERMINAL_SERVERS_CONFIG[0]).json()
        configure_terminals(client, *saved["TERMINAL_SERVER_CONNECTIONS"])
    assert "run_command" in _offered(
        upstream, _plain_turn(member, upstream, terminal_id=terminal_id)
    )


@pytest.mark.parametrize("path", ["/openapi.yaml", "/openapi.json"], ids=["yaml-url", "json-url"])
def test_a_terminal_that_describes_its_tools_in_yaml_serves_the_model(
    admin, terminal, member, upstream, path
):
    terminal.route("GET", path, (200, {"Content-Type": "application/yaml"}, SPEC_AS_YAML.encode()))
    terminal_id = _connect(admin, terminal, member, path=path)

    _, message, result = _command_turn(member, upstream, terminal_id)

    assert result == {"output": "ran"}, f"the command did not run on the terminal: {result}"
    assert message["content"] == "the folder is tidy", message
    assert len(terminal.requests_to("/execute")) == 1


def test_a_terminal_id_that_is_none_of_the_persons_terminals_is_passed_over(
    admin, terminal, member, upstream
):
    terminal_id = _connect(admin, terminal, member)
    known = _plain_turn(member, upstream, terminal_id=terminal_id)
    assert "run_command" in _offered(upstream, known), "the person's own terminal gave no tools"
    terminal.clear()

    unknown = _plain_turn(member, upstream, terminal_id=f"terminal-{uuid.uuid4().hex[:8]}")

    assert "run_command" not in _offered(upstream, unknown), (
        f"a terminal that does not exist gave the model {_offered(upstream, unknown)}"
    )
    assert not terminal.received, (
        f"the person's terminal was called for another id: {[r.path for r in terminal.received]}"
    )


def test_a_model_with_the_terminal_capability_off_gets_nothing_of_a_picked_terminal(
    admin, terminal, member, upstream
):
    terminal_id = _connect(admin, terminal, member)
    with (
        preset_with_capabilities(admin, {"terminal": False}) as without,
        preset_with_capabilities(admin, {"terminal": True}) as with_terminal,
    ):
        allowed = _plain_turn(member, upstream, terminal_id=terminal_id, model=with_terminal)
        refused = _plain_turn(member, upstream, terminal_id=terminal_id, model=without)

    agents_md = f"# AGENTS.md\n\n{AGENTS_MD}"
    assert "run_command" in _offered(upstream, allowed)
    assert agents_md in _texts(upstream, allowed), "the AGENTS.md did not reach the capable model"
    assert "run_command" not in _offered(upstream, refused), (
        f"a model with Terminal off was offered {_offered(upstream, refused)}"
    )
    assert agents_md not in _texts(upstream, refused), (
        "a model with Terminal off was handed the terminal's AGENTS.md"
    )


def _texts(upstream, prompt: str) -> list[str]:
    [request] = _requests_answering(upstream, prompt)
    return [str(entry.get("content")) for entry in request["messages"]]
