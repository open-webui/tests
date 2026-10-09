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

The user's tab is a socket client that answers the server's shell check the way the web client
does. A personal terminal (one the browser calls) carries the setting in the request itself.

Discriminates: passes on dev 1c010b438; with a20b622ba's backend hunks reverted in a backend copy
the working-directory-off test and both `always` tests fail (the description names the folder; the
shell tools are dropped). The default, `auto`, open-shell and AGENTS.md tests pass on both.
"""

from __future__ import annotations

import uuid
from contextlib import contextmanager
from typing import Iterator

import pytest

from harness import upstream as reply
from harness.chat import ask
from harness.listener import json_answer
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
        "/user-terminal/read": {"get": _operation("read_user_terminal", "Read the user's shell.")},
        "/user-terminal/input": {
            "post": _operation("send_user_terminal_input", "Type into the user's shell.")
        },
    },
}
PERSONAL_SPECS = [
    {"name": name, "description": name, "parameters": {"type": "object", "properties": {}}}
    for name in ("run_command", *sorted(SHELL_TOOLS))
]


@pytest.fixture
def terminal(preserve) -> Iterator[FakeTerminalServer]:
    preserve(TERMINAL_SERVERS_CONFIG)
    with serving_terminal() as server:
        server.route("GET", "/openapi.json", json_answer(SPEC))
        server.route("GET", "/files/cwd", json_answer({"cwd": WORKDIR, "home": HOME}))
        server.route("GET", "/files/read", json_answer({"content": AGENTS_MD}))
        server.route("GET", "/api/config", json_answer({"features": {"system": True}}))
        server.route("GET", "/system", json_answer({"prompt": TERMINAL_PROMPT}))
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
def _tab(actor, shell_open: bool) -> Iterator[str]:
    """A signed-in tab answering the server's shell check; yields its socket id."""

    def handle(event: dict):
        data = event.get("data") or {}
        if data.get("type") == "request:terminal:state":
            return {"connected": shell_open}
        return None

    with connected(actor) as session:
        session.client.on("events", handle)
        yield session.client.get_sid(namespace="/")


def _chat_request(actor, upstream, shell_open: bool = False, **options) -> dict:
    """One turn from a tab of `actor`; returns the request the model was sent."""
    prompt = f"check the build {uuid.uuid4().hex[:6]}"
    upstream.queue(reply.text("done", match=reply.answering(prompt)))
    with _tab(actor, shell_open) as session_id, actor.client() as client:
        _, message = ask(client, prompt, session_id=session_id, **options)
    assert message["content"] == "done", message
    return next(
        request
        for request in reversed(upstream.chat_requests())
        if prompt in str(request["messages"][-1].get("content"))
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


@pytest.mark.parametrize("setting", [{}, {"user_shell_tools": "auto"}], ids=["unset", "auto"])
def test_user_shell_tools_auto_follows_the_open_shell(admin, terminal, member, upstream, setting):
    terminal_id = _connect(admin, terminal, member, **setting)

    closed = _offered(_chat_request(member, upstream, terminal_id=terminal_id))
    opened = _offered(_chat_request(member, upstream, shell_open=True, terminal_id=terminal_id))

    assert not SHELL_TOOLS & closed.keys(), f"offered with the shell closed: {sorted(closed)}"
    assert "run_command" in closed
    assert SHELL_TOOLS <= opened.keys(), f"not offered with the shell open: {sorted(opened)}"


@pytest.mark.parametrize("setting", ["always", "auto"])
def test_a_personal_terminal_carries_its_user_shell_tools_setting(make_user, upstream, setting):
    # personal terminals need the direct tool servers permission, which admins have
    owner = make_user(role="admin")
    address = f"http://127.0.0.1:9/{uuid.uuid4().hex[:6]}"
    personal = {
        "url": address,
        "key": "",
        "is_terminal": True,
        "specs": PERSONAL_SPECS,
        "config": {"user_shell_tools": setting},
    }

    request = _chat_request(owner, upstream, terminal_id=address, tool_servers=[personal])

    offered = _offered(request).keys()
    assert "run_command" in offered
    assert (SHELL_TOOLS <= offered) is (setting == "always"), (
        f"a personal terminal set to {setting!r} with its shell closed offered {sorted(offered)}"
    )
