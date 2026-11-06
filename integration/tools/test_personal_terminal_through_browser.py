"""Regression: a personal terminal's skills and AGENTS.md never reached the model.

Fix `7ad0ae468` (issue open-webui/open-webui#30410): a personal terminal is one the browser talks
to, such as Open Terminal on the user's own machine, which the server usually cannot reach. The
server still fetched its skill list, its skills and the AGENTS.md in its home folder itself, so
they were silently missing whenever only the browser could reach the terminal. The server now
asks the user's open tab to read them (`request:terminal` on the tab's socket), the way it runs
the terminal's tools.

The tab here is a socket client answering those requests; the terminal's address is one nothing
listens on, as a machine behind the user's router looks from the server.

Discriminates: passes on dev efe63bd34; with 7ad0ae468 reverted in a backend copy (3f5881c52,
which touches the same lines, undone first) all three narrow tests fail: the server connects to
the terminal itself and the chat ends in a connection error. The admin-terminal test passes on
both.

Issue open-webui/open-webui#33811: the page sends the folder its file browser has open with a new
chat, and the server tried to move a personal terminal there itself, so the reply was only
"Working directory is unavailable". The browser-folder test passes on c165ee14f and on the fix
branch fix/personal-terminal-working-dir and fails on dev da9bca0fc.
"""

from __future__ import annotations

import urllib.parse
from contextlib import contextmanager
from typing import Iterator

import pytest

from harness import upstream as reply
from harness.chat import ask
from harness.listener import json_answer
from harness.socket_client import connected
from harness.terminal_server import TERMINAL_SERVERS_CONFIG, configure_terminals, serving_terminal

pytestmark = [pytest.mark.regression, pytest.mark.api, pytest.mark.requires_source]

UNREACHABLE = "http://127.0.0.1:9"
HOME = "/home/ada"
AGENTS_MD = "Always run the linter before committing."
SKILL_BODY = "Run make ship, then tag the release."
SKILL_LISTING = [
    {
        "id": "terminal:ship-it",
        "name": "ship-it",
        "description": "Ship the app to production",
        "location": f"{HOME}/.skills/ship-it/SKILL.md",
    }
]
SKILL = {
    "name": "ship-it",
    "description": "Ship the app to production",
    "content": SKILL_BODY,
    "location": f"{HOME}/.skills/ship-it/SKILL.md",
    "resources": [],
}
RUN_COMMAND = {
    "name": "run_command",
    "description": "Run a shell command.",
    "parameters": {
        "type": "object",
        "properties": {"command": {"type": "string", "description": "The command."}},
        "required": ["command"],
    },
}


def _terminal_answer(path: str):
    """What the personal terminal answers for `path`, as the tab reads it."""
    parsed = urllib.parse.urlsplit(path)
    query = dict(urllib.parse.parse_qsl(parsed.query))
    if parsed.path == "/skills":
        return SKILL_LISTING
    if parsed.path == "/skills/read" and query.get("name") == "ship-it":
        return SKILL
    if parsed.path == "/files/cwd":
        return {"cwd": HOME, "home": HOME}
    if parsed.path == "/files/read" and query.get("path") == f"{HOME}/AGENTS.md":
        return {"path": f"{HOME}/AGENTS.md", "content": AGENTS_MD}
    return None


@contextmanager
def _browser_tab(actor) -> Iterator[tuple[str, list[str]]]:
    """A tab that reads the personal terminal for the server; yields its sid and the paths."""
    asked: list[str] = []

    def handle(event: dict):
        data = event.get("data") or {}
        if data.get("type") != "request:terminal":
            return None
        request = data.get("data") or {}
        asked.append(request.get("path"))
        return {"data": _terminal_answer(request.get("path") or "")}

    with connected(actor) as session:
        session.client.on("events", handle)
        yield session.client.get_sid(namespace="/"), asked


def _personal_terminal_chat(client, upstream, session_id: str, text: str) -> dict:
    """Chat with the personal terminal selected; returns the request the model was sent last."""
    personal = {"url": UNREACHABLE, "key": "", "is_terminal": True, "specs": [RUN_COMMAND]}
    _, message = ask(
        client, text, session_id=session_id, terminal_id=UNREACHABLE, tool_servers=[personal]
    )
    assert not message.get("error"), (
        f"the chat on a terminal only the browser can reach failed: {message['error']} (#30410)"
    )
    return upstream.chat_requests()[-1]


def _system_text(request: dict) -> str:
    return "\n".join(
        str(entry["content"]) for entry in request["messages"] if entry["role"] == "system"
    )


def test_the_skill_list_is_read_through_the_browser(make_user, upstream):
    owner = make_user(role="admin")
    upstream.queue(reply.text("ready"))
    with _browser_tab(owner) as (session_id, asked), owner.client() as client:
        request = _personal_terminal_chat(client, upstream, session_id, "what can you do?")

    system = _system_text(request)
    assert "/skills" in asked, f"the tab was never asked for the terminal's skills: {asked}"
    assert "<id>terminal:ship-it</id>" in system, (
        "the personal terminal's skills never reached the model (#30410)"
    )


def test_the_home_agents_md_is_read_through_the_browser(make_user, upstream):
    owner = make_user(role="admin")
    upstream.queue(reply.text("ready"))
    with _browser_tab(owner) as (session_id, _), owner.client() as client:
        request = _personal_terminal_chat(client, upstream, session_id, "lint and commit")

    sent = [entry["content"] for entry in request["messages"]]
    assert f"# AGENTS.md\n\n{AGENTS_MD}" in sent, (
        "the AGENTS.md in the personal terminal's home never reached the model (#30410)"
    )


def test_a_skill_the_model_opens_is_read_through_the_browser(make_user, upstream):
    owner = make_user(role="admin")
    upstream.queue(reply.tool_call("view_skill", {"id": "terminal:ship-it"}), reply.text("On it."))
    with _browser_tab(owner) as (session_id, asked), owner.client() as client:
        request = _personal_terminal_chat(client, upstream, session_id, "ship it")

    results = [entry["content"] for entry in request["messages"] if entry["role"] == "tool"]
    assert results and SKILL_BODY in results[-1], (
        f"the model opened the personal terminal's skill and got {results} (#30410)"
    )
    assert "/skills/read?name=ship-it" in asked


def test_a_new_chat_in_the_file_browsers_folder_gets_its_reply(make_user, upstream):
    owner = make_user(role="admin")
    upstream.queue(reply.text("ready"))
    personal = {"url": UNREACHABLE, "key": "", "is_terminal": True, "specs": [RUN_COMMAND]}
    with _browser_tab(owner) as (session_id, _), owner.client() as client:
        _, message = ask(
            client,
            "what is in this folder?",
            session_id=session_id,
            terminal_id=UNREACHABLE,
            terminal_cwd=f"{HOME}/project",
            tool_servers=[personal],
        )

    assert not message.get("error") and message["content"] == "ready", (
        f"a new chat on a personal terminal opened in a folder got no reply: {message} (#33811)"
    )


@pytest.fixture
def admin_terminal(admin, preserve):
    """A terminal the admin connected, which the server reads itself; yields its id."""
    preserve(TERMINAL_SERVERS_CONFIG)
    operation = {"operationId": "run_command", "responses": {"200": {"description": "ok"}}}
    spec = {
        "openapi": "3.0.0",
        "info": {"title": "terminal", "version": "1"},
        "paths": {"/execute": {"post": operation}},
    }
    with serving_terminal() as terminal:
        terminal.route("GET", "/openapi.json", json_answer(spec))
        terminal.route("GET", "/files/cwd", json_answer({"cwd": HOME, "home": HOME}))
        terminal.route("GET", "/files/read", json_answer({"content": AGENTS_MD}))
        connection = terminal.connection()
        with admin.client() as client:
            configure_terminals(client, connection)
        yield connection["id"]


def test_an_admin_terminal_is_still_read_by_the_server(admin_terminal, make_user, upstream):
    owner = make_user(role="admin")
    upstream.queue(reply.text("ready"))
    with owner.client() as client:
        _, message = ask(client, "lint and commit", terminal_id=admin_terminal)

    assert message["content"] == "ready", message
    sent = [entry["content"] for entry in upstream.chat_requests()[-1]["messages"]]
    assert f"# AGENTS.md\n\n{AGENTS_MD}" in sent
