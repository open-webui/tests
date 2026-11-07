"""Regression: the terminal's file browser jumped back to its top folder after every command.

Issue open-webui/open-webui#30051, fix `878c90eaa` (PR open-webui/open-webui#31878). A terminal
can limit its file browser to a top folder. Every `run_command` the model made sent the page a
`terminal:run_command` event, which pointed the browser at `/`: outside that top folder, so the
panel went back to the top folder and closed the open file, and the next command ran there. The
panel now stays in the folder the user picked and only reloads it.

A fake terminal answers what the file browser asks (its config, working folder and top folder,
listings and file contents) and the scripted model runs a command through it.

Discriminates: passes on the dev b859124f9 build, fails on that build with 878c90eaa reverted
(after the command the browser lists the top folder again and the open file is closed).
"""

from __future__ import annotations

import uuid
from urllib.parse import parse_qs, urlparse

import pytest
from playwright.sync_api import Page, expect

from harness import upstream as reply
from harness.listener import json_answer
from harness.terminal_server import TERMINAL_SERVERS_CONFIG, configure_terminals, serving_terminal
from utils.cached_chat import pick_terminal
from utils.chat_ui import expect_reply, send, split_screen

pytestmark = [pytest.mark.regression, pytest.mark.requires_browser, pytest.mark.requires_source]

NAME = "Project shell"
ROOT = "/home/ada/"
NOTES = "/home/ada/notes/"
LISTINGS = {
    ROOT: [{"name": "notes", "type": "directory"}, {"name": "readme.md", "type": "file"}],
    NOTES: [{"name": "todo.txt", "type": "file", "size": 22}],
}
TODO = "buy lamp oil on friday"
RUN_COMMAND = {
    "operationId": "run_command",
    "requestBody": {
        "required": True,
        "content": {
            "application/json": {
                "schema": {
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                    "required": ["command"],
                }
            }
        },
    },
    "responses": {"200": {"description": "ok"}},
}
SPEC = {
    "openapi": "3.0.0",
    "info": {"title": "Open Terminal", "version": "1"},
    "paths": {"/execute": {"post": RUN_COMMAND}},
}


def _listing(request):
    directory = parse_qs(urlparse(request.path).query).get("directory", [ROOT])[0]
    return json_answer({"entries": LISTINGS.get(directory, []), "writable": True})


@pytest.fixture
def terminal(admin, preserve):
    """A connected terminal whose file browser is limited to `ROOT`; yields its connection id."""
    preserve(TERMINAL_SERVERS_CONFIG)
    with serving_terminal() as server:
        server.route("GET", "/openapi.json", json_answer(SPEC))
        server.route("GET", "/api/config", json_answer({"features": {"terminal": True}}))
        server.route(
            "GET",
            "/files/cwd",
            json_answer({"cwd": ROOT, "home": ROOT, "root": {"path": ROOT, "label": "ada"}}),
        )
        server.route("POST", "/files/cwd", json_answer({"cwd": ROOT}))
        server.route("GET", "/files/list", _listing)
        server.route("GET", "/files/read", json_answer({"content": TODO}))
        server.route("POST", "/execute", json_answer({"exit_code": 0, "output": "ok"}))
        connection = server.connection(name=NAME)
        with admin.client() as client:
            configure_terminals(client, connection)
        yield connection["id"]


def _open_terminal_browser(page: Page):
    pick_terminal(page, NAME)
    split_screen(page)
    browser = page.get_by_role("region", name="File browser")
    expect(browser.get_by_text("readme.md")).to_be_visible()
    return browser


def _run_a_command(page: Page, upstream) -> None:
    question = f"tidy up {uuid.uuid4().hex[:6]}"
    upstream.queue(
        reply.tool_call("run_command", {"command": "ls"}, match=reply.answering(question)),
        reply.text("Tidied.", match=reply.answering(question)),
    )
    send(page, question)
    expect_reply(page, "Tidied.")


def test_the_file_browser_stays_in_the_picked_folder_after_a_command(
    page_for, make_user, terminal, upstream
):
    page = page_for(make_user(role="admin"))
    browser = _open_terminal_browser(page)
    browser.get_by_text("notes", exact=True).click()
    expect(browser.get_by_text("todo.txt")).to_be_visible()

    _run_a_command(page, upstream)

    expect(browser.get_by_text("todo.txt")).to_be_visible()
    expect(browser.get_by_text("readme.md")).to_have_count(0)


def test_an_open_file_stays_open_after_a_command(page_for, make_user, terminal, upstream):
    page = page_for(make_user(role="admin"))
    browser = _open_terminal_browser(page)
    browser.get_by_text("notes", exact=True).click()
    browser.get_by_text("todo.txt").click()
    expect(browser.get_by_text(TODO)).to_be_visible()

    _run_a_command(page, upstream)

    expect(browser.get_by_text(TODO)).to_be_visible()
    expect(browser.get_by_text("readme.md")).to_have_count(0)
