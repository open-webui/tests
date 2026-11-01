"""Regression: a stray space in a terminal API key closed the interactive terminal.

Issue open-webui/open-webui#25613, fixed by 58f917031 (PR open-webui/open-webui#25686). The key
a user types for a terminal in their own settings can carry surrounding whitespace. The page
calls that terminal straight from the browser: HTTP calls send the key in the Authorization
header, where a trailing space is dropped, so file listing worked; the terminal WebSocket sends
it inside a JSON `auth` message the server compares exactly, so the shell showed
"[Connection closed]". The fix trims the key in the WebSocket auth message and when the
connection is saved, and the HTTP calls trim it too.

The terminal is the local fake from harness/terminal_server.py, answering the page across
origins. A key saved before the fix, with spaces on both sides, must reach the WebSocket and the
session request trimmed (a leading space survives in a header); a key typed with a trailing
space into the Add Connection form is saved trimmed.

Discriminates: passes on the ef67cc3fa build; on a build with `.trim()` dropped from the
WebSocket auth token the auth test fails, with it dropped from the Bearer header of the terminal
requests the header test fails, and with the trim on save removed the settings test fails.
"""

from __future__ import annotations

import time

import pytest
from playwright.sync_api import Page

from harness.actors import Actor
from harness.listener import json_answer
from harness.terminal_server import FakeTerminalServer, serving_terminal
from utils.cached_chat import open_new_terminal, pick_terminal
from utils.user_menu import open_settings

pytestmark = [pytest.mark.regression, pytest.mark.requires_browser, pytest.mark.requires_source]

NAME = "My laptop"
KEY = "sesame"
HOME = "/home/ada"
SPEC = {"openapi": "3.0.0", "info": {"title": "Open Terminal", "version": "1"}, "paths": {}}


@pytest.fixture
def personal_terminal():
    """A terminal that answers the browser from another origin; yields the fake server."""
    with serving_terminal() as server:
        server.cors = True
        server.route("GET", "/openapi.json", json_answer(SPEC))
        server.route("GET", "/api/config", json_answer({"features": {"terminal": True}}))
        server.route("GET", "/files/cwd", json_answer({"cwd": HOME, "home": HOME}))
        server.route("GET", "/files/list", json_answer({"dir": HOME, "entries": []}))
        server.route("GET", "/api/config", json_answer({"features": {"terminal": True}}))
        server.route("POST", "/api/terminals", json_answer({"id": "session-1"}))
        yield server


def save_personal_terminal(account: Actor, server: FakeTerminalServer, key: str) -> None:
    terminal = {
        "url": server.base_url,
        "key": key,
        "name": NAME,
        "enabled": False,
        "auth_type": "bearer",
        "path": "/openapi.json",
        "config": {"enable": True},
    }
    with account.client() as client:
        saved = client.post(
            "/api/v1/users/user/settings/update",
            json={"ui": {"showChangelog": False, "terminalServers": [terminal]}},
        )
    assert saved.status_code == 200, saved.text


def open_terminal_shell(page: Page) -> None:
    page.reload()
    pick_terminal(page, NAME)
    open_new_terminal(page)


def wait_for_session_auth(server: FakeTerminalServer, timeout: float = 15.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        authed = [session.auth for session in server.sessions if session.auth is not None]
        if authed:
            return authed[-1]
        time.sleep(0.1)
    raise AssertionError(f"the page never opened a terminal session: {server.received}")


@pytest.fixture
def opened_with_a_padded_key(
    page_for, make_user, personal_terminal, direct_integrations
) -> FakeTerminalServer:
    """The dock opened on a terminal whose key was saved with spaces around it."""
    direct_integrations(True)
    account = make_user(role="admin")
    page = page_for(account)
    save_personal_terminal(account, personal_terminal, f"  {KEY}  ")
    open_terminal_shell(page)
    wait_for_session_auth(personal_terminal)
    return personal_terminal


def test_the_terminal_websocket_authenticates_with_the_trimmed_key(opened_with_a_padded_key):
    auth = wait_for_session_auth(opened_with_a_padded_key)

    assert auth["type"] == "auth"
    assert auth["token"] == KEY, (
        f"the terminal WebSocket sent the key as {auth['token']!r}, so the terminal refuses it "
        "and closes the session (#25613)"
    )


def test_every_request_to_the_terminal_carries_the_trimmed_key(opened_with_a_padded_key):
    # the file browser and the terminal; the tool spec is loaded by the tool server client
    sent = {
        f"{request.method} {request.path}": request.headers["authorization"]
        for request in opened_with_a_padded_key.received
        if request.path.startswith(("/api/", "/files/")) and "authorization" in request.headers
    }
    assert "POST /api/terminals" in sent, f"the session was not requested with the key: {sent}"

    untrimmed = {call: header for call, header in sent.items() if header != f"Bearer {KEY}"}
    assert not untrimmed, f"these calls sent the terminal key untrimmed: {untrimmed} (#25613)"


def test_a_key_typed_with_a_trailing_space_is_saved_trimmed(
    page_for, make_user, personal_terminal, direct_integrations
):
    direct_integrations(True)
    account = make_user(role="admin")
    page = page_for(account)
    open_settings(page)
    settings = page.get_by_role("dialog")
    # the personal tab comes before the admin tab of the same name
    settings.get_by_role("tab", name="Integrations").first.click()
    terminals = settings.get_by_role("heading", name="Open Terminal").locator("xpath=../..")
    terminals.get_by_role("button", name="Add Connection").click()

    adding = page.get_by_role("dialog").filter(has_text="Add Terminal Connection")
    adding.get_by_role("textbox", name="Name").fill(NAME)
    adding.get_by_role("textbox", name="URL").fill(personal_terminal.base_url)
    adding.get_by_placeholder("API Key").fill(f"{KEY} ")
    with page.expect_response(
        lambda response: response.url.endswith("/api/v1/users/user/settings/update")
    ):
        adding.get_by_role("button", name="Save").click()

    with account.client() as client:
        stored = client.get("/api/v1/users/user/settings").json()["ui"]["terminalServers"]
    assert [terminal["key"] for terminal in stored] == [KEY], (
        f"the key was saved as {[terminal['key'] for terminal in stored]} (#25613)"
    )
