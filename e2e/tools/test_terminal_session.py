"""Journey: a terminal the admin connects under Integrations opens a live shell from a chat.

A fresh admin adds a terminal connection to a local fake terminal server in the admin settings
and switches it on, each change saving by itself; it is still there, on, when the tab is opened
again. In a chat the admin picks the terminal from the input's Terminal menu, which opens its
file browser, and expands the terminal dock. The page opens a session through the instance's
proxy, and what is typed reaches the fake shell and comes back on screen.

Discriminates: passes on dev ac00d40e3; in a backend copy, with the terminal WebSocket proxy
dropping what the browser sends the typed command never reaches the shell and nothing comes
back.
"""

from __future__ import annotations

import time

import pytest
from playwright.sync_api import Locator, Page, expect

from harness.listener import json_answer
from harness.terminal_server import TERMINAL_SERVERS_CONFIG, serving_terminal
from utils.chat_ui import chat_input
from utils.tooltips import tooltip_button

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

NAME = "Lab shell"
TYPED = "echo lighthouse"


@pytest.fixture
def fake_terminal():
    with serving_terminal() as server:
        server.route("POST", "/api/terminals", json_answer({"id": "session-1"}))
        yield server


def _terminal_section(page: Page) -> Locator:
    page.goto("/admin/settings/integrations")
    settings = page.get_by_role("dialog")
    section = settings.get_by_role("heading", name="Terminal", exact=True).locator("xpath=..")
    expect(section.get_by_text("Open Terminal", exact=True)).to_be_visible()
    return section


def _saves_terminals(response) -> bool:
    # each change to the list is saved on its own, without the tab's Save button
    return response.request.method == "POST" and response.url.endswith(TERMINAL_SERVERS_CONFIG[1])


def _screen_text(terminal: Locator, text: str, timeout: float = 15.0) -> None:
    """Wait until `text` is on the terminal's screen; long lines wrap onto the next row."""
    deadline = time.monotonic() + timeout
    shown = ""
    while time.monotonic() < deadline:
        shown = terminal.inner_text().replace("\n", "")
        if text in shown:
            return
        time.sleep(0.2)
    raise AssertionError(f"the terminal never showed {text!r}; it shows {shown!r}")


def test_a_connected_terminal_opens_a_shell_from_the_chat(
    page_for, make_user, preserve, fake_terminal
):
    preserve(TERMINAL_SERVERS_CONFIG)
    page = page_for(make_user(role="admin"))
    section = _terminal_section(page)
    expect(section.get_by_text("No terminal connections configured")).to_be_visible()
    tooltip_button(section, "Add Connection").click()
    adding = page.get_by_role("dialog").filter(has_text="Add Terminal Connection")
    adding.get_by_role("textbox", name="Name").fill(NAME)
    adding.get_by_role("textbox", name="URL").fill(fake_terminal.base_url)
    adding.get_by_role("combobox").last.select_option(label="None")
    with page.expect_response(_saves_terminals):
        adding.get_by_role("button", name="Save").click()
    expect(section.get_by_text(NAME)).to_be_visible()
    with page.expect_response(_saves_terminals):
        section.get_by_role("switch").click()

    section = _terminal_section(page)
    expect(section.get_by_text(NAME)).to_be_visible()
    expect(section.get_by_role("switch")).to_be_checked()

    page.goto("/")
    expect(chat_input(page)).to_be_visible()
    tooltip_button(page.get_by_role("main"), "Terminal").click()
    page.get_by_role("menu").get_by_role("button", name=NAME).click()
    browser = page.get_by_role("region", name="File browser")
    browser.get_by_role("button", name="Expand terminal").click()
    expect(browser.get_by_role("tab", name="Shell")).to_be_visible()

    browser.get_by_role("textbox", name="Terminal input").press_sequentially(TYPED)
    _screen_text(browser.locator(".xterm-rows"), TYPED)

    assert fake_terminal.requests_to("/api/terminals"), "no session was created on the terminal"
    session = fake_terminal.sessions[-1]
    assert session.path.startswith("/api/terminals/session-1")
    typed = b"".join(part for part in session.messages if isinstance(part, bytes))
    assert typed == TYPED.encode(), session.messages
