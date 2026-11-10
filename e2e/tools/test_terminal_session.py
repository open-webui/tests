"""Journey: a terminal the admin connects under Integrations opens a live shell from a chat.

A fresh admin adds a terminal connection to a local fake terminal server in the admin settings,
with its API key, and it is on from the start (since dev 6f2fc7bc7; before, it had to be
switched on). Switching it off and on again saves each change by itself; it is still there, on,
when the tab is opened again. In a chat the admin picks the terminal from the input's Terminal
menu, which opens its file browser, and opens a New Terminal tab beside it. The page opens a
session through the instance's proxy, and what is typed reaches the fake shell and comes back on
screen. The page runs without WebGL, so the terminal keeps its fallback text rows (f789a0742
draws on a WebGL canvas where it can), the only screen a test can read.

Discriminates: passes on dev ac00d40e3; in a backend copy, with the terminal WebSocket proxy
dropping what the browser sends the typed command never reaches the shell and nothing comes
back.

Show Files on Terminal Select left Settings > Interface in def1d048b (#33855). A person who had
saved it off finds no such switch there, and can still pick a terminal in the chat.
On dev f659c0d12, a frontend copy with the switch put back turns that test red.
"""

from __future__ import annotations

import re

import pytest
from playwright.sync_api import Locator, Page, expect

from harness.listener import json_answer
from harness.terminal_server import (
    TERMINAL_SERVERS_CONFIG,
    configure_terminals,
    read_grant,
    serving_terminal,
)
from utils.cached_chat import open_new_terminal, pick_terminal
from utils.chat_ui import chat_input
from utils.shell_tab import READABLE_SHELL, expect_on_screen, type_into
from utils.terminal_connections import fill_terminal, save_terminal, terminal_dialog
from utils.tooltips import tooltip_button
from utils.user_menu import open_settings

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

NAME = "Lab shell"
TYPED = "echo lighthouse"
# the label, or its translation key where the build has no English text for it
REMOVED_SETTING = re.compile(r"show\s*files\s*on\s*terminal\s*select", re.IGNORECASE)


@pytest.fixture
def fake_terminal():
    with serving_terminal() as server:
        # the side panel offers a shell only to a terminal whose config has it on
        server.route("GET", "/api/config", json_answer({"features": {"terminal": True}}))
        server.route("POST", "/api/terminals", json_answer({"id": "session-1"}))
        # since d4e8493cc the shell looks the session up before it connects
        server.route("GET", "/api/terminals/session-1", json_answer({"id": "session-1"}))
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


def test_a_connected_terminal_opens_a_shell_from_the_chat(
    page_for, make_user, preserve, fake_terminal
):
    preserve(TERMINAL_SERVERS_CONFIG)
    page = page_for(make_user(role="admin"), init_script=READABLE_SHELL)
    section = _terminal_section(page)
    expect(section.get_by_text("No terminal connections configured")).to_be_visible()
    tooltip_button(section, "Add Connection").click()
    adding = terminal_dialog(page)
    fill_terminal(adding, NAME, fake_terminal.base_url, key="lab-shell-key")
    save_terminal(page, adding)
    expect(section.get_by_text(NAME)).to_be_visible()
    # a new connection is on from the start (6f2fc7bc7); off and on again each save by themselves
    for state in (False, True):
        with page.expect_response(_saves_terminals):
            section.get_by_role("switch").click()
        expect(section.get_by_role("switch")).to_be_checked(checked=state)

    section = _terminal_section(page)
    expect(section.get_by_text(NAME)).to_be_visible()
    expect(section.get_by_role("switch")).to_be_checked()

    page.goto("/")
    pick_terminal(page, NAME)
    shell = open_new_terminal(page)

    type_into(shell, TYPED)
    expect_on_screen(shell, TYPED)

    assert fake_terminal.requests_to("/api/terminals"), "no session was created on the terminal"
    session = fake_terminal.sessions[-1]
    assert session.path.startswith("/api/terminals/session-1")
    typed = b"".join(part for part in session.messages if isinstance(part, bytes))
    assert typed == TYPED.encode(), session.messages


def test_a_saved_show_files_choice_is_gone_from_settings_and_a_terminal_still_gets_picked(
    page_for, make_user, admin, preserve, fake_terminal
):
    preserve(TERMINAL_SERVERS_CONFIG)
    person = make_user()
    connection = fake_terminal.connection(
        name=NAME, config={"access_grants": [read_grant(person.id)]}
    )
    with admin.client() as client:
        configure_terminals(client, connection)
    # a choice saved before the setting was removed stays in the account's settings
    with person.client() as client:
        saved = client.post(
            "/api/v1/users/user/settings/update", json={"ui": {"showFilesOnTerminalSelect": False}}
        )
    assert saved.status_code == 200, saved.text

    page = page_for(person)
    expect(chat_input(page)).to_be_visible()
    tooltip_button(page.get_by_role("main"), "Terminal").click()
    page.get_by_role("menu").get_by_role("menuitemradio", name=NAME).click()
    expect(page.get_by_role("main").get_by_role("button", name=NAME, exact=True)).to_be_visible()

    open_settings(page)
    page.get_by_role("tab", name="Interface").click()
    interface = page.locator("#tab-interface")
    expect(interface.get_by_role("button", name="Terminal File Display")).to_be_visible()
    expect(interface.get_by_role("switch", name=REMOVED_SETTING)).to_have_count(0)
