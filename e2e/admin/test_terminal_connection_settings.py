"""Regression: an admin sets Working Directory Context and User Shell Tools on a terminal.

Commit `a20b622ba` added both to the Advanced part of the Add/Edit Terminal Connection dialog
under Admin Settings > Integrations > Terminal. Working Directory Context is a switch, on by
default; User Shell Tools is a select of Automatic (the default) and Always Include. Saving
stores them in the connection's `config` as `working_directory_context` and `user_shell_tools`,
and reopening the connection shows them as saved. A connection saved without touching them
stores the defaults. integration/tools/test_terminal_connection_settings.py covers what each
setting changes for the model.

Discriminates: passes on dev 1c010b438 with its built frontend; on a build with a20b622ba (and the
unrelated 7d205a86d) reverted both tests fail (the dialog has neither control and the connection
is stored without them).
"""

from __future__ import annotations

import secrets
from typing import Iterator

import pytest
from playwright.sync_api import Locator, Page, expect

from harness.listener import json_answer
from harness.terminal_server import TERMINAL_SERVERS_CONFIG, FakeTerminalServer, serving_terminal
from utils.tool_servers import connection_row
from utils.tooltips import tooltip_button

pytestmark = [pytest.mark.regression, pytest.mark.requires_browser, pytest.mark.requires_source]

SPEC = {"openapi": "3.0.0", "info": {"title": "Open Terminal", "version": "1"}, "paths": {}}


@pytest.fixture
def terminal() -> Iterator[FakeTerminalServer]:
    with serving_terminal() as server:
        server.route("GET", "/openapi.json", json_answer(SPEC))
        yield server


@pytest.fixture
def admin_account(make_user, preserve):
    preserve(TERMINAL_SERVERS_CONFIG)
    return make_user(role="admin")


def terminal_section(page: Page) -> Locator:
    page.goto("/admin/settings/integrations")
    heading = page.get_by_role("dialog").get_by_text("Open Terminal", exact=True)
    expect(heading).to_be_visible()
    return heading.locator("xpath=../..")


def open_add_dialog(page: Page) -> Locator:
    tooltip_button(terminal_section(page), "Add Connection").click()
    return terminal_dialog(page, "Add Terminal Connection")


def terminal_dialog(page: Page, title: str) -> Locator:
    dialog = page.get_by_role("dialog").filter(has_text=title)
    expect(dialog).to_be_visible()
    return dialog


def show_advanced(dialog: Locator) -> None:
    dialog.get_by_role("button", name="Advanced", exact=True).click()
    expect(dialog.get_by_label("User Shell Tools")).to_be_visible()


def fill_and_save(page: Page, dialog: Locator, name: str, url: str) -> None:
    dialog.get_by_label("Name", exact=True).fill(name)
    dialog.get_by_label("URL", exact=True).fill(url)
    with page.expect_response(
        lambda response: (
            response.url.endswith(TERMINAL_SERVERS_CONFIG[1]) and response.request.method == "POST"
        )
    ) as saved:
        dialog.get_by_role("button", name="Save", exact=True).click()
    assert saved.value.ok, saved.value.text()


def stored_config(admin_account, name: str) -> dict:
    with admin_account.client() as client:
        connections = client.get(TERMINAL_SERVERS_CONFIG[0]).json()["TERMINAL_SERVER_CONNECTIONS"]
    [connection] = [entry for entry in connections if entry.get("name") == name]
    return connection.get("config") or {}


def test_both_settings_are_saved_and_shown_again(page_for, admin_account, terminal):
    page = page_for(admin_account)
    name = f"Build box {secrets.token_hex(3)}"
    dialog = open_add_dialog(page)
    show_advanced(dialog)
    working_directory = dialog.get_by_role("switch", name="Working Directory Context")
    expect(working_directory).to_be_checked()
    working_directory.click()
    expect(working_directory).not_to_be_checked()
    dialog.get_by_label("User Shell Tools").select_option(label="Always Include")
    expect(dialog.get_by_text("Keep tools listed even while your shell is closed.")).to_be_visible()

    fill_and_save(page, dialog, name, terminal.base_url)

    config = stored_config(admin_account, name)
    assert config.get("working_directory_context") is False, config
    assert config.get("user_shell_tools") == "always", config

    section = terminal_section(page)
    tooltip_button(connection_row(section, name), "Configure").click()
    reopened = terminal_dialog(page, "Edit Terminal Connection")
    show_advanced(reopened)
    expect(reopened.get_by_role("switch", name="Working Directory Context")).not_to_be_checked()
    expect(reopened.get_by_label("User Shell Tools")).to_have_value("always")


def test_a_connection_saved_untouched_stores_the_defaults(page_for, admin_account, terminal):
    page = page_for(admin_account)
    name = f"Build box {secrets.token_hex(3)}"
    dialog = open_add_dialog(page)
    show_advanced(dialog)
    expect(dialog.get_by_role("switch", name="Working Directory Context")).to_be_checked()
    expect(dialog.get_by_label("User Shell Tools")).to_have_value("auto")

    fill_and_save(page, dialog, name, terminal.base_url)

    config = stored_config(admin_account, name)
    assert config.get("working_directory_context") is True, config
    assert config.get("user_shell_tools") == "auto", config
