"""Journey: the default permission Interface Settings Access decides who sees the Interface tab.

Admin Panel > Users > Groups > Default permissions holds Interface Settings Access. With it off an
account's Settings dialog lists no Interface tab, while an admin's still does.

Discriminates: passes on dev 30f3f6a8f; in a frontend copy, reading the permission as always
granted for the Interface tab turns the test red (the account still gets the tab).
"""

from __future__ import annotations

import pytest
from playwright.sync_api import Page, expect

from utils.chat_ui import chat_input

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]


def save_default_permission(page: Page, switch: str, turn_on: bool) -> None:
    page.goto("/admin/users/groups")
    page.get_by_role("button", name="Default permissions").click()
    dialog = page.get_by_role("dialog")
    target = dialog.get_by_role("switch", name=switch)
    if (target.get_attribute("aria-checked") == "true") != turn_on:
        target.click()
    expect(target).to_have_attribute("aria-checked", "true" if turn_on else "false")
    dialog.get_by_role("button", name="Save").click()
    expect(page.get_by_text("Default permissions updated successfully")).to_be_visible()


def open_settings(page: Page) -> Page:
    page.goto("/")
    expect(chat_input(page)).to_be_visible()
    page.get_by_role("navigation", name="Chat history").get_by_label("User menu").click()
    page.get_by_role("button", name="Settings").click()
    expect(page.get_by_role("tab", name="General", exact=True).first).to_be_visible()
    return page


def interface_tabs(page: Page):
    return page.get_by_role("tab", name="Interface", exact=True)


def test_switching_interface_settings_access_off_removes_the_tab_for_accounts_only(
    page_for, admin, make_user, preserve
):
    preserve("permissions")
    admin_page = page_for(admin)
    save_default_permission(admin_page, "Interface Settings Access", turn_on=False)

    account_page = open_settings(page_for(make_user()))
    expect(account_page.get_by_role("tab", name="General", exact=True).first).to_be_visible()
    expect(interface_tabs(account_page)).to_have_count(0)

    open_settings(admin_page)
    expect(interface_tabs(admin_page).first).to_be_visible()
