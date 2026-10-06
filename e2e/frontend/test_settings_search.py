"""Journey: the search box of the Settings dialog finds the tab that holds a setting.

Typing a setting's name narrows the tab list to the tabs that hold it, Enter opens the first,
where the setting is, and Escape brings every tab back. A setting a person may not use is never
found for them: a regular account finds nothing for an admin setting an admin does find, and
nothing for the system prompt once the default permissions take Chat System Prompt away.

Discriminates: passes on dev 30f3f6a8f; in a frontend build whose search matches tab names only,
the narrowing and admin tests fail; in one whose access check lets every setting through, the
system prompt test fails.
"""

from __future__ import annotations

import pytest
from playwright.sync_api import Locator, Page, expect

from utils.chat_ui import chat_input

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]


def open_settings(page: Page) -> Locator:
    expect(chat_input(page)).to_be_visible()
    page.goto("/?settings=general")
    dialog = page.get_by_role("dialog")
    expect(dialog.get_by_role("tab").first).to_be_visible()
    return dialog


def search(dialog: Locator, text: str) -> None:
    dialog.get_by_role("textbox", name="Search").fill(text)


def tab_names(dialog: Locator) -> list[str]:
    return [name.strip() for name in dialog.get_by_role("tab").all_inner_texts()]


def test_a_setting_narrows_the_tabs_and_enter_opens_its_tab(page_for, make_user):
    dialog = open_settings(page_for(make_user()))
    every_tab = tab_names(dialog)

    search(dialog, "widescreen")

    expect(dialog.get_by_role("tab")).to_have_count(1)
    expect(dialog.get_by_role("tab")).to_have_text("Interface")
    dialog.get_by_role("textbox", name="Search").press("Enter")
    expect(dialog.locator("#tab-interface").get_by_text("Widescreen Mode")).to_be_visible()

    dialog.get_by_role("textbox", name="Search").press("Escape")
    expect(dialog.get_by_role("textbox", name="Search")).to_have_value("")
    assert tab_names(dialog) == every_tab


def test_an_admin_setting_is_found_by_an_admin_and_by_nobody_else(page_for, make_user):
    admin_dialog = open_settings(page_for(make_user(role="admin")))
    search(admin_dialog, "JWT Expiration")
    expect(admin_dialog.get_by_role("tab")).to_have_text(["Authentication"])

    dialog = open_settings(page_for(make_user()))
    search(dialog, "JWT Expiration")

    expect(dialog.get_by_role("tab")).to_have_count(0)
    expect(dialog.get_by_role("status")).to_have_text("No matches")


def test_the_system_prompt_is_not_found_without_the_permission(
    page_for, admin, make_user, preserve
):
    dialog = open_settings(page_for(make_user()))
    search(dialog, "System Prompt")
    expect(dialog.get_by_role("tab")).to_have_text(["General"])

    preserve("permissions")
    with admin.client() as client:
        permissions = client.get("/api/v1/users/default/permissions").json()
        permissions["chat"]["system_prompt"] = False
        client.post("/api/v1/users/default/permissions", json=permissions).raise_for_status()
    dialog = open_settings(page_for(make_user()))
    search(dialog, "System Prompt")

    expect(dialog.get_by_role("tab")).to_have_count(0)
