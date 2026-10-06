"""Journey: the feature permissions an admin switches off in the defaults leave a user's app.

Admin Panel > Users > Groups > Default permissions holds, under Features, a switch for Notes,
Calendar, Channels, Folders and Memories. With the feature on for the instance and the switch on,
a user finds it (Notes and Calendar in the user menu, Channels and Folders in the sidebar,
Memories as the Personalization tab of Settings); switched off in the dialog and saved, it is gone
from the next account's app while an admin keeps it.

Discriminates: passes on dev 30f3f6a8f; in a frontend build whose Default permissions dialog
saves the permissions it opened with, every "withdrawn" test fails while the "offered" and admin
ones still pass.
"""

from __future__ import annotations

from typing import Callable

import pytest
from playwright.sync_api import Locator, Page, expect

from utils.chat_ui import chat_input

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

ADMIN_CONFIG = "/api/v1/auths/admin/config"
INSTANCE_FEATURES = {
    "ENABLE_NOTES": True,
    "ENABLE_CALENDAR": True,
    "ENABLE_CHANNELS": True,
    "ENABLE_FOLDERS": True,
    "ENABLE_MEMORIES": True,
}


@pytest.fixture
def features_on(admin, preserve):
    """Every feature on for the instance, so only the permission decides."""
    preserve("admin_config", "permissions")
    with admin.client() as client:
        current = client.get(ADMIN_CONFIG).json()
        saved = client.post(ADMIN_CONFIG, json={**current, **INSTANCE_FEATURES})
    saved.raise_for_status()


def save_default_permission(page: Page, switch: str, turn_on: bool) -> None:
    page.goto("/admin/users/groups")
    page.get_by_role("button", name="Default permissions").click()
    dialog = page.get_by_role("dialog")
    target = dialog.get_by_role("switch", name=switch, exact=True)
    if (target.get_attribute("aria-checked") == "true") != turn_on:
        target.click()
    expect(target).to_have_attribute("aria-checked", "true" if turn_on else "false")
    dialog.get_by_role("button", name="Save").click()
    expect(page.get_by_text("Default permissions updated successfully")).to_be_visible()


def user_menu_link(name: str) -> Callable[[Page], Locator]:
    def locate(page: Page) -> Locator:
        expect(chat_input(page)).to_be_visible()
        page.get_by_role("button", name="User menu").first.click()
        menu = page.get_by_role("menu")
        expect(menu.get_by_role("button", name="Settings")).to_be_visible()
        return menu.get_by_role("link", name=name, exact=True)

    return locate


def sidebar_section(name: str) -> Callable[[Page], Locator]:
    def locate(page: Page) -> Locator:
        expect(chat_input(page)).to_be_visible()
        open_sidebar = page.get_by_role("button", name="Open Sidebar", exact=True)
        if open_sidebar.is_visible():
            open_sidebar.click()
        sidebar = page.get_by_role("navigation", name="Chat history")
        expect(sidebar.get_by_role("button", name="Chats", exact=True)).to_be_visible()
        return sidebar.get_by_role("button", name=name, exact=True)

    return locate


def settings_tab(name: str) -> Callable[[Page], Locator]:
    def locate(page: Page) -> Locator:
        page.goto("/?settings=general")
        dialog = page.get_by_role("dialog")
        expect(dialog.get_by_role("tab").first).to_be_visible()
        return dialog.get_by_role("tab", name=name, exact=True)

    return locate


FEATURES = {
    "Notes": user_menu_link("Notes"),
    "Calendar": user_menu_link("Calendar"),
    "Channels": sidebar_section("Channels"),
    "Folders": sidebar_section("Folders"),
    "Memories": settings_tab("Personalization"),
}


@pytest.mark.parametrize("switch", FEATURES)
def test_the_feature_is_offered_by_default(switch, page_for, make_user, features_on):
    page = page_for(make_user())

    expect(FEATURES[switch](page)).to_be_visible()


@pytest.mark.parametrize("switch", FEATURES)
def test_the_feature_is_withdrawn_once_switched_off(
    switch, page_for, admin, make_user, features_on
):
    save_default_permission(page_for(admin), switch, turn_on=False)

    page = page_for(make_user())

    expect(FEATURES[switch](page)).to_have_count(0)


@pytest.mark.parametrize("switch", ["Notes", "Memories"])
def test_an_admin_keeps_the_feature_when_it_is_switched_off(
    switch, page_for, admin, make_user, features_on
):
    save_default_permission(page_for(admin), switch, turn_on=False)

    page = page_for(make_user(role="admin"))

    expect(FEATURES[switch](page)).to_be_visible()
