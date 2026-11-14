"""Journey: the feature permissions an admin switches off in the defaults leave a user's app.

Admin Panel > Users > Groups > Default permissions holds, under Features, a switch for Notes,
Calendar, Channels, Folders and Memories. With the feature on for the instance and the switch on,
a user finds it (Notes and Calendar in the user menu, Channels and Folders in the sidebar,
Memories as the Personalization tab of Settings); switched off in the dialog and saved, it is gone
from the next account's app while an admin keeps it. With sub-agents on for the instance the
dialog also holds a Sub-agents switch (#34040); saved off, a user's chats are no longer offered
the delegation tool or the timer, while an admin's still are.

The five features share one page: each test saves all five switches in one dialog and walks them
one after the other, so a switch that does nothing shows by name in the failure.

Discriminates: passes on dev 30f3f6a8f; in a frontend build whose Default permissions dialog
saves the permissions it opened with, the "withdrawn" test fails while the "offered" and admin
ones still pass. The Sub-agents test passes on dev 485103cde and fails in a backend copy whose tool
gate skips the permission.
"""

from __future__ import annotations

from typing import Callable

import pytest
from playwright.sync_api import Locator, Page, expect

from harness.tool_calls import offered_tools
from utils.chat_ui import chat_input, close_menu
from utils.user_menu import settings_item

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


def save_default_permissions(page: Page, switches: dict[str, bool]) -> None:
    """Set each switch in the Default permissions dialog and save once."""
    page.goto("/admin/users/groups")
    page.get_by_role("button", name="Default permissions").click()
    dialog = page.get_by_role("dialog")
    for switch, turn_on in switches.items():
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
        expect(settings_item(page)).to_be_visible()
        return menu.get_by_role("menuitem", name=name, exact=True)

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


def expect_every_feature(page: Page, offered: bool) -> None:
    for switch, locate in FEATURES.items():
        feature = locate(page)
        if offered:
            expect(feature, f"the {switch} feature").to_be_visible()
        else:
            expect(feature, f"the {switch} feature").to_have_count(0)
        close_menu(page)


def test_every_feature_is_offered_by_default(page_for, make_user, features_on):
    page = page_for(make_user())

    expect_every_feature(page, offered=True)


def test_every_feature_is_withdrawn_once_switched_off(page_for, admin, make_user, features_on):
    save_default_permissions(page_for(admin), dict.fromkeys(FEATURES, False))

    page = page_for(make_user())

    expect_every_feature(page, offered=False)


def test_an_admin_keeps_every_feature_when_they_are_switched_off(
    page_for, admin, make_user, features_on
):
    save_default_permissions(page_for(admin), dict.fromkeys(FEATURES, False))

    page = page_for(make_user(role="admin"))

    expect_every_feature(page, offered=True)


SUBAGENTS = ("/api/v1/configs/subagents", "/api/v1/configs/subagents")
DELEGATION = {"delegate_task", "timer"}


@pytest.fixture
def sub_agents_on(admin, preserve):
    preserve(SUBAGENTS, "permissions")
    with admin.client() as client:
        current = client.get(SUBAGENTS[0]).json()
        saved = client.post(SUBAGENTS[1], json={**current, "ENABLE_SUBAGENTS": True})
    assert saved.status_code == 200, saved.text


def test_the_sub_agents_switch_saved_off_withdraws_delegation_from_a_user(
    page_for, admin, make_user, upstream, sub_agents_on
):
    save_default_permissions(page_for(admin), {"Sub-agents": False})

    with admin.client() as client:
        stored = client.get("/api/v1/users/default/permissions").json()["features"]
    assert stored["subagents"] is False, f"the dialog saved {stored}"
    with make_user().client() as client:
        offered = offered_tools(client, upstream)
    assert offered, "the model was offered no tools at all, so this shows nothing"
    assert not DELEGATION & offered, f"a user without the permission was offered {offered}"
    with make_user(role="admin").client() as client:
        assert DELEGATION <= offered_tools(client, upstream)
