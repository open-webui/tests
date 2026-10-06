"""Journey: an admin on a phone reaches every admin settings tab, saves one and edits an account.

On a 390 by 844 touch screen the admin settings open in the settings dialog. Its tab strip scrolls
sideways, so each admin tab can be brought onto the screen and tapped, and opens its own panel
within the screen's width. A switch flipped on General and saved with the Save button on the
screen is stored. In the user list the row's buttons sit past the screen's edge in a table that
scrolls sideways; scrolled into view, Edit User opens a dialog that fits the screen and saves.

Discriminates: passes on the ebc6add67 build. In a frontend build of ebc6add67 whose settings tab
strip does not scroll the tabs test goes red at the first admin tab past the screen's edge. In a
backend copy whose admin config update drops `ENABLE_COMMUNITY_SHARING` the saving test goes red,
and in one whose user update drops the name the Edit User test.
"""

from __future__ import annotations

import re
import uuid

import pytest
from playwright.sync_api import Locator, Page, expect

from utils.chat_ui import chat_input
from utils.phone import PHONE, SCREEN, expect_on_screen, expect_reachable, tap_on_screen

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

ADMIN_TABS = [
    "General",
    "Authentication",
    "Interface",
    "Connections",
    "Models",
    "Sub-agents",
    "Integrations",
    "Documents",
    "Audio",
    "Images",
    "Web Search",
    "Code Execution",
    "Pipelines",
    "Evaluations",
    "Analytics",
    "Database",
]


@pytest.fixture
def phone(page_for, make_user) -> Page:
    """A fresh admin's phone, so nothing here touches the shared admin's own settings."""
    page = page_for(make_user(role="admin"), **PHONE)
    expect_on_screen(chat_input(page))
    return page


def admin_settings(page: Page) -> Locator:
    page.goto("/admin/settings/general")
    settings = page.get_by_role("dialog")
    expect_on_screen(settings.get_by_role("tab", name="General", selected=True))
    return settings


def expect_within_the_width(element: Locator) -> None:
    expect(element).to_be_visible()
    box = element.bounding_box()
    assert box["x"] >= 0 and box["x"] + box["width"] <= SCREEN["width"] + 0.5, box


def test_every_admin_tab_can_be_scrolled_onto_the_screen_and_opens_its_panel(phone):
    settings = admin_settings(phone)

    for name in ADMIN_TABS:
        # General, Interface and Audio are personal tabs too; the admin's come after them
        tab = settings.get_by_role("tab", name=name, exact=True).last
        expect_reachable(tab)
        tab.tap()
        expect(tab).to_have_attribute("aria-selected", "true")
        panel = settings.locator("nav + div")
        expect_within_the_width(panel)
        expect_on_screen(panel.get_by_text(re.compile(rf"^{name}")).first)


def test_a_switch_saved_on_a_phone_is_stored(phone, admin, preserve):
    preserve("admin_config")
    with admin.client() as client:
        sharing = client.get("/api/v1/auths/admin/config").json()["ENABLE_COMMUNITY_SHARING"]
    settings = admin_settings(phone)
    switch = settings.get_by_role("switch", name="Community Sharing")
    expect_reachable(switch)
    switch.tap()
    expect(switch).to_have_attribute("aria-checked", str(not sharing).lower())

    tap_on_screen(settings.get_by_role("button", name="Save", exact=True))
    expect(phone.get_by_text("Settings saved successfully!").first).to_be_visible()
    with admin.client() as client:
        stored = client.get("/api/v1/auths/admin/config").json()["ENABLE_COMMUNITY_SHARING"]
    assert stored is (not sharing), f"Community Sharing stored as {stored}"


def test_edit_user_is_reached_by_scrolling_the_row_and_saves_from_the_phone(phone, make_user):
    account = make_user()
    new_name = f"Renamed {uuid.uuid4().hex[:6]}"
    phone.goto("/admin/users")
    users = phone.get_by_role("main")
    search = users.get_by_role("textbox", name="Search")
    expect_on_screen(search)
    search.fill(account.email)
    row = users.get_by_role("row").filter(has_text=account.email)
    expect(row).to_have_count(1)

    edit = row.get_by_role("button", name="Edit User")
    expect_reachable(edit)
    edit.tap()
    editing = phone.get_by_role("dialog").filter(has_text="Edit User")
    expect_within_the_width(editing)
    name = editing.get_by_role("textbox", name="Name")
    expect_reachable(name)
    name.fill(new_name)
    save = editing.get_by_role("button", name="Save")
    expect_reachable(save)
    save.tap()

    expect(editing).to_be_hidden()
    expect(row).to_contain_text(new_name)
    scroll_width = phone.evaluate("document.documentElement.scrollWidth")
    assert scroll_width <= SCREEN["width"], f"the page scrolls sideways ({scroll_width}px wide)"
