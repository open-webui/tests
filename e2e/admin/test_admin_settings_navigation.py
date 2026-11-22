"""Journey: the settings dialog groups an admin's tabs under System, AI, Tools, Quality and Data.

An admin opens Settings from the chat and finds, in the tab list named Admin, the admin tabs in five
named groups, in this order:
System (General, Authentication, Interface), AI (Connections, Models, Sub-agents), Tools
(Integrations, Documents, Web Search, Audio, Images & Videos, then Code Execution and Pipelines
behind a Legacy row that opens on a click, since dev aae047766), Quality (Analytics, alone since
dev 67863f310 removed Evaluations) and Data (Database). A person without the admin role gets no
Admin tab list and none of its tabs. Since dev 7ed2cb56a (open-webui/open-webui#33064) the
personal and the admin tabs sit in two tab lists, Personal and Admin.

Discriminates: in a frontend build with the group map back to the earlier grouping (Interface,
Audio and Images under an Experience heading, Integrations and Documents among the system tabs),
the admin test goes red on the headings and their order.
"""

from __future__ import annotations

import pytest
from playwright.sync_api import Locator, Page, expect

from utils.chat_ui import chat_input
from utils.user_menu import open_settings

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

ADMIN_NAVIGATION = [
    "System",
    "General",
    "Authentication",
    "Interface",
    "AI",
    "Connections",
    "Models",
    "Sub-agents",
    "Tools",
    "Integrations",
    "Documents",
    "Web Search",
    "Audio",
    "Images & Videos",
    "Legacy",
    "Quality",
    "Analytics",
    "Data",
    "Database",
]
LEGACY_TABS = ["Code Execution", "Pipelines"]


def settings_dialog(page: Page) -> Locator:
    """The settings dialog, once its tabs show."""
    dialog = page.get_by_role("dialog")
    expect(dialog.get_by_role("tablist", name="Personal").get_by_role("tab").first).to_be_visible()
    return dialog


def listed(admin_tabs: Locator) -> list[str]:
    return [line.strip() for line in admin_tabs.inner_text().splitlines() if line.strip()]


def test_an_admin_finds_the_admin_tabs_in_named_groups(page_for, make_user):
    page = page_for(make_user(role="admin"), path="/admin/settings/general")
    admin_tabs = settings_dialog(page).get_by_role("tablist", name="Admin")

    assert listed(admin_tabs) == ADMIN_NAVIGATION

    legacy = admin_tabs.get_by_role("button", name="Legacy", exact=True)
    expect(legacy).to_have_attribute("aria-expanded", "false")
    legacy.click()
    expect(legacy).to_have_attribute("aria-expanded", "true")
    after_legacy = ADMIN_NAVIGATION.index("Legacy") + 1
    opened = ADMIN_NAVIGATION[:after_legacy] + LEGACY_TABS + ADMIN_NAVIGATION[after_legacy:]
    assert listed(admin_tabs) == opened


def test_a_person_without_the_admin_role_sees_no_admin_tabs(page_for, make_user):
    page = page_for(make_user())
    expect(chat_input(page)).to_be_visible()
    open_settings(page)
    dialog = settings_dialog(page)

    expect(dialog.get_by_role("tablist", name="Admin")).to_have_count(0)
    for admin_tab in ("Authentication", "Sub-agents", "Pipelines", "Analytics"):
        expect(dialog.get_by_role("tab", name=admin_tab)).to_have_count(0)
