"""Journey: the settings dialog groups an admin's tabs under System, AI, Tools, Quality and Data.

An admin opens Settings from the chat and finds the admin tabs in five named groups, in this order:
System (General, Authentication, Interface), AI (Connections, Models, Sub-agents), Tools
(Integrations, Documents, Audio, Images, Web Search, Code Execution, Pipelines), Quality
(Analytics, Evaluations, in that order since dev 4a145af22) and Data (Database). A person without
the admin role sees none of them.

Discriminates: in a frontend build with the group map back to the earlier grouping (Interface,
Audio and Images under an Experience heading, Integrations and Documents among the system tabs),
the admin test goes red on the headings and their order.
"""

from __future__ import annotations

import pytest
from playwright.sync_api import Locator, Page, expect

from utils.chat_ui import chat_input

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

ADMIN_NAVIGATION = [
    "Admin",
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
    "Audio",
    "Images",
    "Web Search",
    "Code Execution",
    "Pipelines",
    "Quality",
    "Analytics",
    "Evaluations",
    "Data",
    "Database",
]


def tab_list(page: Page) -> Locator:
    """The column holding the tabs and their group headings."""
    first_tab = page.get_by_role("dialog").get_by_role("tab").first
    expect(first_tab).to_be_visible()
    return first_tab.locator("xpath=..")


def open_settings_tabs(page: Page) -> list[str]:
    page.goto("/admin/settings/general")
    return [line.strip() for line in tab_list(page).inner_text().splitlines() if line.strip()]


def test_an_admin_finds_the_admin_tabs_in_named_groups(page_for, make_user):
    page = page_for(make_user(role="admin"))

    lines = open_settings_tabs(page)

    assert lines[lines.index("Admin") :] == ADMIN_NAVIGATION


def test_a_person_without_the_admin_role_sees_no_admin_tabs(page_for, make_user):
    page = page_for(make_user())
    page.goto("/")
    expect(chat_input(page)).to_be_visible()
    page.get_by_role("navigation", name="Chat history").get_by_label("User menu").click()
    page.get_by_role("button", name="Settings").click()
    tabs = tab_list(page)

    assert "Admin" not in tabs.inner_text().splitlines()
    for admin_tab in ("Authentication", "Sub-agents", "Pipelines", "Analytics"):
        expect(tabs.get_by_role("tab", name=admin_tab)).to_have_count(0)
