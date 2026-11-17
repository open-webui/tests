"""Journey: the settings dialog groups an admin's tabs under System, AI, Tools, Quality and Data.

An admin opens Settings from the chat and finds, in the tab list named Admin, the admin tabs in five
named groups, in this order:
System (General, Authentication, Interface), AI (Connections, Models, Sub-agents), Tools
(Integrations, Documents, Audio, Images, Videos since dev c201472cf, Web Search, Code Execution,
Pipelines), Quality (Analytics, Evaluations, in that order since dev 4a145af22) and Data
(Database). A person without the admin role gets no Admin tab list and none of its tabs. Since dev
7ed2cb56a (open-webui/open-webui#33064) the personal and the admin tabs sit in two tab lists,
Personal and Admin.

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
    "Audio",
    "Images",
    "Videos",
    "Web Search",
    "Code Execution",
    "Pipelines",
    "Quality",
    "Analytics",
    "Evaluations",
    "Data",
    "Database",
]


def settings_dialog(page: Page) -> Locator:
    """The settings dialog, once its tabs show."""
    dialog = page.get_by_role("dialog")
    expect(dialog.get_by_role("tablist", name="Personal").get_by_role("tab").first).to_be_visible()
    return dialog


def test_an_admin_finds_the_admin_tabs_in_named_groups(page_for, make_user):
    page = page_for(make_user(role="admin"), path="/admin/settings/general")
    admin_tabs = settings_dialog(page).get_by_role("tablist", name="Admin")

    lines = [line.strip() for line in admin_tabs.inner_text().splitlines() if line.strip()]

    assert lines == ADMIN_NAVIGATION


def test_a_person_without_the_admin_role_sees_no_admin_tabs(page_for, make_user):
    page = page_for(make_user())
    expect(chat_input(page)).to_be_visible()
    open_settings(page)
    dialog = settings_dialog(page)

    expect(dialog.get_by_role("tablist", name="Admin")).to_have_count(0)
    for admin_tab in ("Authentication", "Sub-agents", "Pipelines", "Analytics"):
        expect(dialog.get_by_role("tab", name=admin_tab)).to_have_count(0)
