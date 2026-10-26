"""Regression: the function menu offered Share although Community Sharing was switched off.

Issue open-webui/open-webui#31819, fix `d4a2d11ec` (PR open-webui/open-webui#31820). Every other
Share button in the admin and workspace pages hides when the admin turns Community Sharing off
in the general settings, but the menu on a function in Admin Panel > Functions always showed it.
It now shows Share only while Community Sharing is on.

Discriminates: passes on the dev b859124f9 build, fails on that build with d4a2d11ec reverted
(the function menu offers Share with Community Sharing off).
"""

from __future__ import annotations

import pytest
from playwright.sync_api import Page, expect

from harness.plugins import installed_function
from utils.workspace_list import item_row

pytestmark = [pytest.mark.regression, pytest.mark.requires_browser, pytest.mark.requires_source]

FILTER = """class Filter:
    def inlet(self, body: dict) -> dict:
        return body
"""


def _set_community_sharing(admin, enabled: bool) -> None:
    with admin.client() as client:
        current = client.get("/api/v1/auths/admin/config").json()
        saved = client.post(
            "/api/v1/auths/admin/config", json={**current, "ENABLE_COMMUNITY_SHARING": enabled}
        )
    saved.raise_for_status()


def _function_menu(page: Page, function_id: str):
    page.goto("/admin/functions")
    card = item_row(page, function_id)
    card.get_by_role("button", name="Function Menu").first.click()
    menu = page.get_by_role("menu")
    expect(menu.get_by_role("button", name="Edit")).to_be_visible()
    return menu


@pytest.mark.parametrize("enabled", [False, True], ids=["off", "on"])
def test_the_function_menu_offers_share_only_while_community_sharing_is_on(
    admin_page, admin, preserve, enabled
):
    preserve("admin_config")
    _set_community_sharing(admin, enabled)
    with installed_function(admin, FILTER) as function_id:
        menu = _function_menu(admin_page, function_id)

        expect(menu.get_by_role("button", name="Share")).to_have_count(1 if enabled else 0)
