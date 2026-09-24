"""An admin's UI translation override relabels the interface, and removing it restores the label.

Admin Settings > General > UI Translations stores per-language overrides of interface strings in
the `I18N` admin config. Saving applies them to the running page at once, removing one brings the
bundled string back without a reload, and every page loaded afterwards starts with them.

Browser twin of frontend/i18n/i18n.test.ts, which drives `initI18n` and `updateI18n` directly.

Discriminates: passes on the bbfa876af build; with the overrides left out of the loaded
translations, the relabelled sidebar entry never appears and both tests go red.
"""

from __future__ import annotations

import pytest
from playwright.sync_api import Page, expect

pytestmark = [pytest.mark.requires_browser, pytest.mark.requires_source]

ADMIN_CONFIG = "/api/v1/auths/admin/config"
ORIGINAL = "New Chat"
OVERRIDE = "Start Something"


def sidebar_entry(page: Page, name: str):
    return page.get_by_role("link", name=name, exact=True)


def test_saving_and_removing_an_override_relabels_the_open_page(page_for, make_user, preserve):
    preserve("admin_config")
    page = page_for(make_user(role="admin"))
    page.goto("/admin/settings/general")

    page.get_by_role("button", name="Add translation").click()
    page.get_by_role("textbox", name="Key").fill(ORIGINAL)
    page.get_by_role("textbox", name="Value").fill(OVERRIDE)
    page.get_by_role("button", name="Save").click()
    expect(sidebar_entry(page, OVERRIDE)).to_be_visible()
    expect(sidebar_entry(page, ORIGINAL)).to_have_count(0)

    page.get_by_role("button", name="Delete").click()
    page.get_by_role("button", name="Save").click()
    expect(sidebar_entry(page, ORIGINAL)).to_be_visible()
    expect(sidebar_entry(page, OVERRIDE)).to_have_count(0)


def test_a_saved_override_is_there_for_the_next_page_load(admin, page_for, make_user, preserve):
    preserve("admin_config")
    with admin.client() as client:
        current = client.get(ADMIN_CONFIG).json()
        saved = client.post(ADMIN_CONFIG, json={**current, "I18N": {"en-US": {ORIGINAL: OVERRIDE}}})
    saved.raise_for_status()

    page = page_for(make_user())

    expect(sidebar_entry(page, OVERRIDE)).to_be_visible()
    expect(page.get_by_role("button", name="Search", exact=True)).to_be_visible()
