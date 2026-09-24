"""Regression: the admin's folder sharing toggle reverted after saving the default permissions.

open-webui 0.11.0, fix `3cf3f8e18` (PR #27296, issue #27120): the default permissions endpoint
validated through a schema without `sharing.folders`, so the switch the admin turned on in
Admin Panel > Users > Groups > Default permissions was dropped on save and showed off again on
the next visit. The fix adds the field.

Twin of unit/config/test_sharing_permissions_folders.py.

Discriminates: passes on bbfa876af, fails with the `folders` field removed from
`SharingPermissions` (the switch is off again after the reload).
"""

from __future__ import annotations

import pytest
from playwright.sync_api import Page, expect

pytestmark = [pytest.mark.regression, pytest.mark.requires_browser, pytest.mark.requires_source]


def _folder_sharing_switch(page: Page):
    page.get_by_role("button", name="Default permissions").click()
    return page.get_by_role("dialog").get_by_role("switch", name="Folders Sharing")


def test_enabled_folder_sharing_is_still_on_after_a_reload(page_for, admin, preserve):
    preserve("permissions")
    page = page_for(admin)
    page.goto("/admin/users/groups")

    switch = _folder_sharing_switch(page)
    expect(switch).to_have_attribute("aria-checked", "false")
    switch.click()
    expect(switch).to_have_attribute("aria-checked", "true")
    page.get_by_role("dialog").get_by_role("button", name="Save").click()
    expect(page.get_by_text("Default permissions updated successfully")).to_be_visible()

    page.reload()
    expect(_folder_sharing_switch(page)).to_have_attribute("aria-checked", "true")
