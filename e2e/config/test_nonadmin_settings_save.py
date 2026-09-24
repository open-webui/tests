"""Regression: a regular user's interface setting reverted after a reload.

open-webui 0.10.2, fix `9866a02863` (issue #26627): the settings endpoint answered 500 for every
non-admin save that carried interface settings, while the Settings modal still showed the switch
flipped, so the change was gone on the next reload. The fix reads the default permissions from
the config store.

Twin of unit/config/test_nonadmin_settings_save.py.

Discriminates: passes on bbfa876af, fails with the check reading
`request.app.state.config.USER_PERMISSIONS` again (the switch is off after the reload).
"""

from __future__ import annotations

import pytest
from playwright.sync_api import Page, expect

pytestmark = [pytest.mark.regression, pytest.mark.requires_browser, pytest.mark.requires_source]


def _widescreen_switch(page: Page):
    page.get_by_role("navigation", name="Chat history").get_by_label("User menu").click()
    page.get_by_role("button", name="Settings").click()
    page.get_by_role("tab", name="Interface").click()
    return page.get_by_role("switch", name="Widescreen Mode")


def test_a_regular_users_interface_setting_survives_a_reload(page_for, make_user):
    page = page_for(make_user())

    switch = _widescreen_switch(page)
    expect(switch).to_have_attribute("aria-checked", "false")
    with page.expect_response(lambda response: "/user/settings/update" in response.url):
        switch.click()
    expect(switch).to_have_attribute("aria-checked", "true")

    page.reload()
    expect(_widescreen_switch(page)).to_have_attribute("aria-checked", "true")
