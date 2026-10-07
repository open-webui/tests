"""Journey: the "What's New" dialog opens once for an admin after the instance was updated.

An admin's settings remember the last version whose release notes they closed. When the running
version differs (the instance was updated since), the dialog opens by itself on the next page
load; closing it records the running version, so the next load stays quiet. The admin's own
Settings > Interface switch "Show "What's New" Modal on Login" turns this off and back on. A
regular user never gets the dialog by itself. The tests set the remembered version back to an
older release, as an update leaves it.

Discriminates: passes on dev ebc6add67; in a frontend copy, with the closing button no longer
recording the version the shown-once test fails (the dialog is back after a reload), with the
switch always saved as on the switch test fails and with the role check dropped the regular user
test fails.
"""

from __future__ import annotations

import pytest
from playwright.sync_api import Locator, Page, expect

from harness.actors import Actor
from utils.chat_ui import chat_input

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

OLDER_RELEASE = "0.1.0"
SWITCH = 'Show "What\'s New" Modal on Login'


def _update_ui(account: Actor, **settings) -> None:
    with account.client() as client:
        saved = client.post("/api/v1/users/user/settings/update", json={"ui": settings})
    assert saved.status_code == 200, saved.text


def _seen_version(account: Actor) -> str | None:
    with account.client() as client:
        return client.get("/api/v1/users/user/settings").json()["ui"].get("version")


def _running_version(account: Actor) -> str:
    with account.client() as client:
        return client.get("/api/config").json()["version"]


def whats_new(page: Page) -> Locator:
    return page.get_by_role("dialog").filter(has_text="What's New in")


def load_after_an_update(page: Page, account: Actor) -> None:
    _update_ui(account, version=OLDER_RELEASE)
    page.goto("/")
    expect(chat_input(page)).to_be_visible()


def interface_switch(page: Page) -> Locator:
    page.goto("/?settings=interface")
    switch = page.get_by_role("switch", name=SWITCH)
    expect(switch).to_be_visible()
    return switch


def test_an_admin_sees_whats_new_once_after_an_update(page_for, make_user):
    admin = make_user(role="admin")
    page = page_for(admin)
    _update_ui(admin, showChangelog=True)

    load_after_an_update(page, admin)

    dialog = whats_new(page)
    expect(dialog).to_be_visible()
    with page.expect_response(lambda response: "/users/user/settings/update" in response.url):
        dialog.get_by_role("button", name="Close").click()
    expect(dialog).to_be_hidden()
    assert _seen_version(admin) == _running_version(admin)

    page.reload()
    expect(chat_input(page)).to_be_visible()
    expect(whats_new(page)).to_have_count(0)


def test_the_interface_switch_keeps_whats_new_closed_after_an_update(page_for, make_user):
    admin = make_user(role="admin")
    page = page_for(admin)
    _update_ui(admin, showChangelog=True)
    switch = interface_switch(page)
    expect(switch).to_have_attribute("aria-checked", "true")

    with page.expect_response(lambda response: "/users/user/settings/update" in response.url):
        switch.click()
    expect(switch).to_have_attribute("aria-checked", "false")
    load_after_an_update(page, admin)

    expect(whats_new(page)).to_have_count(0)
    switch = interface_switch(page)
    with page.expect_response(lambda response: "/users/user/settings/update" in response.url):
        switch.click()
    expect(switch).to_have_attribute("aria-checked", "true")
    load_after_an_update(page, admin)
    expect(whats_new(page)).to_be_visible()


def test_a_regular_user_does_not_get_whats_new_by_itself(page_for, make_user):
    account = make_user()
    page = page_for(account)
    _update_ui(account, showChangelog=True)

    load_after_an_update(page, account)

    expect(whats_new(page)).to_have_count(0)
    assert _seen_version(account) == OLDER_RELEASE
