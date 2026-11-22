"""Journey: an admin adds, configures, disables and uninstalls an app in Settings > Integrations.

The admin enters the app's URL and API key, presses Continue, reviews the requested scopes,
goes back and presses Connect. The app then shows in the list with its handle and an Enable
switch that is on. Wrong keys and unreachable URLs show the reason in the form.

Discriminates: passes on dev 4ca8fdc3f; in a frontend copy, never calling Connect after the
app is created leaves its switch off and turns test_an_admin_connects_an_app_from_its_url_and_key
red. Retargeted for 47d2a3c75, which moves the Apps tab into an Apps section of the Integrations
tab, shown only while apps are on, and for fa94487cf (#34733), which sends the old Apps address
there.
"""

from __future__ import annotations

import re

import pytest
from playwright.sync_api import Locator, Page, expect

from utils.setting_rows import setting_button

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]


def _open_apps(page: Page) -> None:
    page.goto("/admin/settings/apps")
    expect(_apps_section(page).get_by_text("Connected apps")).to_be_visible()


def _apps_section(page: Page) -> Locator:
    """The Apps section of Admin Settings > Integrations."""
    heading = page.get_by_role("heading", name="Apps", exact=True)
    return page.get_by_role("dialog").locator("section").filter(has=heading)


def _open_integrations_tab(page: Page) -> None:
    tabs = page.locator("#settings-tabs-container")
    tabs.get_by_role("tab", name="Integrations", exact=True).click()
    # the tool servers load with the apps, so the tab has finished loading once they show
    expect(
        page.get_by_role("dialog").get_by_text("External Tool Servers", exact=True)
    ).to_be_visible()


def _connect(page: Page) -> None:
    """Connect in the app dialog; the Integrations tab behind it has Add Connection buttons too."""
    page.get_by_role("dialog").last.get_by_role("button", name="Connect", exact=True).click()


def _add_app(page: Page, url: str, api_key: str) -> None:
    page.get_by_role("button", name="Add app").click()
    page.locator("#app-url").fill(url)
    page.locator("#app-api-key").fill(api_key)
    page.get_by_role("button", name="Continue").click()


def _back_from_scopes(page: Page) -> None:
    """The scope review closes with Continue (Back before e9635596d)."""
    review = page.get_by_role("region", name="App scopes").locator("xpath=..")
    review.get_by_role("button", name=re.compile("^(Continue|Back)$")).click()


def _row(page: Page, name: str):
    return page.get_by_role("switch", name=f"Enable {name}")


def test_an_admin_connects_an_app_from_its_url_and_key(page_for, apps_admin, reference_app):
    page = page_for(apps_admin)
    _open_apps(page)
    _add_app(page, reference_app.url, reference_app.api_key)
    region = page.get_by_role("region", name="App scopes")
    expect(region).to_be_visible()
    for label in (
        "Read app mentions in joined channels",
        "Read messages and history in joined channels",
        "Send messages and edit own messages in joined channels",
        "Read joined channel details",
    ):
        expect(region.get_by_text(label).first).to_be_visible()
    _back_from_scopes(page)
    expect(page.locator("#app-name")).to_have_value("Reference App")
    expect(page.get_by_text("4 selected")).to_be_visible()
    _connect(page)
    expect(page.get_by_text("App settings updated")).to_be_visible()
    expect(_row(page, "Reference App").first).to_be_checked()


def test_a_wrong_key_shows_why_continue_failed(page_for, apps_admin, reference_app):
    page = page_for(apps_admin)
    _open_apps(page)
    _add_app(page, reference_app.url, "not-the-key")
    expect(page.get_by_role("alert")).to_contain_text("rejected the API key")
    expect(page.locator("#app-url")).to_be_visible()


def test_an_unreachable_url_shows_why_continue_failed(page_for, apps_admin):
    page = page_for(apps_admin)
    _open_apps(page)
    _add_app(page, "http://localhost:9/events", "k" * 32)
    expect(page.get_by_role("alert")).to_contain_text("HTTPS")


def test_an_admin_grants_only_some_scopes(page_for, apps_admin, reference_app):
    page = page_for(apps_admin)
    _open_apps(page)
    _add_app(page, reference_app.url, reference_app.api_key)
    region = page.get_by_role("region", name="App scopes")
    region.get_by_role("switch").nth(3).click()
    _back_from_scopes(page)
    expect(page.get_by_text("3 selected")).to_be_visible()
    page.locator("#app-handle").fill("scoped-app")
    _connect(page)
    expect(page.get_by_text("App settings updated")).to_be_visible()
    with apps_admin.client() as client:
        app = next(a for a in client.get("/api/v1/apps").json() if a["slug"] == "scoped-app")
    assert "channels:read" not in app["data"]["granted_scopes"]
    assert len(app["data"]["granted_scopes"]) == 3


def test_disabling_an_app_flips_its_switch_off(page_for, apps_admin, installed_app):
    with apps_admin.client() as client:
        client.patch(f"/api/v1/apps/{installed_app['id']}", json={"name": "Switch Target"})
    page = page_for(apps_admin)
    _open_apps(page)
    switch = _row(page, "Switch Target")
    expect(switch).to_be_checked()
    switch.click()
    expect(switch).not_to_be_checked()
    with apps_admin.client() as client:
        assert client.get(f"/api/v1/apps/{installed_app['id']}").json()["is_active"] is False


def test_turning_a_disabled_app_back_on_opens_connect(page_for, apps_admin, installed_app):
    with apps_admin.client() as client:
        client.patch(
            f"/api/v1/apps/{installed_app['id']}", json={"name": "Dormant App", "is_active": False}
        )
    page = page_for(apps_admin)
    _open_apps(page)
    _row(page, "Dormant App").click()
    expect(page.get_by_role("heading", name="Configure app")).to_be_visible()
    _connect(page)
    expect(page.get_by_text("App settings updated")).to_be_visible()
    expect(_row(page, "Dormant App")).to_be_checked()


def test_uninstalling_an_app_asks_first_and_marks_it_uninstalled(
    page_for, apps_admin, installed_app
):
    with apps_admin.client() as client:
        client.patch(f"/api/v1/apps/{installed_app['id']}", json={"name": "Leaving App"})
    page = page_for(apps_admin)
    _open_apps(page)
    page.get_by_role("button", name="Configure Leaving App").click()
    page.get_by_role("button", name="Uninstall").click()
    expect(page.get_by_text("Credentials and channel memberships will be removed.")).to_be_visible()
    page.get_by_role("button", name="Confirm").click()
    expect(_row(page, "Leaving App")).not_to_be_checked()
    with apps_admin.client() as client:
        details = client.get(f"/api/v1/apps/{installed_app['id']}").json()
    assert details["configured"] is False


def test_a_manual_install_of_a_post_only_app(page_for, apps_admin, apps_instance):
    from harness.apps import serving_reference_app

    with serving_reference_app(manifest=False, name="Manual App") as app:
        page = page_for(apps_admin)
        _open_apps(page)
        page.get_by_role("button", name="Add app").click()
        page.locator("#app-url").fill(app.url)
        page.locator("#app-api-key").fill(app.api_key)
        page.get_by_role("button", name="Enter details manually").click()
        page.locator("#app-name").fill("Manual App")
        # the handle becomes the manifest id, which the app signs into its connection proof
        page.locator("#app-handle").fill(app.manifest_id)
        page.locator("#app-public-key").fill(app.public_key)
        setting_button(page.get_by_role("dialog").last, "Scopes").click()
        region = page.get_by_role("region", name="App scopes")
        for switch in region.get_by_role("switch").all():
            switch.click()
        _back_from_scopes(page)
        _connect(page)
        expect(page.get_by_text("App settings updated")).to_be_visible()
        expect(_row(page, "Manual App")).to_be_checked()


def test_the_integrations_tab_shows_the_apps_section_while_apps_are_on(page_for, apps_admin):
    page = page_for(apps_admin, path="/admin/settings/general")
    _open_integrations_tab(page)
    apps = _apps_section(page)
    expect(apps.get_by_text("Connected apps")).to_be_visible()
    expect(apps.get_by_role("button", name="Add app")).to_be_visible()


def test_the_integrations_tab_has_no_apps_section_while_apps_are_off(
    page_for, apps_admin, apps_instance, preserve
):
    preserve("admin_config", on=apps_instance)
    with apps_admin.client() as client:
        config = client.get("/api/v1/auths/admin/config").json()
        saved = client.post("/api/v1/auths/admin/config", json={**config, "ENABLE_APPS": False})
    assert saved.status_code == 200, saved.text
    page = page_for(apps_admin, path="/admin/settings/general")
    _open_integrations_tab(page)
    expect(page.get_by_role("heading", name="Apps", exact=True)).to_have_count(0)
    expect(page.get_by_role("button", name="Add app")).to_have_count(0)
