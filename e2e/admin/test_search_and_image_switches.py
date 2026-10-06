"""Journey: the Web Search and Image Generation switches decide what the Integrations menu offers.

The Web Search switch in Admin Settings > Web Search and the Image Generation switch in Admin
Settings > Images, with an engine set up behind each, put Web Search and Image in a user's
Integrations menu of the chat input; switched off and saved, the entry is gone. With the switch
on, the Web Search and Image Generation switches of the default permissions take the entry away
from users while an admin keeps it.

Discriminates: passes on dev 30f3f6a8f; in a frontend build whose Web Search and Images forms
send the stored settings back in place of the edited ones and whose Default permissions dialog
saves the permissions it opened with, every test but the admin one fails.
"""

from __future__ import annotations

import pytest
from playwright.sync_api import Locator, Page, expect

from harness.image_engines import IMAGES_CONFIG, save_image_settings, serve_gemini
from harness.web_retrieval import RETRIEVAL_CONFIG, save_web_settings, serve_search_results
from utils.chat_ui import chat_input

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]


@pytest.fixture
def engines(admin, preserve, listener):
    """A search engine and an image engine set up, both switched off for the instance."""
    preserve(RETRIEVAL_CONFIG, IMAGES_CONFIG, "permissions")
    with admin.client() as client:
        save_web_settings(
            client, **{**serve_search_results(listener, []), "ENABLE_WEB_SEARCH": False}
        )
        save_image_settings(client, **{**serve_gemini(listener), "ENABLE_IMAGE_GENERATION": False})


def switch_on_instance(admin, entry: str) -> None:
    with admin.client() as client:
        if entry == "Web Search":
            save_web_settings(client, ENABLE_WEB_SEARCH=True)
        else:
            save_image_settings(client, ENABLE_IMAGE_GENERATION=True)


TABS = {"Web Search": ("web", "Web Search"), "Image": ("images", "Image Generation")}


def save_tab_switch(page: Page, entry: str, turn_on: bool) -> None:
    tab, switch_name = TABS[entry]
    page.goto(f"/admin/settings/{tab}")
    settings = page.get_by_role("dialog")
    switch = settings.get_by_role("switch", name=switch_name, exact=True)
    expect(switch).to_have_attribute("aria-checked", "false" if turn_on else "true")
    switch.click()
    expect(switch).to_have_attribute("aria-checked", "true" if turn_on else "false")
    settings.get_by_role("button", name="Save", exact=True).click()
    expect(page.get_by_text("Settings saved successfully!").first).to_be_visible()


def save_default_permission(page: Page, switch: str, turn_on: bool) -> None:
    page.goto("/admin/users/groups")
    page.get_by_role("button", name="Default permissions").click()
    dialog = page.get_by_role("dialog")
    target = dialog.get_by_role("switch", name=switch, exact=True)
    if (target.get_attribute("aria-checked") == "true") != turn_on:
        target.click()
    expect(target).to_have_attribute("aria-checked", "true" if turn_on else "false")
    dialog.get_by_role("button", name="Save").click()
    expect(page.get_by_text("Default permissions updated successfully")).to_be_visible()


def menu_entry(page: Page, entry: str) -> Locator:
    expect(chat_input(page)).to_be_visible()
    page.get_by_role("button", name="Integrations", exact=True).last.click()
    menu = page.get_by_role("menu")
    expect(menu).to_be_visible()
    return menu.get_by_role("button", name=entry, exact=True)


@pytest.mark.parametrize("entry", TABS)
def test_a_switched_on_feature_is_offered_in_the_integrations_menu(
    entry, page_for, admin, make_user, engines
):
    save_tab_switch(page_for(admin), entry, turn_on=True)

    page = page_for(make_user())

    expect(menu_entry(page, entry)).to_be_visible()


@pytest.mark.parametrize("entry", TABS)
def test_a_switched_off_feature_leaves_the_integrations_menu(
    entry, page_for, admin, make_user, engines
):
    switch_on_instance(admin, entry)
    save_tab_switch(page_for(admin), entry, turn_on=False)

    page = page_for(make_user())

    expect(menu_entry(page, entry)).to_have_count(0)


@pytest.mark.parametrize("entry", TABS)
def test_the_default_permission_withdraws_the_feature_from_users(
    entry, page_for, admin, make_user, engines
):
    switch_on_instance(admin, entry)
    save_default_permission(page_for(admin), TABS[entry][1], turn_on=False)

    page = page_for(make_user())

    expect(menu_entry(page, entry)).to_have_count(0)


def test_an_admin_keeps_web_search_without_the_default_permission(
    page_for, admin, make_user, engines
):
    switch_on_instance(admin, "Web Search")
    save_default_permission(page_for(admin), "Web Search", turn_on=False)

    page = page_for(make_user(role="admin"))

    expect(menu_entry(page, "Web Search")).to_be_visible()
