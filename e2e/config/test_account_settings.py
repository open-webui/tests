"""Journey: Settings > Account, the profile and the API key.

A name and a Bio saved in the account tab show where people see them after a reload: the name in
the user menu and in Admin Panel > Users, the Bio on the profile card that opens from the avatar
in that list. With the admin's API Keys switch on and the default permission for API Keys on, the
tab offers a key: one created there answers `/api/models` over HTTP, and once it is deleted in
the tab the same key is refused. With the admin switch off, or the permission off, the tab offers
no API keys section at all. A photo uploaded there is the profile picture everybody is served,
and Remove puts the default picture back.

Discriminates: passes on dev 30f3f6a8f; in a frontend copy, the save leaving the name out of the
profile it sends turns the name test red, leaving the Bio out turns the Bio test red, the delete
confirmation not calling the delete route turns the key test red (the key still works), and
`canUseApiKeys` ignoring the permission turns the permission case of the section test red. A save
that sends the stored profile picture in place of the chosen one turns the photo test red.
"""

from __future__ import annotations

import uuid

import httpx
import pytest
from playwright.sync_api import Locator, Page, expect

from harness.actors import Actor

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

ADMIN_CONFIG = "/api/v1/auths/admin/config"


def save_default_permission(page: Page, switch: str, turn_on: bool) -> None:
    page.goto("/admin/users/groups")
    page.get_by_role("button", name="Default permissions").click()
    dialog = page.get_by_role("dialog")
    target = dialog.get_by_role("switch", name=switch)
    if (target.get_attribute("aria-checked") == "true") != turn_on:
        target.click()
    expect(target).to_have_attribute("aria-checked", "true" if turn_on else "false")
    dialog.get_by_role("button", name="Save").click()
    expect(page.get_by_text("Default permissions updated successfully")).to_be_visible()


def _set_api_keys_switch(admin: Actor, enabled: bool) -> None:
    with admin.client() as client:
        current = client.get(ADMIN_CONFIG)
        current.raise_for_status()
        saved = client.post(ADMIN_CONFIG, json={**current.json(), "ENABLE_API_KEYS": enabled})
    assert saved.status_code == 200, saved.text


def _account_settings(page: Page) -> Locator:
    page.get_by_role("navigation", name="Chat history").get_by_label("User menu").click()
    page.get_by_role("button", name="Settings").click()
    page.get_by_role("tab", name="Account").click()
    dialog = page.get_by_role("dialog")
    expect(dialog.get_by_role("textbox", name="Name")).not_to_have_value("")
    return dialog


def _save(page: Page, settings: Locator) -> None:
    with page.expect_response(lambda response: "/auths/update/profile" in response.url):
        settings.get_by_role("button", name="Save", exact=True).click()
    expect(page.get_by_text("Settings saved successfully!").first).to_be_visible()


def _user_row(page: Page, account: Actor) -> Locator:
    page.goto("/admin/users")
    users = page.get_by_role("main")
    users.get_by_role("textbox", name="Search").fill(account.email)
    return users.get_by_role("row").filter(has_text=account.email)


def _models_status(key: str, base_url: str) -> int:
    response = httpx.get(f"{base_url}/api/models", headers={"Authorization": f"Bearer {key}"})
    return response.status_code


@pytest.fixture
def keys_allowed(admin, make_user, page_for, preserve) -> Actor:
    """An account whose default permissions offer API keys, with the admin's switch on."""
    preserve("admin_config", "permissions")
    _set_api_keys_switch(admin, True)
    save_default_permission(page_for(make_user(role="admin")), "API Keys", turn_on=True)
    return make_user()


def test_a_changed_name_shows_in_the_user_menu_and_the_admin_user_list_after_a_reload(
    page_for, make_user
):
    account = make_user()
    page = page_for(account)
    new_name = f"Renamed {uuid.uuid4().hex[:8]}"
    settings = _account_settings(page)
    settings.get_by_role("textbox", name="Name").fill(new_name)
    _save(page, settings)

    page.reload()

    page.get_by_role("navigation", name="Chat history").get_by_label("User menu").click()
    expect(page.get_by_role("button", name=new_name)).to_be_visible()
    expect(_user_row(page_for(make_user(role="admin")), account)).to_contain_text(new_name)


def test_a_saved_bio_shows_on_the_profile_card_in_the_admin_user_list(page_for, make_user):
    account = make_user()
    bio = f"Keeps bees near Villach {uuid.uuid4().hex[:8]}"
    page = page_for(account)
    settings = _account_settings(page)
    settings.get_by_role("textbox", name="Bio").fill(bio)
    _save(page, settings)

    row = _user_row(page_for(make_user(role="admin")), account)
    row.get_by_role("img", name="user").hover()

    expect(row.page.get_by_text(bio)).to_be_visible()


def test_an_api_key_created_in_the_account_settings_works_until_it_is_deleted(
    page_for, keys_allowed
):
    page = page_for(keys_allowed)
    settings = _account_settings(page)
    secrets = settings.locator("section").filter(has_text="API keys")
    secrets.get_by_role("button", name="Show", exact=True).click()

    secrets.get_by_role("button", name="Create new secret key").click()
    expect(page.get_by_text("API Key created.")).to_be_visible()
    key = secrets.get_by_role("textbox").input_value()

    assert key.startswith("sk-")
    assert _models_status(key, keys_allowed.base_url) == 200

    # the menu's trigger is wrapped in a second element with the same name
    secrets.get_by_role("button", name="More").last.click()
    page.get_by_role("button", name="Delete", exact=True).click()
    confirming = page.get_by_role("dialog").filter(has_text="Delete API Key?")
    confirming.get_by_role("button", name="Delete", exact=True).click()
    expect(page.get_by_text("API Key deleted.")).to_be_visible()

    assert _models_status(key, keys_allowed.base_url) == 401


def test_the_account_settings_offer_api_keys_while_the_switch_and_the_permission_are_on(
    page_for, keys_allowed
):
    settings = _account_settings(page_for(keys_allowed))

    expect(settings.locator("section").filter(has_text="API keys")).to_be_visible()


@pytest.mark.parametrize("withdrawn", ["the admin switch", "the permission"])
def test_the_account_settings_offer_no_api_keys_when_either_is_off(
    withdrawn, admin, page_for, make_user, keys_allowed
):
    if withdrawn == "the admin switch":
        _set_api_keys_switch(admin, False)
    else:
        save_default_permission(page_for(make_user(role="admin")), "API Keys", turn_on=False)
    settings = _account_settings(page_for(keys_allowed))

    expect(settings.get_by_role("textbox", name="Name")).to_be_visible()
    expect(settings.locator("section").filter(has_text="API keys")).to_have_count(0)


# a PNG of one red pixel; the account tab redraws any photo as a 250 pixel WebP
RED_PIXEL_PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108020000009077"
    "53de0000000c4944415478da63f8cfc000000301010018dd8db00000000049454e44ae426082"
)


def _profile_image_type(viewer: Actor, account: Actor) -> str:
    with viewer.client() as client:
        served = client.get(f"/api/v1/users/{account.id}/profile/image")
    assert served.status_code == 200, served.text
    return served.headers["content-type"]


def test_an_uploaded_photo_is_the_profile_picture_others_are_served(page_for, make_user, admin):
    account = make_user()
    assert _profile_image_type(admin, account) == "image/png"
    page = page_for(account)
    settings = _account_settings(page)

    with page.expect_file_chooser() as chooser:
        settings.get_by_role("button", name="Upload Photo").click()
    chooser.value.set_files(
        files=[{"name": "portrait.png", "mimeType": "image/png", "buffer": RED_PIXEL_PNG}]
    )
    expect(settings.locator("img[src^='data:image/webp']").first).to_be_visible()
    _save(page, settings)

    assert _profile_image_type(admin, account) == "image/webp"

    page.goto("/")
    settings = _account_settings(page)
    settings.get_by_role("button", name="Remove", exact=True).click()
    _save(page, settings)

    assert _profile_image_type(admin, account) == "image/png"
