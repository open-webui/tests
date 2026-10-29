"""Journey: a control an admin hides under Visible controls is gone for users and kept for admins.

Admin Settings > Interface > Visible controls switches single controls off for every account
but admins. Attach Webpage switched off there and saved leaves a user's + menu without it while
the admin's still offers it. Archive hidden takes both Archive and Unarchive from a user's chat
menus and from Archived Chats, while the user's archived chat stays archived and readable there,
and an admin's own archived chat still offers Unarchive. A setting whose switch is hidden keeps
the value the user saved: Chat Bubble UI switched off before it is hidden still labels the
user's message "You", and the switch is gone from their Interface settings.

Discriminates: passes on the dev 948432b43 build; in a frontend copy with every control read as
visible, every test fails (the user keeps Attach Webpage, Unarchive and the Chat Bubble UI
switch) while the header archive journey on the same build stays green.
"""

from __future__ import annotations

import uuid

import pytest
from playwright.sync_api import Locator, Page, expect

from harness.chat_history import seed_chat
from utils.chat_ui import chat_input, conversation, open_input_menu
from utils.setting_rows import setting_button
from utils.sidebar_chats import chat_menu, chat_row
from utils.user_menu import open_settings

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

VISIBILITY = "/api/v1/configs/interface/visibility"
CHAT_CONFIG = ("/api/v1/chats/config", "/api/v1/chats/config")


@pytest.fixture
def hide(admin):
    """`hide(*ids)` saves a map hiding those controls; the instance's own map is restored after."""
    with admin.client() as client:
        config = client.get("/api/config")
        assert config.status_code == 200, config.text
        before = config.json()["ui"].get("visibility") or {}

        def save(*control_ids: str) -> None:
            saved = client.post(
                VISIBILITY, json={"visibility": {control_id: False for control_id in control_ids}}
            )
            assert saved.status_code == 200, saved.text

        yield save
        client.post(VISIBILITY, json={"visibility": before}).raise_for_status()


def archived_chat(account, title: str) -> str:
    with account.client() as client:
        chat_id, _ = seed_chat(
            client,
            [
                {"role": "user", "content": "Which pier does the ferry use?"},
                {"role": "assistant", "content": "The north pier."},
            ],
        )
        renamed = client.post(f"/api/v1/chats/{chat_id}", json={"chat": {"title": title}})
        assert renamed.status_code == 200, renamed.text
        archived = client.post(f"/api/v1/chats/{chat_id}/archive")
        assert archived.status_code == 200, archived.text
        assert archived.json()["archived"], archived.text
    return chat_id


def header_menu(page: Page, chat_id: str) -> Locator:
    page.goto(f"/c/{chat_id}")
    page.get_by_label("Chat actions").click()
    menu = page.get_by_role("menu")
    expect(menu.get_by_role("menuitem", name="Delete", exact=True)).to_be_visible()
    return menu


def test_attach_webpage_switched_off_leaves_the_users_menu_but_not_the_admins(
    page_for, admin, make_user, hide, preserve
):
    preserve("tasks", CHAT_CONFIG)
    admin_page = page_for(admin, path="/admin/settings/interface")
    settings = admin_page.get_by_role("dialog")
    setting_button(settings, "Visible controls").click()
    controls = settings.locator("#interface-visible-controls")
    controls.get_by_text("Chat", exact=True).click()  # the Chat group opens folded
    switch = controls.get_by_role("switch", name="Attach Webpage", exact=True)
    expect(switch).to_have_attribute("aria-checked", "true")
    switch.click()
    expect(switch).to_have_attribute("aria-checked", "false")
    with admin_page.expect_response(lambda response: VISIBILITY in response.url) as saved:
        settings.get_by_role("button", name="Save", exact=True).click()
    assert saved.value.ok, saved.value.text()

    user_menu = open_input_menu(page_for(make_user()))
    expect(user_menu.get_by_role("menuitem", name="Attach Files")).to_be_visible()
    expect(user_menu.get_by_role("menuitem", name="Attach Webpage")).to_have_count(0)

    admin_page.goto("/")
    expect(chat_input(admin_page)).to_be_visible()
    admin_menu = open_input_menu(admin_page)
    expect(admin_menu.get_by_role("menuitem", name="Attach Webpage")).to_be_visible()


def test_hidden_archive_takes_archive_and_unarchive_from_users_only(
    page_for, admin, make_user, hide
):
    hide("chat.archive")
    account = make_user()
    title = f"Ferry pier {uuid.uuid4().hex[:8]}"
    chat_id = archived_chat(account, title)
    with account.client() as client:
        open_chat, _ = seed_chat(client, [{"role": "user", "content": "Is the pier open?"}])
    page = page_for(account)

    menu = header_menu(page, chat_id)
    expect(menu.get_by_role("menuitem", name="Unarchive")).to_have_count(0)
    expect(menu.get_by_role("menuitem", name="Archive", exact=True)).to_have_count(0)
    page.keyboard.press("Escape")

    page.get_by_role("button", name="Open Sidebar", exact=True).click()
    sidebar = page.get_by_role("navigation", name="Chat history")
    expect(chat_row(sidebar, "Seeded chat")).to_be_visible()
    sidebar_menu = chat_menu(sidebar, "Seeded chat")
    expect(sidebar_menu.get_by_role("menuitem", name="Rename")).to_be_visible()
    expect(sidebar_menu.get_by_role("menuitem", name="Archive")).to_have_count(0)
    page.keyboard.press("Escape")

    open_settings(page)
    settings = page.get_by_role("dialog")
    settings.get_by_role("tab", name="Archived Chats").click()
    expect(settings.get_by_text(title)).to_be_visible()
    expect(settings.get_by_role("button", name="Unarchive Chat")).to_have_count(0)
    with account.client() as client:
        assert client.get(f"/api/v1/chats/{chat_id}").json()["archived"]
        assert not client.get(f"/api/v1/chats/{open_chat}").json()["archived"]

    admin_chat = archived_chat(admin, f"Admin pier {uuid.uuid4().hex[:8]}")
    admin_menu = header_menu(page_for(admin), admin_chat)
    expect(admin_menu.get_by_role("menuitem", name="Unarchive")).to_be_visible()


def test_a_hidden_setting_keeps_the_value_the_user_saved(page_for, make_user, hide):
    account = make_user()
    with account.client() as client:
        stored = client.post(
            "/api/v1/users/user/settings/update", json={"ui": {"chatBubble": False}}
        )
        assert stored.status_code == 200, stored.text
        chat_id, _ = seed_chat(
            client,
            [
                {"role": "user", "content": "Is the lamp lit?"},
                {"role": "assistant", "content": "It is."},
            ],
        )
    hide("settings.chat_bubble")
    page = page_for(account, path=f"/c/{chat_id}")

    own_label = conversation(page).locator(".user-message").get_by_text("You", exact=True)
    expect(own_label).to_be_visible()

    page.goto("/?settings=interface")
    interface = page.locator("#tab-interface")
    expect(interface.get_by_role("switch", name="Widescreen Mode", exact=True)).to_be_visible()
    expect(interface.get_by_role("switch", name="Chat Bubble UI", exact=True)).to_have_count(0)
