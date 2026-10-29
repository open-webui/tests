"""Journey: a person sets a status from the user menu, others see it, and clearing removes it.

The user menu offers "Update your status", a dialog with an emoji picker and a message. Once saved
the menu shows the status with a clear button. Another account sees it on the person's profile
card in a channel and beside the person's name in the sidebar's direct message entry; after the
person clears it, the menu offers to set one again and the other account sees neither.

Discriminates: passes on dev 176d31d1d; in a frontend copy, sending an empty message from the
dialog turns the save, card and sidebar tests red, hiding the status in the direct message entry
turns the sidebar test red, leaving the message out of the profile card turns the card test red,
sending the old values on clear turns the clear test red, and offering the status row whatever
the admin setting turns the switched-off test red.
"""

from __future__ import annotations

import pytest
from playwright.sync_api import Locator, Page, expect

from harness.channel_quotes import ADMIN_CONFIG, enable_channels, group_channel, post_message
from utils.chat_ui import chat_input
from utils.menus import late_menu_entry
from utils.tooltips import tooltip_button
from utils.user_menu import settings_item

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

STATUS = "Sailing until noon"


@pytest.fixture
def people(admin, preserve, make_user):
    """The person who sets a status and the person who looks at it."""
    preserve("admin_config")
    enable_channels(admin)
    return make_user(), make_user()


def _open_user_menu(page: Page) -> None:
    expect(chat_input(page)).to_be_visible()
    page.get_by_role("navigation", name="Chat history").get_by_label("User menu").click()


def _set_status(page: Page, message: str) -> None:
    _open_user_menu(page)
    page.get_by_role("menuitem", name="Update your status").click()
    dialog = page.get_by_role("dialog")
    # the emoji picker button has no name; it is the dialog's second button after Close
    dialog.get_by_role("button").nth(1).click()
    page.get_by_placeholder("Search all emojis").click()
    page.keyboard.type("rocket")
    late_menu_entry(page, "1F680").click()
    dialog.get_by_placeholder("What's on your mind?").fill(message)
    dialog.get_by_role("button", name="Save").click()
    expect(page.get_by_text("Status updated successfully")).to_be_visible()


def _stored_status(actor) -> tuple[str | None, str | None]:
    with actor.client() as client:
        stored = client.get("/api/v1/users/user/status").json()
    return stored["status_emoji"], stored["status_message"]


def _direct_message_entry(page: Page, other_name: str) -> Locator:
    expect(chat_input(page)).to_be_visible()
    page.get_by_role("button", name="Open Sidebar", exact=True).click()
    sidebar = page.get_by_role("navigation", name="Chat history")
    sidebar.get_by_text("Channels", exact=True).click()
    return sidebar.get_by_role("link", name=other_name)


def _open_profile_card(page: Page, channel_id: str, author, text: str) -> None:
    page.goto(f"/channels/{channel_id}")
    message = page.locator("[id^='message-']").filter(has_text=text).first
    # the author's picture has no alt text, so it has no role to find it by
    message.locator(f"img[src$='/users/{author.id}/profile/image']").click()


def _open_direct_message(person, viewer) -> None:
    with person.client() as client:
        client.get(f"/api/v1/channels/users/{viewer.id}").raise_for_status()


def test_a_status_set_from_the_user_menu_is_saved_and_shown_in_the_menu(people, page_for):
    person, _ = people
    page = page_for(person)

    _set_status(page, STATUS)

    _open_user_menu(page)
    menu = page.get_by_role("menu")
    expect(menu.get_by_text(STATUS)).to_be_visible()
    expect(menu.get_by_role("img", name="rocket")).to_be_visible()
    assert _stored_status(person) == ("rocket", STATUS)


def test_another_person_sees_the_status_on_the_profile_card(people, page_for):
    person, viewer = people
    channel_id = group_channel(person, viewer)
    post_message(person, channel_id, "anchoring at the bay")
    _set_status(page_for(person), STATUS)
    viewer_page = page_for(viewer)

    _open_profile_card(viewer_page, channel_id, person, "anchoring at the bay")

    expect(viewer_page.get_by_text(STATUS)).to_be_visible()
    expect(viewer_page.get_by_role("img", name="rocket")).to_be_visible()


def test_another_person_sees_the_status_beside_the_direct_message(people, page_for):
    person, viewer = people
    _open_direct_message(person, viewer)
    _set_status(page_for(person), STATUS)
    viewer_page = page_for(viewer)

    entry = _direct_message_entry(viewer_page, person.name)

    expect(entry.get_by_text(STATUS)).to_be_visible()
    expect(entry.get_by_role("img", name="rocket")).to_be_visible()


def test_clearing_the_status_removes_it_for_everyone(people, page_for):
    person, viewer = people
    _open_direct_message(person, viewer)
    person_page = page_for(person)
    _set_status(person_page, STATUS)

    _open_user_menu(person_page)
    tooltip_button(person_page.get_by_role("menu"), "Clear status", role="menuitem").click()
    expect(person_page.get_by_text("Status cleared successfully")).to_be_visible()

    _open_user_menu(person_page)
    expect(person_page.get_by_role("menuitem", name="Update your status")).to_be_visible()
    assert _stored_status(person) == ("", "")
    viewer_page = page_for(viewer)
    expect(_direct_message_entry(viewer_page, person.name)).to_be_visible()
    expect(viewer_page.get_by_text(STATUS)).to_have_count(0)


def test_the_user_menu_offers_no_status_while_user_status_is_off(admin, people, page_for):
    person, _ = people
    with admin.client() as client:
        current = client.get(ADMIN_CONFIG).json()
        client.post(ADMIN_CONFIG, json={**current, "ENABLE_USER_STATUS": False}).raise_for_status()
    page = page_for(person)

    _open_user_menu(page)

    expect(settings_item(page)).to_be_visible()
    expect(page.get_by_role("menuitem", name="Update your status")).to_have_count(0)
