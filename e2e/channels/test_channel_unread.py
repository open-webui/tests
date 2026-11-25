"""Journey: the unread count on a channel's sidebar entry, live and after a reload.

A member who is in another channel sees the count on a group channel's entry go up with every
message someone else posts there; their own messages, sent from another tab, do not count. The count
reads the same after a reload, and opening the channel clears it for good. A member who picks
"Mentions only" in a channel's options menu sees the entry marked as muted, with no count and no
toast for an ordinary message, still gets a toast when someone @mentions them, and keeps that
choice over a reload; "All messages" brings the count back. The mention is typed by hand as
"@<name>" without picking from the list, and posts as a real mention; a word that names nobody and
the name inside inline code stay plain text.

Discriminates: passes on dev ebc6add67; in a frontend copy, counting the reader's own messages turns
the test red, and so does counting no unread messages in a backend copy. On dev b704d0613, in a
frontend copy, a menu that saves the opposite choice, a toast that ignores the mute and a mute that
also silences mentions each turn the muted channel test red. On dev c143625cc, in a frontend
copy, sending without resolving typed names turns it red, and so does resolving names inside code.
"""

from __future__ import annotations

import uuid

import pytest
from playwright.sync_api import Locator, Page, expect

from harness.channel_quotes import enable_channels, group_channel, post_message
from utils.chat_ui import chat_input
from utils.sidebar_channels import channels_section

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]


@pytest.fixture
def people(admin, preserve, make_user):
    """A reader and a writer who share two group channels: the busy one and a quiet one."""
    preserve("admin_config")
    enable_channels(admin)
    reader, writer = make_user(), make_user()
    return reader, writer, group_channel(reader, writer), group_channel(reader, writer)


def _channel_name(account, channel_id: str) -> str:
    with account.client() as client:
        fetched = client.get(f"/api/v1/channels/{channel_id}")
    fetched.raise_for_status()
    return fetched.json()["name"]


def _sidebar_entry(page: Page, channel_name: str) -> Locator:
    expect(chat_input(page)).to_be_visible()
    return channels_section(page).get_by_role("link", name=channel_name)


def _choose_notifications(page: Page, channel_name: str, choice: str) -> None:
    _sidebar_entry(page, channel_name).hover()
    row = page.locator("#sidebar-channel-item").filter(
        has=page.get_by_role("link", name=channel_name)
    )
    # the menu's trigger wraps its button, and both carry the name
    row.get_by_role("button", name="Channel options").first.click()
    page.get_by_role("menuitemradio", name=choice).click()


def _open_quiet_channel(page_for, reader, quiet_id: str) -> Page:
    page = page_for(reader, path=f"/channels/{quiet_id}")
    expect(chat_input(page)).to_be_visible()
    return page


def test_the_count_adds_up_holds_over_a_reload_and_clears_when_opened(people, page_for):
    reader, writer, busy_id, quiet_id = people
    page = _open_quiet_channel(page_for, reader, quiet_id)
    busy_name = _channel_name(reader, busy_id)
    entry = _sidebar_entry(page, busy_name)
    expect(entry).to_be_visible()

    post_message(writer, busy_id, "first stop is the bakery")
    post_message(reader, busy_id, "noted, from my phone")
    post_message(writer, busy_id, "second stop is the market")

    expect(entry.get_by_title("Unread")).to_have_text("2")
    page.reload()
    entry = _sidebar_entry(page, busy_name)
    expect(entry.get_by_title("Unread")).to_have_text("2")
    entry.click()
    expect(page.get_by_text("second stop is the market")).to_be_visible()
    expect(entry.get_by_title("Unread")).to_have_count(0)
    page.goto(f"/channels/{quiet_id}")
    expect(_sidebar_entry(page, busy_name).get_by_title("Unread")).to_have_count(0)


def _stored_with(account, channel_id: str, text: str) -> str:
    with account.client() as client:
        listed = client.get(f"/api/v1/channels/{channel_id}/messages")
    listed.raise_for_status()
    [content] = [message["content"] for message in listed.json() if text in message["content"]]
    return content


def test_a_channel_muted_from_its_menu_stays_quiet_until_the_reader_is_mentioned(people, page_for):
    reader, writer, busy_id, quiet_id = people
    page = _open_quiet_channel(page_for, reader, quiet_id)
    busy_name = _channel_name(reader, busy_id)
    expect(_sidebar_entry(page, busy_name)).to_be_visible()
    writer_page = page_for(writer, path=f"/channels/{busy_id}")
    expect(chat_input(writer_page)).to_be_visible()

    _choose_notifications(page, busy_name, "Mentions only")

    entry = _sidebar_entry(page, busy_name)
    expect(entry.get_by_text("Mentions only", exact=True)).to_be_visible()
    post_message(writer, busy_id, "first stop is the bakery")
    nobody = f"@nobody{uuid.uuid4().hex[:6]}"
    chat_input(writer_page).click()
    writer_page.keyboard.type(
        f"@{reader.name} bring the map, `@{reader.name}` marks the spot, ask {nobody} too"
    )
    writer_page.keyboard.press("Enter")
    expect(page.get_by_text("bring the map")).to_be_visible()
    expect(page.get_by_text("first stop is the bakery")).to_have_count(0)
    expect(entry.get_by_title("Unread")).to_have_count(0)
    stored = _stored_with(writer, busy_id, "bring the map")
    assert stored.startswith(f"<@U:{reader.id}|{reader.name}> bring the map"), stored
    assert f"`@{reader.name}` marks the spot" in stored, stored
    assert f"ask {nobody} too" in stored, stored
    posted = writer_page.locator("[id^='message-']:not(#message-input-container)")
    sent = posted.filter(has_text="bring the map").first
    expect(sent.locator(".mention")).to_have_text([f"@{reader.name}"])

    page.reload()
    entry = _sidebar_entry(page, busy_name)
    expect(entry.get_by_text("Mentions only", exact=True)).to_be_visible()
    expect(entry.get_by_title("Unread")).to_have_count(0)

    _choose_notifications(page, busy_name, "All messages")

    expect(entry.get_by_text("Mentions only", exact=True)).to_have_count(0)
    expect(entry.get_by_title("Unread")).to_have_text("2")
