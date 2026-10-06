"""Journey: the unread count on a channel's sidebar entry, live and after a reload.

A member who is in another channel sees the count on a group channel's entry go up with every
message someone else posts there; their own messages, sent from another tab, do not count. The count
reads the same after a reload, and opening the channel clears it for good. A reply in a thread is
not counted, so the count does not change on a reload. Messages an outside service posts through a
channel's webhook count for every member, the person who made the webhook too.

Discriminates: passes on dev ebc6add67 except the two tests named below, which are red there; in a
frontend copy, counting the reader's own messages turns the first test red and leaving thread
replies out of the live count turns the thread reply test green; in a backend copy, counting no
unread messages turns the first test red and counting webhook posts for their maker turns the
webhook test green.
"""

from __future__ import annotations

import httpx
import pytest
from playwright.sync_api import Locator, Page, expect

from harness.channel_quotes import enable_channels, group_channel, post_message
from utils.chat_ui import chat_input

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
    sidebar = page.get_by_role("navigation", name="Chat history")
    open_sidebar = page.get_by_role("button", name="Open Sidebar", exact=True)
    if open_sidebar.is_visible():
        open_sidebar.click()
    section = sidebar.get_by_role("button", name="Channels")
    if section.get_attribute("aria-expanded") == "false":
        section.click()
    return sidebar.get_by_role("link", name=channel_name)


def _open_quiet_channel(page_for, reader, quiet_id: str) -> Page:
    page = page_for(reader)
    page.goto(f"/channels/{quiet_id}")
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


def test_a_thread_reply_leaves_the_count_as_a_reload_shows_it(people, page_for):
    """Red on dev ebc6add67: a thread reply raises the live count, and a reload lowers it again.

    The server counts only messages in the channel itself, the sidebar adds one for every
    message that arrives, thread replies included.
    """
    reader, writer, busy_id, quiet_id = people
    parent_id = post_message(reader, busy_id, "who is driving?")
    page = _open_quiet_channel(page_for, reader, quiet_id)
    busy_name = _channel_name(reader, busy_id)
    entry = _sidebar_entry(page, busy_name)
    expect(entry).to_be_visible()

    post_message(writer, busy_id, "I can drive", parent_id=parent_id)
    post_message(writer, busy_id, "leaving at nine")

    # the toast of the later message shows once both have reached the sidebar
    expect(page.get_by_text("leaving at nine")).to_be_visible()
    shown_live = entry.get_by_title("Unread").inner_text()
    page.reload()
    after_reload = _sidebar_entry(page, busy_name).get_by_title("Unread")
    expect(after_reload).to_have_text("1")
    assert shown_live == "1", f"the live count read {shown_live}, counting the thread reply"


def test_webhook_posts_count_for_the_member_who_made_the_webhook(people, page_for):
    """Red on dev ebc6add67: the webhook's maker never has unread webhook posts after a reload.

    A webhook post is stored under the account that made the webhook, and the unread count skips
    a reader's own messages, so for that member every webhook post counts as already read.
    """
    reader, writer, busy_id, quiet_id = people
    with reader.client() as client:
        made = client.post(f"/api/v1/channels/{busy_id}/webhooks/create", json={"name": "Alarm"})
    assert made.status_code == 200, made.text
    webhook = made.json()
    page = _open_quiet_channel(page_for, reader, quiet_id)
    busy_name = _channel_name(reader, busy_id)
    entry = _sidebar_entry(page, busy_name)
    expect(entry).to_be_visible()

    url = f"{reader.base_url}/api/v1/channels/webhooks/{webhook['id']}/{webhook['token']}"
    for content in ("the oven is on", "the oven is still on"):
        posted = httpx.post(url, json={"content": content}, timeout=30)
        assert posted.status_code == 200, posted.text

    expect(entry.get_by_title("Unread")).to_have_text("2")
    page.reload()
    expect(_sidebar_entry(page, busy_name).get_by_title("Unread")).to_have_text("2")
