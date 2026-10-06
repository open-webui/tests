"""Regression: a reply started in one channel followed the reader into the next channel.

open-webui issue #31375, fixed by PR #31376: after pressing Reply on a message and moving to
another channel from the sidebar, the new channel still read "Replying to" the other channel's
message, and sending there failed. Switching channels now drops the pending reply, so the box is
clean and a message sent in the new channel arrives.

The reader stays in one page and switches through the sidebar, so the open channel view is reused.

Discriminates: passes on the dev a5bc78300 build; with the pending reply kept across the switch
the "Replying to" banner is still up in the second channel and the message sent there is refused.
"""

from __future__ import annotations

import re
import time

import pytest
from playwright.sync_api import Locator, Page, expect

from harness.channel_quotes import enable_channels, group_channel, post_message
from utils.chat_ui import chat_input
from utils.tooltips import tooltip_button

pytestmark = [pytest.mark.regression, pytest.mark.requires_browser, pytest.mark.requires_source]


@pytest.fixture
def channels(admin, preserve, make_user):
    """The author, the reader and two group channels they share, each with its name."""
    preserve("admin_config")
    enable_channels(admin)
    author, reader = make_user(), make_user()
    first, second = group_channel(author, reader), group_channel(author, reader)
    return author, reader, first, second


def channel_name(account, channel_id: str) -> str:
    with account.client() as client:
        fetched = client.get(f"/api/v1/channels/{channel_id}")
    fetched.raise_for_status()
    return fetched.json()["name"]


def sidebar_entry(page: Page, name: str) -> Locator:
    sidebar = page.get_by_role("navigation", name="Chat history")
    open_sidebar = page.get_by_role("button", name="Open Sidebar", exact=True)
    if open_sidebar.is_visible():
        open_sidebar.click()
    section = sidebar.get_by_role("button", name="Channels")
    if section.get_attribute("aria-expanded") == "false":
        section.click()
    return sidebar.get_by_role("link", name=name)


def stored(account, channel_id: str) -> list[str]:
    with account.client() as client:
        listed = client.get(f"/api/v1/channels/{channel_id}/messages")
    listed.raise_for_status()
    return [message["content"] for message in listed.json()]


def saved(account, channel_id: str, timeout: float = 10.0) -> list[str]:
    """The channel's messages once one is saved; the page shows a sent message before that."""
    deadline = time.monotonic() + timeout
    while not (messages := stored(account, channel_id)) and time.monotonic() < deadline:
        time.sleep(0.2)
    return messages


def test_switching_channels_drops_the_pending_reply_and_a_message_sent_there_arrives(
    channels, page_for
):
    author, reader, first, second = channels
    post_message(author, first, "the boat leaves at nine")
    page = page_for(reader)
    page.goto(f"/channels/{first}")
    quoted = page.locator("[id^='message-']").filter(has_text="the boat leaves at nine").first
    expect(quoted).to_be_visible()
    quoted.hover()
    tooltip_button(quoted, "Reply").click()
    banner = page.get_by_text(f"Replying to {author.name}")
    expect(banner).to_be_visible()

    sidebar_entry(page, channel_name(reader, second)).click()

    expect(page).to_have_url(re.compile(f"/channels/{second}$"))
    expect(chat_input(page)).to_be_visible()
    expect(
        banner, "#31375: the reply to the other channel's message is still pending"
    ).to_have_count(0)
    chat_input(page).click()
    page.keyboard.type("hello from the second channel")
    page.keyboard.press("Enter")
    expect(page.get_by_text("hello from the second channel")).to_be_visible()
    assert saved(reader, second) == ["hello from the second channel"]
    assert stored(reader, first) == ["the boat leaves at nine"]
