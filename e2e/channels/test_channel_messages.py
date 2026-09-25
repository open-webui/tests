"""Journey: two people in a channel see each other's messages, reactions and thread replies live.

Two fresh accounts share a group channel, each with it open in a browser of their own. A message
one of them sends shows up on the other's screen without a reload; a reaction the other adds
shows on the sender's screen with its count, and a reply in the message's thread shows there as
a reply count that opens the thread.

Discriminates: passes on dev ac00d40e3; in a backend copy, with the new message emitted to no
room the post test fails (the reader sees nothing until a reload), and with the reaction event
sent under a type the page does not handle the reaction test fails at the count.
"""

from __future__ import annotations

import pytest
from playwright.sync_api import Locator, Page, expect

from harness.channel_quotes import enable_channels, group_channel
from utils.chat_ui import chat_input, send
from utils.tooltips import tooltip_button

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]


@pytest.fixture
def channel_pages(admin, preserve, make_user, page_for):
    """The author's and the reader's pages, both open on one group channel."""
    preserve("admin_config")
    enable_channels(admin)
    author, reader = make_user(), make_user()
    channel_id = group_channel(author, reader)
    pages = page_for(author), page_for(reader)
    for page in pages:
        page.goto(f"/channels/{channel_id}")
        expect(chat_input(page)).to_be_visible()
    return pages


def _message(page: Page, text: str) -> Locator:
    return page.locator("[id^='message-']").filter(has_text=text).first


def test_a_message_reaches_the_other_member_without_a_reload(channel_pages):
    author_page, reader_page = channel_pages
    send(author_page, "the ferry leaves at nine")
    expect(_message(reader_page, "the ferry leaves at nine")).to_be_visible()


def test_a_reaction_and_a_thread_reply_reach_the_author_live(channel_pages):
    author_page, reader_page = channel_pages
    send(author_page, "who brings the maps?")
    on_reader = _message(reader_page, "who brings the maps?")
    expect(on_reader).to_be_visible()

    on_reader.hover()
    tooltip_button(on_reader, "Add Reaction").click()
    reader_page.get_by_placeholder("Search all emojis").fill("rocket")
    reader_page.get_by_role("menu").get_by_role("button", name="1F680").click()
    on_author = _message(author_page, "who brings the maps?")
    expect(on_author.get_by_role("button", name="rocket 1")).to_be_visible()

    on_reader.hover()
    tooltip_button(on_reader, "Reply in Thread").click()
    reader_page.get_by_label("Reply to thread...").click()
    reader_page.keyboard.type("I will, and a compass")
    reader_page.keyboard.press("Enter")

    replies = on_author.get_by_role("button", name="1 Replies")
    expect(replies).to_be_visible()
    replies.click()
    expect(author_page.get_by_text("I will, and a compass")).to_be_visible()
