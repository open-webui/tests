"""Regression: deleting a channel message left its quote on the replies of everyone watching.

open-webui issue #30313, fixed by b988f06ce (PR #30314): when a channel message was deleted, the
channel and thread views removed it from the list and nothing else. Every reply kept quoting the
deleted text and its author, and a reply someone was composing to it kept its "Replying to" chip,
so sending it was refused. The fix clears both when the deletion event arrives.

Two accounts share a channel: the reader quotes the author's message, the author deletes it, and
the reader's open view has to drop the quote and the chip without a reload. Both are cleared in
the same update that removes the deleted row, so the test reads them once the row is gone.

Twin of unit/frontend/test_deleted_message_quote_clearing.py.

Discriminates: passes on the bbfa876af build; with the deletion branch of Channel.svelte and
Thread.svelte back to filtering the row only, the quote and the chip stay on the reader's screen.
"""

from __future__ import annotations

import pytest
from playwright.sync_api import Locator, Page, expect

from harness.channel_quotes import delete_message, enable_channels, group_channel, post_message
from utils.chat_ui import send

pytestmark = [pytest.mark.regression, pytest.mark.requires_browser, pytest.mark.requires_source]

QUOTED = "the words about to be deleted"


@pytest.fixture
def channel(admin, preserve, make_user):
    """A group channel of two fresh accounts: its id, the author and the reader."""
    preserve("admin_config")
    enable_channels(admin)
    author, reader = make_user(), make_user()
    return group_channel(author, reader), author, reader


def message_rows(page: Page, rendered_id: str) -> Locator:
    return page.locator(f"[id='message-{rendered_id}']")


def toolbar_button(message: Locator, tooltip: str) -> Locator:
    """An icon-only button of a message's hover toolbar, told apart by the tooltip it shows."""
    message.hover()
    buttons = message.get_by_role("button")
    index = buttons.evaluate_all(
        "(buttons, tooltip) => buttons.findIndex("
        "(button) => button.parentElement?._tippy?.props.content === tooltip)",
        tooltip,
    )
    assert index >= 0, f"no button in the message toolbar shows the tooltip {tooltip!r}"
    return buttons.nth(index)


def left_on_screen(page: Page, deleted_rendered_id: str, author_name: str) -> dict:
    """What still points at the deleted message once its own row is gone."""
    expect(message_rows(page, deleted_rendered_id)).to_have_count(0)
    return {
        "copies of its text": page.get_by_text(QUOTED).count(),
        "reply chip": page.get_by_text(f"Replying to {author_name}").is_visible(),
    }


def test_deleting_a_message_clears_its_quote_and_the_reply_chip(channel, page_for):
    channel_id, author, reader = channel
    quoted_id = post_message(author, channel_id, QUOTED)
    reader_page, author_page = page_for(reader), page_for(author)
    for page in (reader_page, author_page):
        page.goto(f"/channels/{channel_id}")
        expect(page.get_by_text(QUOTED)).to_be_visible()

    toolbar_button(message_rows(reader_page, quoted_id).first, "Reply").click()
    send(reader_page, "my answer to that")
    expect(reader_page.get_by_text("my answer to that")).to_be_visible()
    expect(reader_page.get_by_text(QUOTED)).to_have_count(2)  # the message and the quote of it
    toolbar_button(message_rows(reader_page, quoted_id).first, "Reply").click()
    expect(reader_page.get_by_text(f"Replying to {author.name}")).to_be_visible()

    toolbar_button(message_rows(author_page, quoted_id).first, "Delete").click()
    author_page.get_by_role("button", name="Confirm").click()

    assert left_on_screen(reader_page, quoted_id, author.name) == {
        "copies of its text": 0,
        "reply chip": False,
    }
    expect(reader_page.get_by_text("my answer to that")).to_be_visible()


def test_the_thread_panel_clears_them_too(channel, page_for):
    channel_id, author, reader = channel
    root_id = post_message(author, channel_id, "a thread starter")
    quoted_id = post_message(author, channel_id, QUOTED, parent_id=root_id)
    post_message(
        reader, channel_id, "my answer in the thread", parent_id=root_id, reply_to_id=quoted_id
    )
    reader_page = page_for(reader)
    reader_page.goto(f"/channels/{channel_id}")
    starter = message_rows(reader_page, root_id).first
    expect(starter).to_contain_text("a thread starter")

    toolbar_button(starter, "Reply in Thread").click()
    expect(reader_page.get_by_text(QUOTED)).to_have_count(2)  # the message and the quote of it
    toolbar_button(message_rows(reader_page, f"{root_id}-{quoted_id}").first, "Reply").click()
    expect(reader_page.get_by_text(f"Replying to {author.name}")).to_be_visible()

    delete_message(author, channel_id, quoted_id)

    assert left_on_screen(reader_page, f"{root_id}-{quoted_id}", author.name) == {
        "copies of its text": 0,
        "reply chip": False,
    }
    expect(reader_page.get_by_text("my answer in the thread")).to_be_visible()
