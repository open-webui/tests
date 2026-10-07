"""Journey: a reply that streams while the person looks elsewhere, or watches it from two tabs.

A reply keeps going on the server whatever the page does. It finishes while the person reads
another chat and shows whole, finished, when they come back through the sidebar; it finishes
after they closed the tab and shows whole when they open the chat again, which still answers the
next message. A second tab opened on the chat while the reply streams follows it to the end with
the same text as the first, and Stop in that second tab stops the reply in both.

Twin of integration/chat/test_stream_interruptions.py for the closed tab.

Discriminates: passes on dev ebc6add67; the follow test fails on a backend copy that sends a
reply's events only to the tab that asked, the Stop test on one whose stop endpoint leaves the
reply running, the closed-tab test on one that cancels every reply when a socket disconnects and
the other-chat test on one that never saves a finished reply.
"""

from __future__ import annotations

import re

import pytest
from playwright.sync_api import Page, expect

from harness import upstream as reply
from harness.inflight import SLOW_PIECES, wait_until_no_reply_runs
from utils.chat_ui import (
    REPLY_TIMEOUT_MS,
    expect_reply,
    last_reply,
    regenerate_buttons,
    send,
    stop_button,
    typing_cursor,
)

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

WHOLE_REPLY = "".join(SLOW_PIECES).strip()
STORY = "tell me a long story"


def _start_story(page: Page, upstream) -> str:
    """Send the story prompt and return the chat's address once the reply is streaming."""
    upstream.queue(reply.text(SLOW_PIECES, chunk_delay=0.3, match=reply.answering(STORY)))
    send(page, STORY)
    expect(last_reply(page)).to_contain_text("part-1")
    expect(page).to_have_url(re.compile(r"/c/[\w-]+$"))
    return page.url


def _wait_for_the_reply_to_finish(account, chat_url: str) -> None:
    with account.client() as client:
        wait_until_no_reply_runs(client, chat_url.rsplit("/", 1)[1])


def _expect_whole_and_finished(page: Page) -> None:
    expect(last_reply(page)).to_contain_text(WHOLE_REPLY, timeout=REPLY_TIMEOUT_MS)
    expect(stop_button(page)).to_be_hidden()
    expect(typing_cursor(last_reply(page))).to_have_count(0)
    expect(regenerate_buttons(page).last).to_be_visible()


def _open_sidebar(page: Page):
    page.get_by_role("button", name="Open Sidebar", exact=True).click()
    return page.get_by_role("navigation", name="Chat history")


def test_a_reply_that_finished_in_another_chat_shows_on_return(page_for, make_user, upstream):
    account = make_user()
    page = page_for(account)
    upstream.queue(reply.text("Rome is the capital.", match=reply.answering("capital of Italy")))
    send(page, "capital of Italy")
    expect_reply(page, "Rome is the capital.")
    page.goto("/")

    story_url = _start_story(page, upstream)
    sidebar = _open_sidebar(page)
    sidebar.get_by_role("button", name="capital of Italy").click()
    expect_reply(page, "Rome is the capital.")
    _wait_for_the_reply_to_finish(account, story_url)

    sidebar.get_by_role("button", name=STORY).click()
    _expect_whole_and_finished(page)


def test_a_second_tab_follows_a_streaming_reply_to_its_end(page_for, make_user, upstream):
    page = page_for(make_user())
    chat_url = _start_story(page, upstream)

    second_tab = page.context.new_page()
    second_tab.goto(chat_url)
    expect(last_reply(second_tab)).to_contain_text("part-1")
    expect(stop_button(second_tab)).to_be_visible()

    _expect_whole_and_finished(second_tab)
    _expect_whole_and_finished(page)


def test_stop_in_a_second_tab_stops_the_reply_in_both(page_for, make_user, upstream):
    page = page_for(make_user())
    chat_url = _start_story(page, upstream)
    second_tab = page.context.new_page()
    second_tab.goto(chat_url)
    expect(stop_button(second_tab)).to_be_visible()

    stop_button(second_tab).click()

    expect(stop_button(second_tab)).to_be_hidden()
    expect(stop_button(page)).to_be_hidden()
    second_tab.wait_for_timeout(1500)  # five more piece intervals
    stopped_text = last_reply(second_tab).inner_text().strip()
    next_piece = SLOW_PIECES[stopped_text.count("part-")].strip()
    expect(last_reply(page)).to_contain_text(stopped_text)
    expect(last_reply(page)).not_to_contain_text(next_piece)
    expect(last_reply(second_tab)).not_to_contain_text(next_piece)


def test_a_reply_finishes_after_its_tab_was_closed(page_for, make_user, upstream):
    account = make_user()
    page = page_for(account)
    chat_url = _start_story(page, upstream)
    browser_context = page.context

    page.close()
    _wait_for_the_reply_to_finish(account, chat_url)
    reopened = browser_context.new_page()
    reopened.goto(chat_url)

    _expect_whole_and_finished(reopened)
    upstream.queue(reply.text("Glad you liked it.", match=reply.answering("thanks")))
    send(reopened, "thanks")
    expect_reply(reopened, "Glad you liked it.")
