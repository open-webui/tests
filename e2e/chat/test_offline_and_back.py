"""Journey: the browser loses its network and gets it back, while idle and while a reply streams.

The browser context goes offline, the way a laptop leaving the Wi-Fi does. The page warns that
the connection was lost and says it reconnected once the network is back. An idle page then
answers the next message. A reply that keeps streaming on the server while the page is offline
is picked back up with no piece missing once the page is back, and it shows whole and finished
at its end, whether it was still streaming when the network returned or had finished meanwhile.

The pick-up test is red on dev ebc6add67: a page that reconnects while its reply still streams
only listens on, so the pieces sent while it was offline stay missing from the reply until it
ends, where the docs promise that a reconnecting browser picks a reply back up where it left off.

Discriminates: passes on dev ebc6add67 except the pick-up test, which passes on a build that
reloads the chat on every reconnect with a reply pending; every test fails on a build whose
socket does not reconnect.
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

CONNECTION_LOST = "Connection lost. Reconnecting..."
RECONNECTED = "Reconnected"
WHOLE_REPLY = "".join(SLOW_PIECES).strip()
PROMPT = "tell me a long story"


@pytest.fixture
def chat_page(page_for, make_user):
    return page_for(make_user())


def _expect_whole_and_finished(page) -> None:
    expect(last_reply(page)).to_contain_text(WHOLE_REPLY, timeout=REPLY_TIMEOUT_MS)
    expect(stop_button(page)).to_be_hidden()
    stuck = "the reply still shows its blinking cursor after the page came back online"
    expect(typing_cursor(last_reply(page)), stuck).to_have_count(0)
    expect(regenerate_buttons(page).last).to_be_visible()


def _go_offline_and_back(page: Page) -> None:
    page.context.set_offline(True)
    expect(page.get_by_text(CONNECTION_LOST)).to_be_visible()
    page.context.set_offline(False)
    expect(page.get_by_text(RECONNECTED)).to_be_visible()


def test_an_idle_page_reconnects_and_answers_the_next_message(chat_page, upstream):
    upstream.queue(reply.text("Hello there.", match=reply.answering("hello")))
    send(chat_page, "hello")
    expect_reply(chat_page, "Hello there.")

    _go_offline_and_back(chat_page)

    upstream.queue(reply.text("Back online.", match=reply.answering("are you there?")))
    send(chat_page, "are you there?")
    expect_reply(chat_page, "Back online.")


def test_a_reply_still_streaming_when_the_network_returns_ends_whole(chat_page, upstream):
    upstream.queue(reply.text(SLOW_PIECES, chunk_delay=0.7, match=reply.answering(PROMPT)))
    send(chat_page, PROMPT)
    expect(last_reply(chat_page)).to_contain_text("part-1")

    _go_offline_and_back(chat_page)

    expect(stop_button(chat_page)).to_be_visible()  # the reply had not finished yet
    _expect_whole_and_finished(chat_page)


def test_a_reconnected_page_picks_a_running_reply_back_up(chat_page, upstream):
    upstream.queue(reply.text(SLOW_PIECES, chunk_delay=1.0, match=reply.answering(PROMPT)))
    send(chat_page, PROMPT)
    expect(last_reply(chat_page)).to_contain_text("part-1")

    _go_offline_and_back(chat_page)
    expect(last_reply(chat_page)).to_contain_text(SLOW_PIECES[15].strip(), timeout=REPLY_TIMEOUT_MS)
    shown_mid_stream = last_reply(chat_page).inner_text()
    expect(stop_button(chat_page)).to_be_visible()  # the reply had not finished yet

    assert "".join(SLOW_PIECES[:16]).strip() in shown_mid_stream, (
        f"the pieces streamed while the page was offline are missing: {shown_mid_stream!r}"
    )


def test_a_reply_that_finished_while_offline_shows_whole(page_for, make_user, upstream):
    account = make_user()
    chat_page = page_for(account)
    upstream.queue(reply.text(SLOW_PIECES, chunk_delay=0.2, match=reply.answering(PROMPT)))
    send(chat_page, PROMPT)
    expect(last_reply(chat_page)).to_contain_text("part-1")
    expect(chat_page).to_have_url(re.compile(r"/c/[\w-]+$"))

    chat_page.context.set_offline(True)
    expect(chat_page.get_by_text(CONNECTION_LOST)).to_be_visible()
    with account.client() as client:
        wait_until_no_reply_runs(client, chat_page.url.rsplit("/", 1)[1])
    chat_page.context.set_offline(False)

    expect(chat_page.get_by_text(RECONNECTED)).to_be_visible()
    _expect_whole_and_finished(chat_page)
