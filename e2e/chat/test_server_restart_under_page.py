"""Journey: the server restarts while a chat page stays open in the browser.

The instance is stopped and started again on the same address and data, as a redeploy does.
While it is down the page warns that the connection was lost; once it is back the page says it
reconnected and the next message is answered in the same chat. A reply that was streaming when
the server went down cannot finish, so after the reconnect the page shows it as finished: no
blinking cursor, no Stop button, Regenerate offered.

The interrupted-reply test is red on dev ebc6add67: after reconnecting, the page reloads the
chat and marks the unfinished reply done, but never redraws it, so the reply keeps its blinking
cursor and offers no Regenerate until the page is reloaded by hand.

Discriminates: passes on dev ebc6add67 except the interrupted-reply test, which passes on a
build that redraws the chat after marking it done; the reconnect tests fail on a build whose
socket does not reconnect (no "Reconnected", the next reply never shows).
"""

from __future__ import annotations

import re

import pytest
from playwright.sync_api import Page, expect

from harness import upstream as reply
from harness.actors import create_user
from harness.inflight import SLOW_PIECES
from utils.chat_ui import (
    expect_reply,
    last_reply,
    regenerate_buttons,
    send,
    stop_button,
    typing_cursor,
)

pytestmark = [
    pytest.mark.journey,
    pytest.mark.requires_browser,
    pytest.mark.requires_source,
    pytest.mark.slow,
]

CONNECTION_LOST = "Connection lost. Reconnecting..."
RECONNECTED = "Reconnected"
RESTART_TIMEOUT_MS = 120_000


@pytest.fixture
def restartable(instance_with):
    return instance_with({"WEBUI_NAME": "Restarting Open WebUI"})


def _chat_with_one_answer(page: Page, provider) -> None:
    provider.queue(reply.text("Hello there.", match=reply.answering("hello")))
    send(page, "hello")
    expect_reply(page, "Hello there.")
    expect(page).to_have_url(re.compile(r"/c/[\w-]+$"))


def _restart_under(page: Page, server) -> None:
    server.stop()
    expect(page.get_by_text(CONNECTION_LOST)).to_be_visible()
    server.start()
    expect(page.get_by_text(RECONNECTED)).to_be_visible(timeout=RESTART_TIMEOUT_MS)


def test_an_open_page_reconnects_and_the_next_message_is_answered(restartable, page_for):
    page = page_for(create_user(restartable))
    _chat_with_one_answer(page, restartable.upstream)
    chat_url = page.url

    _restart_under(page, restartable)

    restartable.upstream.queue(reply.text("Still here.", match=reply.answering("still there?")))
    send(page, "still there?")
    expect_reply(page, "Still here.")
    assert page.url == chat_url


def test_a_reply_cut_off_by_a_restart_is_shown_finished(restartable, page_for):
    page = page_for(create_user(restartable))
    prompt = "tell me a long story"
    restartable.upstream.queue(
        reply.text(SLOW_PIECES, chunk_delay=0.5, match=reply.answering(prompt))
    )
    send(page, prompt)
    expect(last_reply(page)).to_contain_text("part-1")

    _restart_under(page, restartable)

    expect(stop_button(page)).to_be_hidden()
    stuck = "the interrupted reply still shows its blinking cursor after the reconnect"
    expect(typing_cursor(last_reply(page)), stuck).to_have_count(0)
    expect(regenerate_buttons(page).last).to_be_visible()


def test_after_a_restart_cut_a_reply_off_the_next_message_is_answered(restartable, page_for):
    page = page_for(create_user(restartable))
    prompt = "tell me a long story"
    restartable.upstream.queue(
        reply.text(SLOW_PIECES, chunk_delay=0.5, match=reply.answering(prompt)),
        reply.text("A short one, then.", match=reply.answering("a short one")),
    )
    send(page, prompt)
    expect(last_reply(page)).to_contain_text("part-1")

    _restart_under(page, restartable)
    expect(stop_button(page)).to_be_hidden()

    send(page, "a short one")
    expect_reply(page, "A short one, then.")
