"""Journey: the browser loses its network and gets it back, while idle and while a reply streams.

The browser context goes offline, the way a laptop leaving the Wi-Fi does. The page warns that
the connection was lost and says it reconnected once the network is back. An idle page then
answers the next message. A reply that keeps streaming on the server while the page is offline
shows whole and finished at its end, whether it was still streaming when the network returned or
had finished meanwhile.

One test pins a fixed bug (issue open-webui/open-webui#32338, fix c92bc089b, PR
open-webui/open-webui#32341): a page signed in on the form without a reload reconnected with the
signed-out state it loaded with, the server dropped that connection as soon as the open chat
joined its room, and the page reconnected in a loop until reloaded.

Every test but the sign-in one runs in the main chat and in a side chat
(`utils/chat_surface.py`); the sign-in test is about the sign-in page's sockets. On dev
c15f28649 the two tests of a reply that went on while offline are red in a side chat, which had
no Regenerate under the reply (open-webui/open-webui#34014, fix PR open-webui/open-webui#34064).

Discriminates: passes on dev ebc6add67; every test fails on a build whose socket does not
reconnect. The sign-in test passes on the dev 5032e834e build and fails on that build with
c92bc089b reverted (over 300 more sockets in the 5 s after reconnecting).
"""

from __future__ import annotations

import re

import pytest
from playwright.sync_api import Page, expect

from harness import upstream as reply
from harness.inflight import SLOW_PIECES, wait_until_no_reply_runs
from utils.chat_surface import ChatSurface
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
# the socket retries up to 5 s apart, so it may only reconnect that long after the network returns
RECONNECT_TIMEOUT_MS = 15_000
WHOLE_REPLY = "".join(SLOW_PIECES).strip()
PROMPT = "tell me a long story"


@pytest.fixture
def chat_page(chat_for, make_user):
    return chat_for(make_user())


def _expect_whole_and_finished(chat: ChatSurface) -> None:
    expect(last_reply(chat)).to_contain_text(WHOLE_REPLY, timeout=REPLY_TIMEOUT_MS)
    expect(stop_button(chat)).to_be_hidden()
    stuck = "the reply still shows its blinking cursor after the page came back online"
    expect(typing_cursor(last_reply(chat)), stuck).to_have_count(0)
    expect(regenerate_buttons(chat).last).to_be_visible()


def _go_offline_and_back(page: Page) -> None:
    page.context.set_offline(True)
    expect(page.get_by_text(CONNECTION_LOST)).to_be_visible()
    page.context.set_offline(False)
    expect(page.get_by_text(RECONNECTED)).to_be_visible(timeout=RECONNECT_TIMEOUT_MS)


def test_an_idle_page_reconnects_and_answers_the_next_message(chat_page, upstream):
    upstream.queue(reply.text("Hello there.", match=reply.answering("hello")))
    send(chat_page, "hello")
    expect_reply(chat_page, "Hello there.")

    _go_offline_and_back(chat_page.page)

    upstream.queue(reply.text("Back online.", match=reply.answering("are you there?")))
    send(chat_page, "are you there?")
    expect_reply(chat_page, "Back online.")


def test_a_reply_still_streaming_when_the_network_returns_ends_whole(chat_page, upstream):
    upstream.queue(reply.text(SLOW_PIECES, chunk_delay=0.7, match=reply.answering(PROMPT)))
    send(chat_page, PROMPT)
    expect(last_reply(chat_page)).to_contain_text("part-1")

    _go_offline_and_back(chat_page.page)

    expect(stop_button(chat_page)).to_be_visible()  # the reply had not finished yet
    _expect_whole_and_finished(chat_page)


def test_a_reply_that_finished_while_offline_shows_whole(chat_for, make_user, upstream):
    account = make_user()
    chat_page = chat_for(account)
    upstream.queue(reply.text(SLOW_PIECES, chunk_delay=0.2, match=reply.answering(PROMPT)))
    send(chat_page, PROMPT)
    expect(last_reply(chat_page)).to_contain_text("part-1")
    expect(chat_page.page).to_have_url(re.compile(r"/c/[\w-]+$"))

    chat_page.page.context.set_offline(True)
    expect(chat_page.page.get_by_text(CONNECTION_LOST)).to_be_visible()
    with account.client() as client:
        wait_until_no_reply_runs(client, chat_page.page.url.rsplit("/", 1)[1])
    chat_page.page.context.set_offline(False)

    expect(chat_page.page.get_by_text(RECONNECTED)).to_be_visible(timeout=RECONNECT_TIMEOUT_MS)
    _expect_whole_and_finished(chat_page)


@pytest.mark.regression
def test_a_page_signed_in_on_the_form_reconnects_once_and_stays(page, make_user, upstream):
    account = make_user()
    sockets: list[str] = []
    page.on("websocket", lambda socket: sockets.append(socket.url))
    page.goto("/auth")
    page.get_by_label("Email").fill(account.email)
    page.get_by_label("Password", exact=True).fill(account.password)
    page.get_by_role("button", name="Sign in", exact=True).click()
    upstream.queue(reply.text("Signed in and here.", match=reply.answering("hello after sign in")))
    send(page, "hello after sign in")
    expect_reply(page, "Signed in and here.")
    expect(page).to_have_url(re.compile(r"/c/[\w-]+$"))

    _go_offline_and_back(page)
    reconnected = len(sockets)
    page.wait_for_timeout(5000)  # bounded: a reconnect loop opens several sockets a second

    assert len(sockets) == reconnected, (
        f"the page opened {len(sockets) - reconnected} more sockets after it reconnected"
    )
