"""Journey: the model provider goes down in the middle of a reply.

The scripted provider streams the first three pieces of a reply and then drops the connection
without finishing the stream, the way a provider that crashes or loses its network does. The
page shows what arrived with the error under it and the reply as finished, and Regenerate asks
again and replaces the broken reply with a whole one.

Discriminates: passes on dev ebc6add67; the error test fails on a build whose reply hides its
error and the regenerate test on a build whose Regenerate ignores a failed reply.
"""

from __future__ import annotations

import re

import pytest
from playwright.sync_api import expect

from harness import upstream as reply
from harness.inflight import SLOW_PIECES
from utils.chat_ui import (
    expect_reply,
    last_reply,
    regenerate_buttons,
    replies,
    send,
    stop_button,
    typing_cursor,
)

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

PROMPT = "tell me the story of the lighthouse"
RECEIVED = "part-0 part-1 part-2"
NEVER_SENT = "part-3"
DROP_ERROR = re.compile("payload is not completed", re.IGNORECASE)


@pytest.fixture
def dropped_reply(page_for, make_user, upstream):
    """A page whose reply lost its provider after three pieces, with a whole answer queued next."""
    upstream.queue(
        reply.text(SLOW_PIECES, chunk_delay=0.2, hang_up_after=3, match=reply.answering(PROMPT)),
        reply.text("The lighthouse keeper slept well.", match=reply.answering(PROMPT)),
    )
    page = page_for(make_user())
    send(page, PROMPT)
    expect(last_reply(page)).to_contain_text("part-2")
    expect(stop_button(page)).to_be_hidden()
    return page


def test_the_reply_keeps_what_arrived_and_shows_the_error(dropped_reply):
    shown = last_reply(dropped_reply)

    expect(shown).to_contain_text(DROP_ERROR)
    expect(shown).to_contain_text(RECEIVED)
    expect(shown).not_to_contain_text(NEVER_SENT)
    expect(typing_cursor(shown)).to_have_count(0)


def test_regenerate_replaces_the_dropped_reply_with_a_whole_one(dropped_reply):
    last_reply(dropped_reply).hover()
    regenerate_buttons(dropped_reply).last.click()
    dropped_reply.get_by_text("Try Again").click()

    expect_reply(dropped_reply, "The lighthouse keeper slept well.")
    expect(last_reply(dropped_reply)).not_to_contain_text(DROP_ERROR)
    expect(replies(dropped_reply)).to_have_count(1)
