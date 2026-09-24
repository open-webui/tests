"""Regression: after a failed reply the chat turned away the next message.

dbb17a572: when the provider answered with an error, the reply was left unfinished and sending
the next message showed "Oops! There was an error in the previous response." and sent nothing,
so the chat was stuck until a reload. The fix finishes the failed reply when its error arrives and
removes the block in Chat.svelte that refused to send after an errored reply without content.

Twin of unit/chat/test_stream_error_close.py.

Discriminates: passes on the bbfa876af build; with the toast-and-return block restored in
Chat.svelte the second message is refused with that toast and never reaches the provider.
"""

from __future__ import annotations

import pytest
from playwright.sync_api import expect

from harness import upstream as reply
from utils.chat_ui import expect_reply, send, stop_button

pytestmark = [pytest.mark.regression, pytest.mark.requires_browser, pytest.mark.requires_source]

REFUSAL = "Oops! There was an error in the previous response."


@pytest.fixture
def chat_page(page_for, make_user):
    return page_for(make_user())


def test_a_provider_error_is_shown_and_the_next_message_still_sends(chat_page, upstream):
    upstream.queue(reply.error(500, "the provider is down"), reply.text("back again"))
    send(chat_page, "hello?")
    expect_reply(chat_page, "the provider is down")
    expect(stop_button(chat_page)).to_be_hidden()

    send(chat_page, "hello again")
    expect_reply(chat_page, "back again")
    expect(chat_page.get_by_text(REFUSAL)).to_have_count(0)
    assert upstream.chat_requests()[-1]["messages"][-1]["content"] == "hello again"
