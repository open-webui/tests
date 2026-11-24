"""Regression: after a failed reply the chat turned away the next message.

dbb17a572: when the provider answered with an error, the reply was left unfinished and sending
the next message showed "Oops! There was an error in the previous response." and sent nothing,
so the chat was stuck until a reload. The fix finishes the failed reply when its error arrives and
removes the block in Chat.svelte that refused to send after an errored reply without content.

Twin of unit/chat/test_stream_error_close.py.

The test runs in the main chat and in a side chat (`utils/chat_surface.py`) and asserts in both
that the error shows in the reply, as the main chat shows it. PR open-webui/open-webui#34064 names
it as a difference that a side chat shows errors as notifications.

#34950: the error box ran a provider's message spread over several lines into one paragraph. The
error here has several lines and a long unbroken token, and the box shows each line on its own and
wraps the token inside the box.

Discriminates: passes on the bbfa876af build; with the toast-and-return block restored in
Chat.svelte the second message is refused with that toast and never reaches the provider. The
line checks fail on dev c6ef4a523 (the lines read as one) and pass with the fix for #34950.
"""

from __future__ import annotations

import pytest
from playwright.sync_api import expect

from harness import upstream as reply
from utils.chat_ui import expect_reply, last_reply, send, stop_button

pytestmark = [pytest.mark.regression, pytest.mark.requires_browser, pytest.mark.requires_source]

REFUSAL = "Oops! There was an error in the previous response."
ERROR_LINES = [
    "Your request was blocked by the provider.",
    "Please rephrase your message or switch to a different model.",
    "---",
    "request_id=demo-123",
    "trace=" + "a1b2c3d4" * 40,
]


@pytest.fixture
def chat_page(chat_for, make_user):
    return chat_for(make_user())


def test_a_provider_error_is_shown_and_the_next_message_still_sends(chat_page, upstream):
    upstream.queue(reply.error(400, "\n".join(ERROR_LINES)), reply.text("back again"))
    send(chat_page, "hello?")
    expect_reply(chat_page, ERROR_LINES[0])
    expect(stop_button(chat_page)).to_be_hidden()

    error_box = last_reply(chat_page).get_by_text(ERROR_LINES[0])
    shown_lines = error_box.inner_text().splitlines()
    assert shown_lines == ERROR_LINES, f"the error box shows its lines as {shown_lines}"
    overflow = error_box.evaluate("box => box.scrollWidth - box.clientWidth")
    assert overflow <= 0, f"the long token runs {overflow}px out of the error box"

    send(chat_page, "hello again")
    expect_reply(chat_page, "back again")
    expect(chat_page.page.get_by_text(REFUSAL)).to_have_count(0)
    assert reply.conversation(upstream.chat_requests()[-1])[-1]["content"] == "hello again"
