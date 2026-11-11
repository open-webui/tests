"""Journey: the token usage a provider reports is shown under its reply, live and after a reload.

When the provider ends a reply with its token counts, the reply's info button shows them in its
tooltip, with the digits grouped for the page's language since e84dba2d3 (#33896), and they are
still there when the chat is opened again. A reply the provider reported no usage for has no info
button.

`test_the_reported_usage_shows_in_the_reply_info_and_after_a_reload` failed on dev 9bbb95048 in CI
with the reply left blank and passed 3 of 3 locally: since de73bb830 the reply in a new chat
sometimes stays blank until a reload although the server saved it whole
(open-webui/open-webui#32091).

Discriminates: passes on dev 30f3f6a8f and, grouped, on a072aaf31; in a backend copy that stores
and sends no usage the first test fails.

Both tests run in the main chat and in every way a side chat starts and every state it can be in
(`utils/chat_surface.py`). In a side chat the info button showed only after a reload
(open-webui/open-webui#34016, fix PR open-webui/open-webui#34064): on dev c15f28649 the first
test is red in every side chat and passes on the PR's build. A temporary side chat is left out of
it, as the chat is gone after the reload the test makes.
"""

from __future__ import annotations

import pytest
from playwright.sync_api import Locator, expect

from harness import upstream as reply
from utils.chat_surface import EVERYWHERE, TEMPORARY_SIDE, ChatSurface
from utils.chat_ui import expect_reply, last_reply, send

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

USAGE = {"prompt_tokens": 1234, "completion_tokens": 56, "total_tokens": 1290}


def _info_button(chat: ChatSurface) -> Locator:
    whole_reply = last_reply(chat).locator("xpath=ancestor::*[starts-with(@id, 'message-')][1]")
    return whole_reply.locator("[id^='info-']")


def _usage_tooltip(chat: ChatSurface) -> Locator:
    info = _info_button(chat)
    expect(info, f"the reply in the {chat.place} shows no usage").to_have_count(1)
    info.hover()
    # the tooltip of a button the pointer crossed can still be fading out
    return chat.page.get_by_role("tooltip").filter(has_text="total_tokens")


@pytest.mark.chat_surfaces(*(place for place in EVERYWHERE if place != TEMPORARY_SIDE))
def test_the_reported_usage_shows_in_the_reply_info_and_after_a_reload(
    chat_for, make_user, upstream
):
    """In a side chat: open-webui/open-webui#34016."""
    chat = chat_for(make_user(), locale="en-US")
    upstream.queue(reply.text("Counted.", usage=USAGE, match=reply.answering("count my tokens")))
    send(chat, "count my tokens")
    expect_reply(chat, "Counted.")

    tooltip = _usage_tooltip(chat)
    expect(tooltip).to_contain_text("prompt_tokens: 1,234")
    expect(tooltip).to_contain_text("completion_tokens: 56")

    chat = chat.reload()
    expect_reply(chat, "Counted.")
    tooltip = _usage_tooltip(chat)
    expect(tooltip).to_contain_text("total_tokens: 1,290")


def test_a_reply_without_usage_has_no_info_button(chat_for, make_user, upstream):
    chat = chat_for(make_user())
    upstream.queue(reply.text("Not counted.", match=reply.answering("no counting")))
    send(chat, "no counting")
    expect_reply(chat, "Not counted.")

    expect(_info_button(chat)).to_have_count(0)
