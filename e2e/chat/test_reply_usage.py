"""Journey: the token usage a provider reports is shown under its reply, live and after a reload.

When the provider ends a reply with its token counts, the reply's info button shows them in its
tooltip, and they are still there when the chat is opened again. A reply the provider reported
no usage for has no info button.

Discriminates: passes on dev 30f3f6a8f; in a backend copy that stores and sends no usage the
first test fails.
"""

from __future__ import annotations

import pytest
from playwright.sync_api import Locator, Page, expect

from harness import upstream as reply
from utils.chat_ui import expect_reply, last_reply, send

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

USAGE = {"prompt_tokens": 1234, "completion_tokens": 56, "total_tokens": 1290}


def _info_button(page: Page) -> Locator:
    whole_reply = last_reply(page).locator("xpath=ancestor::*[starts-with(@id, 'message-')][1]")
    return whole_reply.locator("[id^='info-']")


def _usage_tooltip(page: Page) -> Locator:
    info = _info_button(page)
    expect(info, "the reply shows no usage").to_have_count(1)
    info.hover()
    return page.get_by_role("tooltip")


def test_the_reported_usage_shows_in_the_reply_info_and_after_a_reload(
    page_for, make_user, upstream
):
    page = page_for(make_user())
    upstream.queue(reply.text("Counted.", usage=USAGE, match=reply.answering("count my tokens")))
    send(page, "count my tokens")
    expect_reply(page, "Counted.")

    tooltip = _usage_tooltip(page)
    expect(tooltip).to_contain_text("prompt_tokens: 1234")
    expect(tooltip).to_contain_text("completion_tokens: 56")

    page.reload()
    expect_reply(page, "Counted.")
    tooltip = _usage_tooltip(page)
    expect(tooltip).to_contain_text("total_tokens: 1290")


def test_a_reply_without_usage_has_no_info_button(page_for, make_user, upstream):
    page = page_for(make_user())
    upstream.queue(reply.text("Not counted.", match=reply.answering("no counting")))
    send(page, "no counting")
    expect_reply(page, "Not counted.")

    expect(_info_button(page)).to_have_count(0)
