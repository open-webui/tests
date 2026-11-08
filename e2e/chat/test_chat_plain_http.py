"""Regression: the chat page loads and answers when Open WebUI is opened over plain HTTP by address.

Opened at an address such as http://192.168.1.20:8080, the page is not a secure context and the
browser offers no `crypto.randomUUID`. Since d4e8493cc (dev, after 0.13.0) the chat page makes the
id of its saved tab layout with it, so over plain HTTP by address the new chat page stops at a
spinner with "crypto.randomUUID is not a function" and no chat can be started or opened. The
browser here reaches the instance by a host name that is neither `localhost` nor a loopback
address, as a phone or another computer on the network does.

Red on 0.13.1, dev e8c0ad85e (the chat input never shows). Discriminates: passes on a build of
e8c0ad85e whose saved tab layout makes its id with the app's uuid helper.
"""

from __future__ import annotations

import pytest
from playwright.sync_api import expect

from harness import upstream as reply
from utils.chat_ui import REPLY_TIMEOUT_MS, chat_input, expect_reply, send

pytestmark = [pytest.mark.regression, pytest.mark.requires_browser, pytest.mark.requires_source]

QUESTION = "Which bridge crosses the river first?"
ANSWER = "The old stone bridge does."


def test_a_new_chat_is_answered_over_plain_http(lan_page_for, make_user, upstream):
    page = lan_page_for(make_user())
    errors: list[str] = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.reload()

    # the same cold load `send` allows for
    expect(chat_input(page)).to_be_visible(timeout=REPLY_TIMEOUT_MS)
    upstream.queue(reply.text(ANSWER, match=reply.answering(QUESTION)))
    send(page, QUESTION)
    expect_reply(page, ANSWER)
    assert not errors, f"the chat page failed over plain HTTP: {errors}"
