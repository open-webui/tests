"""Regression: after a reply was stopped before any text, the chat refused every next message.

Fix 743a46bdc (PR #31892, issue #25083) leaves empty assistant messages out of what is sent to
the model. Before, a reply stopped before its first token stayed in the chat as an empty assistant
message and went back to the provider on every later turn; a provider that refuses empty assistant
messages then failed each new message in that chat. Here the provider answers only a request
without an empty assistant message and fails every other one.

Twin of integration/chat/test_empty_reply_replay.py.

`test_a_chat_answers_again_after_a_reply_stopped_before_any_text` is red on dev 62f70a844: since
de73bb830 a reply stopped with Stop never ends on the page, the Stop button stays and the next
message is not answered (open-webui/open-webui#32081).

Discriminates: passes on dev b859124f9, fails with 743a46bdc reverted (the second message gets the
provider's refusal instead of the answer).
"""

from __future__ import annotations

import pytest
from playwright.sync_api import expect

from harness import upstream as reply
from utils.chat_ui import expect_reply, send, stop_button

pytestmark = [pytest.mark.regression, pytest.mark.requires_browser, pytest.mark.requires_source]


def has_no_empty_assistant_message(body: dict) -> bool:
    return all(
        entry.get("content")
        for entry in body.get("messages", [])
        if entry.get("role") == "assistant"
    )


def answering_without_empty_replies(prompt: str):
    answering = reply.answering(prompt)
    return lambda body: answering(body) and has_no_empty_assistant_message(body)


def test_a_chat_answers_again_after_a_reply_stopped_before_any_text(page_for, make_user, upstream):
    upstream.reset("error")
    upstream.queue(
        reply.text("too late", delay=10, match=reply.answering("first question")),
        reply.text("here you go", match=answering_without_empty_replies("second question")),
    )
    page = page_for(make_user())

    send(page, "first question")
    expect(stop_button(page)).to_be_visible()
    stop_button(page).click()
    expect(stop_button(page)).to_be_hidden()
    send(page, "second question")

    expect_reply(page, "here you go")
