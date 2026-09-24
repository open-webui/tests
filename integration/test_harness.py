"""The harness itself: a scripted reply reaches the stored chat, and the request is recorded.

Every chat test builds on these two facts. When they fail, the harness or the chat entrypoint
it drives has changed, and the failures of the tests built on it say nothing about regressions.
"""

from __future__ import annotations

import pytest

from harness import upstream as reply
from harness.chat import ask

pytestmark = [pytest.mark.api, pytest.mark.requires_source]


def test_a_scripted_reply_is_stored_on_the_chat(user, upstream):
    upstream.queue(reply.text("scripted answer"))
    with user.client() as client:
        _, message = ask(client, "hello")
    assert message["content"] == "scripted answer"


def test_the_provider_records_the_prompt_it_was_sent(user, upstream):
    with user.client() as client:
        ask(client, "a distinctive prompt")
    sent = upstream.chat_requests()[-1]["messages"]
    assert any(entry.get("content") == "a distinctive prompt" for entry in sent)
