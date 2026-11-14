"""Filters as the chat page shows their work: an outlet edit.

Unpinned, no issue filed: an `outlet` filter that edits the reply edits both its `content` and
the text of its `output` message items, since the chat page renders a reply from its `output`.
The edit shows live, is stored and still shows after a reload.

Twin of integration/chat/test_filter_context_and_handlers.py (its mid-request cases). The
mid-request case of fix `7d694570a` (a filter switched off mid-reply still runs its stream stage)
was dropped here: the integration twin pins it and the page adds nothing to it.

Discriminates: passes on dev ef67cc3fa and fails in a backend copy that drops the outlet's
`output` edit (the page shows the unedited reply).

The test runs in the main chat and in a side chat. In a side chat it is red on dev c15f28649 and
green on the side chat rework build (fix PR open-webui/open-webui#34064), with no issue filed: the
side chat keeps showing the unedited reply until a reload.
"""

from __future__ import annotations

import re
import time

import pytest
from playwright.sync_api import expect

from harness import upstream as reply
from harness.plugins import installed_function
from utils.chat_ui import expect_reply, send

pytestmark = [pytest.mark.regression, pytest.mark.requires_browser, pytest.mark.requires_source]

HIGHLIGHTING_OUTLET = """
class Filter:
    async def outlet(self, body):
        for message in body["messages"]:
            if message["role"] != "assistant":
                continue
            message["content"] += " [highlighted]"
            for item in message.get("output", []):
                if item["type"] == "message":
                    for part in item["content"]:
                        if part["type"] == "output_text":
                            part["text"] += " [highlighted]"
        return body
"""


def stored_reply(person, chat_id: str, expected: str) -> dict:
    """The stored reply once the outlet stage has rewritten it, or as it stands at the deadline."""
    deadline = time.monotonic() + 10
    with person.client() as client:
        while True:
            history = client.get(f"/api/v1/chats/{chat_id}").json()["chat"]["history"]
            message = history["messages"][history["currentId"]]
            if message["content"] == expected or time.monotonic() > deadline:
                return message
            time.sleep(0.2)


def output_text(message: dict) -> str:
    return "".join(
        part["text"]
        for item in message.get("output", [])
        if item["type"] == "message"
        for part in item["content"]
        if part["type"] == "output_text"
    )


def test_an_outlet_edit_to_the_reply_shows_on_the_page(chat_for, admin, make_user, upstream):
    """In a side chat: red on dev, no issue filed (fix PR open-webui/open-webui#34064)."""
    prompt = "say something plain"
    upstream.queue(reply.text("plain answer", match=reply.answering(prompt)))
    person = make_user()
    chat = chat_for(person)
    with installed_function(admin, HIGHLIGHTING_OUTLET, is_global=True):
        send(chat, prompt)
        expect_reply(chat, "plain answer [highlighted]")
        expect(chat.page).to_have_url(re.compile(r"/c/[0-9a-f-]+$"))
        chat_id = chat.page.url.rsplit("/", 1)[-1]
        stored = stored_reply(person, chat_id, "plain answer [highlighted]")
        assert stored["content"] == "plain answer [highlighted]"
        assert output_text(stored) == "plain answer [highlighted]"

        chat = chat.reload()
        expect_reply(chat, "plain answer [highlighted]")
