"""Filters as the chat page shows their work: one switched off mid-reply, and an outlet edit.

1. Fix commit `7d694570a` in `utils/filter.py`: the active filters are read once per request and
   kept on it, where every filter stage of a chat turn used to read them from the database
   again. So a filter an admin switches off mid-reply still runs its `stream` stage to the end
   of that reply. Here a second filter's inlet switches it off, which makes the moment exact.
2. Unpinned, no issue filed: an `outlet` filter that edits the reply edits both its `content`
   and the text of its `output` message items, since the chat page renders a reply from its
   `output`. The edit shows live, is stored and still shows after a reload.

Twin of integration/chat/test_filter_context_and_handlers.py (its mid-request cases).

Discriminates: both pass on dev ef67cc3fa. The first fails with 7d694570a reverted in a backend
copy (the reply arrives without the stream mark); the second fails in a backend copy that drops
the outlet's `output` edit (the page shows the unedited reply).

Both tests run in the main chat and in a side chat. In a side chat the outlet test is red on dev
c15f28649 and green on the side chat rework build (fix PR open-webui/open-webui#34064), with no
issue filed: the side chat keeps showing the unedited reply until a reload.
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

SWITCHING_FILTER = """
import httpx


class Filter:
    async def inlet(self, body):
        async with httpx.AsyncClient(base_url={base_url!r}) as client:
            response = await client.post(
                "/api/v1/functions/id/{target}/toggle",
                headers={{"Authorization": "Bearer {token}"}},
            )
            response.raise_for_status()
        return body
"""

LATE_STREAM_FILTER = """
class Filter:
    def stream(self, event):
        for choice in event.get("choices", []):
            if choice.get("delta", {}).get("content"):
                choice["delta"]["content"] += " [late stream]"
        return event
"""

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


def test_the_reply_keeps_the_filter_switched_off_during_it(
    chat_for, instance, admin, make_user, upstream
):
    prompt = "tell me about filters"
    upstream.queue(reply.text("filtered reply", match=reply.answering(prompt)))
    chat = chat_for(make_user())
    with installed_function(admin, LATE_STREAM_FILTER, is_global=True) as late_filter:
        switching = SWITCHING_FILTER.format(
            base_url=instance.base_url, target=late_filter, token=admin.token
        )
        with installed_function(admin, switching, is_global=True):
            send(chat, prompt)
            expect_reply(chat, "filtered reply [late stream]")


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
