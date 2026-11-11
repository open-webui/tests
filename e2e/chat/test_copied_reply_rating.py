"""A reply saved as a copy carried the original's rating, open-webui/open-webui#30961.

Fix commit `693f666c6` (PR open-webui/open-webui#30962). Editing a reply and choosing Save As
Copy spread the original message into the copy, its `annotation` and `feedbackId` included. The
copy showed the original's thumbs, and rating it updated the original's feedback row, so the
original's rating was overwritten. The copy now starts unrated.

Discriminates: passes on the efe63bd34 build, fails on it with `693f666c6` reverted (the copy
holds the original's feedback id and rating it rewrites the original's feedback).

Both tests run in the main chat and in every way a side chat starts and every state it can be in
(`utils/chat_surface.py`) except a temporary one, which offers no rating in the main chat either.
A side chat reply had neither Edit nor the thumbs (open-webui/open-webui#34014, fix PR
open-webui/open-webui#34064): on dev c15f28649 every side chat variant is red.
"""

from __future__ import annotations

import json
import time

import pytest

from harness import upstream as reply
from utils.chat_surface import EVERYWHERE, TEMPORARY_SIDE, ChatSurface
from utils.chat_ui import conversation, expect_reply, last_reply, send

pytestmark = [pytest.mark.regression, pytest.mark.requires_browser, pytest.mark.requires_source]

QUESTION = "suggest a name for a cat"
ANSWER = "Call it Pepper."
COPY = "Call it Biscuit."


def is_feedback_write(response) -> bool:
    return response.request.method == "POST" and "/evaluations/feedback" in response.url


def rate(chat: ChatSurface, verdict: str):
    last_reply(chat).hover()
    with chat.page.expect_response(is_feedback_write) as written:
        conversation(chat).get_by_role("button", name=verdict).last.click()
    return written.value


def save_as_copy(chat: ChatSurface, text: str) -> None:
    last_reply(chat).hover()
    conversation(chat).get_by_role("button", name="Edit").last.click()
    editor = conversation(chat).locator("textarea").last
    editor.fill(text)
    conversation(chat).get_by_role("button", name="Save As Copy").click()
    expect_reply(chat, text)


def reads(message: dict, text: str) -> bool:
    # a reply may be stored as `output` items with an empty `content`
    return message.get("content") == text or text in json.dumps(message.get("output") or [])


def stored_message(client, chat_id: str, content: str) -> dict:
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        messages = client.get(f"/api/v1/chats/{chat_id}").json()["chat"]["history"]["messages"]
        found = [message for message in messages.values() if reads(message, content)]
        if found:
            return found[0]
        time.sleep(0.2)
    raise AssertionError(f"no stored message reads {content!r}")


def feedback_ratings(client) -> dict[str, int]:
    listed = client.get("/api/v1/evaluations/feedbacks/user").json()["items"]
    return {item["meta"]["message_id"]: item["data"]["rating"] for item in listed}


@pytest.fixture
def rated_reply(make_user, chat_for, upstream):
    """A chat whose reply the account rated thumbs up."""
    account = make_user()
    chat = chat_for(account)
    upstream.queue(reply.text(ANSWER, match=reply.answering(QUESTION)))
    send(chat, QUESTION)
    expect_reply(chat, ANSWER)
    created = rate(chat, "Good Response")
    assert created.url.endswith("/evaluations/feedback"), created.url
    return account, chat


# a temporary chat offers no rating, in the main chat either; the model count cannot matter, as
# Save As Copy and the thumbs act on one reply
EVERY_RATED_PLACE = tuple(place for place in EVERYWHERE if place != TEMPORARY_SIDE)


@pytest.mark.chat_surfaces(*EVERY_RATED_PLACE)
def test_a_copy_of_a_rated_reply_starts_unrated(rated_reply):
    """In a side chat: open-webui/open-webui#34014."""
    account, chat = rated_reply
    save_as_copy(chat, COPY)
    # the address follows the selected tab, which is the chat's own
    chat_id = chat.page.url.rsplit("/", 1)[-1]

    with account.client() as client:
        copy = stored_message(client, chat_id, COPY)
        original = stored_message(client, chat_id, ANSWER)

    assert original.get("feedbackId")
    assert not copy.get("feedbackId"), "the copy holds the original's feedback id"
    assert not (copy.get("annotation") or {}).get("rating"), "the copy holds the original's rating"


@pytest.mark.chat_surfaces(*EVERY_RATED_PLACE)
def test_rating_the_copy_leaves_the_originals_feedback_alone(rated_reply):
    """In a side chat: open-webui/open-webui#34014."""
    account, chat = rated_reply
    save_as_copy(chat, COPY)
    # the address follows the selected tab, which is the chat's own
    chat_id = chat.page.url.rsplit("/", 1)[-1]

    written = rate(chat, "Bad Response")

    assert written.url.endswith("/evaluations/feedback"), "rating the copy rewrote a feedback row"
    with account.client() as client:
        original = stored_message(client, chat_id, ANSWER)
        copy = stored_message(client, chat_id, COPY)
        ratings = feedback_ratings(client)
    assert ratings == {original["id"]: 1, copy["id"]: -1}
