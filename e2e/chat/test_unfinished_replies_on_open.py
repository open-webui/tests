"""Regression: older unfinished replies kept a loading cursor forever after opening a chat.

Issue open-webui/open-webui#14806, fix 394295ec7 (PR open-webui/open-webui#31895). Opening a
chat only finished the newest reply that never completed, so an older one (from a cancelled
generation, or a turn an earlier version left unfinished) kept spinning and never offered its
Copy and Regenerate buttons. With no generation running, every unfinished reply now shows as
finished on load, except one waiting for the user to answer a tool's question.

Discriminates: passes on the dev b859124f9 build, fails on that build with 394295ec7 reverted
(the older unfinished reply has no Copy or Regenerate buttons).

Every test runs in the main chat and in a side chat (`utils/chat_surface.py`), a stored chat
dragged into a tab beside a new chat. On dev c15f28649 a chat in a tab drew its replies without
the reply rows of the main chat (no filed issue), so the side variants are red there; they pass
on dev e791cbdde, with PR open-webui/open-webui#34064.
"""

from __future__ import annotations

import json

import pytest
from playwright.sync_api import Locator, expect

from harness.chat_history import seed_chat
from utils.chat_surface import ChatSurface
from utils.chat_ui import conversation

pytestmark = [pytest.mark.regression, pytest.mark.requires_browser, pytest.mark.requires_source]

OLDER = "The first answer was cut off mid"
MIDDLE = "A complete second answer."
NEWEST = "The last answer was cut off mid"

# the whole message row: its buttons sit beside the text, not inside the element holding it
MESSAGE = "div.flex.w-full[id^='message-']"

ASK_USER_CALL = {
    "type": "function_call",
    "id": "fc_plan",
    "call_id": "call_plan",
    "name": "ask_user",
    "status": "pending",
    "arguments": json.dumps(
        {
            "questions": [
                {
                    "id": "plan",
                    "header": "Plan",
                    "question": "Which plan fits you best?",
                    "options": [
                        {"label": "Basic", "description": "The cheapest one"},
                        {"label": "Pro", "description": "The fastest one"},
                    ],
                }
            ]
        }
    ),
}


def open_chat(chat_for, make_user, messages: list[dict]) -> ChatSurface:
    owner = make_user()
    with owner.client() as client:
        chat_id, _ = seed_chat(client, messages)
    chat = chat_for(owner, chat_id=chat_id)
    return chat


def reply_with(chat: ChatSurface, text: str) -> Locator:
    return conversation(chat).locator(MESSAGE, has_text=text)


def expect_finished(reply: Locator) -> None:
    expect(reply).to_be_visible()
    reply.hover()
    expect(reply.get_by_role("button", name="Copy")).to_have_count(1)
    expect(reply.get_by_role("button", name="Regenerate").first).to_be_attached()


def test_an_older_unfinished_reply_is_shown_as_finished(chat_for, make_user):
    chat = open_chat(
        chat_for,
        make_user,
        [
            {"role": "user", "content": "first question"},
            {"role": "assistant", "content": OLDER, "done": False},
            {"role": "user", "content": "second question"},
            {"role": "assistant", "content": MIDDLE},
        ],
    )

    expect(reply_with(chat, MIDDLE)).to_be_visible()
    expect_finished(reply_with(chat, OLDER))


def test_every_unfinished_reply_in_a_longer_chat_is_shown_as_finished(chat_for, make_user):
    chat = open_chat(
        chat_for,
        make_user,
        [
            {"role": "user", "content": "first question"},
            {"role": "assistant", "content": OLDER, "done": False},
            {"role": "user", "content": "second question"},
            {"role": "assistant", "content": MIDDLE},
            {"role": "user", "content": "third question"},
            {"role": "assistant", "content": NEWEST, "done": False},
        ],
    )

    expect_finished(reply_with(chat, NEWEST))
    expect_finished(reply_with(chat, MIDDLE))
    expect_finished(reply_with(chat, OLDER))


def test_a_reply_waiting_for_the_users_answer_stays_pending(chat_for, make_user):
    chat = open_chat(
        chat_for,
        make_user,
        [
            {"role": "user", "content": "first question"},
            {"role": "assistant", "content": OLDER, "done": False},
            {"role": "user", "content": "help me choose a plan"},
            {
                "role": "assistant",
                "content": "",
                "done": False,
                "output": [ASK_USER_CALL],
            },
        ],
    )

    expect(chat.panel.get_by_text("Which plan fits you best?")).to_be_visible()
    expect_finished(reply_with(chat, OLDER))
    waiting = conversation(chat).locator(MESSAGE).last
    waiting.hover()
    expect(waiting.get_by_role("button", name="Regenerate")).to_have_count(0)
