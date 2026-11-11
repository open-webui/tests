"""A refused `ask_user` call left the user with a dead chat and no reply.

Fix commit `be958d7b0` (PR #29252, issue #29077). When the model asked two `ask_user`
questions in one turn, Open WebUI refused them but stopped the turn there: the model was never
told and the page showed no answer. The refusal now goes back to the model, whose next reply
reaches the page. A single well-formed question still opens the question card.

The card follows the Chat Direction setting (open-webui/open-webui#32784, fix aee4cbf46): with the
default Auto a question in Hebrew reads right to left and an English one left to right, and the
Other box reads right to left once Hebrew is typed into it.

Twin of unit/chat/test_ask_user_refusal.py, next to integration/chat/test_ask_user_refusal.py.

Every test runs in the main chat and in a side chat (`utils/chat_surface.py`).

Discriminates: with be958d7b0 reverted the refused turn never shows a reply; the question card
shows on both. On the 0.12.4 pre-fix build (aee4cbf46 reverted among other fixes) the Hebrew
card test goes red (its question reads left to right); the English card stays left to right on
both.
"""

from __future__ import annotations

import json

import pytest
from playwright.sync_api import expect

from harness import upstream as reply
from harness.upstream import Reply
from utils.chat_ui import expect_reply, send

pytestmark = [pytest.mark.regression, pytest.mark.requires_browser, pytest.mark.requires_source]

PLAN_QUESTION = {
    "id": "plan",
    "header": "Plan",
    "question": "Which plan fits you best?",
    "options": [
        {"label": "Basic", "description": "The cheapest one"},
        {"label": "Pro", "description": "The fastest one"},
    ],
}


HEBREW_QUESTION = {
    "id": "plan",
    "header": "תוכנית",
    "question": "איזו תוכנית מתאימה לך ביותר?",
    "options": [
        {"label": "בסיסית", "description": "הזולה ביותר"},
        {"label": "מקצועית", "description": "המהירה ביותר"},
    ],
    "allow_other": True,
}
HEBREW_ANSWER = "אחרת לגמרי"


def ask_user_call(call_id: str, question: dict = PLAN_QUESTION) -> dict:
    arguments = json.dumps({"questions": [question]})
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": "ask_user", "arguments": arguments},
    }


@pytest.fixture
def chat_page(chat_for, make_user):
    return chat_for(make_user())


def test_a_refused_double_question_still_gets_a_reply(chat_page, upstream):
    two_questions = Reply(tool_calls=[ask_user_call("call_a"), ask_user_call("call_b")])
    upstream.queue(two_questions, reply.text("Let me ask one question at a time."))

    send(chat_page, "help me choose a plan")

    expect_reply(chat_page, "Let me ask one question at a time.")


def test_a_single_question_opens_the_question_card(chat_page, upstream):
    upstream.queue(Reply(tool_calls=[ask_user_call("call_plan")]))

    send(chat_page, "help me choose a plan")

    question = chat_page.panel.get_by_text("Which plan fits you best?")
    expect(question).to_be_visible(timeout=30_000)
    expect(chat_page.panel.get_by_text("The cheapest one")).to_be_visible()
    expect(chat_page.panel.get_by_text("The fastest one")).to_be_visible()
    expect(question).to_have_css("direction", "ltr")


def test_a_hebrew_question_card_reads_right_to_left(chat_page, upstream):
    upstream.queue(Reply(tool_calls=[ask_user_call("call_plan", HEBREW_QUESTION)]))

    send(chat_page, "help me choose a plan in hebrew")

    question = chat_page.panel.get_by_text(HEBREW_QUESTION["question"])
    expect(question).to_be_visible(timeout=30_000)
    expect(question, "the Hebrew question reads left to right").to_have_css("direction", "rtl")
    other = chat_page.panel.get_by_placeholder("Type your answer")
    other.fill(HEBREW_ANSWER)
    expect(other, "Hebrew typed as Other reads left to right").to_have_css("direction", "rtl")
