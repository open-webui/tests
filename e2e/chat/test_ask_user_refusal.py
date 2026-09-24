"""A refused `ask_user` call left the user with a dead chat and no reply.

Fix commit `be958d7b0` (PR #29252, issue #29077). When the model asked two `ask_user`
questions in one turn, Open WebUI refused them but stopped the turn there: the model was never
told and the page showed no answer. The refusal now goes back to the model, whose next reply
reaches the page. A single well-formed question still opens the question card.

Twin of unit/chat/test_ask_user_refusal.py, next to integration/chat/test_ask_user_refusal.py.

Discriminates: with be958d7b0 reverted the refused turn never shows a reply; the question card
shows on both.
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


def ask_user_call(call_id: str) -> dict:
    arguments = json.dumps({"questions": [PLAN_QUESTION]})
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": "ask_user", "arguments": arguments},
    }


@pytest.fixture
def chat_page(page_for, make_user):
    page = page_for(make_user())
    page.goto("/")
    return page


def test_a_refused_double_question_still_gets_a_reply(chat_page, upstream):
    two_questions = Reply(tool_calls=[ask_user_call("call_a"), ask_user_call("call_b")])
    upstream.queue(two_questions, reply.text("Let me ask one question at a time."))

    send(chat_page, "help me choose a plan")

    expect_reply(chat_page, "Let me ask one question at a time.")


def test_a_single_question_opens_the_question_card(chat_page, upstream):
    upstream.queue(Reply(tool_calls=[ask_user_call("call_plan")]))

    send(chat_page, "help me choose a plan")

    expect(chat_page.get_by_text("Which plan fits you best?")).to_be_visible(timeout=30_000)
    expect(chat_page.get_by_text("The cheapest one")).to_be_visible()
    expect(chat_page.get_by_text("The fastest one")).to_be_visible()
