"""Journey: a model picked with @ in the composer answers in place of the chat's model.

Typing @ and part of a model's name in the message box lists matching models; the one picked shows
above the box and answers the next messages, with its own system prompt, under its own name, until
the pick is cleared with Escape. The chat's own model is left as it was and answers again after
that.

Discriminates: passes on dev 30f3f6a8f; in a frontend build that sends every message to the chat's
selected model whatever was picked with @ the first test fails, and in one where Escape keeps the
pick both fail.
"""

from __future__ import annotations

import uuid

import pytest
from playwright.sync_api import Page, expect

from harness import upstream as reply
from harness.upstream import MOCK_MODEL_ID
from utils.chat_ui import chat_input, conversation, expect_reply

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

HARBOUR_PROMPT = "You are the harbour master and answer in nautical terms."


@pytest.fixture
def harbour_master(make_user):
    """A fresh admin and a model of theirs with its own system prompt; yields (account, name)."""
    account = make_user(role="admin")
    suffix = uuid.uuid4().hex[:6]
    model_id, name = f"harbour-{suffix}", f"Harbourmaster {suffix}"
    form = {
        "id": model_id,
        "name": name,
        "base_model_id": MOCK_MODEL_ID,
        "meta": {},
        "params": {"system": HARBOUR_PROMPT},
    }
    with account.client() as client:
        created = client.post("/api/v1/models/create", json=form)
        assert created.status_code == 200, created.text
        yield account, name
        client.post("/api/v1/models/model/delete", json={"id": model_id})


def _pick(page: Page, name: str) -> None:
    expect(chat_input(page)).to_be_visible()
    chat_input(page).click()
    page.keyboard.type(f"@{name.split()[0]}")
    page.get_by_role("button").filter(has_text=name).first.click()
    # the pick shows above the message box
    expect(page.locator("form").get_by_text(name, exact=True)).to_be_visible()


def _ask(page: Page, upstream, answer: str) -> dict:
    question = f"when is the tide? {uuid.uuid4().hex[:6]}"
    upstream.queue(reply.text(answer, match=reply.answering(question)))
    chat_input(page).click()
    page.keyboard.type(question)
    page.keyboard.press("Enter")
    expect_reply(page, answer)
    return next(filter(reply.answering(question), upstream.chat_requests()))


def _system_text(request: dict) -> str:
    return "\n".join(
        str(entry["content"]) for entry in request["messages"] if entry["role"] == "system"
    )


def test_the_picked_model_answers_until_the_pick_is_cleared(page_for, harbour_master, upstream):
    account, name = harbour_master
    page = page_for(account)
    _pick(page, name)

    first = _ask(page, upstream, "High water at noon, skipper.")
    assert HARBOUR_PROMPT in _system_text(first)
    expect(conversation(page).get_by_text(name, exact=True).last).to_be_visible()
    second = _ask(page, upstream, "Low water at six, skipper.")
    assert HARBOUR_PROMPT in _system_text(second)

    chat_input(page).click()
    page.keyboard.press("Escape")
    expect(page.locator("form").get_by_text(name, exact=True)).to_have_count(0)
    third = _ask(page, upstream, "Around six in the evening.")
    assert HARBOUR_PROMPT not in _system_text(third)
    assert third["model"] == MOCK_MODEL_ID


def test_a_pick_cleared_before_sending_leaves_the_chats_model_answering(
    page_for, harbour_master, upstream
):
    account, name = harbour_master
    page = page_for(account)
    _pick(page, name)
    page.keyboard.press("Escape")
    expect(page.locator("form").get_by_text(name, exact=True)).to_have_count(0)

    request = _ask(page, upstream, "Around noon.")
    assert HARBOUR_PROMPT not in _system_text(request)
