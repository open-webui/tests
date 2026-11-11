"""Journey: a model picked with @ in the composer answers in place of the chat's model.

Typing @ and part of a model's name in the message box lists matching models; the one picked shows
above the box and answers the next message, with its own system prompt, under its own name. Sending
makes it the chat's model (dev 3aaecdd0e): the selector shows it and it answers the messages after,
until another model is chosen in the selector. A pick cleared with Escape before sending leaves the
chat's own model answering. A model chosen in the selector of an open chat is saved with the chat
at once (dev 3aaecdd0e), so a reload before the next message still shows it and it answers.

Discriminates: passes on dev 34d360134; in a frontend build that sends every message to the chat's
selected model whatever was picked with @ both tests' first replies come from the wrong model, in
one where the pick stays a one-off above the box (dev cac14f985) the first test fails, and in one
where Escape keeps the pick the second fails. On dev c5981aaea, in a build that saves a chat's
model only with a message, both reload tests fail (the chat keeps storing the old model).

The reload test for a chat started on the home page failed from 193cf225f: such a chat never took
in its stored state, so the selector's pick was not saved (open-webui/open-webui#33376, fixed on
dev by open-webui/open-webui#33377).

Every test runs in the main chat and in a side chat. In a side chat three are red on dev
c15f28649 and green on the side chat rework build (fix PR open-webui/open-webui#34064), with no
issue filed: the model picked with @ stays a one-off above the box after sending (no chat model
change), and a model picked in the selector of an open side chat is not saved with the chat, so
the chat still stores the old model and a reload shows it again.
"""

from __future__ import annotations

import re
import uuid

import pytest
from playwright.sync_api import expect

from harness import upstream as reply
from harness.backends import wait_until
from harness.chat_history import seed_chat
from harness.upstream import MOCK_MODEL_ID
from utils.chat_surface import ChatSurface
from utils.chat_ui import chat_input, conversation, expect_reply, send
from utils.model_selector import select_model

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

HARBOUR_PROMPT = "You are the harbour master and answer in nautical terms."


@pytest.fixture
def harbour_master(make_user):
    """A fresh admin and their model with its own system prompt; yields (account, name, id)."""
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
        yield account, name, model_id
        client.post("/api/v1/models/model/delete", json={"id": model_id})


def _pick(chat: ChatSurface, name: str) -> None:
    expect(chat_input(chat)).to_be_visible()
    chat_input(chat).click()
    chat.page.keyboard.type(f"@{name.split()[0]}")
    chat.page.get_by_role("button").filter(has_text=name).first.click()
    # the pick shows above the message box
    expect(chat.panel.locator("form").get_by_text(name, exact=True)).to_be_visible()


def _ask(chat: ChatSurface, upstream, answer: str) -> dict:
    question = f"when is the tide? {uuid.uuid4().hex[:6]}"
    upstream.queue(reply.text(answer, match=reply.answering(question)))
    chat_input(chat).click()
    chat.page.keyboard.type(question)
    chat.page.keyboard.press("Enter")
    expect_reply(chat, answer)
    return next(filter(reply.answering(question), upstream.chat_requests()))


def _system_text(request: dict) -> str:
    return "\n".join(
        str(entry["content"]) for entry in request["messages"] if entry["role"] == "system"
    )


def test_the_picked_model_becomes_the_chats_model_once_sent(chat_for, harbour_master, upstream):
    """In a side chat: red on dev, no issue filed (fix PR open-webui/open-webui#34064)."""
    account, name, _ = harbour_master
    chat = chat_for(account)
    _pick(chat, name)

    first = _ask(chat, upstream, "High water at noon, skipper.")
    assert HARBOUR_PROMPT in _system_text(first)
    expect(conversation(chat).get_by_text(name, exact=True).last).to_be_visible()
    expect(chat.panel.get_by_role("button", name="Remove Selected Model")).to_have_count(0)
    expect(chat.panel.get_by_role("button", name=f"Selected model: {name}")).to_be_visible()
    second = _ask(chat, upstream, "Low water at six, skipper.")
    assert HARBOUR_PROMPT in _system_text(second)

    select_model(chat, MOCK_MODEL_ID)
    expect(
        chat.panel.get_by_role("button", name=f"Selected model: {MOCK_MODEL_ID}")
    ).to_be_visible()
    third = _ask(chat, upstream, "Around six in the evening.")
    assert HARBOUR_PROMPT not in _system_text(third)
    assert third["model"] == MOCK_MODEL_ID


def test_a_pick_cleared_before_sending_leaves_the_chats_model_answering(
    chat_for, harbour_master, upstream
):
    account, name, _ = harbour_master
    chat = chat_for(account)
    _pick(chat, name)
    chat.page.keyboard.press("Escape")
    expect(chat.panel.locator("form").get_by_text(name, exact=True)).to_have_count(0)

    request = _ask(chat, upstream, "Around noon.")
    assert HARBOUR_PROMPT not in _system_text(request)


def _stored_models(account, chat_id: str) -> list[str]:
    with account.client() as client:
        stored = client.get(f"/api/v1/chats/{chat_id}")
    assert stored.status_code == 200, stored.text
    return stored.json()["chat"].get("models")


def _pick_in_selector_and_reload(
    chat: ChatSurface, account, chat_id: str, name: str, model_id: str
) -> ChatSurface:
    expect(
        chat.panel.get_by_role("button", name=f"Selected model: {MOCK_MODEL_ID}")
    ).to_be_visible()
    select_model(chat, name)
    expect(chat.panel.get_by_role("button", name=f"Selected model: {name}")).to_be_visible()
    assert wait_until(lambda: _stored_models(account, chat_id) == [model_id], 10), (
        f"the chat still stores {_stored_models(account, chat_id)} after picking {model_id}"
    )
    chat = chat.reload()
    expect(chat.panel.get_by_role("button", name=f"Selected model: {name}")).to_be_visible()
    return chat


def test_a_model_picked_in_the_selector_of_an_open_chat_stays_after_a_reload(
    chat_for, harbour_master, upstream
):
    """In a side chat: red on dev, no issue filed (fix PR open-webui/open-webui#34064)."""
    account, name, model_id = harbour_master
    with account.client() as client:
        chat_id, _ = seed_chat(
            client, [{"role": "user", "content": "ahoy"}, {"role": "assistant", "content": "aye"}]
        )
    chat = chat_for(account, chat_id=chat_id)
    expect_reply(chat, "aye")

    chat = _pick_in_selector_and_reload(chat, account, chat_id, name, model_id)
    request = _ask(chat, upstream, "Slack water at three.")
    assert HARBOUR_PROMPT in _system_text(request)


def test_a_model_picked_in_the_selector_of_a_chat_started_on_the_home_page_stays_after_a_reload(
    chat_for, harbour_master, upstream
):
    """In a side chat: red on dev, no issue filed (fix PR open-webui/open-webui#34064)."""
    account, name, model_id = harbour_master
    chat = chat_for(account)
    question = f"ahoy {uuid.uuid4().hex[:6]}"
    upstream.queue(reply.text("aye", match=reply.answering(question)))
    send(chat, question)
    expect_reply(chat, "aye")
    expect(chat.page).to_have_url(re.compile(r"/c/"))
    chat_id = chat.page.url.rsplit("/c/", 1)[1]

    _pick_in_selector_and_reload(chat, account, chat_id, name, model_id)
