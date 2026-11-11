"""Regression: Ctrl+Shift+Enter sent the typed message when "Ctrl+Enter to Send" was on.

Issue open-webui/open-webui#31468, fix 41ca360cc (PR open-webui/open-webui#31864). With Enter Key
Behavior set to Ctrl+Enter to Send, the chat input counted Ctrl+Shift+Enter as Ctrl+Enter and
submitted the message as well as running Generate Message Pair, so the model was asked for a
reply the user did not want. The chord now only adds the message and a placeholder reply, and
Ctrl+Enter still sends.

With two models chosen, the pair gets a placeholder reply from each of them, side by side, also
after a reload (feature 83dcc2dbe, PR open-webui/open-webui#32309); that test is a journey.

Discriminates: passes on the dev b859124f9 build, fails on that build with 41ca360cc reverted
(the model receives the typed message after Ctrl+Shift+Enter). The two-model test passes on the
dev 5032e834e build and fails on that build with 83dcc2dbe reverted (one placeholder reply).

Every test runs in the main chat and in a side chat (`utils/chat_surface.py`), the side chat asking
both default models picked in its own selector for the pair test. In a side chat the two pair
tests are red on dev c15f28649 and on dev e791cbdde: the shortcut adds the pair to the main chat
beside it and leaves the side chat empty (issue draft pending; PR open-webui/open-webui#34064
does not change it). On c15f28649 a side chat also had no Compare (open-webui/open-webui#34030).
"""

from __future__ import annotations

import re
import uuid

import pytest
from playwright.sync_api import expect

from harness import upstream as reply
from harness.python_tools import EVERYONE_READS
from harness.upstream import MOCK_MODEL_ID
from utils.chat_surface import ChatSurface
from utils.chat_ui import (
    chat_input,
    conversation,
    expect_reply,
    replies,
    start_on_default_models,
)

pytestmark = [pytest.mark.regression, pytest.mark.requires_browser, pytest.mark.requires_source]


@pytest.fixture
def ctrl_enter_page(chat_for, make_user) -> ChatSurface:
    account = make_user()
    with account.client() as client:
        client.post(
            "/api/v1/users/user/settings/update", json={"ui": {"ctrlEnterToSend": True}}
        ).raise_for_status()
    chat = chat_for(account)
    expect(chat_input(chat)).to_be_visible()
    chat_input(chat).click()
    return chat


def asked_about(upstream, prompt: str) -> list[dict]:
    return [request for request in upstream.chat_requests() if reply.answering(prompt)(request)]


def test_ctrl_shift_enter_adds_the_pair_without_asking_the_model(ctrl_enter_page, upstream):
    """In a side chat: issue draft pending (the pair goes to the main chat)."""
    chat = ctrl_enter_page
    upstream.queue(reply.text("An unwanted answer", match=reply.answering("pair me")))
    chat.page.keyboard.type("pair me")

    chat.page.keyboard.press("Control+Shift+Enter")

    expect(conversation(chat).get_by_text("pair me")).to_have_count(1)
    expect(conversation(chat).get_by_text("[RESPONSE")).to_be_visible()
    expect(chat_input(chat)).not_to_contain_text("pair me")
    chat.page.wait_for_timeout(1000)  # bounded: a request the chord sent would be here by now
    assert asked_about(upstream, "pair me") == []
    expect(conversation(chat).get_by_text("An unwanted answer")).to_have_count(0)


def test_ctrl_enter_still_sends_the_message(ctrl_enter_page, upstream):
    chat = ctrl_enter_page
    upstream.queue(reply.text("Sent and answered", match=reply.answering("send me")))
    chat.page.keyboard.type("send me")

    chat.page.keyboard.press("Control+Enter")

    expect_reply(chat, "Sent and answered")
    assert len(asked_about(upstream, "send me")) == 1


def test_plain_enter_does_not_send_with_the_setting_on(ctrl_enter_page, upstream):
    chat = ctrl_enter_page
    chat.page.keyboard.type("first line")

    chat.page.keyboard.press("Enter")
    chat.page.keyboard.type("second line")

    expect(chat_input(chat)).to_contain_text("second line")
    chat.page.wait_for_timeout(1000)  # bounded: an Enter that sent would have reached the provider
    assert asked_about(upstream, "first line") == []


@pytest.fixture
def two_models_page(chat_for, make_user, admin) -> ChatSurface:
    """A fresh account whose chats start on the scripted model and a second model beside it."""
    model_id = f"second-{uuid.uuid4().hex[:8]}"
    form = {
        "id": model_id,
        "base_model_id": MOCK_MODEL_ID,
        "name": model_id,
        "meta": {},
        "params": {},
        "access_grants": [EVERYONE_READS],
    }
    with admin.client() as client:
        created = client.post("/api/v1/models/create", json=form)
        assert created.status_code == 200, created.text
        account = make_user()
        with account.client() as account_client:
            account_client.post(
                "/api/v1/users/user/settings/update",
                json={"ui": {"models": [MOCK_MODEL_ID, model_id]}},
            ).raise_for_status()
        chat = chat_for(account)
        expect(chat_input(chat)).to_be_visible()
        start_on_default_models(chat)
        chat_input(chat).click()
        yield chat
        client.post("/api/v1/models/model/delete", json={"id": model_id})


@pytest.mark.journey
def test_the_pair_adds_a_placeholder_reply_for_every_selected_model(two_models_page, upstream):
    """In a side chat: issue draft pending (the pair goes to the main chat)."""
    chat = two_models_page
    chat.page.keyboard.type("pair us both")

    chat.page.keyboard.press("Control+Shift+Enter")

    expect(conversation(chat).get_by_text("pair us both")).to_have_count(1)
    expect(replies(chat)).to_have_count(2)
    expect(conversation(chat).get_by_text("[RESPONSE")).to_have_count(2)
    chat.page.wait_for_timeout(1000)  # bounded: a request the chord sent would be here by now
    assert asked_about(upstream, "pair us both") == []

    expect(chat.page).to_have_url(re.compile(r"/c/[\w-]+$"))
    chat = chat.reload()
    expect(conversation(chat).get_by_text("[RESPONSE")).to_have_count(2)
