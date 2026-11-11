"""Journey: Merge Responses turns the answers of two models into one merged reply.

With two models chosen for a chat, both answer a question side by side and the last of them
offers Merge Responses. The merge asks the task model with the question and both answers in its
prompt, shows the merged text under the answers and keeps it with the chat after a reload.

Discriminates: passes on dev 30f3f6a8f; in a backend copy whose merge prompt leaves out the
answers the merge test fails on what the model was sent, and in a frontend build that does not
save the merged reply the reload test fails.

Both tests run in the main chat and in every way a side chat starts and every state it can be in
(`utils/chat_surface.py`), the side chat asking both models picked in its own model selector. A
side chat had no Compare in its model selector and showed one model's answer only
(open-webui/open-webui#34030, fix PR open-webui/open-webui#34064), so the side chat variants are
red on dev c15f28649. A temporary side chat is left out of the reload test, as the chat is gone
after the reload.
"""

from __future__ import annotations

import re
import uuid

import pytest
from playwright.sync_api import Locator, expect

from harness import upstream as reply
from harness.upstream import MOCK_MODEL_ID
from utils.chat_surface import EVERYWHERE, TEMPORARY_SIDE, ChatSurface
from utils.chat_ui import conversation, send, start_on_default_models
from utils.tooltips import tooltip_button

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

MERGE_TASK = "synthesize these responses"
QUESTION = "Which coast is better for a sailing holiday?"
ANSWERS = ("The west coast has steadier winds.", "The south coast has calmer harbours.")
MERGED = "Steady winds out west, calm harbours down south."


@pytest.fixture
def two_models(make_user):
    """A fresh admin whose chats start on the scripted model and a second model beside it."""
    account = make_user(role="admin")
    model_id = f"second-opinion-{uuid.uuid4().hex[:6]}"
    form = {"id": model_id, "name": "Second Opinion", "base_model_id": MOCK_MODEL_ID}
    with account.client() as client:
        created = client.post("/api/v1/models/create", json={**form, "meta": {}, "params": {}})
        assert created.status_code == 200, created.text
        chosen = client.post(
            "/api/v1/users/user/settings/update", json={"ui": {"models": [MOCK_MODEL_ID, model_id]}}
        )
        assert chosen.status_code == 200, chosen.text
        yield account
        client.post("/api/v1/models/model/delete", json={"id": model_id})


def _replies_to_the_last_question(chat: ChatSurface) -> Locator:
    """The replies after the chat's last question; a side chat can hold earlier turns."""
    question = conversation(chat).locator(".chat-user").last
    return question.locator("xpath=following::*[contains(@class, 'chat-assistant')]")


def _merged(chat: ChatSurface, upstream) -> None:
    upstream.queue(
        reply.text(MERGED, match=reply.answering(MERGE_TASK)),
        *(reply.text(answer, match=reply.answering(QUESTION)) for answer in ANSWERS),
    )
    send(chat, QUESTION)
    expect(_replies_to_the_last_question(chat)).to_have_count(2)
    for answer in ANSWERS:
        expect(conversation(chat).get_by_text(answer)).to_be_visible()
    tooltip_button(conversation(chat), "Merge Responses").click()
    expect(conversation(chat).get_by_text(MERGED)).to_be_visible()


@pytest.mark.chat_surfaces(*EVERYWHERE)
def test_the_merge_is_written_from_the_question_and_both_answers(chat_for, two_models, upstream):
    """In a side chat: open-webui/open-webui#34030."""
    chat = chat_for(two_models)
    start_on_default_models(chat)
    _merged(chat, upstream)

    [merge_request] = [
        body for body in upstream.chat_requests() if reply.answering(MERGE_TASK)(body)
    ]
    prompt = str(merge_request["messages"][-1]["content"])
    assert QUESTION in prompt
    for answer in ANSWERS:
        assert answer in prompt, f"the merge was not given the answer {answer!r}"
    expect(conversation(chat).get_by_text("Merged Response")).to_be_visible()


# a temporary chat is not kept, so it has nothing to show after the reload
@pytest.mark.chat_surfaces(*(place for place in EVERYWHERE if place != TEMPORARY_SIDE))
def test_the_merged_reply_is_kept_with_the_chat(chat_for, two_models, upstream):
    """In a side chat: open-webui/open-webui#34030."""
    chat = chat_for(two_models)
    start_on_default_models(chat)
    _merged(chat, upstream)
    expect(chat.page).to_have_url(re.compile(r"/c/"))

    chat = chat.reload()
    expect(conversation(chat).get_by_text(ANSWERS[0])).to_be_visible()
    expect(conversation(chat).get_by_text(MERGED)).to_be_visible()
