"""Journey: Merge Responses turns the answers of two models into one merged reply.

With two models chosen for a chat, both answer a question side by side and the last of them
offers Merge Responses. The merge asks the task model with the question and both answers in its
prompt, shows the merged text under the answers and keeps it with the chat after a reload.

Discriminates: passes on dev 30f3f6a8f; in a backend copy whose merge prompt leaves out the
answers the merge test fails on what the model was sent, and in a frontend build that does not
save the merged reply the reload test fails.
"""

from __future__ import annotations

import re
import uuid

import pytest
from playwright.sync_api import Page, expect

from harness import upstream as reply
from harness.upstream import MOCK_MODEL_ID
from utils.chat_ui import conversation, replies, send
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


def _merged(page: Page, upstream) -> None:
    upstream.queue(
        reply.text(MERGED, match=reply.answering(MERGE_TASK)),
        *(reply.text(answer, match=reply.answering(QUESTION)) for answer in ANSWERS),
    )
    send(page, QUESTION)
    expect(replies(page)).to_have_count(2)
    for answer in ANSWERS:
        expect(conversation(page).get_by_text(answer)).to_be_visible()
    tooltip_button(conversation(page), "Merge Responses").click()
    expect(conversation(page).get_by_text(MERGED)).to_be_visible()


def test_the_merge_is_written_from_the_question_and_both_answers(page_for, two_models, upstream):
    page = page_for(two_models)
    _merged(page, upstream)

    [merge_request] = [
        body for body in upstream.chat_requests() if reply.answering(MERGE_TASK)(body)
    ]
    prompt = str(merge_request["messages"][-1]["content"])
    assert QUESTION in prompt
    for answer in ANSWERS:
        assert answer in prompt, f"the merge was not given the answer {answer!r}"
    expect(conversation(page).get_by_text("Merged Response")).to_be_visible()


def test_the_merged_reply_is_kept_with_the_chat(page_for, two_models, upstream):
    page = page_for(two_models)
    _merged(page, upstream)
    expect(page).to_have_url(re.compile(r"/c/"))

    page.reload()
    expect(conversation(page).get_by_text(ANSWERS[0])).to_be_visible()
    expect(conversation(page).get_by_text(MERGED)).to_be_visible()
