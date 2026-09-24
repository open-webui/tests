"""A reply its parent did not list lost its branch navigation in the chat view.

Fix `59d3c5b06` (#29299, open-webui 0.11.3). The chat view builds a reply's `1/2` sibling switcher
from the parent's `childrenIds`; a reply stored without that link showed as the only answer and
its sibling was unreachable. Reading the chat now relinks every orphaned message first.

Twin of unit/models/test_chat_branch_links.py.

Discriminates: passes on upstream dev `bbfa876af`; with the relink loop removed from
`_repair_chat_current_id` the reply shows no `2/2` switcher.
"""

from __future__ import annotations

import pytest
from playwright.sync_api import expect

from harness.upstream import MOCK_MODEL_ID
from utils.chat_ui import conversation, expect_reply

pytestmark = [pytest.mark.regression, pytest.mark.requires_browser, pytest.mark.requires_source]


def _message(message_id: str, parent_id: str | None, children: list[str], content: str) -> dict:
    role = "user" if parent_id is None else "assistant"
    return {
        "id": message_id,
        "parentId": parent_id,
        "childrenIds": children,
        "role": role,
        "content": content,
        "timestamp": 1_700_000_000,
        **({"model": MOCK_MODEL_ID, "done": True} if role == "assistant" else {}),
    }


def test_an_orphaned_reply_gets_its_sibling_switcher(page_for, make_user):
    owner = make_user()
    messages = [
        _message("question", None, ["first"], "Which colour?"),
        _message("first", "question", [], "The first answer"),
        _message("second", "question", [], "The second answer"),
    ]
    chat = {
        "title": "Two answers",
        "models": [MOCK_MODEL_ID],
        "history": {"currentId": "second", "messages": {entry["id"]: entry for entry in messages}},
    }
    with owner.client() as client:
        created = client.post("/api/v1/chats/new", json={"chat": chat})
    assert created.status_code == 200, created.text

    page = page_for(owner)
    page.goto(f"/c/{created.json()['id']}")
    expect_reply(page, "The second answer")
    expect(conversation(page).get_by_text("2/2")).to_be_visible()

    conversation(page).get_by_role("button", name="Previous message").click()
    expect_reply(page, "The first answer")
    expect(conversation(page).get_by_text("The second answer")).to_be_hidden()
