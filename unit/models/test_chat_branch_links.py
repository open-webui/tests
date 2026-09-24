"""Saving a message must list it under its parent (the save side of `59d3c5b06`, #29299).

`upsert_message_to_history` wrote a message's `parentId` but never added it to the parent's
`childrenIds`, so the chat's branch structure broke. Open WebUI 0.11.3 links it on save and also
relinks every orphan on each chat read; the read side is pinned over HTTP and in the browser by
integration/ and e2e/models/test_chat_branch_links.py. Every HTTP path reads, and so repairs,
the chat before it saves a message, which hides the save side from HTTP; it is pinned here.

Discriminates: passes on upstream dev `bbfa876af`; with the `_add_child_id_to_parent` call removed
from `upsert_message_to_history` both narrow tests fail (the parent's `childrenIds` stays empty).
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.regression


@pytest.fixture(scope="module")
def chats(owui_module):
    return owui_module("open_webui.models.chats").Chats


def message(message_id: str, parent_id: str | None, role: str) -> dict:
    return {
        "id": message_id,
        "parentId": parent_id,
        "childrenIds": [],
        "role": role,
        "content": f"{role} {message_id}",
        "timestamp": 10,
    }


def test_saving_a_reply_lists_it_under_its_parent(chats):
    history = {"currentId": "question", "messages": {"question": message("question", None, "user")}}

    chats.upsert_message_to_history(
        history=history, message_id="reply", message={"parentId": "question", "content": "hi"}
    )

    assert history["messages"]["question"]["childrenIds"] == ["reply"], (
        "a saved reply was never added to its parent's childrenIds"
    )


def test_saving_an_orphaned_message_again_lists_it_under_its_parent(chats):
    history = {
        "currentId": "reply",
        "messages": {
            "question": message("question", None, "user"),
            "reply": message("reply", "question", "assistant"),
        },
    }

    chats.upsert_message_to_history(
        history=history, message_id="reply", message={"content": "edited"}
    )

    assert history["messages"]["question"]["childrenIds"] == ["reply"]
    assert history["messages"]["reply"]["content"] == "edited"


def test_saving_a_root_message_links_nothing(chats):
    history = {"currentId": None, "messages": {}}

    saved = chats.upsert_message_to_history(
        history=history, message_id="root", message={"parentId": None, "content": "hi"}
    )

    assert saved["parentId"] is None
    assert saved["childrenIds"] == []
    assert history["currentId"] == "root"
