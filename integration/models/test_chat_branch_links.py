"""A reply missing from its parent's `childrenIds` broke the chat's branch structure.

Fix `59d3c5b06` (#29299, open-webui 0.11.3) in `models/chats.py`. Saving a message wrote its
`parentId` but never added it to the parent's `childrenIds`, so branch navigation, exports and
every walk down the tree lost what hung below it. `_repair_chat_current_id`, which runs on every
chat read and before every message write, now relinks each orphaned message, and saving a message
links it to its parent.

Twin of unit/models/test_chat_branch_links.py, which keeps the save-side link that the read-side
repair hides from HTTP.

Discriminates: passes on upstream dev `bbfa876af`; with the relink loop removed from
`_repair_chat_current_id` and the link removed from `upsert_message_to_history`, both narrow tests
and the three broad tests fail (the parent keeps listing only the children it was stored with).
"""

from __future__ import annotations

import httpx
import pytest

pytestmark = [pytest.mark.regression, pytest.mark.api, pytest.mark.requires_source]


def message(message_id: str, parent_id: str | None, children: list[str], role: str) -> dict:
    return {
        "id": message_id,
        "parentId": parent_id,
        "childrenIds": list(children),
        "role": role,
        "content": f"{role} {message_id}",
        "timestamp": 1_700_000_000,
    }


def history(current_id: str, *messages: dict) -> dict:
    return {"currentId": current_id, "messages": {entry["id"]: entry for entry in messages}}


def create_chat(client: httpx.Client, chat_history: dict) -> str:
    created = client.post(
        "/api/v1/chats/new", json={"chat": {"title": "Branches", "history": chat_history}}
    )
    assert created.status_code == 200, created.text
    return created.json()["id"]


def read_messages(client: httpx.Client, chat_id: str) -> dict:
    response = client.get(f"/api/v1/chats/{chat_id}")
    assert response.status_code == 200, response.text
    return response.json()["chat"]["history"]["messages"]


@pytest.fixture
def client(make_user):
    with make_user().client() as owner_client:
        yield owner_client


def test_reading_a_chat_lists_a_reply_under_its_parent(client):
    chat_id = create_chat(
        client,
        history(
            "reply",
            message("question", None, [], "user"),
            message("reply", "question", [], "assistant"),
        ),
    )

    assert read_messages(client, chat_id)["question"]["childrenIds"] == ["reply"], (
        "a chat stored with the parent-to-child link missing stayed broken when opened"
    )


def test_editing_an_orphaned_reply_lists_it_under_its_parent(client):
    chat_id = create_chat(
        client,
        history(
            "reply",
            message("question", None, [], "user"),
            message("reply", "question", [], "assistant"),
        ),
    )

    edited = client.post(f"/api/v1/chats/{chat_id}/messages/reply", json={"content": "edited"})
    assert edited.status_code == 200, edited.text

    saved = edited.json()["chat"]["history"]["messages"]
    assert saved["reply"]["content"] == "edited"
    assert saved["question"]["childrenIds"] == ["reply"]


def test_every_orphan_is_relinked_and_no_link_is_invented(client):
    chat_id = create_chat(
        client,
        history(
            "leaf",
            message("root", None, ["branch_a"], "user"),
            message("branch_a", "root", [], "assistant"),
            message("branch_b", "root", [], "assistant"),
            message("leaf", "branch_b", [], "user"),
        ),
    )

    messages = read_messages(client, chat_id)
    for message_id, entry in messages.items():
        if entry["parentId"]:
            assert message_id in messages[entry["parentId"]]["childrenIds"], message_id
        for child_id in entry["childrenIds"]:
            assert messages[child_id]["parentId"] == message_id, "a child link was invented"


def test_relinking_appends_to_the_listed_children_and_happens_once(client):
    chat_id = create_chat(
        client,
        history(
            "late",
            message("root", None, ["second", "first"], "user"),
            message("first", "root", [], "assistant"),
            message("second", "root", [], "assistant"),
            message("late", "root", [], "assistant"),
        ),
    )

    first_read = read_messages(client, chat_id)
    assert first_read["root"]["childrenIds"] == ["second", "first", "late"]
    assert read_messages(client, chat_id) == first_read


def test_a_parent_stored_without_a_children_list_gets_one(client):
    root = message("root", None, [], "user")
    del root["childrenIds"]
    chat_id = create_chat(client, history("reply", root, message("reply", "root", [], "assistant")))

    assert read_messages(client, chat_id)["root"]["childrenIds"] == ["reply"]


@pytest.mark.parametrize(
    "stored",
    [
        history(
            "reply",
            message("root", None, ["reply"], "user"),
            message("reply", "root", [], "assistant"),
        ),
        history("root", message("root", None, [], "user")),
        history("orphan", message("orphan", "gone", [], "assistant")),
    ],
    ids=["consistent", "single-root", "parent-not-in-history"],
)
def test_a_history_with_nothing_to_relink_is_read_back_unchanged(client, stored):
    chat_id = create_chat(client, stored)

    assert read_messages(client, chat_id) == stored["messages"]
