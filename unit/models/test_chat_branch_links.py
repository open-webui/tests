"""Regression: a reply was not listed under its parent message, so the branch
structure of a chat broke.

`59d3c5b06` (#29299, shipped in 0.11.3) in `open_webui/models/chats.py`.
`upsert_message_to_history` wrote the new message with its `parentId` but never
appended its id to the parent's `childrenIds`, so branch arrows, exports and any
later walk down the tree lost everything below that point. The new
`_add_child_id_to_parent` helper writes the link on save, and
`_repair_chat_current_id` now relinks every orphaned message when a chat is
read, so chats already stored with the link missing heal on open.

Discriminates: passes on v0.11.3, fails on v0.11.2 (a saved or edited message
never reaches its parent's `childrenIds`, and reading a chat whose links are
already missing leaves them missing).
"""

from __future__ import annotations

import copy

import pytest

pytestmark = pytest.mark.regression


@pytest.fixture(scope="module")
def chat_models(owui_module):
    return owui_module("open_webui.models.chats")


def message(
    message_id: str, parent_id: str | None, children: list[str], role: str, timestamp: int
) -> dict:
    return {
        "id": message_id,
        "parentId": parent_id,
        "childrenIds": list(children),
        "role": role,
        "content": f"{role} {message_id}",
        "timestamp": timestamp,
    }


def orphaned_history() -> dict:
    """`reply` points at `question` but `question` does not list it."""
    return {
        "currentId": "reply",
        "messages": {
            "question": message("question", None, [], "user", 10),
            "reply": message("reply", "question", [], "assistant", 11),
        },
    }


# --- Narrow: the missing parent-to-child link ---------------------------------


@pytest.mark.asyncio
async def test_reading_a_chat_relinks_a_reply_to_its_parent(chat_models):
    chat = {"history": orphaned_history()}

    changed = chat_models.Chats._repair_chat_current_id(chat)

    assert chat["history"]["messages"]["question"]["childrenIds"] == ["reply"], (
        "a chat stored with the parent-to-child link missing stayed broken on "
        "read, so the branch below that message was invisible"
    )
    assert changed is True


@pytest.mark.asyncio
async def test_saving_a_reply_lists_it_under_its_parent(chat_models):
    history = {
        "currentId": "question",
        "messages": {"question": message("question", None, [], "user", 10)},
    }

    chat_models.Chats.upsert_message_to_history(
        history, "reply", {"parentId": "question", "content": "hi"}
    )

    assert history["messages"]["question"]["childrenIds"] == ["reply"], (
        "a newly saved reply was never added to its parent's childrenIds, so the "
        "conversation lost everything below the parent"
    )
    assert history["messages"]["reply"]["parentId"] == "question"


# --- Broad: the whole history is consistent after a read ----------------------


@pytest.mark.asyncio
async def test_repair_relinks_every_orphan_across_branches(chat_models):
    messages = {
        "root": message("root", None, ["branch_a"], "user", 1),
        "branch_a": message("branch_a", "root", [], "assistant", 2),
        "branch_b": message("branch_b", "root", [], "assistant", 3),
        "leaf": message("leaf", "branch_b", [], "user", 4),
    }
    chat = {"history": {"currentId": "leaf", "messages": messages}}

    chat_models.Chats._repair_chat_current_id(chat)

    repaired = chat["history"]["messages"]
    for message_id, entry in repaired.items():
        parent_id = entry["parentId"]
        if parent_id:
            assert message_id in repaired[parent_id]["childrenIds"]
        for child_id in entry["childrenIds"]:
            assert repaired[child_id]["parentId"] == message_id, "repair invented a child link"


@pytest.mark.asyncio
async def test_repair_is_idempotent_and_keeps_existing_child_order(chat_models):
    messages = {
        "root": message("root", None, ["second", "first"], "user", 1),
        "first": message("first", "root", [], "assistant", 2),
        "second": message("second", "root", [], "assistant", 3),
        "late": message("late", "root", [], "assistant", 4),
    }
    chat = {"history": {"currentId": "late", "messages": messages}}

    chat_models.Chats._repair_chat_current_id(chat)
    after_first_pass = copy.deepcopy(chat)
    chat_models.Chats._repair_chat_current_id(chat)

    assert messages["root"]["childrenIds"] == ["second", "first", "late"], (
        "repair reordered children that were already listed instead of appending "
        "only the missing one"
    )
    assert chat == after_first_pass


@pytest.mark.asyncio
async def test_updating_an_existing_message_also_relinks_it(chat_models):
    history = {
        "currentId": "reply",
        "messages": {
            "question": message("question", None, [], "user", 10),
            "reply": message("reply", "question", [], "assistant", 11),
        },
    }

    chat_models.Chats.upsert_message_to_history(history, "reply", {"content": "edited"})

    assert history["messages"]["question"]["childrenIds"] == ["reply"], (
        "editing a message left it unlisted under its parent, so a chat broken "
        "before the fix stayed broken through every later write to it"
    )
    assert history["currentId"] == "reply"


@pytest.mark.asyncio
async def test_repair_rebuilds_a_parent_whose_children_list_is_missing(chat_models):
    messages = {
        "root": {"id": "root", "parentId": None, "role": "user", "timestamp": 1},
        "reply": message("reply", "root", [], "assistant", 2),
    }
    chat = {"history": {"currentId": "reply", "messages": messages}}

    changed = chat_models.Chats._repair_chat_current_id(chat)

    assert messages["root"].get("childrenIds") == ["reply"], (
        "a parent stored without a childrenIds list kept no children at all"
    )
    assert changed is True


@pytest.mark.asyncio
async def test_repair_leaves_a_consistent_history_untouched(chat_models):
    messages = {
        "root": message("root", None, ["reply"], "user", 1),
        "reply": message("reply", "root", [], "assistant", 2),
    }
    chat = {"history": {"currentId": "reply", "messages": messages}}
    before = copy.deepcopy(chat)

    changed = chat_models.Chats._repair_chat_current_id(chat)

    assert chat == before
    assert changed is False


# --- Nearby: histories with nothing to relink ---------------------------------


@pytest.mark.asyncio
async def test_repair_ignores_a_chat_with_no_history_key(chat_models):
    assert chat_models.Chats._repair_chat_current_id({"title": "Untitled"}) is False


@pytest.mark.asyncio
async def test_repair_ignores_an_empty_history(chat_models):
    chat = {"history": {"currentId": None, "messages": {}}}

    assert chat_models.Chats._repair_chat_current_id(chat) is False
    assert chat["history"]["messages"] == {}


@pytest.mark.asyncio
async def test_repair_leaves_a_single_root_message_alone(chat_models):
    messages = {"root": message("root", None, [], "user", 1)}
    chat = {"history": {"currentId": "root", "messages": messages}}
    before = copy.deepcopy(chat)

    changed = chat_models.Chats._repair_chat_current_id(chat)

    assert chat == before
    assert changed is False


@pytest.mark.asyncio
async def test_repair_ignores_a_parent_that_is_not_in_the_history(chat_models):
    messages = {"orphan": message("orphan", "gone", [], "assistant", 1)}
    chat = {"history": {"currentId": "orphan", "messages": messages}}
    before = copy.deepcopy(chat)

    changed = chat_models.Chats._repair_chat_current_id(chat)

    assert chat == before
    assert changed is False


@pytest.mark.asyncio
async def test_saving_a_root_message_needs_no_parent_link(chat_models):
    history = {"currentId": None, "messages": {}}

    saved = chat_models.Chats.upsert_message_to_history(
        history, "root", {"parentId": None, "content": "hi"}
    )

    assert saved["parentId"] is None
    assert saved["childrenIds"] == []
    assert history["currentId"] == "root"
