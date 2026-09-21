"""Regression: a forked chat lands outside the source folder unless the caller
can write to it.

open-webui 0.11.4 fix `48fb2b84b` (#30069): `fork_chat_by_id` in
`open_webui/routers/chats.py` copied `chat.folder_id` into the new chat
unconditionally. A fork is created owned by the CALLER, but the source
folder can be one the caller cannot write to (a folder shared read-only, or
another user's folder reached through a shared chat), so the fork silently
landed in a folder the caller could not manage. The fix keeps the folder
only when `has_folder_write_access` allows it, matching what chat creation
and chat moves already do.

The test drives the real route with the model boundary stubbed and asserts
on the folder_id handed to `insert_new_chat`.

Discriminates: passes on v0.11.4, fails on v0.11.3 (the fork is inserted with
the source's folder_id even though the caller cannot write to it).
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

pytest.importorskip("fastapi")

pytestmark = pytest.mark.regression

OWNER = "alice"
FOLDER_ID = "folder-shared"


def _source_chat() -> SimpleNamespace:
    return SimpleNamespace(
        id="chat-1",
        user_id=OWNER,
        title="Source",
        chat={
            "history": {
                "currentId": "m2",
                "messages": {
                    "m1": {
                        "id": "m1",
                        "parentId": None,
                        "childrenIds": ["m2"],
                        "role": "user",
                        "content": "question",
                    },
                    "m2": {
                        "id": "m2",
                        "parentId": "m1",
                        "childrenIds": [],
                        "role": "assistant",
                        "content": "answer",
                        "done": True,
                    },
                },
            }
        },
        meta={},
        variables={},
        pinned=False,
        current_message_id="m2",
        folder_id=FOLDER_ID,
    )


@pytest.fixture
def fork_world(owui_module):
    router_module = owui_module("open_webui.routers.chats")

    inserted = {"folder_ids": []}
    write_access = {"allowed": False}

    async def fake_insert_new_chat(new_id, user_id, form_data, db=None, **kwargs):
        inserted["folder_ids"].append(form_data.folder_id)
        source = _source_chat()
        fork = SimpleNamespace(
            id=new_id,
            user_id=user_id,
            title="Source (fork)",
            chat=source.chat,
            meta={"forked_from": "chat-1", "forked_from_message_id": "m2"},
            variables={},
            folder_id=form_data.folder_id,
            created_at=0,
            updated_at=0,
            archived=False,
            pinned=False,
        )
        return fork

    async def fake_write_access(user_id, folder_id, db=None):
        return write_access["allowed"]

    patches = [
        patch.object(
            router_module.Chats,
            "get_chat_by_id_and_user_id",
            AsyncMock(return_value=_source_chat()),
        ),
        patch.object(
            router_module.Chats,
            "get_messages_map_by_chat_id",
            AsyncMock(return_value=_source_chat().chat["history"]["messages"]),
        ),
        patch.object(router_module.Chats, "insert_new_chat", fake_insert_new_chat),
        patch.object(
            router_module.Chats,
            "update_chat_variables_by_id",
            AsyncMock(return_value=None),
        ),
        patch.object(router_module.Chats, "toggle_chat_pinned_by_id", AsyncMock()),
        patch.object(router_module, "require_chat_import_permission", AsyncMock()),
        patch.object(router_module, "has_active_tasks", AsyncMock(return_value=False)),
        patch.object(router_module, "has_folder_write_access", fake_write_access),
        patch.object(router_module, "publish_event", AsyncMock()),
    ]
    for p in patches:
        p.start()
    try:
        yield router_module, inserted, write_access
    finally:
        for p in reversed(patches):
            p.stop()


async def _fork(router_module, form=None):
    return await router_module.fork_chat_by_id(
        request=SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(redis=None))),
        id="chat-1",
        form_data=form,
        user=SimpleNamespace(id=OWNER, role="user"),
        db=None,
    )


# ── Narrow: the bug itself ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_fork_of_an_unwritable_folder_lands_outside_any_folder(fork_world):
    router_module, inserted, write_access = fork_world
    write_access["allowed"] = False

    await _fork(router_module)

    assert inserted["folder_ids"] == [None], (
        "the fork was created inside a folder the caller cannot write to, so the "
        "copy sits somewhere its owner cannot manage it (#30069)"
    )


@pytest.mark.asyncio
async def test_fork_of_a_writable_folder_stays_in_the_folder(fork_world):
    router_module, inserted, write_access = fork_world
    write_access["allowed"] = True

    await _fork(router_module)

    assert inserted["folder_ids"] == [FOLDER_ID], (
        "a caller with write access to the source folder must keep the fork in it"
    )


# ── Broad: the permission check actually runs for the source folder ────────


@pytest.mark.asyncio
async def test_folder_write_access_is_checked_for_the_source_folder(fork_world):
    router_module, inserted, write_access = fork_world
    checked = {"folder_ids": []}

    async def recording_access(user_id, folder_id, db=None):
        checked["folder_ids"].append(folder_id)
        return write_access["allowed"]

    with patch.object(router_module, "has_folder_write_access", recording_access):
        await _fork(router_module)

    assert checked["folder_ids"] == [FOLDER_ID], (
        "the fork never asked about write access to the source folder, so the "
        "folder id was carried over blindly (#30069)"
    )


# ── Nearby: forking a chat with no folder stays folderless ──────────────────


@pytest.mark.asyncio
async def test_fork_of_a_folderless_chat_stays_folderless(fork_world):
    router_module, inserted, write_access = fork_world

    source = _source_chat()
    source.folder_id = None
    with patch.object(
        router_module.Chats,
        "get_chat_by_id_and_user_id",
        AsyncMock(return_value=source),
    ):
        await _fork(router_module)

    assert inserted["folder_ids"] == [None]
