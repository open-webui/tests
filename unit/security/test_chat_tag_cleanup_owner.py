"""Regression: an admin deleting someone else's chat tidies the owner's tags.

open-webui 0.11.4 fix `2690d04ca` (#30171): `delete_chat_by_id` in
`open_webui/routers/chats.py` called `Chats.delete_orphan_tags_for_user` with
`user.id`, the deleting admin's id, instead of `chat.user_id`, the owner's.
Deleting a user's last chat carrying a tag removed the tag row from the
ADMIN's tag table and left the owner's row behind, so the owner kept tags
nothing pointed at (and the admin could lose an unrelated tag of their own
that happened to share the id).

The test drives the real route with the model boundary stubbed, and asserts
on the exact user id handed to `delete_orphan_tags_for_user`.

Discriminates: passes on v0.11.4, fails on v0.11.3 (the tag cleanup is called
with the admin's id, not the owner's).
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

pytest.importorskip("fastapi")

pytestmark = pytest.mark.regression

ADMIN = SimpleNamespace(id="root", role="admin")
OWNER = "alice"
CHAT_ID = "chat-1"


@pytest.fixture
def calls(owui_module):
    return {"tag_user_ids": []}


@pytest.fixture
def chats_world(owui_module, calls):
    router_module = owui_module("open_webui.routers.chats")

    chat = SimpleNamespace(
        id=CHAT_ID,
        user_id=OWNER,
        title="Victim chat",
        chat={"history": {"messages": {}}},
        meta={"tags": ["work"]},
        pinned=False,
        variables={},
    )

    async def fake_delete_orphan_tags(tag_ids, user_id, threshold=0, db=None):
        calls["tag_user_ids"].append(user_id)

    patches = [
        patch.object(router_module.Chats, "get_chat_by_id", AsyncMock(return_value=chat)),
        patch.object(router_module.Chats, "delete_chat_by_id", AsyncMock(return_value=True)),
        patch.object(router_module.Chats, "delete_chat_by_id_and_user_id", AsyncMock()),
        patch.object(
            router_module.Chats,
            "delete_orphan_tags_for_user",
            fake_delete_orphan_tags,
        ),
        patch.object(
            router_module.Chats,
            "get_internal_chat_ids_by_parent_id",
            AsyncMock(return_value=[]),
        ),
        patch.object(router_module, "stop_item_tasks", AsyncMock()),
        patch.object(router_module, "publish_event", AsyncMock()),
    ]
    for p in patches:
        p.start()
    try:
        yield router_module
    finally:
        for p in reversed(patches):
            p.stop()


async def _delete(router_module, user=ADMIN):
    return await router_module.delete_chat_by_id(
        request=SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(redis=None))),
        id=CHAT_ID,
        user=user,
        db=None,
    )


# ── Narrow: the bug itself ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_admin_deleting_another_users_chat_cleans_the_owners_tags(chats_world, calls):
    """The exact bug: an admin deletes alice's chat, the tag cleanup must run
    against alice's account, not the admin's."""
    await _delete(chats_world)

    assert calls["tag_user_ids"] == [OWNER], (
        f"the admin's delete of alice's chat tidied tags for "
        f"{calls['tag_user_ids']} instead of the owner's account, leaving the "
        "owner with tags nothing points at (#30171)"
    )


@pytest.mark.asyncio
async def test_owner_deleting_their_own_chat_still_cleans_their_tags(chats_world, calls):
    owner = SimpleNamespace(id=OWNER, role="user")

    # The non-admin path checks the chat.delete permission first.
    with patch.object(
        chats_world, "has_permission", AsyncMock(return_value=True)
    ), patch.object(
        chats_world.Chats,
        "get_chat_by_id_and_user_id",
        AsyncMock(
            return_value=SimpleNamespace(
                id=CHAT_ID,
                user_id=OWNER,
                title="Own chat",
                chat={},
                meta={"tags": ["work"]},
                pinned=False,
                variables={},
            )
        ),
    ):
        await _delete(chats_world, user=owner)

    assert calls["tag_user_ids"] == [OWNER], (
        "the owner's own delete must keep tidying the owner's tags"
    )


# ── Nearby: the delete itself still happens for the right chat ─────────────


@pytest.mark.asyncio
async def test_admin_delete_still_removes_the_chat(chats_world):
    result = await _delete(chats_world)
    assert result is True, "the admin's delete must still go through"
