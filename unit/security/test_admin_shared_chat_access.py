"""Regression: turning off blanket admin chat access must not revoke a share an
admin was deliberately given.

open-webui 0.11.0 fix `a35b37adc` (#27127): `get_chat_by_id` branched on the
caller's role. Admins went down a path that returned the chat only when
`ENABLE_ADMIN_CHAT_ACCESS` was on, and never fell through to the access-grant and
shared-folder checks that every non-admin got. With the setting off, an admin who
had been explicitly added as a recipient of a chat, directly or through a shared
folder, was refused it while any other recipient could open it. The fix tries the
admin-only path first, then lets every role fall through to the grant and folder
checks, so the setting still closes admin access to chats nobody shared.

The regression lived entirely inside the 0.11.0 cycle: `7088d245b` introduced the
role branch and `a35b37adc` removed it, so v0.10.2 predates the bug and passes
these tests. Discrimination was confirmed against `a35b37adc~1`, where the four
share-with-an-admin cases raise HTTPException 401 instead of returning the chat.

Discriminates: passes on v0.11.0, fails on a35b37adc~1 (admin plus setting off
raises HTTPException 401 instead of returning the deliberately shared chat).
Does NOT discriminate against v0.10.2, which never had the bug.
"""

from contextlib import ExitStack, contextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException

pytestmark = pytest.mark.regression

CHAT_ID = "chat-shared-with-admin"
OWNER_ID = "alice"
FOLDER_ID = "folder-shared-with-admin"
SHARE_ID = "share-token"


@pytest.fixture(scope="module")
def chats_router(owui_module):
    return owui_module("open_webui.routers.chats")


@pytest.fixture(scope="module")
def chat_models(owui_module):
    return owui_module("open_webui.models.chats")


def build_chat(chat_models, folder_id: str | None = None, meta: dict | None = None):
    return chat_models.ChatModel(
        id=CHAT_ID,
        user_id=OWNER_ID,
        title="Quarterly numbers",
        chat={"messages": []},
        created_at=1,
        updated_at=2,
        folder_id=folder_id,
        meta=meta or {},
    )


@contextmanager
def chat_backend(
    chats_router,
    *,
    stored_chat,
    admin_chat_access: bool,
    owned_by_caller: bool = False,
    has_grant: bool = False,
    folder_readable: bool = False,
):
    """Stub the DB boundary `get_chat_by_id` reads, leaving its logic untouched."""
    mod = chats_router
    folder = SimpleNamespace(id=FOLDER_ID, user_id=OWNER_ID) if stored_chat else None
    with ExitStack() as stack:
        stack.enter_context(patch.object(mod, "ENABLE_ADMIN_CHAT_ACCESS", admin_chat_access))
        stack.enter_context(
            patch.object(
                mod.Chats,
                "get_chat_by_id_and_user_id",
                AsyncMock(return_value=stored_chat if owned_by_caller else None),
            )
        )
        stack.enter_context(
            patch.object(mod.Chats, "get_chat_by_id", AsyncMock(return_value=stored_chat))
        )
        stack.enter_context(
            patch.object(mod.AccessGrants, "has_access", AsyncMock(return_value=has_grant))
        )
        stack.enter_context(
            patch.object(mod.Folders, "get_folder_by_id", AsyncMock(return_value=folder))
        )
        stack.enter_context(
            patch.object(mod, "has_folder_access", AsyncMock(return_value=folder_readable))
        )
        if hasattr(mod, "get_chat_context_usage"):
            stack.enter_context(
                patch.object(mod, "get_chat_context_usage", AsyncMock(return_value={}))
            )
        yield


def returned_chat_id(result) -> str:
    """v0.11.0 returns a dict (it adds `context_usage`), v0.10.2 a ChatResponse."""
    return result["id"] if isinstance(result, dict) else result.id


def make_user(role: str, user_id: str = "dana"):
    return SimpleNamespace(id=user_id, role=role)


# --- Narrow: the reported bug, plus the security property that must survive it ---


@pytest.mark.asyncio
async def test_admin_recipient_reads_shared_chat_with_blanket_access_off(chats_router, chat_models):
    stored_chat = build_chat(chat_models)
    with chat_backend(
        chats_router, stored_chat=stored_chat, admin_chat_access=False, has_grant=True
    ):
        result = await chats_router.get_chat_by_id(CHAT_ID, user=make_user("admin"), db=None)

    assert returned_chat_id(result) == CHAT_ID, (
        "an admin who was explicitly added as a recipient was refused a chat any "
        "other recipient can open, so the admin role took away access the owner "
        "granted (#27127)"
    )


@pytest.mark.asyncio
async def test_admin_reads_chat_in_shared_folder_with_blanket_access_off(chats_router, chat_models):
    stored_chat = build_chat(chat_models, folder_id=FOLDER_ID)
    with chat_backend(
        chats_router, stored_chat=stored_chat, admin_chat_access=False, folder_readable=True
    ):
        result = await chats_router.get_chat_by_id(CHAT_ID, user=make_user("admin"), db=None)

    assert returned_chat_id(result) == CHAT_ID, (
        "an admin lost a chat reachable through a folder shared with them, while a "
        "non-admin with the same folder access keeps it (#27127)"
    )


@pytest.mark.asyncio
async def test_admin_without_any_share_stays_denied_with_blanket_access_off(
    chats_router, chat_models
):
    """The half the setting exists for: no grant, no folder, no access."""
    stored_chat = build_chat(chat_models)
    with chat_backend(chats_router, stored_chat=stored_chat, admin_chat_access=False):
        with pytest.raises(HTTPException) as excinfo:
            await chats_router.get_chat_by_id(CHAT_ID, user=make_user("admin"), db=None)

    assert excinfo.value.status_code == 401, (
        "ENABLE_ADMIN_CHAT_ACCESS=false must still keep an arbitrary user's chat out "
        "of reach of an admin nobody shared it with (#27127)"
    )


# --- Broad: an explicit share outranks the blanket-access toggle, for every role
# and every read path ---


@pytest.mark.asyncio
@pytest.mark.parametrize("role", ["admin", "user"])
@pytest.mark.parametrize("share_kind", ["direct_grant", "shared_folder"])
async def test_explicit_share_beats_blanket_access_toggle_for_every_role(
    chats_router, chat_models, role, share_kind
):
    is_direct = share_kind == "direct_grant"
    stored_chat = build_chat(chat_models, folder_id=None if is_direct else FOLDER_ID)
    with chat_backend(
        chats_router,
        stored_chat=stored_chat,
        admin_chat_access=False,
        has_grant=is_direct,
        folder_readable=not is_direct,
    ):
        result = await chats_router.get_chat_by_id(CHAT_ID, user=make_user(role), db=None)

    assert returned_chat_id(result) == CHAT_ID, (
        f"a {role} holding a {share_kind} was refused the chat; the blanket-access "
        "setting must only close the admin-only route, never revoke a share (#27127)"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("role", ["admin", "user"])
async def test_shared_link_path_honours_the_grant_with_blanket_access_off(
    chats_router, chat_models, role
):
    """Same invariant on the sibling read path, the public share link."""
    mod = chats_router
    stored_chat = build_chat(chat_models)
    shared = SimpleNamespace(id=SHARE_ID, chat_id=CHAT_ID, user_id=OWNER_ID)
    with ExitStack() as stack:
        stack.enter_context(patch.object(mod, "ENABLE_ADMIN_CHAT_ACCESS", False))
        stack.enter_context(
            patch.object(mod.SharedChats, "get_by_id", AsyncMock(return_value=shared))
        )
        stack.enter_context(
            patch.object(
                mod.Chats, "get_chat_by_share_id", AsyncMock(return_value=stored_chat)
            )
        )
        stack.enter_context(
            patch.object(mod.AccessGrants, "has_access", AsyncMock(return_value=True))
        )
        stack.enter_context(
            patch.object(
                mod.AccessGrants, "has_anyone_access", AsyncMock(return_value=False)
            )
        )
        result = await mod.get_shared_chat_by_id(SHARE_ID, user=make_user(role), db=None)

    assert result.id == CHAT_ID, (
        f"a {role} recipient could not open the share link they were granted (#27127)"
    )


# --- Nearby: behaviour that was already correct and must stay that way ---


@pytest.mark.asyncio
async def test_admin_reads_any_chat_when_blanket_access_is_on(chats_router, chat_models):
    stored_chat = build_chat(chat_models)
    with chat_backend(chats_router, stored_chat=stored_chat, admin_chat_access=True):
        result = await chats_router.get_chat_by_id(CHAT_ID, user=make_user("admin"), db=None)

    assert returned_chat_id(result) == CHAT_ID


@pytest.mark.asyncio
async def test_non_admin_without_share_is_denied(chats_router, chat_models):
    stored_chat = build_chat(chat_models, folder_id=FOLDER_ID)
    with chat_backend(chats_router, stored_chat=stored_chat, admin_chat_access=True):
        with pytest.raises(HTTPException) as excinfo:
            await chats_router.get_chat_by_id(CHAT_ID, user=make_user("user"), db=None)

    assert excinfo.value.status_code == 401


@pytest.mark.asyncio
async def test_owner_reads_own_chat_regardless_of_the_setting(chats_router, chat_models):
    stored_chat = build_chat(chat_models)
    with chat_backend(
        chats_router, stored_chat=stored_chat, admin_chat_access=False, owned_by_caller=True
    ):
        result = await chats_router.get_chat_by_id(
            CHAT_ID, user=make_user("user", OWNER_ID), db=None
        )

    assert returned_chat_id(result) == CHAT_ID


@pytest.mark.asyncio
async def test_missing_chat_is_denied_not_crashed(chats_router):
    with chat_backend(chats_router, stored_chat=None, admin_chat_access=True):
        with pytest.raises(HTTPException) as excinfo:
            await chats_router.get_chat_by_id("no-such-chat", user=make_user("admin"), db=None)

    assert excinfo.value.status_code == 401
