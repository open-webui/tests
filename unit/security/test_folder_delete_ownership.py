"""Regression: deleting a folder is reserved for the folder owner or an admin.

open-webui 0.11.0 fix `915ef7d07` (#27003): `delete_folder_by_id` branched its
authorization on `folder.parent_id`. A root folder required admin, but a
subfolder only required *write* access, and a write grant on a shared root
folder is inherited by every descendant subfolder. Folder deletion cascades
into the folder owner's chats and the entire subtree, so a write-collaborator
on a shared folder could permanently destroy chats belonging to someone else.
The fix removes the root/subfolder split and requires owner-or-admin at every
depth.

Discriminates: passes on v0.11.0, fails on v0.10.2 (the collaborator's delete
of a shared subfolder returns True and wipes the owner's chats instead of
raising HTTPException 403).
"""

from contextlib import ExitStack, contextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

pytestmark = pytest.mark.regression

OWNER = "alice"
COLLABORATOR = "mallory"
ADMIN = "root"


def _folder(id: str, user_id: str = OWNER, parent_id: str | None = None) -> SimpleNamespace:
    return SimpleNamespace(id=id, user_id=user_id, parent_id=parent_id, name=id)


def _user(id: str, role: str = "user") -> SimpleNamespace:
    return SimpleNamespace(id=id, role=role)


class _FolderStore:
    """In-memory stand-in for the `Folders` model layer."""

    def __init__(self, *folders: SimpleNamespace) -> None:
        self.by_id = {folder.id: folder for folder in folders}
        self.deleted_ids: list[str] = []

    async def get_folder_by_id(self, id, db=None):
        return self.by_id.get(id)

    async def get_folder_by_id_and_user_id(self, id, user_id, db=None):
        folder = self.by_id.get(id)
        return folder if folder and folder.user_id == user_id else None

    async def get_folder_ids_by_id_and_user_id_in_subtree(self, id, user_id, db=None):
        return [entry.id for entry in self._subtree(id) if entry.user_id == user_id]

    async def get_folders_by_parent_id_and_user_id(self, parent_id, user_id, db=None):
        return [
            entry
            for entry in self.by_id.values()
            if entry.parent_id == parent_id and entry.user_id == user_id
        ]

    async def delete_folder_by_id_and_user_id(self, id, user_id, db=None):
        self.deleted_ids.append(id)
        self.by_id.pop(id, None)
        return [id]

    def _subtree(self, id):
        root = self.by_id.get(id)
        if not root:
            return []
        found = [root]
        pending = [id]
        while pending:
            parent_id = pending.pop()
            for entry in self.by_id.values():
                if entry.parent_id == parent_id:
                    found.append(entry)
                    pending.append(entry.id)
        return found


@contextmanager
def _router_boundary_patched(mod, store: _FolderStore):
    """Replace only the I/O boundary the delete handler touches."""
    chats = SimpleNamespace(
        count_chats_by_folder_ids_and_user_id=AsyncMock(return_value=0),
        delete_chats_by_user_id_and_folder_id=AsyncMock(),
        move_chats_by_user_id_and_folder_id=AsyncMock(),
    )
    config = SimpleNamespace(
        get_many=AsyncMock(return_value={"folders.enable": True, "user.permissions": {}}),
        get=AsyncMock(return_value={}),
    )
    patches = [
        patch.object(mod, "Folders", store),
        patch.object(mod, "Chats", chats),
        patch.object(mod, "Config", config),
        patch.object(mod, "AccessGrants", SimpleNamespace(revoke_all_access=AsyncMock())),
        patch.object(mod, "Automations", SimpleNamespace(clear_folder_ids=AsyncMock())),
        patch.object(mod, "has_permission", AsyncMock(return_value=True)),
        patch.object(mod, "publish_event", AsyncMock()),
    ]
    with ExitStack() as stack:
        for item in patches:
            stack.enter_context(item)
        yield chats


async def _delete(mod, store, user, folder_id, delete_contents=True):
    with _router_boundary_patched(mod, store):
        return await mod.delete_folder_by_id(
            SimpleNamespace(app=SimpleNamespace()),
            folder_id,
            delete_contents=delete_contents,
            user=user,
            db=None,
        )


def _shared_tree() -> _FolderStore:
    """Owner's folders: a shared root plus two levels of subfolder beneath it."""
    return _FolderStore(
        _folder("root"),
        _folder("child", parent_id="root"),
        _folder("grandchild", parent_id="child"),
    )


@pytest.mark.asyncio
async def test_write_collaborator_cannot_delete_a_shared_subfolder(owui_module):
    """The bug: write access on a shared folder is inherited by subfolders."""
    from fastapi import HTTPException

    mod = owui_module("open_webui.routers.folders")
    store = _shared_tree()

    with pytest.raises(HTTPException) as excinfo:
        await _delete(mod, store, _user(COLLABORATOR), "child")

    assert excinfo.value.status_code == 403, (
        "a write-collaborator deleted a subfolder of a folder someone else owns; "
        "the delete cascades into the owner's chats and subtree (#27003)"
    )
    assert store.deleted_ids == [], (
        "the owner's folders were destroyed before the authorization check took "
        "effect (#27003)"
    )


@pytest.mark.asyncio
async def test_write_collaborator_delete_does_not_move_the_owners_chats(owui_module):
    """With delete_contents=False the same path force-moved the owner's chats out."""
    from fastapi import HTTPException

    mod = owui_module("open_webui.routers.folders")
    store = _shared_tree()

    with _router_boundary_patched(mod, store) as chats:
        with pytest.raises(HTTPException) as excinfo:
            await mod.delete_folder_by_id(
                SimpleNamespace(app=SimpleNamespace()),
                "child",
                delete_contents=False,
                user=_user(COLLABORATOR),
                db=None,
            )

    assert excinfo.value.status_code == 403
    chats.move_chats_by_user_id_and_folder_id.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("folder_id", ["root", "child", "grandchild"])
async def test_non_owner_is_denied_at_every_depth(owui_module, folder_id):
    """The invariant: depth must not change who may delete."""
    from fastapi import HTTPException

    mod = owui_module("open_webui.routers.folders")
    store = _shared_tree()

    with pytest.raises(HTTPException) as excinfo:
        await _delete(mod, store, _user(COLLABORATOR), folder_id)

    assert excinfo.value.status_code == 403, (
        f"deleting '{folder_id}' was allowed for a non-owner; only the owner or an "
        "admin may delete a folder at any nesting level (#27003)"
    )
    assert store.deleted_ids == [], (
        f"deleting '{folder_id}' as a non-owner destroyed the owner's folders "
        f"{store.deleted_ids} (#27003)"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("folder_id", ["root", "child", "grandchild"])
async def test_owner_can_delete_their_own_folder_at_every_depth(owui_module, folder_id):
    """The fix must not lock owners out of their own folders."""
    mod = owui_module("open_webui.routers.folders")
    store = _shared_tree()

    result = await _delete(mod, store, _user(OWNER), folder_id)

    assert result is True
    assert folder_id in store.deleted_ids


@pytest.mark.asyncio
@pytest.mark.parametrize("folder_id", ["root", "child", "grandchild"])
async def test_admin_can_delete_another_users_folder_at_every_depth(owui_module, folder_id):
    mod = owui_module("open_webui.routers.folders")
    store = _shared_tree()

    result = await _delete(mod, store, _user(ADMIN, role="admin"), folder_id)

    assert result is True
    assert folder_id in store.deleted_ids


@pytest.mark.asyncio
async def test_a_collaborator_can_delete_a_subfolder_they_own(owui_module):
    """Write access is not what grants deletion, ownership is."""
    mod = owui_module("open_webui.routers.folders")
    store = _FolderStore(
        _folder("root"),
        _folder("their-own", user_id=COLLABORATOR, parent_id="root"),
    )

    result = await _delete(mod, store, _user(COLLABORATOR), "their-own")

    assert result is True


@pytest.mark.asyncio
async def test_unknown_folder_is_reported_as_missing(owui_module):
    """A folder that does not exist gives 404, not the authorization 403."""
    from fastapi import HTTPException

    mod = owui_module("open_webui.routers.folders")

    with pytest.raises(HTTPException) as excinfo:
        await _delete(mod, _shared_tree(), _user(COLLABORATOR), "no-such-folder")

    assert excinfo.value.status_code == 404


@pytest.mark.asyncio
async def test_unknown_folder_is_reported_as_missing_for_admins(owui_module):
    from fastapi import HTTPException

    mod = owui_module("open_webui.routers.folders")

    with pytest.raises(HTTPException) as excinfo:
        await _delete(mod, _shared_tree(), _user(ADMIN, role="admin"), "no-such-folder")

    assert excinfo.value.status_code == 404
