"""Regression: deleting a folder is reserved for its owner or an admin.

open-webui 0.11.0, fix `915ef7d07` (#27003): `DELETE /api/v1/folders/{id}` required an admin for a
root folder but only write access for a subfolder, and write access on a shared folder is
inherited by every folder below it. The delete cascades into the owner's chats, so a collaborator
with write access to a shared folder could destroy chats that belong to someone else. The fix
requires owner or admin at every depth.

open-webui 0.11.4, fix `b64cb0604` (#30163): the chat delete permission was checked even with
`delete_contents=false`, where the chats move out of the folder instead of being deleted, so an
account without `chat.delete` could not remove a folder while keeping its chats.

Twin of unit/security/test_folder_delete_ownership.py.

Discriminates: passes on bbfa876af; fails with 915ef7d07 reverted (the collaborator's deletes of
the shared subfolders answer 200 and the owner's chat is gone) and with b64cb0604 reverted (the
keep-chats delete answers 403); the other tests pass on both.
"""

from __future__ import annotations

import uuid

import pytest

pytestmark = [pytest.mark.regression, pytest.mark.api, pytest.mark.requires_source]


def _create_folder(actor, parent_id: str | None = None) -> str:
    with actor.client() as client:
        created = client.post(
            "/api/v1/folders/",
            json={"name": f"folder-{uuid.uuid4().hex[:8]}", "parent_id": parent_id},
        )
    assert created.status_code == 200, created.text
    return created.json()["id"]


def _create_chat(actor, folder_id: str) -> str:
    with actor.client() as client:
        created = client.post(
            "/api/v1/chats/new", json={"chat": {"title": "kept safe"}, "folder_id": folder_id}
        )
    assert created.status_code == 200, created.text
    return created.json()["id"]


def _stored_chat(owner, chat_id: str) -> dict | None:
    with owner.client() as client:
        found = client.get(f"/api/v1/chats/{chat_id}")
    return found.json() if found.status_code == 200 else None


def _delete_folder(actor, folder_id: str, delete_contents: bool = True):
    with actor.client() as client:
        return client.delete(
            f"/api/v1/folders/{folder_id}",
            params={"delete_contents": str(delete_contents).lower()},
        )


def _folder_exists(owner, folder_id: str) -> bool:
    with owner.client() as client:
        return client.get(f"/api/v1/folders/{folder_id}").status_code == 200


@pytest.fixture
def shared_tree(admin, make_user):
    """The admin's root, child and grandchild folders, a chat in the child and a user with
    write access to the root."""
    root = _create_folder(admin)
    child = _create_folder(admin, parent_id=root)
    grandchild = _create_folder(admin, parent_id=child)
    chat_id = _create_chat(admin, child)
    collaborator = make_user()
    grants = [
        {"principal_type": "user", "principal_id": collaborator.id, "permission": permission}
        for permission in ("read", "write")
    ]
    with admin.client() as client:
        shared = client.post(
            f"/api/v1/folders/{root}/access/update", json={"access_grants": grants}
        )
    assert shared.status_code == 200, shared.text
    yield {"root": root, "child": child, "grandchild": grandchild, "chat_id": chat_id}, collaborator
    _delete_folder(admin, root)


@pytest.fixture
def chat_delete_denied(admin, preserve):
    preserve("permissions")
    with admin.client() as client:
        permissions = client.get("/api/v1/users/default/permissions").json()
        permissions["chat"]["delete"] = False
        saved = client.post("/api/v1/users/default/permissions", json=permissions)
    assert saved.status_code == 200, saved.text


# narrow (#27003): write access to a shared subfolder is not a licence to delete it


@pytest.mark.parametrize("delete_contents", [True, False], ids=["delete-chats", "keep-chats"])
def test_write_collaborator_cannot_delete_a_shared_subfolder(admin, shared_tree, delete_contents):
    folders, collaborator = shared_tree

    deleted = _delete_folder(collaborator, folders["child"], delete_contents)

    assert deleted.status_code == 403, (
        f"a write collaborator deleted a subfolder someone else owns (HTTP {deleted.status_code}); "
        "the delete cascades into the owner's chats (#27003)"
    )
    chat = _stored_chat(admin, folders["chat_id"])
    assert chat is not None, "the owner's chat was deleted by a collaborator (#27003)"
    assert chat["folder_id"] == folders["child"], "the owner's chat was moved out of its folder"


# broad (#27003): depth does not change who may delete


@pytest.mark.parametrize("depth", ["root", "child", "grandchild"])
def test_write_collaborator_cannot_delete_at_any_depth(admin, shared_tree, depth):
    folders, collaborator = shared_tree

    deleted = _delete_folder(collaborator, folders[depth])

    assert deleted.status_code == 403, f"a write collaborator deleted the shared {depth} (#27003)"
    assert _folder_exists(admin, folders[depth])


# narrow (#30163): keeping the chats needs no chat delete permission


def test_user_without_chat_delete_can_delete_a_folder_keeping_its_chats(
    chat_delete_denied, make_user
):
    owner = make_user()
    folder_id = _create_folder(owner)
    chat_id = _create_chat(owner, folder_id)

    deleted = _delete_folder(owner, folder_id, delete_contents=False)

    assert deleted.status_code == 200, (
        f"a keep-chats folder delete was refused for lack of chat.delete (HTTP "
        f"{deleted.status_code}), so the folder can never be removed (#30163)"
    )
    assert not _folder_exists(owner, folder_id)
    assert _stored_chat(owner, chat_id)["folder_id"] is None


# broad (#30163): destroying chats still needs the permission


def test_user_without_chat_delete_cannot_delete_a_folder_with_its_chats(
    chat_delete_denied, make_user
):
    owner = make_user()
    folder_id = _create_folder(owner)
    chat_id = _create_chat(owner, folder_id)

    deleted = _delete_folder(owner, folder_id, delete_contents=True)

    assert deleted.status_code == 403
    assert _folder_exists(owner, folder_id)
    assert _stored_chat(owner, chat_id) is not None


# nearby: owners and admins keep deleting, missing folders are 404


@pytest.mark.parametrize("depth", ["root", "child", "grandchild"])
def test_owner_can_delete_their_folder_at_any_depth(admin, shared_tree, depth):
    folders, _ = shared_tree

    deleted = _delete_folder(admin, folders[depth])

    assert deleted.status_code == 200, deleted.text
    assert not _folder_exists(admin, folders[depth])


@pytest.mark.parametrize("depth", ["root", "child"])
def test_admin_can_delete_another_users_folder_at_any_depth(admin, make_user, depth):
    owner = make_user()
    root = _create_folder(owner)
    folders = {"root": root, "child": _create_folder(owner, parent_id=root)}

    deleted = _delete_folder(admin, folders[depth])

    assert deleted.status_code == 200, deleted.text
    assert not _folder_exists(owner, folders[depth])


def test_collaborator_can_delete_their_own_folder_inside_the_shared_one(shared_tree):
    folders, collaborator = shared_tree
    own_folder = _create_folder(collaborator)
    with collaborator.client() as client:
        moved = client.post(
            f"/api/v1/folders/{own_folder}/update/parent", json={"parent_id": folders["root"]}
        )
    assert moved.status_code == 200, moved.text

    assert _delete_folder(collaborator, own_folder).status_code == 200


def test_user_without_chat_delete_can_delete_an_empty_folder(chat_delete_denied, make_user):
    owner = make_user()
    folder_id = _create_folder(owner)

    assert _delete_folder(owner, folder_id, delete_contents=True).status_code == 200


def test_admin_delete_with_chats_ignores_the_chat_delete_permission(chat_delete_denied, admin):
    folder_id = _create_folder(admin)
    chat_id = _create_chat(admin, folder_id)

    assert _delete_folder(admin, folder_id, delete_contents=True).status_code == 200
    assert _stored_chat(admin, chat_id) is None


@pytest.mark.parametrize("role", ["user", "admin"])
def test_unknown_folder_is_not_found(make_user, role):
    assert _delete_folder(make_user(role=role), "no-such-folder").status_code == 404
