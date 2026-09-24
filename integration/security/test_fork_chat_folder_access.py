"""Regression: a forked chat lands in the source chat's folder only when the user may write there.

open-webui 0.11.4, fix `48fb2b84b` (#30069): `POST /api/v1/chats/{id}/fork` copied the source
chat's `folder_id` into the fork unchanged. The fork is a new chat owned by the caller, and the
folder may be one the caller can no longer write to (someone else's folder whose grant was
reduced to read), so the fork landed in a folder the caller could not manage. The fix keeps the
folder only when the caller has write access, as chat creation and chat moves already require.

Twin of unit/security/test_fork_chat_folder_access.py.

Discriminates: passes on bbfa876af; with the write check of 48fb2b84b removed the fork after the
grant is reduced to read is stored in the shared folder; the other tests pass on both.
"""

from __future__ import annotations

import uuid

import pytest

pytestmark = [pytest.mark.regression, pytest.mark.api, pytest.mark.requires_source]

CONVERSATION = {
    "title": "Source",
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
    },
}


def _create_folder(actor) -> str:
    with actor.client() as client:
        created = client.post("/api/v1/folders/", json={"name": f"folder-{uuid.uuid4().hex[:8]}"})
    assert created.status_code == 200, created.text
    return created.json()["id"]


def _grant(owner, folder_id: str, member, permission: str) -> None:
    grant = {"principal_type": "user", "principal_id": member.id, "permission": permission}
    with owner.client() as client:
        updated = client.post(
            f"/api/v1/folders/{folder_id}/access/update", json={"access_grants": [grant]}
        )
    assert updated.status_code == 200, updated.text


def _create_chat(actor, folder_id: str | None = None) -> str:
    with actor.client() as client:
        created = client.post("/api/v1/chats/new", json={"chat": CONVERSATION})
        assert created.status_code == 200, created.text
        chat_id = created.json()["id"]
        if folder_id:
            moved = client.post(f"/api/v1/chats/{chat_id}/folder", json={"folder_id": folder_id})
            assert moved.status_code == 200, moved.text
    return chat_id


def _fork_folder(actor, chat_id: str) -> str | None:
    """The folder the stored fork ended up in."""
    with actor.client() as client:
        forked = client.post(f"/api/v1/chats/{chat_id}/fork", json={"message_id": None})
        assert forked.status_code == 200, forked.text
        stored = client.get(f"/api/v1/chats/{forked.json()['id']}")
    assert stored.status_code == 200, stored.text
    return stored.json()["folder_id"]


@pytest.fixture
def shared_folder(make_user):
    """Another user's folder holding a member's chat, moved there while the member could write."""
    owner, member = make_user(), make_user()
    folder_id = _create_folder(owner)
    _grant(owner, folder_id, member, "write")
    chat_id = _create_chat(member, folder_id)
    return owner, member, folder_id, chat_id


# narrow: once the grant is only read, the fork lands outside the folder


def test_fork_of_a_chat_in_a_folder_now_read_only_lands_outside_it(shared_folder):
    owner, member, folder_id, chat_id = shared_folder
    _grant(owner, folder_id, member, "read")

    fork_folder = _fork_folder(member, chat_id)

    assert fork_folder is None, (
        f"the fork was stored in folder {fork_folder}, where its owner can only read; it sits "
        "somewhere they cannot manage it (#30069)"
    )


# broad: no way of placing a chat puts it in a folder the user cannot write to


def test_moving_or_creating_a_chat_in_a_read_only_folder_is_refused(shared_folder):
    owner, member, folder_id, _ = shared_folder
    _grant(owner, folder_id, member, "read")
    elsewhere = _create_chat(member)

    with member.client() as client:
        moved = client.post(f"/api/v1/chats/{elsewhere}/folder", json={"folder_id": folder_id})
        created = client.post(
            "/api/v1/chats/new", json={"chat": CONVERSATION, "folder_id": folder_id}
        )

    assert moved.status_code == 404, f"a chat was moved into a read-only folder: {moved.text}"
    assert created.status_code == 404, f"a chat was created in a read-only folder: {created.text}"


def test_clone_of_a_chat_in_a_read_only_folder_lands_outside_it(shared_folder):
    owner, member, folder_id, chat_id = shared_folder
    _grant(owner, folder_id, member, "read")

    with member.client() as client:
        cloned = client.post(f"/api/v1/chats/{chat_id}/clone", json={})

    assert cloned.status_code == 200, cloned.text
    assert cloned.json()["folder_id"] is None


# nearby: a writable folder keeps the fork, a folderless chat stays folderless


def test_fork_of_a_chat_in_a_writable_shared_folder_stays_in_it(shared_folder):
    _, member, folder_id, chat_id = shared_folder

    assert _fork_folder(member, chat_id) == folder_id, (
        "a member with write access to the source folder lost the fork from it"
    )


def test_fork_of_a_chat_in_the_users_own_folder_stays_in_it(make_user):
    owner = make_user()
    folder_id = _create_folder(owner)
    chat_id = _create_chat(owner, folder_id)

    assert _fork_folder(owner, chat_id) == folder_id


def test_fork_of_a_folderless_chat_stays_folderless(make_user):
    owner = make_user()

    assert _fork_folder(owner, _create_chat(owner)) is None
