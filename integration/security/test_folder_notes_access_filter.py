"""Regression: a folder's attached notes were never access-checked.

open-webui 0.11.0, fix `f89b50198` (#26739): `get_accessible_folder_files` reduces a folder's
attached knowledge (`data.files`) to the entries the caller may read. It checked `file` and
`collection` entries but kept every `note` entry as it was, so a note the caller had lost access
to, or that no longer existed, stayed in the knowledge handed to the model. The fix keeps a note
only when the caller owns it or holds a read grant. Over HTTP the filter shows in two places:
`GET /api/v1/folders/` saves the filtered list back to each folder, and creating or updating a
folder refuses entries the caller cannot read.

Twin of unit/security/test_folder_notes_access_filter.py.

Discriminates: passes on bbfa876af; with the note branch reduced to keeping every note, as
before f89b50198, the deleted and the revoked note stay attached and a folder naming another
user's private note is created and updated with 200; the other tests pass on both.
"""

from __future__ import annotations

import uuid

import pytest

pytestmark = [pytest.mark.regression, pytest.mark.api, pytest.mark.requires_source]


def _create_note(client, access_grants: list[dict] | None = None) -> str:
    created = client.post(
        "/api/v1/notes/create",
        json={
            "title": f"note-{uuid.uuid4().hex[:8]}",
            "data": {"content": {"md": "private thoughts"}},
            "access_grants": access_grants or [],
        },
    )
    assert created.status_code == 200, created.text
    return created.json()["id"]


def _read_grant(reader) -> dict:
    return {"principal_type": "user", "principal_id": reader.id, "permission": "read"}


def _note_entry(note_id: str) -> dict:
    return {"type": "note", "id": note_id}


def _create_folder(client, files: list[dict]):
    return client.post(
        "/api/v1/folders/",
        json={"name": f"folder-{uuid.uuid4().hex[:8]}", "data": {"files": files}},
    )


def _attached_after_listing(client, folder_id: str) -> list[dict]:
    """The folder's knowledge once the sidebar listing has filtered and saved it."""
    listed = client.get("/api/v1/folders/")
    assert listed.status_code == 200, listed.text
    folder = client.get(f"/api/v1/folders/{folder_id}")
    assert folder.status_code == 200, folder.text
    return folder.json()["data"]["files"]


@pytest.fixture
def owner(make_user):
    return make_user()


@pytest.fixture
def author(make_user):
    return make_user()


# narrow: a note the owner can no longer read leaves the folder's knowledge


def test_a_deleted_note_leaves_the_folder_knowledge(owner):
    with owner.client() as client:
        kept_note, deleted_note = _create_note(client), _create_note(client)
        entries = [_note_entry(kept_note), _note_entry(deleted_note)]
        folder = _create_folder(client, entries)
        assert folder.status_code == 200, folder.text
        assert client.delete(f"/api/v1/notes/{deleted_note}/delete").status_code == 200

        attached = _attached_after_listing(client, folder.json()["id"])

    assert attached == [_note_entry(kept_note)], (
        f"the folder still hands {attached} to the model after one of its notes was deleted; "
        "note entries must be checked like files and collections (#26739)"
    )


def test_a_note_whose_share_was_revoked_leaves_the_folder_knowledge(owner, author):
    with author.client() as client:
        shared_note = _create_note(client, [_read_grant(owner)])
    with owner.client() as client:
        folder = _create_folder(client, [_note_entry(shared_note)])
        assert folder.status_code == 200, folder.text
    with author.client() as client:
        revoked = client.post(
            f"/api/v1/notes/{shared_note}/access/update", json={"access_grants": []}
        )
        assert revoked.status_code == 200, revoked.text

    with owner.client() as client:
        attached = _attached_after_listing(client, folder.json()["id"])

    assert attached == [], (
        f"a note whose share was revoked is still attached for the former reader: {attached} "
        "(#26739)"
    )


# broad: no entry type the user cannot read can be attached


@pytest.mark.parametrize("surface", ["create", "update"])
@pytest.mark.parametrize("entry_type", ["note", "file", "collection"])
def test_attaching_an_entry_the_user_cannot_read_is_refused(owner, author, entry_type, surface):
    with author.client() as client:
        private_note = _create_note(client)
    unreadable = {"type": entry_type, "id": private_note if entry_type == "note" else "no-such-id"}

    with owner.client() as client:
        if surface == "create":
            attached = _create_folder(client, [unreadable])
        else:
            folder = _create_folder(client, [])
            assert folder.status_code == 200, folder.text
            attached = client.post(
                f"/api/v1/folders/{folder.json()['id']}/update",
                json={"data": {"files": [unreadable]}},
            )

    assert attached.status_code == 403, (
        f"a folder {surface} attached a {entry_type} the user cannot read (HTTP "
        f"{attached.status_code}); its content would reach the model as folder knowledge (#26739)"
    )


# nearby: notes the user may read stay attached, admins are not filtered


def test_a_note_shared_with_the_user_stays_in_the_folder_knowledge(owner, author):
    with author.client() as client:
        shared_note = _create_note(client, [_read_grant(owner)])
    with owner.client() as client:
        own_note = _create_note(client)
        entries = [_note_entry(shared_note), _note_entry(own_note)]
        folder = _create_folder(client, entries)
        assert folder.status_code == 200, folder.text

        attached = _attached_after_listing(client, folder.json()["id"])

    assert attached == entries, f"a readable note was dropped from the folder: {attached}"


@pytest.fixture
def collection_shared_with(admin):
    """`collection_shared_with(reader)` is an admin's knowledge base the reader may read."""
    created_ids = []

    def create(reader) -> str:
        with admin.client() as client:
            created = client.post(
                "/api/v1/knowledge/create",
                json={
                    "name": "shared kb",
                    "description": "",
                    "access_grants": [_read_grant(reader)],
                },
            )
        assert created.status_code == 200, created.text
        created_ids.append(created.json()["id"])
        return created_ids[-1]

    yield create
    with admin.client() as client:
        for knowledge_id in created_ids:
            client.delete(f"/api/v1/knowledge/{knowledge_id}/delete")


def test_readable_entries_of_every_type_stay_in_the_folder_knowledge(owner, collection_shared_with):
    with owner.client() as client:
        uploaded = client.post(
            "/api/v1/files/",
            params={"process": "false"},
            files={"file": ("notes.txt", b"plain text", "text/plain")},
        )
        assert uploaded.status_code == 200, uploaded.text
        entries = [
            {"type": "file", "id": uploaded.json()["id"]},
            {"type": "collection", "id": collection_shared_with(owner)},
            _note_entry(_create_note(client)),
        ]
        folder = _create_folder(client, entries)
        assert folder.status_code == 200, folder.text

        attached = _attached_after_listing(client, folder.json()["id"])

    assert attached == entries, f"a readable entry was dropped from the folder: {attached}"


def test_an_admin_may_attach_another_users_private_note(admin, author):
    with author.client() as client:
        private_note = _create_note(client)

    with admin.client() as client:
        folder = _create_folder(client, [_note_entry(private_note)])
        if folder.status_code == 200:
            client.delete(f"/api/v1/folders/{folder.json()['id']}")

    assert folder.status_code == 200, (
        f"an admin, who can read every note, could not attach one: {folder.text}"
    )
