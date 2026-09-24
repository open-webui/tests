"""Regression: a folder's knowledge stays bounded by what its OWNER can read.

open-webui 0.11.0 fix `56183fcb1` (PR #27464), two halves. `POST /api/v1/folders/{id}/update`
checked attached files against the caller, so a collaborator with write access on someone
else's folder could park a private file there, and an admin acting on the folder passed for any
file; `POST /api/v1/folders/` with a shared `parent_id` had no check at all, and entries of an
unknown type or without an id were kept unchecked. When a chat in the folder read its knowledge,
both the native path (the `<attached_knowledge>` list sent to the model) and the legacy RAG path
used the stored entries as they were, so a knowledge base the owner had since lost access to
kept feeding the chat. The fix requires the owner to read every attached entry and re-checks
the entries against the owner whenever a chat uses them.

Twin of unit/security/test_shared_folder_file_access.py.

Discriminates: passes on dev bbfa876af, fails with the fix reverted (update checked against the
caller, the subfolder owner check removed, unknown entries kept, folder knowledge taken from the
stored entries in middleware and retrieval): the collaborator, admin, subfolder and unknown-entry
attaches return 200, and the revoked knowledge base reaches the model on both chat paths.
"""

from __future__ import annotations

import re
import uuid

import httpx
import pytest

from harness.chat import ask

pytestmark = [pytest.mark.regression, pytest.mark.api, pytest.mark.requires_source]


def _upload(client: httpx.Client, content: bytes, process: bool = False) -> str:
    query = "process=true&process_in_background=false" if process else "process=false"
    uploaded = client.post(
        f"/api/v1/files/?{query}", files={"file": ("notes.txt", content, "text/plain")}
    )
    assert uploaded.status_code == 200, uploaded.text
    return uploaded.json()["id"]


def _create_folder(client: httpx.Client, **fields) -> httpx.Response:
    return client.post(
        "/api/v1/folders/", json={"name": f"folder-{uuid.uuid4().hex[:8]}", **fields}
    )


def _folder_of(owner) -> str:
    with owner.client() as client:
        created = _create_folder(client)
    assert created.status_code == 200, created.text
    return created.json()["id"]


def _shared_folder(owner, collaborator) -> str:
    """A folder of `owner` that `collaborator` may write to."""
    folder_id = _folder_of(owner)
    grant = {"principal_type": "user", "principal_id": collaborator.id, "permission": "write"}
    with owner.client() as client:
        shared = client.post(
            f"/api/v1/folders/{folder_id}/access/update", json={"access_grants": [grant]}
        )
    assert shared.status_code == 200, shared.text
    return folder_id


def _attach(client: httpx.Client, folder_id: str, entries: list) -> httpx.Response:
    return client.post(f"/api/v1/folders/{folder_id}/update", json={"data": {"files": entries}})


def _files_in_folder(owner, folder_id: str) -> list:
    with owner.client() as client:
        folder = client.get(f"/api/v1/folders/{folder_id}")
    assert folder.status_code == 200, folder.text
    return (folder.json().get("data") or {}).get("files") or []


def _subfolders_of(owner, folder_id: str) -> list[dict]:
    with owner.client() as client:
        folders = client.get("/api/v1/folders/")
    assert folders.status_code == 200, folders.text
    return [folder for folder in folders.json() if folder.get("parent_id") == folder_id]


def _read_grants(*readers) -> list[dict]:
    return [
        {"principal_type": "user", "principal_id": reader.id, "permission": "read"}
        for reader in readers
    ]


@pytest.fixture
def knowledge_base(admin):
    """`knowledge_base(readers, document)`: an admin's knowledge base, deleted afterwards."""
    client = admin.client()
    created_ids: list[str] = []

    def create(readers: list, document: str = "") -> str:
        created = client.post(
            "/api/v1/knowledge/create",
            json={
                "name": f"kb-{uuid.uuid4().hex[:8]}",
                "description": "",
                "access_grants": _read_grants(*readers),
            },
        )
        assert created.status_code == 200, created.text
        knowledge_id = created.json()["id"]
        created_ids.append(knowledge_id)
        if document:
            file_id = _upload(client, document.encode(), process=True)
            added = client.post(
                f"/api/v1/knowledge/{knowledge_id}/file/add", json={"file_id": file_id}
            )
            assert added.status_code == 200, added.text
        return knowledge_id

    yield create
    for knowledge_id in created_ids:
        client.delete(f"/api/v1/knowledge/{knowledge_id}/delete")
    client.close()


def _revoke_all_grants(admin, knowledge_id: str) -> None:
    with admin.client() as client:
        revoked = client.post(
            f"/api/v1/knowledge/{knowledge_id}/access/update", json={"access_grants": []}
        )
    assert revoked.status_code == 200, revoked.text


def _collections(*knowledge_ids: str) -> list[dict]:
    return [{"type": "collection", "id": knowledge_id} for knowledge_id in knowledge_ids]


# ---------------------------------------------------------------- narrow: attaching


def test_collaborator_cannot_attach_a_file_the_owner_cannot_read(make_user):
    owner, collaborator = make_user(), make_user()
    folder_id = _shared_folder(owner, collaborator)

    with collaborator.client() as client:
        private_file_id = _upload(client, b"private notes")
        attached = _attach(client, folder_id, [{"type": "file", "id": private_file_id}])

    assert attached.status_code == 403, (
        f"a collaborator parked a private file in someone else's folder (#27464): "
        f"HTTP {attached.status_code} {attached.text}"
    )
    assert _files_in_folder(owner, folder_id) == []


def test_admin_cannot_attach_a_file_the_folder_owner_cannot_read(make_user, admin):
    owner, other_user = make_user(), make_user()
    folder_id = _folder_of(owner)
    with other_user.client() as client:
        private_file_id = _upload(client, b"private notes")

    with admin.client() as client:
        attached = _attach(client, folder_id, [{"type": "file", "id": private_file_id}])

    assert attached.status_code == 403, (
        f"an admin widened a folder beyond what its owner can read (#27464): "
        f"HTTP {attached.status_code} {attached.text}"
    )
    assert _files_in_folder(owner, folder_id) == []


def test_collaborator_cannot_seed_a_subfolder_with_a_file_the_owner_cannot_read(make_user):
    owner, collaborator = make_user(), make_user()
    folder_id = _shared_folder(owner, collaborator)

    with collaborator.client() as client:
        private_file_id = _upload(client, b"private notes")
        created = _create_folder(
            client,
            parent_id=folder_id,
            data={"files": [{"type": "file", "id": private_file_id}]},
        )

    assert created.status_code == 403, (
        f"a subfolder owned by the folder owner was pre-loaded with a file the owner cannot "
        f"read (#27464): HTTP {created.status_code} {created.text}"
    )
    assert _subfolders_of(owner, folder_id) == []


# ---------------------------------------------------------------- narrow: chatting


def test_a_chat_in_the_folder_lists_only_knowledge_the_owner_still_reads(
    make_user, admin, knowledge_base, upstream
):
    """Native tool calling: the model is told which knowledge it may search."""
    owner = make_user()
    kept, revoked = knowledge_base([owner]), knowledge_base([owner])
    folder_id = _folder_of(owner)
    with owner.client() as client:
        assert _attach(client, folder_id, _collections(kept, revoked)).status_code == 200
    _revoke_all_grants(admin, revoked)

    with owner.client() as client:
        ask(client, "what do my notes say?", folder_id=folder_id)

    prompt = str(upstream.chat_requests()[-1]["messages"])
    listed = re.findall(r'<knowledge type="collection" id="([^"]+)"', prompt)
    assert kept in listed, f"the folder's readable knowledge was not offered: {prompt}"
    assert revoked not in listed, (
        "a chat in the folder offered the model a knowledge base its owner can no longer read "
        "(#27464)"
    )


def test_folder_retrieval_skips_knowledge_the_owner_can_no_longer_read(
    make_user, admin, knowledge_base, upstream
):
    """Legacy tool calling: the folder's documents are retrieved into the prompt."""
    owner = make_user()
    kept = knowledge_base([owner], "The kept notes mention the harbour lighthouse.")
    revoked = knowledge_base([owner], "The revoked notes mention the salary table.")
    folder_id = _folder_of(owner)
    with owner.client() as client:
        assert _attach(client, folder_id, _collections(kept, revoked)).status_code == 200
    _revoke_all_grants(admin, revoked)

    with owner.client() as client:
        ask(
            client,
            "what do my notes mention?",
            folder_id=folder_id,
            params={"function_calling": "legacy"},
        )

    prompt = str(upstream.chat_requests()[-1]["messages"])
    assert "harbour lighthouse" in prompt, (
        f"the folder's readable document was not retrieved: {prompt}"
    )
    assert "salary table" not in prompt, (
        "a chat in the folder was fed a document from a knowledge base its owner can no longer "
        "read (#27464)"
    )


# ---------------------------------------------------------------- broad


@pytest.mark.parametrize(
    "entry",
    [{"type": "folder", "id": "nested"}, {"type": "file"}],
    ids=["unknown-type", "missing-id"],
)
def test_entries_that_cannot_be_access_checked_are_refused(make_user, entry):
    owner = make_user()
    folder_id = _folder_of(owner)
    with owner.client() as client:
        attached = _attach(client, folder_id, [entry])

    assert attached.status_code == 403, (
        f"an entry with no access check to run was stored as folder knowledge: "
        f"HTTP {attached.status_code} {attached.text}"
    )
    assert _files_in_folder(owner, folder_id) == []


# ---------------------------------------------------------------- nearby


def test_owner_can_attach_their_own_file(make_user):
    owner = make_user()
    folder_id = _folder_of(owner)
    with owner.client() as client:
        file_id = _upload(client, b"my notes")
        attached = _attach(client, folder_id, [{"type": "file", "id": file_id}])

    assert attached.status_code == 200, attached.text
    assert _files_in_folder(owner, folder_id) == [{"type": "file", "id": file_id}]


def test_collaborator_can_attach_a_knowledge_base_the_owner_can_read(make_user, knowledge_base):
    owner, collaborator = make_user(), make_user()
    folder_id = _shared_folder(owner, collaborator)
    knowledge_id = knowledge_base([owner, collaborator])

    with collaborator.client() as client:
        attached = _attach(client, folder_id, _collections(knowledge_id))

    assert attached.status_code == 200, (
        f"a collaborator was blocked from attaching something the owner can read: "
        f"HTTP {attached.status_code} {attached.text}"
    )
    assert _files_in_folder(owner, folder_id) == _collections(knowledge_id)


def test_collaborator_can_still_create_an_empty_subfolder(make_user):
    owner, collaborator = make_user(), make_user()
    folder_id = _shared_folder(owner, collaborator)

    with collaborator.client() as client:
        created = _create_folder(client, parent_id=folder_id)

    assert created.status_code == 200, created.text
    assert created.json()["user_id"] == owner.id
    assert [folder["id"] for folder in _subfolders_of(owner, folder_id)] == [created.json()["id"]]


def test_user_without_a_grant_cannot_update_the_folder(make_user):
    owner, stranger = make_user(), make_user()
    folder_id = _folder_of(owner)

    with stranger.client() as client:
        attached = _attach(client, folder_id, [])

    assert attached.status_code == 404
