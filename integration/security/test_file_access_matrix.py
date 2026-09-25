"""Journey: who may open, edit, rename and delete a file shared through a knowledge base.

A user uploads a text file into a knowledge base of theirs and shares the knowledge base with a
reader (read) and a writer (read and write), directly or through a group. A stranger is refused
the file everywhere; the reader may open it, its content and its extracted text but not change,
rename, reprocess or delete it; the writer and the admin may do all of it. Refusals answer 404,
so a file id cannot be probed. Every refused write leaves the owner's file as it was. A batch
reprocess is judged per file: the caller's own file is processed and the owner's is refused.

Discriminates: in a backend copy, asking for `read` instead of the requested access in the
knowledge branch of `has_access_to_file` turns the rename, content update and delete rows red
(the reader gets 200 and the owner's file changes), and dropping the ownership and access check
from the file content route turns its rows red (the stranger downloads the file). Dropping the
access check from `/retrieval/process/file` turns its row red (the stranger rewrites the file's
text) and dropping the ownership check from the batch route turns both batch refusals red.
"""

from __future__ import annotations

import httpx
import pytest

from harness.access import Shareable, attempts, cast, create_shared, reads

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source]


def _file_in_shared_knowledge(client: httpx.Client, grants: list[dict]) -> str:
    uploaded = client.post(
        "/api/v1/files/",
        params={"process": "true", "process_in_background": "false"},
        files={"file": ("plan.txt", b"the owner's plan", "text/plain")},
    )
    assert uploaded.status_code == 200, uploaded.text
    file_id = uploaded.json()["id"]
    created = client.post("/api/v1/knowledge/create", json={"name": "Plans", "description": ""})
    assert created.status_code == 200, created.text
    knowledge_id = created.json()["id"]
    added = client.post(f"/api/v1/knowledge/{knowledge_id}/file/add", json={"file_id": file_id})
    assert added.status_code == 200, added.text
    shared = client.post(
        f"/api/v1/knowledge/{knowledge_id}/access/update", json={"access_grants": grants}
    )
    assert shared.status_code == 200, shared.text
    return file_id


FILE = Shareable(
    create_shared=_file_in_shared_knowledge,
    owner_permissions={"workspace": {"knowledge": True}},
)
OWNERS_FILE = reads("/api/v1/files/{id}", "/api/v1/files/{id}/content")

REFUSED, ALLOWED = 404, 200
READ = {"owner": ALLOWED, "stranger": REFUSED, "reader": ALLOWED, "writer": ALLOWED}
WRITE = {"owner": ALLOWED, "stranger": REFUSED, "reader": REFUSED, "writer": ALLOWED}

# method, path, body, what each account gets
MATRIX = [
    ("GET", "/api/v1/files/{id}", None, READ),
    ("GET", "/api/v1/files/{id}/content", None, READ),
    ("GET", "/api/v1/files/{id}/data/content", None, READ),
    ("POST", "/api/v1/files/{id}/data/content/update", {"content": "rewritten"}, WRITE),
    ("POST", "/api/v1/files/{id}/rename", {"filename": "renamed.txt"}, WRITE),
    ("DELETE", "/api/v1/files/{id}", None, WRITE),
    (
        "POST",
        "/api/v1/retrieval/process/file",
        lambda actor, fields: {"file_id": fields["id"], "content": "rewritten"},
        WRITE,
    ),
]


@pytest.mark.parametrize("via", ["user", "group"])
@pytest.mark.parametrize(
    "method, path, body, expected", MATRIX, ids=[f"{row[0]} {row[1]}" for row in MATRIX]
)
def test_each_account_gets_what_the_knowledge_grant_allows(
    method, path, body, expected, via, admin, make_user
):
    accounts = cast(FILE, admin, make_user, via=via)

    answered = attempts(accounts, method, path, body, look=OWNERS_FILE)

    assert {role: attempt.status for role, attempt in answered.items()} == {
        **expected,
        "admin": ALLOWED,
    }, f"{method} {path} shared via {via}"
    for role, attempt in answered.items():
        if attempt.status != ALLOWED:
            assert attempt.after == attempt.before, f"a refused {role} changed the owner's file"


def _own_file(client: httpx.Client, text: str) -> dict:
    uploaded = client.post(
        "/api/v1/files/",
        params={"process": "true", "process_in_background": "false"},
        files={"file": ("mine.txt", text.encode(), "text/plain")},
    )
    assert uploaded.status_code == 200, uploaded.text
    return client.get(f"/api/v1/files/{uploaded.json()['id']}").json()


def _batch(client: httpx.Client, collection: str, *files: dict) -> dict:
    processed = client.post(
        "/api/v1/retrieval/process/files/batch",
        json={"collection_name": collection, "files": list(files)},
    )
    assert processed.status_code == 200, processed.text
    body = processed.json()
    return {entry["file_id"]: entry["status"] for entry in body["results"] + body["errors"]}


@pytest.mark.parametrize("role", ["stranger", "reader"])
def test_a_batch_processes_the_callers_own_file_and_refuses_the_owners(role, admin, make_user):
    accounts = cast(FILE, admin, make_user)
    owners_file_id = create_shared(accounts)
    with accounts.owner.client() as owner_client:
        before = OWNERS_FILE(owner_client, {"id": owners_file_id})

    with accounts.actor(role).client() as client:
        own = _own_file(client, "my own notes")
        disguised = {**own, "id": owners_file_id, "data": {"content": "rewritten"}}
        statuses = _batch(client, f"file-{own['id']}", own, disguised)
    with accounts.owner.client() as owner_client:
        after = OWNERS_FILE(owner_client, {"id": owners_file_id})

    assert statuses == {own["id"]: "completed", owners_file_id: "failed"}, statuses
    assert after == before, f"a refused {role}'s batch changed the owner's file"


def test_an_admin_batch_processes_a_users_file(admin, make_user):
    with make_user().client() as client:
        users_file = _own_file(client, "a user's notes")

    with admin.client() as client:
        statuses = _batch(client, f"file-{users_file['id']}", users_file)

    assert statuses == {users_file["id"]: "completed"}
