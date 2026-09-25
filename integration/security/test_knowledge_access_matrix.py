"""Journey: who may find, read, change, share, fill, empty and delete a shared knowledge base.

An owner allowed to create knowledge bases shares one with a reader (read) and a writer (read
and write), directly or through a group. A stranger is refused every route and finds nothing in
the listing or the search; the reader opens the base and its files and changes nothing; the
writer and the admin may do all of it. After a refused write the owner still sees the same
name, grants and files. A public grant from an account without `sharing.public_knowledge` is
dropped, and an account without `workspace.knowledge` cannot create a base at all.

Discriminates: in a backend copy, dropping the owner-or-write check from the knowledge update
handler turns the `/update` rows red (the stranger and the reader get 200 and the name
changes), asking for `write` instead of `read` in the get handler turns the `GET /{id}` rows and
the kept public grant red (the reader and the public get 401), and skipping
`filter_allowed_access_grants` on create whenever grants are sent turns the dropped public grant
red (it is stored).
"""

from __future__ import annotations

import uuid

import httpx
import pytest

from harness.access import ROLES, Shareable, attempts, cast, grant
from harness.actors import Actor
from harness.knowledge_bases import knowledge_base

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source]

KNOWLEDGE = Shareable(
    create_path="/api/v1/knowledge/create",
    create_body=lambda: {"name": f"shared {uuid.uuid4().hex[:8]}", "description": "kept"},
    access_path="/api/v1/knowledge/{id}/access/update",
)
EVERYONE_READS = grant("user", "*", "read")

OK = 200
READ = {"owner", "reader", "writer", "admin"}
WRITE = {"owner", "writer", "admin"}


def _upload(actor: Actor, text: str) -> str:
    with actor.client() as client:
        uploaded = client.post(
            "/api/v1/files/",
            params={"process": "true", "process_in_background": "false"},
            files={"file": (f"{uuid.uuid4().hex[:8]}.txt", text.encode(), "text/plain")},
        )
    assert uploaded.status_code == 200, uploaded.text
    return uploaded.json()["id"]


def _with_owner_file(owner: Actor, knowledge_id: str) -> dict:
    file_id = _upload(owner, "The owner's filed document.")
    with owner.client() as client:
        added = client.post(f"/api/v1/knowledge/{knowledge_id}/file/add", json={"file_id": file_id})
    assert added.status_code == 200, added.text
    return {"file_id": file_id}


def _own_new_file(actor: Actor, fields: dict) -> dict:
    return {"file_id": _upload(actor, "A document of the sender's own.")}


def _owner_view(client: httpx.Client, fields: dict) -> dict | None:
    knowledge = client.get(f"/api/v1/knowledge/{fields['id']}")
    if knowledge.status_code != 200:
        return None
    files = client.get(f"/api/v1/knowledge/{fields['id']}/files")
    assert files.status_code == 200, files.text
    body = knowledge.json()
    return {
        "name": body["name"],
        "description": body["description"],
        "grants": sorted(
            (entry["principal_id"], entry["permission"]) for entry in body["access_grants"]
        ),
        "files": sorted(entry["id"] for entry in files.json()["items"]),
    }


# method, path, body, setup, the route's refusal code, who gets through
MATRIX = [
    ("GET", "/api/v1/knowledge/{id}", None, None, 401, READ),
    ("GET", "/api/v1/knowledge/{id}/files", None, None, 400, READ),
    (
        "POST",
        "/api/v1/knowledge/{id}/update",
        {"name": "renamed", "description": "changed"},
        None,
        400,
        WRITE,
    ),
    ("POST", "/api/v1/knowledge/{id}/access/update", {"access_grants": []}, None, 400, WRITE),
    ("POST", "/api/v1/knowledge/{id}/file/add", _own_new_file, None, 400, WRITE),
    (
        "POST",
        "/api/v1/knowledge/{id}/file/remove",
        lambda actor, fields: {"file_id": fields["file_id"]},
        _with_owner_file,
        400,
        WRITE,
    ),
    ("POST", "/api/v1/knowledge/{id}/reset", None, _with_owner_file, 400, WRITE),
    ("DELETE", "/api/v1/knowledge/{id}/delete", None, _with_owner_file, 400, WRITE),
]


def _set_permissions(admin: Actor, **sections: dict) -> None:
    with admin.client() as client:
        permissions = client.get("/api/v1/users/default/permissions").json()
        for section, flags in sections.items():
            permissions[section].update(flags)
        saved = client.post("/api/v1/users/default/permissions", json=permissions)
    assert saved.status_code == 200, saved.text


@pytest.fixture
def knowledge_allowed(admin, preserve):
    preserve("permissions")
    _set_permissions(admin, workspace={"knowledge": True}, sharing={"public_knowledge": False})


@pytest.mark.parametrize("via", ["user", "group"])
@pytest.mark.parametrize(
    "method, path, body, setup, refused, allowed",
    MATRIX,
    ids=[f"{row[0]} {row[1].removeprefix('/api/v1/knowledge/')}" for row in MATRIX],
)
def test_each_account_gets_what_its_grant_allows(
    method, path, body, setup, refused, allowed, via, knowledge_allowed, admin, make_user
):
    accounts = cast(KNOWLEDGE, admin, make_user, via=via)

    answered = attempts(accounts, method, path, body, setup=setup, look=_owner_view)

    assert {role: attempt.status for role, attempt in answered.items()} == {
        role: OK if role in allowed else refused for role in ROLES
    }, f"{method} {path} shared via {via}"
    for role, attempt in answered.items():
        if attempt.status != OK:
            assert attempt.after == attempt.before, f"a refused {role} changed the knowledge base"


@pytest.mark.parametrize("via", ["user", "group"])
@pytest.mark.parametrize("listing", ["list", "search"])
def test_the_listing_shows_own_and_granted_bases_only(
    listing, via, knowledge_allowed, admin, make_user
):
    accounts = cast(KNOWLEDGE, admin, make_user, via=via)
    name = f"findable {uuid.uuid4().hex[:8]}"
    with accounts.owner.client() as client:
        created = client.post(
            KNOWLEDGE.create_path,
            json={"name": name, "description": "", "access_grants": accounts.grants},
        )
    assert created.status_code == 200, created.text
    knowledge_id = created.json()["id"]

    seen = {}
    for role in ROLES:
        with accounts.actor(role).client() as client:
            if listing == "list":
                listed = client.get("/api/v1/knowledge/")
            else:
                listed = client.get("/api/v1/knowledge/search", params={"query": name})
        assert listed.status_code == 200, listed.text
        entry = next((item for item in listed.json()["items"] if item["id"] == knowledge_id), None)
        seen[role] = entry and entry["write_access"]

    assert seen == {"owner": True, "stranger": None, "reader": False, "writer": True, "admin": True}


def test_a_public_grant_without_the_permission_is_dropped(knowledge_allowed, make_user):
    owner, stranger = make_user(), make_user()
    with owner.client() as client:
        created = client.post(
            KNOWLEDGE.create_path,
            json={"name": "public?", "description": "", "access_grants": [EVERYONE_READS]},
        )
        assert created.status_code == 200, created.text
        knowledge_id = created.json()["id"]
        reshared = client.post(
            f"/api/v1/knowledge/{knowledge_id}/access/update",
            json={"access_grants": [EVERYONE_READS]},
        )
    assert reshared.status_code == 200, reshared.text

    assert created.json()["access_grants"] == []
    assert reshared.json()["access_grants"] == []
    with stranger.client() as client:
        assert client.get(f"/api/v1/knowledge/{knowledge_id}").status_code == 401
        listed = client.get("/api/v1/knowledge/").json()["items"]
    assert knowledge_id not in [item["id"] for item in listed]


def test_a_public_grant_with_the_permission_is_kept(knowledge_allowed, admin, make_user):
    _set_permissions(admin, sharing={"public_knowledge": True})
    owner, stranger = make_user(), make_user()
    # every account would see it, so it goes when the test ends
    with (
        owner.client() as client,
        knowledge_base(client, "public", [EVERYONE_READS]) as knowledge_id,
        stranger.client() as stranger_client,
    ):
        opened = stranger_client.get(f"/api/v1/knowledge/{knowledge_id}")

    assert opened.status_code == 200, opened.text
    assert opened.json()["write_access"] is False


def test_creating_needs_the_workspace_knowledge_permission(admin, preserve, make_user):
    preserve("permissions")
    _set_permissions(admin, workspace={"knowledge": False})
    account = make_user()

    with account.client() as client:
        refused = client.post(KNOWLEDGE.create_path, json=KNOWLEDGE.create_body())
        listed = client.get("/api/v1/knowledge/")

    assert refused.status_code == 401, refused.text
    assert [item for item in listed.json()["items"] if item["user_id"] == account.id] == []
    with make_user(role="admin").client() as client:
        assert client.post(KNOWLEDGE.create_path, json=KNOWLEDGE.create_body()).status_code == 200
