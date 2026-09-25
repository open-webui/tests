"""Journey: who may open, rename, move, share and list the chats of a shared folder.

The owner shares a folder with a reader (read) and a writer (read and write), directly or through
a group. A stranger is refused every route; the reader may open the folder and list its chats
(read-only) but not rename, move or re-share it; the writer and the admin may rename and re-share
it. Moving stays the owner's alone: the move only looks in the caller's own folder tree, so it
answers 404 to everyone else, the admin included. After a refused write the owner still sees the
same name, parent and grants. With folders switched off every route refuses everyone, and without
`features.folders` every route refuses a user but not the admin. Deleting is pinned by
test_folder_delete_ownership.py.

Discriminates: in a backend copy, asking for `read` instead of `write` in the folder rename
handler's shared-access check turns the `/update` rows red (the reader gets 200 and the name
changes), and dropping the owner-admin-or-write check from the access update handler turns the
`/access/update` rows red (the stranger and the reader get 200 and the grants are gone).
"""

from __future__ import annotations

import uuid

import httpx
import pytest

from harness.access import ROLES, Shareable, attempts, cast
from harness.actors import Actor

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source]

FOLDER = Shareable(
    create_path="/api/v1/folders/",
    create_body=lambda: {"name": f"shared {uuid.uuid4().hex[:8]}"},
    access_path="/api/v1/folders/{id}/access/update",
)
CONVERSATION = {"title": "Filed", "history": {"currentId": None, "messages": {}}}

OK = 200
READ = {"owner", "reader", "writer", "admin"}
WRITE = {"owner", "writer", "admin"}
OWNER = {"owner"}


def _another_owner_folder(owner: Actor, folder_id: str) -> dict:
    with owner.client() as client:
        created = client.post("/api/v1/folders/", json=FOLDER.create_body())
    assert created.status_code == 200, created.text
    return {"target_id": created.json()["id"]}


def _owner_chat_inside(owner: Actor, folder_id: str) -> dict:
    with owner.client() as client:
        created = client.post("/api/v1/chats/new", json={"chat": CONVERSATION})
        assert created.status_code == 200, created.text
        chat_id = created.json()["id"]
        moved = client.post(f"/api/v1/chats/{chat_id}/folder", json={"folder_id": folder_id})
    assert moved.status_code == 200, moved.text
    return {"chat_id": chat_id}


def _fresh_name(actor: Actor, fields: dict) -> dict:
    return FOLDER.create_body()


def _owner_view(client: httpx.Client, fields: dict) -> dict:
    folder = client.get(f"/api/v1/folders/{fields['id']}")
    assert folder.status_code == 200, folder.text
    body = folder.json()
    return {
        "name": body["name"],
        "parent_id": body["parent_id"],
        "grants": sorted(
            (entry["principal_id"], entry["permission"]) for entry in body["access_grants"]
        ),
    }


# method, path, body, setup, the route's refusal code, who gets through
MATRIX = [
    ("GET", "/api/v1/folders/{id}", None, None, 404, READ),
    ("GET", "/api/v1/folders/{id}/shared/chats", None, _owner_chat_inside, 403, READ),
    ("POST", "/api/v1/folders/{id}/update", _fresh_name, None, 404, WRITE),
    (
        "POST",
        "/api/v1/folders/{id}/update/parent",
        lambda actor, fields: {"parent_id": fields["target_id"]},
        _another_owner_folder,
        404,
        OWNER,
    ),
    ("POST", "/api/v1/folders/{id}/access/update", {"access_grants": []}, None, 403, WRITE),
]


@pytest.mark.parametrize("via", ["user", "group"])
@pytest.mark.parametrize(
    "method, path, body, setup, refused, allowed",
    MATRIX,
    ids=[f"{row[0]} {row[1].removeprefix('/api/v1/folders/')}" for row in MATRIX],
)
def test_each_account_gets_what_its_grant_allows(
    method, path, body, setup, refused, allowed, via, admin, make_user
):
    accounts = cast(FOLDER, admin, make_user, via=via)

    answered = attempts(accounts, method, path, body, setup=setup, look=_owner_view)

    assert {role: attempt.status for role, attempt in answered.items()} == {
        role: OK if role in allowed else refused for role in ROLES
    }, f"{method} {path} shared via {via}"
    for role, attempt in answered.items():
        if attempt.status != OK:
            assert attempt.after == attempt.before, f"a refused {role} changed the folder"


def test_a_reader_lists_the_owners_chats_read_only(admin, make_user):
    accounts = cast(FOLDER, admin, make_user)
    with accounts.owner.client() as client:
        created = client.post(FOLDER.create_path, json=FOLDER.create_body())
        folder_id = created.json()["id"]
        client.post(
            FOLDER.access_path.format(id=folder_id), json={"access_grants": accounts.grants}
        ).raise_for_status()
    chat_id = _owner_chat_inside(accounts.owner, folder_id)["chat_id"]

    with accounts.reader.client() as client:
        listed = client.get(f"/api/v1/folders/{folder_id}/shared/chats")

    assert listed.status_code == 200, listed.text
    assert listed.json()["folder_permission"] == "read"
    assert [(chat["id"], chat["readonly"]) for chat in listed.json()["chats"]] == [(chat_id, True)]


def _set_admin_config(admin: Actor, **changes) -> None:
    with admin.client() as client:
        config = client.get("/api/v1/auths/admin/config").json()
        client.post("/api/v1/auths/admin/config", json={**config, **changes}).raise_for_status()


def _set_folder_permission(admin: Actor, permitted: bool) -> None:
    with admin.client() as client:
        permissions = client.get("/api/v1/users/default/permissions").json()
        permissions["features"]["folders"] = permitted
        client.post("/api/v1/users/default/permissions", json=permissions).raise_for_status()


@pytest.mark.parametrize("switch", ["feature off", "permission off"])
@pytest.mark.parametrize(
    "method, path, body, setup, refused, allowed",
    MATRIX,
    ids=[f"{row[0]} {row[1].removeprefix('/api/v1/folders/')}" for row in MATRIX],
)
def test_switched_off_folders_refuse_every_route(
    method, path, body, setup, refused, allowed, switch, admin, preserve, make_user
):
    owner = make_user()
    with owner.client() as client:
        folder_id = client.post(FOLDER.create_path, json=FOLDER.create_body()).json()["id"]
    fields = {"id": folder_id, **(setup(owner, folder_id) if setup else {})}
    preserve("admin_config", "permissions")
    if switch == "feature off":
        _set_admin_config(admin, ENABLE_FOLDERS=False)
    else:
        _set_folder_permission(admin, False)

    answered = {}
    for role, actor in (("owner", owner), ("admin", admin)):
        payload = body(actor, fields) if callable(body) else body
        with actor.client() as client:
            response = client.request(method, path.format(**fields), json=payload)
        answered[role] = response.status_code

    # without the permission the admin still meets the route's own check
    admin_answer = OK if "admin" in allowed else refused
    assert answered == {"owner": 403, "admin": 403 if switch == "feature off" else admin_answer}
