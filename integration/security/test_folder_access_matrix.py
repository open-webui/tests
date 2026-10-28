"""Journey: who may open, rename, move, share and list the chats of a shared folder.

The owner shares a folder with a reader (read) and a writer (read and write), directly or through
a group. A stranger is refused every route; the reader may open the folder and list its chats
(read-only) but not rename, move or re-share it; the writer and the admin may rename it, and only
the owner and the admin may re-share it (8145774e3). Moving stays the owner's alone: the move
only looks in the caller's own folder tree, so it answers 404 to everyone else, the admin
included. After a refused write the owner still sees the same name, parent and grants. With
folders switched off every route refuses everyone, and without `features.folders` every route
refuses a user but not the admin. Deleting is pinned by test_folder_delete_ownership.py.

Since 0.12.4 (226207314) the owner may also rename and move a chat a member started in the
shared folder. The rename takes a title alone, trimmed, and refuses a blank one with 400 and
anything more (history, other chat fields, variables, a folder) with 401. The move stays inside
the shared folder's tree: a private folder above it, any other folder of the owner's or the
member's, and no folder at all are refused. Other members, a stranger and the admin get neither,
and the owner's move leaves the member's chat as read or unread as it was.

Discriminates: in a backend copy, asking for `read` instead of `write` in the folder rename
handler's shared-access check turns the `/update` rows red (the reader gets 200 and the name
changes), and dropping the owner-admin-or-write check from the access update handler turns the
`/access/update` rows red (the stranger and the reader get 200 and the grants are gone), and
restoring the write-grant branch there turns the writer's `/access/update` row red. The member's
chat tests pass on dev 948432b43; in backend copies of it, refusing the owner in the chat update
handler turns the rename and blank title tests red, `has_chat_move_access` answering False turns
the move within and unread tests red, dropping the title-only check turns every
more-than-the-title row red, dropping the trim turns the rename test red, dropping the blank check
turns the blank title rows red, taking the topmost folder as the tree whatever is shared turns
the two private parent rows red, letting any folder owner pass the tree walk turns every other
account's move row red, looking the folder up without its owner in the update handler turns every
other account's rename row red, and marking the chat read on any move turns the unread test red.
"""

from __future__ import annotations

import uuid

import httpx
import pytest

from harness.access import ROLES, Shareable, attempts, cast, grant
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
SHARE = {"owner", "admin"}


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
    ("POST", "/api/v1/folders/{id}/access/update", {"access_grants": []}, None, 403, SHARE),
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


MEMBERS_HISTORY = {
    "currentId": "a1",
    "messages": {
        "q1": {
            "id": "q1",
            "parentId": None,
            "childrenIds": ["a1"],
            "role": "user",
            "content": "when is high tide?",
        },
        "a1": {
            "id": "a1",
            "parentId": "q1",
            "childrenIds": [],
            "role": "assistant",
            "content": "High tide at noon.",
            "done": True,
        },
    },
}
REFUSED_CHAT = 401


def _folder(owner: Actor, parent_id: str | None = None) -> str:
    form = FOLDER.create_body()
    if parent_id:
        form["parent_id"] = parent_id
    with owner.client() as client:
        created = client.post("/api/v1/folders/", json=form)
    assert created.status_code == 200, created.text
    return created.json()["id"]


def _share(owner: Actor, folder_id: str, grants: list[dict]) -> None:
    with owner.client() as client:
        shared = client.post(
            FOLDER.access_path.format(id=folder_id), json={"access_grants": grants}
        )
    assert shared.status_code == 200, shared.text


@pytest.fixture
def harbour(admin, make_user):
    """The owner's private folder holding a shared folder with a sub-folder, and a member's chat.

    The writer starts the chat and files it in the shared folder; a second writer, the reader, a
    stranger and the admin stand by.
    """
    accounts = cast(FOLDER, admin, make_user)
    second_writer = make_user()
    private_parent = _folder(accounts.owner)
    shared = _folder(accounts.owner, parent_id=private_parent)
    sub_folder = _folder(accounts.owner, parent_id=shared)
    _share(
        accounts.owner,
        shared,
        accounts.grants
        + [grant("user", second_writer.id, "read"), grant("user", second_writer.id, "write")],
    )
    with accounts.writer.client() as client:
        created = client.post(
            "/api/v1/chats/new",
            json={"chat": {"title": "Tide log", "history": MEMBERS_HISTORY}},
        )
        assert created.status_code == 200, created.text
        chat_id = created.json()["id"]
        filed = client.post(f"/api/v1/chats/{chat_id}/folder", json={"folder_id": shared})
    assert filed.status_code == 200, filed.text
    return {
        "accounts": accounts,
        "second_writer": second_writer,
        "private_parent": private_parent,
        "shared": shared,
        "sub_folder": sub_folder,
        "chat_id": chat_id,
    }


def _members_view(harbour: dict) -> dict:
    with harbour["accounts"].writer.client() as client:
        read = client.get(f"/api/v1/chats/{harbour['chat_id']}")
        listed = client.get(
            "/api/v1/chats/", params={"include_folders": True, "include_pinned": True}
        )
    assert read.status_code == 200, read.text
    assert listed.status_code == 200, listed.text
    body = read.json()
    [in_sidebar] = [chat for chat in listed.json() if chat["id"] == harbour["chat_id"]]
    return {
        "title": body["title"],
        "folder_id": body["folder_id"],
        "history": body["chat"]["history"],
        "variables": body["chat"].get("variables"),
        "last_read_at": in_sidebar["last_read_at"],
    }


def _rename(actor: Actor, chat_id: str, body: dict) -> httpx.Response:
    with actor.client() as client:
        return client.post(f"/api/v1/chats/{chat_id}", json=body)


def _move(actor: Actor, chat_id: str, folder_id: str | None) -> httpx.Response:
    with actor.client() as client:
        return client.post(f"/api/v1/chats/{chat_id}/folder", json={"folder_id": folder_id})


def test_the_owner_renames_a_members_chat_in_their_shared_folder(harbour):
    before = _members_view(harbour)

    renamed = _rename(
        harbour["accounts"].owner, harbour["chat_id"], {"chat": {"title": "  Low tide  "}}
    )

    assert renamed.status_code == OK, renamed.text
    after = _members_view(harbour)
    assert after["title"] == "Low tide", f"the member sees the title {after['title']!r}"
    assert after["history"] == before["history"], "the owner's rename changed the conversation"
    assert after["folder_id"] == harbour["shared"]


@pytest.mark.parametrize("title", ["   ", "", None, 7], ids=["spaces", "empty", "null", "number"])
def test_a_blank_title_from_the_owner_is_refused(harbour, title):
    renamed = _rename(harbour["accounts"].owner, harbour["chat_id"], {"chat": {"title": title}})

    assert renamed.status_code == 400, f"the owner's title {title!r} got {renamed.status_code}"
    assert _members_view(harbour)["title"] == "Tide log"


TAMPERED_HISTORY = {
    "currentId": "a1",
    "messages": {
        **MEMBERS_HISTORY["messages"],
        "a1": {**MEMBERS_HISTORY["messages"]["a1"], "content": "the owner's words"},
    },
}

BEYOND_THE_TITLE = {
    "history": {"chat": {"title": "Low tide", "history": TAMPERED_HISTORY}},
    "history alone": {"chat": {"history": TAMPERED_HISTORY}},
    "another chat field": {"chat": {"title": "Low tide", "models": ["someone-elses-model"]}},
    "variables": {"chat": {"title": "Low tide"}, "variables": {"{{tide}}": "low"}},
    "folder": {"chat": {"title": "Low tide"}, "folder_id": "elsewhere"},
}


@pytest.mark.parametrize("body", BEYOND_THE_TITLE.values(), ids=BEYOND_THE_TITLE.keys())
def test_the_owner_cannot_change_more_than_the_title_of_a_members_chat(harbour, body):
    before = _members_view(harbour)

    changed = _rename(harbour["accounts"].owner, harbour["chat_id"], body)

    assert changed.status_code == REFUSED_CHAT, f"the owner sent {body} and got {changed.text}"
    assert _members_view(harbour) == before, "a refused change from the owner was stored"


OTHERS = ["reader", "second writer", "stranger", "admin"]


def _other(harbour: dict, role: str) -> Actor:
    if role == "second writer":
        return harbour["second_writer"]
    return harbour["accounts"].actor(role)


@pytest.mark.parametrize("role", OTHERS)
def test_no_one_but_the_folder_owner_renames_a_members_chat(harbour, role):
    renamed = _rename(_other(harbour, role), harbour["chat_id"], {"chat": {"title": "Low tide"}})

    assert renamed.status_code == REFUSED_CHAT, f"the {role} renamed the chat: {renamed.text}"
    assert _members_view(harbour)["title"] == "Tide log"


def test_the_owner_moves_a_members_chat_within_the_shared_folder(harbour):
    owner, chat_id = harbour["accounts"].owner, harbour["chat_id"]

    down = _move(owner, chat_id, harbour["sub_folder"])
    assert down.status_code == OK, down.text
    assert _members_view(harbour)["folder_id"] == harbour["sub_folder"]

    back = _move(owner, chat_id, harbour["shared"])
    assert back.status_code == OK, back.text
    assert _members_view(harbour)["folder_id"] == harbour["shared"]


def _outside_targets(harbour: dict) -> dict[str, str | None]:
    accounts = harbour["accounts"]
    owner = accounts.owner
    other_shared = _folder(owner)
    _share(owner, other_shared, accounts.grants)
    return {
        "the private parent": harbour["private_parent"],
        "a folder beside it in the private parent": _folder(owner, harbour["private_parent"]),
        "the owner's other private folder": _folder(owner),
        "the owner's other shared folder": other_shared,
        "the member's own folder": _folder(accounts.writer),
        "no folder": None,
    }


OUTSIDE = [
    "the private parent",
    "a folder beside it in the private parent",
    "the owner's other private folder",
    "the owner's other shared folder",
    "the member's own folder",
    "no folder",
]


@pytest.mark.parametrize("target", OUTSIDE)
def test_the_owner_cannot_move_a_members_chat_out_of_the_shared_folder(harbour, target):
    target_id = _outside_targets(harbour)[target]

    moved = _move(harbour["accounts"].owner, harbour["chat_id"], target_id)

    assert moved.status_code == REFUSED_CHAT, f"the owner moved it to {target}: {moved.text}"
    assert _members_view(harbour)["folder_id"] == harbour["shared"]


@pytest.mark.parametrize("role", OTHERS)
def test_no_one_but_the_folder_owner_moves_a_members_chat(harbour, role):
    moved = _move(_other(harbour, role), harbour["chat_id"], harbour["sub_folder"])

    assert moved.status_code == REFUSED_CHAT, f"the {role} moved the chat: {moved.text}"
    assert _members_view(harbour)["folder_id"] == harbour["shared"]


def test_a_move_by_the_owner_leaves_the_members_chat_unread(harbour):
    member, chat_id = harbour["accounts"].writer, harbour["chat_id"]
    with member.client() as client:
        marked = client.post(f"/api/v1/chats/{chat_id}/unread")
    assert marked.status_code == OK, marked.text
    assert _members_view(harbour)["last_read_at"] == 0

    moved = _move(harbour["accounts"].owner, chat_id, harbour["sub_folder"])

    assert moved.status_code == OK, moved.text
    after = _members_view(harbour)
    assert after["last_read_at"] == 0, f"the owner's move marked it read at {after['last_read_at']}"
    # the member's own move does mark it read
    assert _move(member, chat_id, harbour["shared"]).status_code == OK
    assert _members_view(harbour)["last_read_at"] > 0
