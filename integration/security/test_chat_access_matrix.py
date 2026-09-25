"""Journey: who may read, change, share and delete someone else's chat.

A chat belongs to its owner. The only grant a chat takes is the read share the share dialog
stores, so the reader and the writer (read plus write, stored the same way) may open the chat
and its tags and nothing more; a stranger is refused everything. An admin reaches every chat
while `ENABLE_ADMIN_CHAT_ACCESS` is on and is refused like a stranger on the read and owner
routes while it is off. Every refused write leaves the owner's copy as it was.

Discriminates: in a backend copy, reading the chat with `Chats.get_chat_by_id` in place of the
owner lookup in the update handler turns the `POST /api/v1/chats/{id}` rows red (the stranger,
reader and writer get 200), dropping the non-owner check from the message update handler turns
its rows red (the stranger rewrites the owner's message), and toggling the pin before the owner
check turns the pin rows red on the refused-but-changed check alone.
"""

from __future__ import annotations

import uuid

import pytest

from harness.access import Shareable, attempts, cast, reads
from harness.actors import Actor, admin_of, create_user
from harness.upstream import MOCK_MODEL_ID

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source]

MESSAGE_ID = "assistant-1"
RESTRICTED_ADMIN_ENV = {"ENABLE_ADMIN_CHAT_ACCESS": "false", "BYPASS_ADMIN_ACCESS_CONTROL": "false"}


def _chat() -> dict:
    user_message = {
        "id": "user-1",
        "parentId": None,
        "childrenIds": [MESSAGE_ID],
        "role": "user",
        "content": "the owner's question",
    }
    reply = {
        "id": MESSAGE_ID,
        "parentId": "user-1",
        "childrenIds": [],
        "role": "assistant",
        "content": "the owner's answer",
        "model": MOCK_MODEL_ID,
        "done": True,
    }
    return {
        "chat": {
            "title": f"private {uuid.uuid4().hex[:8]}",
            "models": [MOCK_MODEL_ID],
            "history": {
                "currentId": MESSAGE_ID,
                "messages": {"user-1": user_message, MESSAGE_ID: reply},
            },
            "messages": [user_message, reply],
        }
    }


CHAT = Shareable(
    create_path="/api/v1/chats/new",
    create_body=_chat,
    access_path="/api/v1/chats/shared/{id}/access/update",
)
OWNERS_CHAT = reads("/api/v1/chats/{id}")


def _tagged(owner: Actor, chat_id: str) -> dict:
    with owner.client() as client:
        client.post(f"/api/v1/chats/{chat_id}/tags", json={"name": "kept"}).raise_for_status()
    return {}


def _owners_folder(owner: Actor, chat_id: str) -> dict:
    with owner.client() as client:
        created = client.post("/api/v1/folders/", json={"name": f"folder {uuid.uuid4().hex[:8]}"})
    assert created.status_code == 200, created.text
    return {"folder_id": created.json()["id"]}


REFUSED, ALLOWED = 401, 200
READ = {"owner": ALLOWED, "stranger": REFUSED, "reader": ALLOWED, "writer": ALLOWED}
OWNER = {"owner": ALLOWED, "stranger": REFUSED, "reader": REFUSED, "writer": REFUSED}
# the delete and share-grant handlers answer 404 to anyone but the owner and an admin
OWNER_OR_NOT_FOUND = {"owner": ALLOWED, "stranger": 404, "reader": 404, "writer": 404}
MESSAGE = f"/api/v1/chats/{{id}}/messages/{MESSAGE_ID}"

# method, path, body, setup, what each account gets, what the admin gets with access on
MATRIX = [
    ("GET", "/api/v1/chats/{id}", None, None, READ, ALLOWED),
    ("GET", "/api/v1/chats/{id}/tags", None, None, READ, ALLOWED),
    ("GET", "/api/v1/chats/{id}/pinned", None, None, OWNER, REFUSED),
    ("POST", "/api/v1/chats/{id}", {"chat": {"title": "renamed"}}, None, OWNER, REFUSED),
    ("POST", "/api/v1/chats/{id}/pin", None, None, OWNER, REFUSED),
    ("POST", "/api/v1/chats/{id}/tags", {"name": "added"}, None, OWNER, REFUSED),
    ("DELETE", "/api/v1/chats/{id}/tags", {"name": "kept"}, _tagged, OWNER, REFUSED),
    (
        "POST",
        "/api/v1/chats/{id}/folder",
        lambda actor, fields: {"folder_id": fields["folder_id"]},
        _owners_folder,
        OWNER,
        REFUSED,
    ),
    ("POST", MESSAGE, {"content": "rewritten"}, None, OWNER, ALLOWED),
    ("DELETE", MESSAGE, None, None, OWNER, ALLOWED),
    ("POST", "/api/v1/chats/{id}/share", None, None, OWNER, REFUSED),
    ("DELETE", "/api/v1/chats/{id}/share", None, None, OWNER, REFUSED),
    (
        "POST",
        "/api/v1/chats/shared/{id}/access/update",
        {"access_grants": []},
        None,
        OWNER_OR_NOT_FOUND,
        ALLOWED,
    ),
    ("POST", "/api/v1/chats/{id}/clone", {}, None, OWNER, REFUSED),
    ("POST", "/api/v1/chats/{id}/archive", None, None, OWNER, REFUSED),
    ("DELETE", "/api/v1/chats/{id}", None, None, OWNER_OR_NOT_FOUND, ALLOWED),
]
ROW_IDS = [f"{row[0]} {row[1]}" for row in MATRIX]


def _assert_matrix(accounts, method, path, body, setup, expected):
    answered = attempts(accounts, method, path, body, setup=setup, look=OWNERS_CHAT)

    assert {role: attempt.status for role, attempt in answered.items()} == expected, (
        f"{method} {path}"
    )
    for role, attempt in answered.items():
        if attempt.status != ALLOWED:
            assert attempt.after == attempt.before, f"a refused {role} changed the owner's chat"


@pytest.mark.parametrize("via", ["user", "group"])
@pytest.mark.parametrize("method, path, body, setup, expected, admin_gets", MATRIX, ids=ROW_IDS)
def test_each_account_gets_what_the_chat_share_allows(
    method, path, body, setup, expected, admin_gets, via, admin, make_user
):
    accounts = cast(CHAT, admin, make_user, via=via)

    _assert_matrix(accounts, method, path, body, setup, {**expected, "admin": admin_gets})


# left out: with access off an admin still edits and deletes messages, sets share grants and
# deletes the chat, and the first three answer with the whole chat
RESTRICTED_ROWS = [row for row in MATRIX if row[5] == REFUSED or row[4] is READ]


@pytest.mark.parametrize(
    "method, path, body, setup, expected, admin_gets",
    RESTRICTED_ROWS,
    ids=[f"{row[0]} {row[1]}" for row in RESTRICTED_ROWS],
)
def test_an_admin_without_chat_access_is_refused_like_a_stranger(
    method, path, body, setup, expected, admin_gets, instance_with
):
    launched = instance_with(RESTRICTED_ADMIN_ENV)
    accounts = cast(CHAT, admin_of(launched), lambda: create_user(launched))

    _assert_matrix(accounts, method, path, body, setup, {**expected, "admin": REFUSED})
