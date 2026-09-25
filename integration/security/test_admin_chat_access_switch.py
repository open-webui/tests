"""Regression: with admin chat access off, an admin still reached other users' chats.

Issue #31413, open on dev ac00d40e3: `ENABLE_ADMIN_CHAT_ACCESS=false` makes opening another
user's chat answer 401, but three chat routes let any admin through without reading the setting.
Editing a message (`POST /chats/{id}/messages/{message_id}`) overwrites it and answers with the
whole chat, deleting a message that does not exist answers with the whole chat and changes
nothing, and `POST /chats/shared/{id}/access/update` lets the admin grant themselves read access,
after which opening the chat works too. A fourth path, added to the issue: continuing the
owner's chat through the chat completion request is accepted, the model is sent the owner's
earlier messages and the admin's turn is saved into the owner's chat.

Discriminates: fails on dev ac00d40e3 (the three routes answer 200 to the admin, the grant opens
the chat and the continuation is accepted); passes with those routes and the existing-chat
ownership check of the chat completion request treating an admin as a stranger unless
`ENABLE_ADMIN_CHAT_ACCESS` is on. The tests with the setting on and the owner's tests pass on both.
"""

from __future__ import annotations

import pytest

from harness import upstream as reply
from harness.actors import admin_of, create_user
from harness.chat import ask, send_message

pytestmark = [
    pytest.mark.regression,
    pytest.mark.api,
    pytest.mark.requires_source,
    pytest.mark.slow,
]

RESTRICTED_ADMIN_ENV = {"ENABLE_ADMIN_CHAT_ACCESS": "false"}
ANSWER = "the owner's private answer"
REFUSED = (401, 403, 404)


def _owners_chat(owner, upstream) -> tuple[str, str]:
    """A chat with one answered question; returns its id and the reply's id."""
    upstream.queue(reply.text(ANSWER))
    with owner.client() as client:
        turn, stored = ask(client, "the owner's private question")
    assert stored["content"] == ANSWER
    return turn.chat_id, turn.assistant_message_id


def _stored_reply(owner, chat_id: str, message_id: str) -> str:
    with owner.client() as client:
        chat = client.get(f"/api/v1/chats/{chat_id}").json()["chat"]
    return chat["history"]["messages"][message_id]["content"]


def _edit_message(account, chat_id: str, message_id: str):
    with account.client() as client:
        return client.post(
            f"/api/v1/chats/{chat_id}/messages/{message_id}", json={"content": "edited"}
        )


def _delete_missing_message(account, chat_id: str):
    with account.client() as client:
        return client.delete(f"/api/v1/chats/{chat_id}/messages/does-not-exist")


def _grant_self_read(account, chat_id: str):
    grant = {"principal_type": "user", "principal_id": account.id, "permission": "read"}
    with account.client() as client:
        return client.post(
            f"/api/v1/chats/shared/{chat_id}/access/update", json={"access_grants": [grant]}
        )


def _open_chat(account, chat_id: str) -> int:
    with account.client() as client:
        return client.get(f"/api/v1/chats/{chat_id}").status_code


@pytest.fixture
def restricted(instance_with):
    """An instance with ENABLE_ADMIN_CHAT_ACCESS off, its admin and a chat owner on it."""
    launched = instance_with(RESTRICTED_ADMIN_ENV)
    return admin_of(launched), create_user(launched), launched.upstream


# narrow: every route on another user's chat refuses the admin while the setting is off


def test_admin_cannot_open_another_users_chat(restricted):
    admin, owner, upstream = restricted
    chat_id, _ = _owners_chat(owner, upstream)

    assert _open_chat(admin, chat_id) == 401


def test_admin_cannot_edit_a_message_in_another_users_chat(restricted):
    admin, owner, upstream = restricted
    chat_id, message_id = _owners_chat(owner, upstream)

    edited = _edit_message(admin, chat_id, message_id)

    assert edited.status_code in REFUSED, (
        f"with admin chat access off an admin edited another user's message: HTTP "
        f"{edited.status_code} (#31413)"
    )
    assert ANSWER not in edited.text
    assert _stored_reply(owner, chat_id, message_id) == ANSWER


def test_admin_cannot_read_another_users_chat_by_deleting_a_missing_message(restricted):
    admin, owner, upstream = restricted
    chat_id, _ = _owners_chat(owner, upstream)

    deleted = _delete_missing_message(admin, chat_id)

    assert deleted.status_code in REFUSED, (
        f"with admin chat access off an admin read another user's chat through a message "
        f"delete: HTTP {deleted.status_code} (#31413)"
    )
    assert ANSWER not in deleted.text


def test_admin_cannot_grant_themselves_another_users_chat(restricted):
    admin, owner, upstream = restricted
    chat_id, _ = _owners_chat(owner, upstream)

    granted = _grant_self_read(admin, chat_id)

    assert granted.status_code in REFUSED, (
        f"with admin chat access off an admin granted themselves another user's chat: HTTP "
        f"{granted.status_code} (#31413)"
    )
    assert _open_chat(admin, chat_id) == 401, "the admin's own grant opened the chat (#31413)"


def test_admin_cannot_continue_another_users_chat(restricted):
    admin, owner, upstream = restricted
    chat_id, message_id = _owners_chat(owner, upstream)
    with owner.client() as client:
        before = client.get(f"/api/v1/chats/{chat_id}").json()["chat"]["history"]
    upstream.reset()

    with admin.client() as client:
        try:
            send_message(client, "the admin's message", chat_id=chat_id, parent_id=message_id)
            accepted = True
        except AssertionError:
            accepted = False
    with owner.client() as client:
        after = client.get(f"/api/v1/chats/{chat_id}").json()["chat"]["history"]

    assert not accepted, (
        "with admin chat access off an admin continued another user's chat (#31413)"
    )
    assert after == before
    assert upstream.chat_requests() == [], "the owner's messages were sent to the model (#31413)"


# nearby: the owner keeps every route, and an admin reaches the chat with the setting on


def test_the_owner_edits_deletes_and_shares_with_the_setting_off(restricted):
    _, owner, upstream = restricted
    chat_id, message_id = _owners_chat(owner, upstream)

    assert _edit_message(owner, chat_id, message_id).status_code == 200
    assert _stored_reply(owner, chat_id, message_id) == "edited"
    assert _delete_missing_message(owner, chat_id).status_code == 200
    assert _grant_self_read(owner, chat_id).status_code == 200


def test_admin_reaches_another_users_chat_with_the_setting_on(admin, make_user, upstream):
    owner = make_user()
    chat_id, message_id = _owners_chat(owner, upstream)

    assert _open_chat(admin, chat_id) == 200
    assert ANSWER in _delete_missing_message(admin, chat_id).text
    assert _grant_self_read(admin, chat_id).status_code == 200
    edited = _edit_message(admin, chat_id, message_id)
    assert edited.status_code == 200, edited.text
    assert _stored_reply(owner, chat_id, message_id) == "edited"
