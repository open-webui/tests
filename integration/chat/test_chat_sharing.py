"""Journey: sharing a chat, the way the share dialog and the shared chat page use it.

The owner shares a chat, which stores a snapshot under a new share id, and gives a user read
access to it. That user opens the share, a signed-out visitor and a stranger are refused, and
messages sent after sharing stay out of the snapshot until the owner shares again. Unsharing one
chat or deleting all shares revokes the links, an account without `chat.share` cannot share, and
cloning a shared chat gives the reader a copy of their own.

Discriminates: in a backend copy, `delete_shared_chat_by_id` leaving the snapshot and grants in
place fails the unshare test, `unshare_all_chats` deleting no rows fails the delete-all test,
`share_chat_by_id` skipping the `chat.share` check fails the permission test, and
`clone_shared_chat_by_id` importing the copy under the sharer's id fails the clone test.
"""

from __future__ import annotations

import uuid

import httpx
import pytest

from harness import upstream as reply
from harness.access import grant
from harness.chat import ask

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source]

PERMISSIONS = "/api/v1/users/default/permissions"


def _start_chat(owner, upstream, question: str) -> tuple[str, str]:
    """A chat with one answered question; returns its id and the reply's id."""
    answer = f"answer to {question}"
    upstream.queue(reply.text(answer))
    with owner.client() as client:
        turn, stored = ask(client, question)
    assert stored["content"] == answer
    return turn.chat_id, turn.assistant_message_id


def _share(owner, chat_id: str, *readers) -> str:
    """Share the chat and give `readers` read access, as the share dialog does."""
    with owner.client() as client:
        shared = client.post(f"/api/v1/chats/{chat_id}/share")
        assert shared.status_code == 200, shared.text
        grants = [grant("user", reader.id, "read") for reader in readers]
        granted = client.post(
            f"/api/v1/chats/shared/{chat_id}/access/update", json={"access_grants": grants}
        )
    assert granted.status_code == 200, granted.text
    return shared.json()["share_id"]


def _open_share(account, share_id: str) -> httpx.Response:
    with account.client() as client:
        return client.get(f"/api/v1/chats/share/{share_id}")


def _stored_chat(owner, chat_id: str) -> dict:
    with owner.client() as client:
        stored = client.get(f"/api/v1/chats/{chat_id}")
    assert stored.status_code == 200, stored.text
    return stored.json()


def test_a_share_is_read_by_the_user_it_was_shared_with(make_user, upstream):
    owner, reader = make_user(), make_user()
    chat_id, _ = _start_chat(owner, upstream, "what is the capital of Norway?")

    share_id = _share(owner, chat_id, reader)
    opened = _open_share(reader, share_id)

    assert share_id and share_id != chat_id
    assert _stored_chat(owner, chat_id)["share_id"] == share_id
    assert opened.status_code == 200, opened.text
    assert "answer to what is the capital of Norway?" in str(opened.json()["chat"])


def test_a_share_refuses_signed_out_visitors_and_strangers(instance, make_user, upstream):
    owner, reader = make_user(), make_user()
    chat_id, _ = _start_chat(owner, upstream, "private question")
    share_id = _share(owner, chat_id, reader)

    signed_out = httpx.get(f"{instance.base_url}/api/v1/chats/share/{share_id}", timeout=60.0)
    stranger = _open_share(make_user(), share_id)

    assert signed_out.status_code == 401, signed_out.text
    assert stranger.status_code == 401, stranger.text


def test_a_share_is_a_snapshot_until_shared_again(make_user, upstream):
    owner, reader = make_user(), make_user()
    chat_id, reply_id = _start_chat(owner, upstream, "first question")
    share_id = _share(owner, chat_id, reader)
    upstream.queue(reply.text("second answer"))
    with owner.client() as client:
        ask(client, "second question", chat_id=chat_id, parent_id=reply_id)

    before_resharing = _open_share(reader, share_id).json()
    assert _share(owner, chat_id, reader) == share_id
    after_resharing = _open_share(reader, share_id).json()

    assert "second answer" not in str(before_resharing["chat"])
    assert "second answer" in str(after_resharing["chat"])


def test_unsharing_revokes_the_share(make_user, upstream):
    owner, reader = make_user(), make_user()
    chat_id, _ = _start_chat(owner, upstream, "soon unshared")
    share_id = _share(owner, chat_id, reader)

    with owner.client() as client:
        unshared = client.delete(f"/api/v1/chats/{chat_id}/share")
        grants = client.get(f"/api/v1/chats/shared/{chat_id}/access")

    assert unshared.status_code == 200, unshared.text
    assert _open_share(reader, share_id).status_code == 401
    assert _stored_chat(owner, chat_id)["share_id"] is None
    assert grants.json() == []


def test_deleting_all_shares_revokes_every_share(make_user, upstream):
    owner, reader = make_user(), make_user()
    chat_ids = [_start_chat(owner, upstream, f"chat {n}")[0] for n in range(2)]
    share_ids = [_share(owner, chat_id, reader) for chat_id in chat_ids]
    other_owner = make_user()
    other_chat, _ = _start_chat(other_owner, upstream, "someone else's share")
    other_share = _share(other_owner, other_chat, reader)

    with owner.client() as client:
        deleted = client.delete("/api/v1/chats/share/all")
        listed = client.get("/api/v1/chats/shared")

    assert deleted.status_code == 200, deleted.text
    assert [_open_share(reader, share_id).status_code for share_id in share_ids] == [401, 401]
    assert [_stored_chat(owner, chat_id)["share_id"] for chat_id in chat_ids] == [None, None]
    assert listed.json() == []
    assert _open_share(reader, other_share).status_code == 200, "another user's share was revoked"


def test_a_user_without_the_share_permission_cannot_share(admin, make_user, upstream, preserve):
    preserve("permissions")
    owner = make_user()
    chat_id, _ = _start_chat(owner, upstream, "not for sharing")
    with admin.client() as client:
        permissions = client.get(PERMISSIONS).json()
        barred = {**permissions, "chat": {**permissions["chat"], "share": False}}
        client.post(PERMISSIONS, json=barred).raise_for_status()

    with owner.client() as client:
        refused = client.post(f"/api/v1/chats/{chat_id}/share")

    assert refused.status_code == 401, refused.text
    assert _stored_chat(owner, chat_id)["share_id"] is None


def test_cloning_a_share_gives_the_reader_their_own_copy(make_user, upstream):
    owner, reader = make_user(), make_user()
    chat_id, _ = _start_chat(owner, upstream, "worth keeping")
    share_id = _share(owner, chat_id, reader)

    with reader.client() as client:
        cloned = client.post(f"/api/v1/chats/{share_id}/clone/shared")
        assert cloned.status_code == 200, cloned.text
        clone = cloned.json()
        renamed = client.post(
            f"/api/v1/chats/{clone['id']}", json={"chat": {"title": f"mine {uuid.uuid4().hex}"}}
        )
        own_chats = {chat["id"] for chat in client.get("/api/v1/chats/").json()}

    assert clone["user_id"] == reader.id
    assert clone["id"] not in (chat_id, share_id)
    assert "answer to worth keeping" in str(clone["chat"])
    assert clone["id"] in own_chats
    assert renamed.status_code == 200, renamed.text
    assert _stored_chat(owner, chat_id)["title"] != renamed.json()["title"]
