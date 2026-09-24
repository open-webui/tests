"""Regression: deleting someone else's chat must not cancel its reply.

open-webui 0.11.0 fix `4f93c3e36` (#27006): `DELETE /api/v1/chats/{id}` cancelled the chat's
in-flight tasks as its very first statement, before checking that the caller was an admin,
held `chat.delete` and owned the chat. Anyone who knew a chat id could cut off another user's
streaming reply; the delete itself was refused afterwards. The fix authorizes first.

Twin of unit/security/test_chat_delete_task_cancellation.py.

Discriminates: passes on dev bbfa876af; with the task cancellation moved back in front of the
authorization the refused deletes still cut the reply short.
"""

from __future__ import annotations

import pytest

from harness.chat import ask, wait_for_reply
from harness.inflight import LAST_PIECE, start_slow_reply

pytestmark = [pytest.mark.regression, pytest.mark.api, pytest.mark.requires_source]


def _assert_finished_in_full(client, turn, what: str) -> None:
    message = wait_for_reply(client, turn, timeout=20)
    assert LAST_PIECE in message["content"], (
        f"{what} cut the chat's streaming reply short (#27006): {message['content']!r}"
    )


def test_a_strangers_delete_leaves_the_reply_running(admin, make_user, upstream):
    with admin.client() as owner_client, make_user().client() as stranger:
        turn = start_slow_reply(owner_client, upstream)

        refused = stranger.delete(f"/api/v1/chats/{turn.chat_id}")
        assert refused.status_code == 404, refused.text

        _assert_finished_in_full(owner_client, turn, "a stranger's refused delete")
        assert owner_client.get(f"/api/v1/chats/{turn.chat_id}").status_code == 200


def test_a_delete_without_the_delete_permission_leaves_the_reply_running(
    admin, make_user, upstream, preserve
):
    preserve("permissions")
    with admin.client() as admin_client:
        permissions = admin_client.get("/api/v1/users/default/permissions").json()
        permissions["chat"]["delete"] = False
        admin_client.post("/api/v1/users/default/permissions", json=permissions).raise_for_status()

    with make_user().client() as owner_client:
        turn = start_slow_reply(owner_client, upstream)

        refused = owner_client.delete(f"/api/v1/chats/{turn.chat_id}")
        assert refused.status_code == 401, refused.text

        _assert_finished_in_full(owner_client, turn, "a delete refused for lack of chat.delete")


def test_a_strangers_archive_leaves_the_reply_running(admin, make_user, upstream):
    with admin.client() as owner_client, make_user().client() as stranger:
        turn = start_slow_reply(owner_client, upstream)

        refused = stranger.post(f"/api/v1/chats/{turn.chat_id}/archive")
        assert refused.status_code == 401, refused.text

        _assert_finished_in_full(owner_client, turn, "a stranger's refused archive")


def test_a_bulk_delete_only_removes_the_callers_chats(make_user, upstream):
    owner, caller = make_user(), make_user()
    with owner.client() as owner_client, caller.client() as caller_client:
        owner_turn, _ = ask(owner_client, "keep me")
        caller_turn, _ = ask(caller_client, "delete me")

        assert caller_client.delete("/api/v1/chats/").json() is True

        assert caller_client.get(f"/api/v1/chats/{caller_turn.chat_id}").status_code != 200
        assert owner_client.get(f"/api/v1/chats/{owner_turn.chat_id}").status_code == 200


def test_the_owners_delete_still_stops_their_reply(admin, make_user, upstream):
    with make_user().client() as owner_client, admin.client() as admin_client:
        turn = start_slow_reply(owner_client, upstream)
        running = owner_client.get(f"/api/tasks/chat/{turn.chat_id}").json()["task_ids"]
        assert running, "the slow reply has no task to stop"

        deleted = owner_client.delete(f"/api/v1/chats/{turn.chat_id}")
        assert deleted.status_code == 200 and deleted.json() is True, deleted.text

        still_running = set(admin_client.get("/api/tasks").json()["tasks"])
        assert not still_running & set(running)


def test_an_admin_can_delete_another_users_chat(admin, make_user, upstream):
    with make_user().client() as owner_client, admin.client() as admin_client:
        turn, _ = ask(owner_client, "hello")

        deleted = admin_client.delete(f"/api/v1/chats/{turn.chat_id}")

        assert deleted.status_code == 200 and deleted.json() is True, deleted.text
        assert owner_client.get(f"/api/v1/chats/{turn.chat_id}").status_code != 200


def test_deleting_a_missing_chat_is_not_found(make_user):
    with make_user().client() as client:
        assert client.delete("/api/v1/chats/no-such-chat").status_code == 404
