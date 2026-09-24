"""Socket-layer repairs from 0.11.1 that a connected client can see.

* `ce3c175e26`: `SocketSessionEventSink` is registered in `EVENT_SINKS`, so a
  `user.role_updated` or `user.deleted` event from any path disconnects the user's open
  Socket.IO sessions. The client reconnects and re-authenticates, so the role and permissions
  the socket layer cached for that session are read afresh instead of outliving the change.
* `5735123f5` (PR #28669): `yjs_document_update` cancelled the pending debounced note save
  before knowing whether a replacement would be scheduled, so the content-less resync update a
  client sends after rejoining dropped the edits made just before it.

Twin of unit/chat/test_socket_runtime.py.

Discriminates: passes on dev bbfa876af; with `SocketSessionEventSink` dropped from `EVENT_SINKS`
the role-change and deletion tests fail, and with the unconditional cancel restored ahead of the
update the resync test fails.
"""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager

import pytest

from harness.socket_client import connected

pytestmark = [pytest.mark.regression, pytest.mark.api, pytest.mark.requires_source]

DISCONNECT_TIMEOUT = 10
QUIET_PERIOD = 2


@contextmanager
def watched_socket(account):
    """The account's live socket, with an event that is set once the server drops it."""
    with connected(account) as socket:
        dropped = threading.Event()
        socket.client.on("disconnect", lambda *args: dropped.set())
        yield dropped


def test_a_role_change_disconnects_the_users_live_socket(admin, make_user):
    account = make_user()
    with watched_socket(account) as dropped, admin.client() as client:
        client.post(f"/api/v1/users/{account.id}/update", json={"role": "admin"}).raise_for_status()

        assert dropped.wait(DISCONNECT_TIMEOUT), (
            "the socket kept the role it authenticated with after the admin changed it"
        )


def test_deleting_the_account_disconnects_its_live_socket(admin, make_user):
    account = make_user()
    with watched_socket(account) as dropped, admin.client() as client:
        client.delete(f"/api/v1/users/{account.id}").raise_for_status()

        assert dropped.wait(DISCONNECT_TIMEOUT), "a deleted account kept its live socket"


def test_an_update_that_keeps_the_role_leaves_the_socket_connected(admin, make_user):
    account = make_user()
    with watched_socket(account) as dropped, admin.client() as client:
        client.post(
            f"/api/v1/users/{account.id}/update", json={"name": "Renamed User"}
        ).raise_for_status()

        assert not dropped.wait(QUIET_PERIOD)


SAVE_WAIT = 5.0


def _note_content(client, note_id: str) -> dict:
    return client.get(f"/api/v1/notes/{note_id}").json()["data"]["content"]


def _edit_then(account, followed_by: dict) -> dict:
    """Edit a note over its live document, then send `followed_by`; the content saved after."""
    with account.client() as client:
        note = client.post(
            "/api/v1/notes/create", json={"title": "draft", "data": {"content": {"md": "old"}}}
        )
        assert note.status_code == 200, note.text
        document_id = f"note:{note.json()['id']}"
        with connected(account) as socket:
            socket.call("ydoc:document:join", {"document_id": document_id})
            edit = {"update": [1, 2, 3], "data": {"content": {"md": "edited"}}}
            socket.call("ydoc:document:update", {"document_id": document_id, **edit})
            socket.call("ydoc:document:update", {"document_id": document_id, **followed_by})

            deadline = time.monotonic() + SAVE_WAIT
            while time.monotonic() < deadline:
                content = _note_content(client, note.json()["id"])
                if content != {"md": "old"}:
                    return content
                time.sleep(0.2)
            return content


def test_a_resync_update_keeps_the_pending_note_save(make_user):
    content = _edit_then(make_user(), followed_by={"update": [4, 5, 6]})

    assert content == {"md": "edited"}, (
        "the content-less resync update cancelled the pending save of the edit (PR #28669)"
    )


def test_a_later_edit_replaces_the_pending_note_save(make_user):
    later = {"update": [4, 5, 6], "data": {"content": {"md": "edited again"}}}

    assert _edit_then(make_user(), followed_by=later) == {"md": "edited again"}
