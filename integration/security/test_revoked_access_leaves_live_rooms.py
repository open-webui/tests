"""Regression: an account whose access was revoked still got a channel's or note's live updates.

Fix `e93a59f4d` (0.11.5): a socket joins the rooms of the channels and notes its account may
read (`channel:{id}` through `user-join`, `note:{id}` through `note:join`, `doc_note:{id}` when
the editor opens the note), and nothing made it leave them again. Once the owner took the grant
away, the open tab still got every new channel message, every edit of the note and every
keystroke in its live document. Saving grants that drop someone now makes that account's sessions
leave the rooms. Deleting a channel or note also closes its rooms; that half is not pinned, since
nothing is sent to the room of a deleted channel or note.

Discriminates: passes on dev efe63bd34; with e93a59f4d reverted in a backend copy every revoked
case receives the update after the grant is gone. The kept-reader and admin cases pass on both.
"""

from __future__ import annotations

import threading
import time
import uuid

import pytest

from harness.channel_quotes import enable_channels, post_message
from harness.socket_client import SocketSession, connected

pytestmark = [pytest.mark.regression, pytest.mark.api, pytest.mark.requires_source]

ARRIVAL_TIMEOUT = 15.0
QUIET_PERIOD = 2.0


def _grant(account, permission: str = "read") -> dict:
    return {"principal_type": "user", "principal_id": account.id, "permission": permission}


def _arrivals(session: SocketSession, event: str, marker: str) -> threading.Event:
    """Set once `event` arrives carrying `marker` anywhere in its payload."""
    arrived = threading.Event()

    def record(*payload) -> None:
        if marker in str(payload):
            arrived.set()

    session.client.on(event, record)
    return arrived


def _received(flag: threading.Event, expected: bool) -> bool:
    return flag.wait(ARRIVAL_TIMEOUT if expected else QUIET_PERIOD)


# --- channels --------------------------------------------------------------------------------


@pytest.fixture
def announcements(admin, preserve):
    """`announcements(*readers)` opens a standard channel the readers may read; returns its id."""
    preserve("admin_config")
    enable_channels(admin)

    def open_channel(*readers) -> str:
        with admin.client() as client:
            created = client.post(
                "/api/v1/channels/create",
                json={
                    "name": f"news-{uuid.uuid4().hex[:6]}",
                    "type": None,
                    "access_grants": [_grant(reader) for reader in readers],
                },
            )
        assert created.status_code == 200, created.text
        return created.json()["id"]

    return open_channel


def _join_channels(session: SocketSession, account) -> None:
    """What the web client sends after connecting: it joins the rooms of its channels."""
    session.call("user-join", {"auth": {"token": account.token}})


def _set_channel_readers(admin, channel_id: str, *readers) -> None:
    with admin.client() as client:
        updated = client.post(
            f"/api/v1/channels/{channel_id}/access/update",
            json={"access_grants": [_grant(reader) for reader in readers]},
        )
    assert updated.status_code == 200, updated.text


def test_a_reader_removed_from_a_channel_stops_getting_its_messages(
    admin, make_user, announcements
):
    removed, kept = make_user(), make_user()
    channel_id = announcements(removed, kept)

    with connected(removed) as removed_tab, connected(kept) as kept_tab:
        _join_channels(removed_tab, removed)
        _join_channels(kept_tab, kept)
        _set_channel_readers(admin, channel_id, kept)
        marker = f"after the change {uuid.uuid4().hex[:6]}"
        to_removed = _arrivals(removed_tab, "events:channel", marker)
        to_kept = _arrivals(kept_tab, "events:channel", marker)
        post_message(admin, channel_id, marker)

        kept_received = _received(to_kept, expected=True)
        removed_received = _received(to_removed, expected=False)

    assert kept_received, "a reader who kept the grant stopped getting the channel's messages"
    assert not removed_received, (
        "a reader whose grant was taken away still got the channel's new messages in the tab "
        "that was open (e93a59f4d)"
    )


def test_a_reader_still_gets_messages_before_the_grant_is_removed(admin, make_user, announcements):
    reader = make_user()
    channel_id = announcements(reader)

    with connected(reader) as tab:
        _join_channels(tab, reader)
        marker = f"while shared {uuid.uuid4().hex[:6]}"
        arrived = _arrivals(tab, "events:channel", marker)
        post_message(admin, channel_id, marker)
        received = _received(arrived, expected=True)

    assert received, "a reader with a grant got no live channel messages"


# --- notes -----------------------------------------------------------------------------------


def _create_note(owner) -> str:
    with owner.client() as client:
        created = client.post(
            "/api/v1/notes/create",
            json={"title": f"shared {uuid.uuid4().hex[:6]}", "data": {"content": {"md": ""}}},
        )
    assert created.status_code == 200, created.text
    return created.json()["id"]


def _share_note(owner, note_id: str, *readers) -> None:
    with owner.client() as client:
        shared = client.post(
            f"/api/v1/notes/{note_id}/access/update",
            json={"access_grants": [_grant(reader) for reader in readers]},
        )
    assert shared.status_code == 200, shared.text


def _follow_note(session: SocketSession, account, note_id: str) -> None:
    """Open the note the way the note page does: follow its edits and its live document."""
    session.call("note:join", {"auth": {"token": account.token}, "note_id": note_id})
    session.join_note(note_id)


def test_a_reader_removed_from_a_note_stops_getting_its_edits(make_user):
    owner, removed, kept = make_user(), make_user(), make_user()
    note_id = _create_note(owner)
    _share_note(owner, note_id, removed, kept)

    with connected(removed) as removed_tab, connected(kept) as kept_tab:
        _follow_note(removed_tab, removed, note_id)
        _follow_note(kept_tab, kept, note_id)
        _share_note(owner, note_id, kept)
        title = f"renamed {uuid.uuid4().hex[:6]}"
        to_removed = _arrivals(removed_tab, "events:note", title)
        to_kept = _arrivals(kept_tab, "events:note", title)
        with owner.client() as client:
            renamed = client.post(f"/api/v1/notes/{note_id}/update", json={"title": title})
        assert renamed.status_code == 200, renamed.text

        kept_received = _received(to_kept, expected=True)
        removed_received = _received(to_removed, expected=False)

    assert kept_received, "a reader who kept the grant stopped getting the note's edits"
    assert not removed_received, (
        "a reader whose grant was taken away still got the note's edits in the tab that was "
        "open (e93a59f4d)"
    )


def test_a_reader_removed_from_a_note_stops_getting_live_typing(make_user):
    owner, removed, kept = make_user(), make_user(), make_user()
    note_id = _create_note(owner)
    _share_note(owner, note_id, removed, kept)

    with (
        connected(owner) as owners_tab,
        connected(removed) as removed_tab,
        connected(kept) as kept_tab,
    ):
        owners_tab.join_note(note_id)
        removed_tab.join_note(note_id)
        kept_tab.join_note(note_id)
        _share_note(owner, note_id, kept)
        owners_tab.edit_note(note_id, "typed after the change")
        kept_update = kept_tab.note_update(note_id, timeout=ARRIVAL_TIMEOUT)
        time.sleep(QUIET_PERIOD)

    assert kept_update, "a reader who kept the grant stopped getting live typing"
    assert removed_tab.document_updates == [], (
        "a reader whose grant was taken away still received the owner's typing in the live "
        "document (e93a59f4d)"
    )


def test_an_admin_keeps_getting_a_notes_edits_after_the_grants_change(admin, make_user):
    owner = make_user()
    note_id = _create_note(owner)
    _share_note(owner, note_id, admin)

    with connected(admin) as admins_tab:
        _follow_note(admins_tab, admin, note_id)
        _share_note(owner, note_id)
        title = f"renamed {uuid.uuid4().hex[:6]}"
        arrived = _arrivals(admins_tab, "events:note", title)
        with owner.client() as client:
            renamed = client.post(f"/api/v1/notes/{note_id}/update", json={"title": title})
        assert renamed.status_code == 200, renamed.text
        received = _received(arrived, expected=True)

    assert received, "an admin, who may read every note, was made to leave the note's room"
