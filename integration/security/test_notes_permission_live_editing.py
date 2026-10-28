"""Regression: a user without the Notes permission still opened and edited notes live.

Fix `6d409da9d` (#31552): the notes routes refuse a user whose Notes permission the admin took
away, but the socket that carries live note editing never asked. Such a user still opened a
note's live document (`ydoc:document:join`), received what others typed, had their own typing
saved to the note, and followed the note's changes through `note:join`. Both socket events now
check the Notes permission first. A user whose group grants Notes and an admin still edit live.

Discriminates: passes on dev a5bc78300; with 6d409da9d reverted in a backend copy the restricted
user's typing is saved to the note and the owner's typing and the note's rename reach their tab.
The allowed cases pass on both.
"""

from __future__ import annotations

import threading
import time
import uuid

import pytest

from harness.access import make_group
from harness.socket_client import SocketSession, connected, note_text

pytestmark = [pytest.mark.regression, pytest.mark.api, pytest.mark.requires_source]

PERMISSIONS = "/api/v1/users/default/permissions"
NOTES_GRANTED = {"features": {"notes": True}}
ARRIVAL_TIMEOUT = 15.0
QUIET_PERIOD = 2.0
# the server saves half a second after the last edit
SAVE_WAIT = 10.0
REFUSED_SAVE_WAIT = 3.0


def _create_note(owner) -> str:
    with owner.client() as client:
        created = client.post(
            "/api/v1/notes/create",
            json={"title": f"plans {uuid.uuid4().hex[:6]}", "data": {"content": {"md": ""}}},
        )
    assert created.status_code == 200, created.text
    return created.json()["id"]


def _share_note(owner, note_id: str, *readers) -> None:
    grants = [
        {"principal_type": "user", "principal_id": reader.id, "permission": "read"}
        for reader in readers
    ]
    with owner.client() as client:
        shared = client.post(
            f"/api/v1/notes/{note_id}/access/update", json={"access_grants": grants}
        )
    assert shared.status_code == 200, shared.text


@pytest.fixture
def notes_off_by_default(admin, preserve):
    """`notes_off_by_default()` takes Notes out of the default user permissions."""
    preserve("permissions")

    def switch_off() -> None:
        with admin.client() as client:
            current = client.get(PERMISSIONS)
            assert current.status_code == 200, current.text
            permissions = current.json()
            permissions["features"] = {**permissions["features"], "notes": False}
            saved = client.post(PERMISSIONS, json=permissions)
        assert saved.status_code == 200, saved.text

    return switch_off


def _saved_markdown(admin, note_id: str, expected: str, wait: float) -> str:
    """The note's stored markdown once it reads `expected`, or as it is when the wait ends."""
    deadline = time.monotonic() + wait
    with admin.client() as client:
        while True:
            note = client.get(f"/api/v1/notes/{note_id}")
            assert note.status_code == 200, note.text
            markdown = ((note.json().get("data") or {}).get("content") or {}).get("md")
            if markdown == expected or time.monotonic() > deadline:
                return markdown
            time.sleep(0.2)


def _renames(session: SocketSession, title: str) -> threading.Event:
    arrived = threading.Event()

    def record(*payload) -> None:
        if title in str(payload):
            arrived.set()

    session.client.on("events:note", record)
    return arrived


def _follow_note(session: SocketSession, account, note_id: str) -> None:
    """Open the note the way the note page does: follow its changes and its live document."""
    session.call("note:join", {"auth": {"token": account.token}, "note_id": note_id})
    session.join_note(note_id)


def test_a_user_without_the_permission_cannot_save_their_note_by_typing_live(
    admin, make_user, notes_off_by_default
):
    owner = make_user()
    note_id = _create_note(owner)
    notes_off_by_default()

    with connected(owner) as tab:
        tab.join_note(note_id)
        tab.edit_note(note_id, "typed without Notes")
        stored = _saved_markdown(admin, note_id, "typed without Notes", REFUSED_SAVE_WAIT)

    assert tab.document_states == [], (
        "a user without the Notes permission opened a note's live document (#31552)"
    )
    assert stored == "", (
        f"a user without the Notes permission saved {stored!r} into their note by typing live "
        "(#31552)"
    )


def test_a_user_without_the_permission_gets_no_live_edits_of_a_shared_note(
    admin, make_user, notes_off_by_default
):
    owner, restricted, granted = make_user(), make_user(), make_user()
    make_group(admin, [owner, granted], NOTES_GRANTED)
    note_id = _create_note(owner)
    _share_note(owner, note_id, restricted, granted)
    notes_off_by_default()

    with (
        connected(owner) as owners_tab,
        connected(restricted) as restricted_tab,
        connected(granted) as granted_tab,
    ):
        owners_tab.join_note(note_id)
        _follow_note(restricted_tab, restricted, note_id)
        _follow_note(granted_tab, granted, note_id)
        owners_tab.edit_note(note_id, "the owner typing")
        granted_update = granted_tab.note_update(note_id, timeout=ARRIVAL_TIMEOUT)

        title = f"renamed {uuid.uuid4().hex[:6]}"
        to_restricted = _renames(restricted_tab, title)
        to_granted = _renames(granted_tab, title)
        with owner.client() as client:
            renamed = client.post(f"/api/v1/notes/{note_id}/update", json={"title": title})
        assert renamed.status_code == 200, renamed.text
        granted_renamed = to_granted.wait(ARRIVAL_TIMEOUT)
        restricted_renamed = to_restricted.wait(QUIET_PERIOD)

    assert note_text(granted_update["update"]) == "<paragraph>the owner typing</paragraph>"
    assert granted_renamed, "a reader whose group grants Notes stopped getting the note's changes"
    assert restricted_tab.document_updates == [], (
        "a reader without the Notes permission received the owner's live typing (#31552)"
    )
    assert not restricted_renamed, (
        "a reader without the Notes permission followed the note's changes through note:join "
        "(#31552)"
    )


def test_a_user_whose_group_grants_notes_still_saves_by_typing_live(
    admin, make_user, notes_off_by_default
):
    owner = make_user()
    make_group(admin, [owner], NOTES_GRANTED)
    note_id = _create_note(owner)
    notes_off_by_default()

    with connected(owner) as tab:
        tab.join_note(note_id)
        tab.edit_note(note_id, "typed with Notes from the group")
        stored = _saved_markdown(admin, note_id, "typed with Notes from the group", SAVE_WAIT)

    assert stored == "typed with Notes from the group"


def test_an_admin_still_edits_a_users_note_live(admin, make_user, notes_off_by_default):
    owner = make_user()
    note_id = _create_note(owner)
    notes_off_by_default()

    with connected(admin) as tab:
        tab.join_note(note_id)
        tab.edit_note(note_id, "typed by an admin")
        stored = _saved_markdown(admin, note_id, "typed by an admin", SAVE_WAIT)

    assert tab.note_state(note_id) == ""
    assert stored == "typed by an admin"
