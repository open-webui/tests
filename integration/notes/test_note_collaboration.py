"""Journey: editing a note live, the way the note editor talks to the server over the socket.

A tab joins the note's Yjs document and types into it. The server keeps the update so a tab
that opens the note later starts from the edited document, passes it on at once to a tab that
already has the note open, and saves the editor's content snapshot to the note. An account that
may only read the note has its edits dropped.

Discriminates: in a backend copy, `yjs_document_update` not appending the update to the stored
document, broadcasting it only to the sender and never scheduling `document_save_handler` each
turn one test red.
"""

from __future__ import annotations

import time
import uuid

import pytest

from harness.socket_client import connected, note_text

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source]

SAVE_WAIT = 10.0
# the server saves half a second after the last edit
REFUSED_SAVE_WAIT = 2.0


def _create_note(owner) -> str:
    with owner.client() as client:
        created = client.post(
            "/api/v1/notes/create",
            json={"title": f"live {uuid.uuid4().hex[:8]}", "data": {"content": {"md": ""}}},
        )
    assert created.status_code == 200, created.text
    return created.json()["id"]


def _saved_markdown(owner, note_id: str, expected: str, wait: float = SAVE_WAIT) -> str:
    """The note's stored markdown once it reads `expected`, or as it is when the wait ends."""
    deadline = time.monotonic() + wait
    with owner.client() as client:
        while True:
            note = client.get(f"/api/v1/notes/{note_id}")
            assert note.status_code == 200, note.text
            markdown = ((note.json().get("data") or {}).get("content") or {}).get("md")
            if markdown == expected or time.monotonic() > deadline:
                return markdown
            time.sleep(0.2)


def test_a_tab_that_opens_the_note_later_starts_from_the_edit(make_user):
    owner = make_user()
    note_id = _create_note(owner)

    with connected(owner) as first:
        first.join_note(note_id)
        first.edit_note(note_id, "written live")
        with connected(owner) as later:
            later.join_note(note_id)
            document = later.note_state(note_id)

    assert document == "<paragraph>written live</paragraph>"


def test_an_open_tab_gets_the_edit_at_once(make_user):
    owner = make_user()
    note_id = _create_note(owner)

    with connected(owner) as first, connected(owner) as second:
        first.join_note(note_id)
        second.join_note(note_id)
        first.edit_note(note_id, "seen elsewhere")
        passed_on = second.note_update(note_id)

    assert note_text(passed_on["update"]) == "<paragraph>seen elsewhere</paragraph>"
    assert first.document_updates == [], "the sender got its own edit back"


def test_the_editors_content_is_saved_to_the_note(make_user):
    owner = make_user()
    note_id = _create_note(owner)

    with connected(owner) as tab:
        tab.join_note(note_id)
        tab.edit_note(note_id, "kept on disk")

        assert _saved_markdown(owner, note_id, "kept on disk") == "kept on disk"


def test_a_reader_cannot_edit_the_note(make_user):
    owner, reader = make_user(), make_user()
    note_id = _create_note(owner)
    grant = {"principal_type": "user", "principal_id": reader.id, "permission": "read"}
    with owner.client() as client:
        shared = client.post(
            f"/api/v1/notes/{note_id}/access/update", json={"access_grants": [grant]}
        )
    assert shared.status_code == 200, shared.text

    with connected(reader) as tab:
        tab.join_note(note_id)
        tab.edit_note(note_id, "not allowed")
        with connected(owner) as owners_tab:
            owners_tab.join_note(note_id)
            document = owners_tab.note_state(note_id)

    assert document == ""
    assert _saved_markdown(owner, note_id, "not allowed", wait=REFUSED_SAVE_WAIT) == ""
