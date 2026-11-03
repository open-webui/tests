"""Regression: a large paste into a note was lost after a reload.

Issue open-webui/open-webui#26140, fix 914ff1f11 (PR open-webui/open-webui#31893). Every live edit
of a note carries the whole note several times over: the Yjs update and the editor's markdown,
HTML and JSON snapshot. Pasting around 150 KB of text pushed that one socket message past the
Socket.IO server's 1 MB default, so the server dropped the connection, the edit was never stored
and a reload showed the note without it. The server now takes messages up to 16 MiB, the size it
already accepts for a websocket frame.

Issue open-webui/open-webui#33606: a later cap of 2 MiB per live document (PR
open-webui/open-webui#31603) refused the first document of any larger note, such as one holding
a pasted screenshot. The editor seeds the document from the saved note when it opens, the
server answered that seed with nothing, and the note opened empty and took no typing. The cap
now matches the 16 MiB message limit.

Discriminates: passes on dev b859124f9, fails on that backend with 914ff1f11 reverted (the large
edit never reaches the note or its live document). The note above 2 MiB passes with the fix for
#33606 and fails on dev f3ef8b949 (the seed is answered with nothing).
"""

from __future__ import annotations

import time
import uuid

import pytest
import socketio

from harness.socket_client import connected, note_edit, note_text

pytestmark = [pytest.mark.regression, pytest.mark.api, pytest.mark.requires_source]

SAVE_WAIT = 15.0
LARGE_TEXT = " ".join(f"word{number}" for number in range(40_000))
# About 3.2 MB, past the old 2 MiB cap and still inside one 16 MiB socket message.
NOTE_ABOVE_2_MIB = " ".join(f"word{number}" for number in range(300_000))


def _create_note(owner) -> str:
    with owner.client() as client:
        created = client.post(
            "/api/v1/notes/create",
            json={"title": f"large {uuid.uuid4().hex[:8]}", "data": {"content": {"md": ""}}},
        )
    assert created.status_code == 200, created.text
    return created.json()["id"]


def _saved_markdown(owner, note_id: str, expected: str) -> str:
    """The note's stored markdown once it reads `expected`, or as it is when the wait ends."""
    deadline = time.monotonic() + SAVE_WAIT
    with owner.client() as client:
        while True:
            note = client.get(f"/api/v1/notes/{note_id}")
            assert note.status_code == 200, note.text
            markdown = ((note.json().get("data") or {}).get("content") or {}).get("md")
            if markdown == expected or time.monotonic() > deadline:
                return markdown
            time.sleep(0.2)


def _send_edit(tab, note_id: str, text: str) -> None:
    try:
        tab.edit_note(note_id, text)
    except socketio.exceptions.TimeoutError:
        pytest.fail(f"#26140: the server dropped the edit of {len(text)} characters unanswered")


def test_a_large_paste_is_saved_to_the_note(make_user):
    owner = make_user()
    note_id = _create_note(owner)

    with connected(owner) as tab:
        tab.join_note(note_id)
        _send_edit(tab, note_id, LARGE_TEXT)

        saved = _saved_markdown(owner, note_id, LARGE_TEXT)

    assert saved == LARGE_TEXT, f"#26140: the note holds {len(saved or '')} characters"


def test_a_tab_opening_the_note_later_starts_from_the_large_paste(make_user):
    owner = make_user()
    note_id = _create_note(owner)

    with connected(owner) as first:
        first.join_note(note_id)
        _send_edit(first, note_id, LARGE_TEXT)
        with connected(owner) as later:
            later.join_note(note_id)
            document = later.note_state(note_id)

    assert document == f"<paragraph>{LARGE_TEXT}</paragraph>"


def test_a_note_above_2_mib_opens_with_its_content(make_user):
    owner = make_user()
    note_id = _create_note(owner)
    seed = {
        "document_id": f"note:{note_id}",
        "update": note_edit(NOTE_ABOVE_2_MIB),
        "initialize": True,
    }

    with connected(owner) as tab:
        tab.join_note(note_id)
        opened = tab.call("ydoc:document:update", seed)
        assert opened is not None, (
            f"#33606: the server refused a {len(NOTE_ABOVE_2_MIB)} character note; it opens empty"
        )
        with connected(owner) as later:
            later.join_note(note_id)
            document = later.note_state(note_id)

    assert note_text(opened) == f"<paragraph>{NOTE_ABOVE_2_MIB}</paragraph>"
    assert document == f"<paragraph>{NOTE_ABOVE_2_MIB}</paragraph>"
