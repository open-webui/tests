"""Regression: a file on a shared note opens for the note's readers when the note owner owns it.

A reader opening a note's file chip makes the file dialog's requests, `GET /api/v1/files/{id}`
for the file record and `GET /api/v1/files/{id}/content` for its bytes. Both were refused for
anyone but the uploader (open-webui/open-webui#32011); fix b612c8847 lets read access on a note
open the files attached to it, but only files the note's owner owns, so a file a writer attaches
from their own account stays closed to the note's readers. People without a grant on the note,
and readers after the file is detached or the note unshared, are refused as before.

Discriminates: passes on dev b5a20423e; in a backend copy with b612c8847's change to
utils/access_control/files.py undone every test but the stranger's fails, since the reader is
refused the owner's file from the start.
"""

from __future__ import annotations

import uuid

import pytest

from harness.access import grant
from harness.actors import Actor

pytestmark = [pytest.mark.regression, pytest.mark.api, pytest.mark.requires_source]

FILE_TEXT = "The pilot boat waits at the lighthouse at noon."


def upload(owner: Actor, name: str = "pilot.txt") -> dict:
    with owner.client() as client:
        uploaded = client.post(
            "/api/v1/files/", files={"file": (name, FILE_TEXT.encode(), "text/plain")}
        )
    assert uploaded.status_code == 200, uploaded.text
    return uploaded.json()


def note_entry(uploaded: dict) -> dict:
    """The file the way the note editor stores an upload on the note."""
    return {
        "type": "file",
        "id": uploaded["id"],
        "url": uploaded["id"],
        "name": uploaded["filename"],
        "status": "uploaded",
        "size": len(FILE_TEXT),
        "collection_name": f"file-{uploaded['id']}",
        "file": uploaded,
    }


def save_files(actor: Actor, note_id: str, entries: list[dict]) -> None:
    with actor.client() as client:
        saved = client.post(
            f"/api/v1/notes/{note_id}/update",
            json={"title": "Harbour", "data": {"files": entries}},
        )
    assert saved.status_code == 200, saved.text


def share(owner: Actor, note_id: str, grants: list[dict]) -> None:
    with owner.client() as client:
        shared = client.post(
            f"/api/v1/notes/{note_id}/access/update", json={"access_grants": grants}
        )
    assert shared.status_code == 200, shared.text


def opens_chip(actor: Actor, file_id: str) -> tuple[int, int]:
    """Statuses of the two requests the file dialog makes for a chip."""
    with actor.client() as client:
        return (
            client.get(f"/api/v1/files/{file_id}").status_code,
            client.get(f"/api/v1/files/{file_id}/content").status_code,
        )


def read_file(actor: Actor, file_id: str) -> str:
    with actor.client() as client:
        record = client.get(f"/api/v1/files/{file_id}")
        content = client.get(f"/api/v1/files/{file_id}/content")
    assert record.status_code == 200, record.text
    assert content.status_code == 200, content.text
    assert record.json()["id"] == file_id
    return content.text


@pytest.fixture
def shared_note(make_user) -> tuple[Actor, Actor, str, dict]:
    owner, reader = make_user(), make_user()
    uploaded = upload(owner)
    with owner.client() as client:
        created = client.post(
            "/api/v1/notes/create",
            json={
                "title": f"Harbour {uuid.uuid4().hex[:6]}",
                "data": {"content": {"md": "see the file"}, "files": [note_entry(uploaded)]},
                "access_grants": [grant("user", reader.id, "read")],
            },
        )
    assert created.status_code == 200, created.text
    return owner, reader, created.json()["id"], uploaded


def test_a_reader_opens_the_file_the_note_owner_attached(shared_note):
    _, reader, _, uploaded = shared_note
    assert read_file(reader, uploaded["id"]) == FILE_TEXT


def test_someone_without_a_grant_on_the_note_cannot_open_its_file(shared_note, make_user):
    _, _, _, uploaded = shared_note
    stranger = make_user()
    statuses = opens_chip(stranger, uploaded["id"])
    assert all(status >= 400 for status in statuses), statuses


def test_a_reader_is_refused_again_once_the_owner_detaches_the_file(shared_note):
    owner, reader, note_id, uploaded = shared_note
    assert read_file(reader, uploaded["id"]) == FILE_TEXT

    save_files(owner, note_id, [])

    statuses = opens_chip(reader, uploaded["id"])
    assert all(status >= 400 for status in statuses), statuses


def test_a_reader_is_refused_again_once_the_owner_unshares_the_note(shared_note):
    owner, reader, note_id, uploaded = shared_note
    assert read_file(reader, uploaded["id"]) == FILE_TEXT

    share(owner, note_id, [])

    statuses = opens_chip(reader, uploaded["id"])
    assert all(status >= 400 for status in statuses), statuses


def test_a_file_a_writer_attaches_from_their_own_account_stays_closed_to_readers(make_user):
    owner, writer, reader = make_user(), make_user(), make_user()
    owners_file = upload(owner, "pilot.txt")
    with owner.client() as client:
        created = client.post(
            "/api/v1/notes/create",
            json={
                "title": f"Harbour {uuid.uuid4().hex[:6]}",
                "data": {"content": {"md": "see the file"}, "files": [note_entry(owners_file)]},
                "access_grants": [
                    grant("user", reader.id, "read"),
                    grant("user", writer.id, "read"),
                    grant("user", writer.id, "write"),
                ],
            },
        )
    assert created.status_code == 200, created.text
    note_id = created.json()["id"]
    writers_file = upload(writer, "tide.txt")

    save_files(writer, note_id, [note_entry(owners_file), note_entry(writers_file)])

    assert read_file(reader, owners_file["id"]) == FILE_TEXT
    statuses = opens_chip(reader, writers_file["id"])
    assert all(status >= 400 for status in statuses), statuses
