"""Journey: a chat opened from a note is given the note's attached files, whoever opened it.

A note carries its own files, and every message in a chat opened from the note gets their text as
context (docs: "Attached files feed the note's chat"); read access on the note is enough to open
such a chat. The owner uploads a text file, puts it on a note shared read-only with a reader, and
each of them opens the note's chat and asks; the test reads what the provider was sent.

The reader's chat is not given the file: the note adds its files to the request, but the retrieval
step lets through only files the asker owns or reaches through a knowledge base, a channel, a
shared chat or a model, never through a note. That test stays red until a note's grant counts.
Twin of the note chat tests in e2e/notes/test_note_files.py.

Discriminates: passes on dev ebc6add67 except the reader test; in a backend copy whose note chat
leaves out the note's files the owner test fails too.
"""

from __future__ import annotations

import json
import uuid

import pytest

from harness import upstream as reply
from harness.access import grant
from harness.actors import Actor
from harness.chat import ask

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source]

FILE_TEXT = "The pilot boat waits at the lighthouse at noon."


def upload(owner: Actor) -> dict:
    with owner.client() as client:
        uploaded = client.post(
            "/api/v1/files/", files={"file": ("pilot.txt", FILE_TEXT.encode(), "text/plain")}
        )
    assert uploaded.status_code == 200, uploaded.text
    return uploaded.json()


def note_entry(uploaded: dict) -> dict:
    """The file the way the note editor stores an upload on the note."""
    return {
        "type": "file",
        "id": uploaded["id"],
        "url": uploaded["id"],
        "name": "pilot.txt",
        "status": "uploaded",
        "size": len(FILE_TEXT),
        "collection_name": f"file-{uploaded['id']}",
        "file": uploaded,
    }


def create_note(owner: Actor, files: list[dict], grants: list[dict]) -> str:
    with owner.client() as client:
        created = client.post(
            "/api/v1/notes/create",
            json={
                "title": f"Harbour {uuid.uuid4().hex[:6]}",
                "data": {"content": {"md": "see the file"}, "files": files},
                "access_grants": grants,
            },
        )
    assert created.status_code == 200, created.text
    return created.json()["id"]


def sent_in_note_chat(asker: Actor, note_id: str, upstream) -> str:
    """Open the note's chat as `asker`, ask once and return what the provider was sent."""
    question = f"when does the pilot boat wait? {uuid.uuid4().hex[:6]}"
    upstream.queue(reply.text("Noted.", match=reply.answering(question)))
    with asker.client() as client:
        opened = client.post(f"/api/v1/notes/{note_id}/chat")
        assert opened.status_code == 200, opened.text
        ask(client, question, chat_id=opened.json()["id"])
    [request] = [body for body in upstream.chat_requests() if reply.answering(question)(body)]
    return json.dumps(request["messages"])


@pytest.fixture
def shared_note(make_user) -> tuple[Actor, Actor, str]:
    owner, reader = make_user(), make_user()
    note_id = create_note(owner, [note_entry(upload(owner))], [grant("user", reader.id, "read")])
    return owner, reader, note_id


def test_the_owners_note_chat_is_given_the_notes_file(shared_note, upstream):
    owner, _, note_id = shared_note
    assert FILE_TEXT in sent_in_note_chat(owner, note_id, upstream)


def test_a_readers_note_chat_is_given_the_notes_file(shared_note, upstream):
    _, reader, note_id = shared_note
    sent = sent_in_note_chat(reader, note_id, upstream)
    assert FILE_TEXT in sent, "a reader's chat on the shared note was not given the note's file"
