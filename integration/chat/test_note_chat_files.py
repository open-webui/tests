"""Journey: a chat opened from a note is given the note's attached files, whoever opened it.

A note carries its own files, and every message in a chat opened from the note gets their text as
context (docs: "Attached files feed the note's chat"); read access on the note is enough to open
such a chat. The owner uploads a text file, attaches it in a note shared read-only with a reader,
and each of them opens the note's chat and asks with the files the note editor sends along (the
attachments on the open page, since b34885efc); the test reads what the provider was sent.

The reader's chat is given the file too: the retrieval step lets through files the asker owns,
reaches through a knowledge base, a channel, a shared chat or a model, or reaches through a note
whose owner owns the file (fix b612c8847, issue open-webui/open-webui#32011).
Twin of the note chat tests in e2e/notes/test_note_files.py.

Discriminates: passes on dev b5a20423e; in a backend copy with b612c8847's change to
utils/access_control/files.py undone the reader test fails. Retargeted for b34885efc, where files
sit in the page body and the client sends them with each message.
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


def note_content(*uploads: dict) -> dict:
    """A note body with the files attached the way the note editor stores them in the page."""
    blocks = [{"type": "paragraph", "content": [{"type": "text", "text": "see the file"}]}]
    html = "<p>see the file</p>"
    for uploaded in uploads:
        attrs = {"id": uploaded["id"], "name": uploaded["filename"], "size": len(FILE_TEXT)}
        blocks.append({"type": "fileAttachment", "attrs": attrs})
        html += (
            f'<div data-file-id="{attrs["id"]}" data-file-size="{attrs["size"]}">'
            f'<a href="/api/v1/files/{attrs["id"]}/content">{attrs["name"]}</a></div>'
        )
    return {"json": {"type": "doc", "content": blocks}, "html": html, "md": "see the file"}


def create_note(owner: Actor, uploaded: dict, grants: list[dict]) -> str:
    with owner.client() as client:
        created = client.post(
            "/api/v1/notes/create",
            json={
                "title": f"Harbour {uuid.uuid4().hex[:6]}",
                "data": {"content": note_content(uploaded)},
                "access_grants": grants,
            },
        )
    assert created.status_code == 200, created.text
    return created.json()["id"]


def page_files(asker: Actor, note_id: str) -> list[dict]:
    """The files the note editor sends with a message: the attachments on the open page."""
    with asker.client() as client:
        note = client.get(f"/api/v1/notes/{note_id}")
    assert note.status_code == 200, note.text
    [page] = note.json()["data"]["pages"]
    return [
        {**block["attrs"], "type": "file"}
        for block in page["content"]["json"]["content"]
        if block["type"] == "fileAttachment"
    ]


def sent_in_note_chat(asker: Actor, note_id: str, upstream) -> str:
    """Open the note's chat as `asker`, ask once and return what the provider was sent."""
    question = f"when does the pilot boat wait? {uuid.uuid4().hex[:6]}"
    upstream.queue(reply.text("Noted.", match=reply.answering(question)))
    files = page_files(asker, note_id)
    assert files, "the note's page lists no attached file"
    with asker.client() as client:
        opened = client.post(f"/api/v1/notes/{note_id}/chat")
        assert opened.status_code == 200, opened.text
        ask(client, question, chat_id=opened.json()["id"], files=files)
    [request] = [body for body in upstream.chat_requests() if reply.answering(question)(body)]
    return json.dumps(request["messages"])


@pytest.fixture
def shared_note(make_user) -> tuple[Actor, Actor, str]:
    owner, reader = make_user(), make_user()
    note_id = create_note(owner, upload(owner), [grant("user", reader.id, "read")])
    return owner, reader, note_id


def test_the_owners_note_chat_is_given_the_notes_file(shared_note, upstream):
    owner, _, note_id = shared_note
    assert FILE_TEXT in sent_in_note_chat(owner, note_id, upstream)


def test_a_readers_note_chat_is_given_the_notes_file(shared_note, upstream):
    _, reader, note_id = shared_note
    sent = sent_in_note_chat(reader, note_id, upstream)
    assert FILE_TEXT in sent, "a reader's chat on the shared note was not given the note's file"
