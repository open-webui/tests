"""Journey: files on a note, as its owner, a reader and a writer meet them.

Upload files in the note's menu puts a file on the note as a chip above the text; the chip opens
the file's text, and the note's chat is given that text on every message. A reader of the shared
note sees the chip and opens it but cannot detach it, and the note's chat is the reader's to use
as well; a writer detaches the file for everyone. A file over the admin's Max Upload Size is
refused with the limit named, and a picture pasted into the note is shrunk to the size the
person's Image Compression asks for.

A reader's chat on the note is not given the note's file (docs: "Attached files feed the note's
chat", and read access is enough to open one): the file belongs to the owner and the retrieval
step only lets through files the asker owns or reaches through a knowledge base, a channel, a
shared chat or a model, never through a note. That test stays red until the note grants count.

Discriminates: passes on the dev ebc6add67 build except the reader's chat test. In a frontend
build whose file dialog shows "No content" for the text, whose note chip offers its remove button
to a reader and never saves a writer's removal, whose note upload skips the size check and whose
note paste skips image compression, every test but the owner's chat and the uncompressed paste
fails; in a backend copy whose note chat leaves out the note's files the owner's chat test fails.
"""

from __future__ import annotations

import base64
import io
import json
import time
import uuid
from dataclasses import dataclass
from typing import Callable

import pytest
from PIL import Image
from playwright.sync_api import Locator, Page, expect

from harness import upstream as reply
from harness.access import grant
from harness.actors import Actor
from harness.web_retrieval import RETRIEVAL_CONFIG
from utils.chat_ui import expect_reply, send

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

FILE_TEXT = "The pilot boat waits at the lighthouse at noon."
FILE_NAME = "pilot.txt"

PASTE_IMAGE = """(editor, pngBase64) => {
    const bytes = Uint8Array.from(atob(pngBase64), (c) => c.charCodeAt(0));
    const clipboard = new DataTransfer();
    clipboard.items.add(new File([bytes], 'buoy.png', { type: 'image/png' }));
    editor.dispatchEvent(
        new ClipboardEvent('paste', { clipboardData: clipboard, bubbles: true, cancelable: true })
    );
}"""


def create_note(owner: Actor, grants: list[dict] | None = None) -> str:
    with owner.client() as client:
        created = client.post(
            "/api/v1/notes/create",
            json={
                "title": f"Harbour {uuid.uuid4().hex[:6]}",
                "data": {"content": {"md": "see the file"}},
                "access_grants": grants or [],
            },
        )
    assert created.status_code == 200, created.text
    return created.json()["id"]


def stored_files(owner: Actor, note_id: str) -> list[dict]:
    with owner.client() as client:
        note = client.get(f"/api/v1/notes/{note_id}").json()
    return (note.get("data") or {}).get("files") or []


def eventually(read: Callable[[], object], expected: object, timeout: float = 15.0) -> None:
    """Wait until `read()` returns `expected`; the editor saves a moment after a change."""
    deadline = time.monotonic() + timeout
    current = read()
    while current != expected and time.monotonic() < deadline:
        time.sleep(0.2)
        current = read()
    assert current == expected, current


def stored_file_names(owner: Actor, note_id: str) -> list[str]:
    return [entry.get("name") for entry in stored_files(owner, note_id)]


def open_note(page: Page, note_id: str) -> Locator:
    page.goto(f"/notes/{note_id}")
    editor = page.get_by_role("main").get_by_label("Write something...")
    expect(editor).to_contain_text("see the file")
    return editor


def open_note_menu(page: Page) -> Locator:
    # the "..." button has no name; it is the last menu button in the editor's header
    page.get_by_role("main").locator("[aria-haspopup=true]:visible").last.click()
    menu = page.get_by_role("menu")
    expect(menu.get_by_role("button", name="Download")).to_be_visible()
    return menu


def upload_to_note(page: Page, name: str, content: bytes) -> None:
    menu = open_note_menu(page)
    with page.expect_file_chooser() as chooser:
        menu.get_by_role("button", name="Upload files").click()
    chooser.value.set_files({"name": name, "mimeType": "text/plain", "buffer": content})


def file_chip(page: Page, name: str) -> Locator:
    return page.get_by_role("main").get_by_role("button").filter(has_text=name).first


@dataclass
class SharedNote:
    owner: Actor
    reader: Actor
    writer: Actor
    note_id: str


@pytest.fixture
def note_with_file(page_for, make_user) -> SharedNote:
    """A note shared with a reader and a writer, whose owner uploaded the file in its menu."""
    owner, reader, writer = make_user(), make_user(), make_user()
    grants = [
        grant("user", reader.id, "read"),
        grant("user", writer.id, "read"),
        grant("user", writer.id, "write"),
    ]
    note_id = create_note(owner, grants)
    page = page_for(owner)
    open_note(page, note_id)
    upload_to_note(page, FILE_NAME, FILE_TEXT.encode())
    expect(file_chip(page, FILE_NAME)).to_be_visible()
    eventually(lambda: stored_file_names(owner, note_id), [FILE_NAME])
    return SharedNote(owner, reader, writer, note_id)


def ask_note_chat(page: Page, upstream, question: str) -> dict:
    """Ask in the note's chat panel; returns what the provider was sent for the question."""
    upstream.queue(reply.text("Noted.", match=reply.answering(question)))
    page.get_by_role("main").get_by_role("button", name="Chat", exact=True).click()
    send(page, question)
    expect_reply(page, "Noted.")
    [request] = [body for body in upstream.chat_requests() if reply.answering(question)(body)]
    return request


def test_an_uploaded_file_shows_on_the_note_and_opens_its_text(note_with_file, page_for):
    page = page_for(note_with_file.owner)
    open_note(page, note_with_file.note_id)

    file_chip(page, FILE_NAME).click()

    dialog = page.get_by_role("dialog").filter(has_text=FILE_NAME)
    expect(dialog.get_by_text(FILE_TEXT)).to_be_visible()
    expect(dialog.get_by_text("1 Extracted Lines")).to_be_visible()


def test_the_owners_note_chat_is_given_the_notes_file(note_with_file, page_for, upstream):
    page = page_for(note_with_file.owner)
    open_note(page, note_with_file.note_id)

    request = ask_note_chat(
        page, upstream, f"when does the pilot boat wait? {uuid.uuid4().hex[:6]}"
    )

    assert FILE_TEXT in json.dumps(request["messages"])


def test_a_reader_opens_the_notes_file_but_cannot_detach_it(note_with_file, page_for):
    page = page_for(note_with_file.reader)
    open_note(page, note_with_file.note_id)
    expect(page.get_by_role("main").get_by_text("Read-Only Access")).to_be_visible()

    chip = file_chip(page, FILE_NAME)
    chip.hover()
    expect(chip.get_by_role("button", name="Remove File")).to_have_count(0)
    chip.click()
    dialog = page.get_by_role("dialog").filter(has_text=FILE_NAME)
    expect(dialog.get_by_text(FILE_TEXT)).to_be_visible()
    page.keyboard.press("Escape")
    expect(open_note_menu(page).get_by_role("button", name="Upload files")).to_have_count(0)


def test_a_readers_note_chat_is_given_the_notes_file(note_with_file, page_for, upstream):
    page = page_for(note_with_file.reader)
    open_note(page, note_with_file.note_id)

    request = ask_note_chat(
        page, upstream, f"when does the pilot boat wait? {uuid.uuid4().hex[:6]}"
    )

    assert FILE_TEXT in json.dumps(request["messages"]), (
        "a reader's chat on the shared note was not given the note's file; the owner's is"
    )


def test_a_writer_detaches_the_file_for_everyone(note_with_file, page_for):
    page = page_for(note_with_file.writer)
    open_note(page, note_with_file.note_id)

    chip = file_chip(page, FILE_NAME)
    chip.hover()
    chip.get_by_role("button", name="Remove File").click()

    expect(page.get_by_role("main").get_by_text(FILE_NAME)).to_have_count(0)
    eventually(lambda: stored_file_names(note_with_file.owner, note_with_file.note_id), [])
    owner_page = page_for(note_with_file.owner)
    open_note(owner_page, note_with_file.note_id)
    expect(owner_page.get_by_role("main").get_by_text(FILE_NAME)).to_have_count(0)


def test_a_file_over_the_max_upload_size_is_refused_on_a_note(page_for, make_user, admin, preserve):
    preserve(RETRIEVAL_CONFIG)
    with admin.client() as client:
        saved = client.post(RETRIEVAL_CONFIG[1], json={"FILE_MAX_SIZE": 1})
    assert saved.status_code == 200, saved.text
    owner = make_user()
    note_id = create_note(owner)
    page = page_for(owner)
    open_note(page, note_id)

    upload_to_note(page, "atlas.txt", b"x" * (2 * 1024 * 1024))

    expect(page.get_by_text("File size should not exceed 1 MB.")).to_be_visible()
    expect(page.get_by_role("main").get_by_text("atlas.txt")).to_have_count(0)
    assert stored_files(owner, note_id) == []


def buoy_png() -> str:
    picture = io.BytesIO()
    Image.new("RGB", (96, 64), (200, 30, 30)).save(picture, format="PNG")
    return base64.b64encode(picture.getvalue()).decode()


def stored_image_size(owner: Actor, note_id: str) -> tuple[int, int] | None:
    images = [entry for entry in stored_files(owner, note_id) if entry.get("type") == "image"]
    if not images:
        return None
    encoded = images[0]["url"].split(",", 1)[1]
    return Image.open(io.BytesIO(base64.b64decode(encoded))).size


@pytest.mark.parametrize(
    ("compression", "expected_size"),
    [(None, (96, 64)), ({"width": 24, "height": 16}, (24, 16))],
    ids=["at-its-own-size", "compressed"],
)
def test_a_picture_pasted_into_a_note_follows_the_persons_image_compression(
    page_for, make_user, compression, expected_size
):
    owner = make_user()
    if compression:
        with owner.client() as client:
            saved = client.post(
                "/api/v1/users/user/settings/update",
                json={"ui": {"imageCompression": True, "imageCompressionSize": compression}},
            )
        assert saved.status_code == 200, saved.text
    note_id = create_note(owner)
    page = page_for(owner)
    editor = open_note(page, note_id)

    editor.click()
    editor.evaluate(PASTE_IMAGE, buoy_png())

    expect(editor.locator("img")).to_have_count(1)
    eventually(lambda: stored_image_size(owner, note_id), expected_size)
