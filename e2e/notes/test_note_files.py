"""Journey: files on a note, as its owner, a reader and a writer meet them.

Upload files in the note's menu puts a file into the note as a chip in its text; the chip opens
the file's text, and the note's chat is given that text on every message. A reader of the shared
note sees the chip and opens it but cannot detach it, and the note's chat is the reader's to use
as well; a writer detaches the file for everyone. A file over the admin's Max Upload Size is
refused with the limit named, and a picture pasted into the note is shrunk to the size the
person's Image Compression asks for.

A reader's chat on the note is given the note's file too (docs: "Attached files feed the note's
chat", and read access is enough to open one): the file belongs to the owner and fix b612c8847
(issue open-webui/open-webui#32011) lets read access on a note open the files its owner attached.
The chip dialog shows the text the note stored beside the chip, but its Preview of a spreadsheet
or CSV reads the file itself, and a reader only gets that table with the fix.

Discriminates: in a backend copy with b612c8847's change to utils/access_control/files.py undone
the reader's chat test and the reader's CSV preview test fail (the dialog cannot load the CSV);
the plain reader chip test passes there, since the note carries the text. In a frontend
build whose file dialog shows "No content" for the text, whose note chip offers its remove button
to a reader and never saves a writer's removal, whose note upload skips the size check and whose
note paste skips image compression, every test but the owner's chat fails; in a backend copy
whose note chat leaves out the note's files the owner's chat test fails.

A picture picked with Upload files in the note's menu shows in the note and is stored with it.
Before 0.12.4 it was uploaded and left out of the note (open-webui/open-webui#32793, fix
4604f2a3b). That test fails in the pre-fix build of the 0.12.4 changes, which has the commit
reverted.

One test pins a fixed bug (PR open-webui/open-webui#32328, fix 998ad4c9f): when the note saved
while a file was still uploading, the server's echo of that save replaced the page's file list,
and the finished upload then updated an entry no longer in it. The note kept the file as
uploading with no id, even after a reload. The upload is held until the echo arrives. It passes
on the dev 5032e834e build and fails on that build with 998ad4c9f reverted.

Since dev b34885efc a note keeps its files as blocks in its text and its pasted pictures inline
as data URLs, where the tests now read them; an upload still in flight stays out of the saved
note until it finishes. The reader's chip test is red on dev b34885efc: the chip in the
reader's read-only editor offers its remove button.
"""

from __future__ import annotations

import base64
import io
import json
import time
import uuid
from dataclasses import dataclass
from typing import Callable
from urllib.parse import urlparse

import pytest
from PIL import Image
from playwright.sync_api import Locator, Page, Request, Route, expect

from harness import upstream as reply
from harness.access import grant
from harness.actors import Actor
from harness.web_retrieval import RETRIEVAL_CONFIG
from utils.chat_ui import expect_reply, send
from utils.note_content import note_files, note_images

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

FILE_TEXT = "The pilot boat waits at the lighthouse at noon."
FILE_NAME = "pilot.txt"
SHEET_NAME = "tides.csv"
SHEET_CSV = "Harbour,High tide\nLighthouse,06:40\n"

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


def stored_note(owner: Actor, note_id: str) -> dict:
    with owner.client() as client:
        return client.get(f"/api/v1/notes/{note_id}").json()


def stored_files(owner: Actor, note_id: str) -> list[dict]:
    return note_files(stored_note(owner, note_id))


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


def upload_to_note(page: Page, name: str, content: bytes, mime_type: str = "text/plain") -> None:
    menu = open_note_menu(page)
    with page.expect_file_chooser() as chooser:
        menu.get_by_role("button", name="Upload files").click()
    chooser.value.set_files({"name": name, "mimeType": mime_type, "buffer": content})


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
    expect(chip.locator("xpath=..").get_by_role("button", name="Remove File")).to_have_count(0)
    chip.click()
    dialog = page.get_by_role("dialog").filter(has_text=FILE_NAME)
    expect(dialog.get_by_text(FILE_TEXT)).to_be_visible()
    page.keyboard.press("Escape")
    expect(open_note_menu(page).get_by_role("button", name="Upload files")).to_have_count(0)


def test_a_reader_previews_a_spreadsheet_the_owner_put_on_the_note(page_for, make_user):
    owner, reader = make_user(), make_user()
    note_id = create_note(owner, [grant("user", reader.id, "read")])
    owner_page = page_for(owner)
    open_note(owner_page, note_id)
    upload_to_note(owner_page, SHEET_NAME, SHEET_CSV.encode(), "text/csv")
    expect(file_chip(owner_page, SHEET_NAME)).to_be_visible()
    eventually(lambda: stored_file_names(owner, note_id), [SHEET_NAME])

    page = page_for(reader)
    open_note(page, note_id)
    file_chip(page, SHEET_NAME).click()
    dialog = page.get_by_role("dialog").filter(has_text=SHEET_NAME)
    dialog.get_by_role("button", name="Preview", exact=True).click()

    expect(dialog.get_by_role("cell", name="06:40")).to_be_visible()
    expect(dialog.get_by_text("Failed to load Excel/CSV file")).to_have_count(0)


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
    chip.locator("xpath=..").get_by_role("button", name="Remove File").click()

    expect(page.get_by_role("main").get_by_text(FILE_NAME)).to_have_count(0)
    eventually(lambda: stored_file_names(note_with_file.owner, note_with_file.note_id), [])
    owner_page = page_for(note_with_file.owner)
    open_note(owner_page, note_with_file.note_id)
    expect(owner_page.get_by_role("main").get_by_text(FILE_NAME)).to_have_count(0)


def _is_upload(request: Request) -> bool:
    return request.method == "POST" and urlparse(request.url).path == "/api/v1/files/"


def stored_uploads(owner: Actor, note_id: str) -> list[tuple[str, bool]]:
    return [(entry.get("name"), bool(entry.get("id"))) for entry in stored_files(owner, note_id)]


def _wait_on_page(page: Page, check: Callable[[], bool]) -> None:
    """Wait up to 15 s for `check()` while the page keeps running its routes and socket frames."""
    for _ in range(75):
        if check():
            return
        page.wait_for_timeout(200)


@pytest.mark.regression
def test_a_file_still_uploading_when_the_note_saves_finishes_attaching(page_for, make_user):
    owner = make_user()
    note_id = create_note(owner)
    page = page_for(owner)
    note_echoes: list[str] = []
    page.on(
        "websocket",
        lambda socket: socket.on(
            "framereceived",
            lambda frame: note_echoes.append(str(frame)) if "events:note" in str(frame) else None,
        ),
    )
    held_uploads: list[Route] = []

    def hold_uploads(route: Route) -> None:
        if _is_upload(route.request):
            held_uploads.append(route)
        else:
            route.continue_()

    page.route("**/api/v1/files/**", hold_uploads)
    open_note(page, note_id)
    with page.expect_request(_is_upload):
        upload_to_note(page, FILE_NAME, FILE_TEXT.encode())

    # the title saves the note while the upload is held, and the server echoes that save back
    title = page.get_by_role("main").get_by_placeholder("Title")
    title.fill("Harbour pilots")
    title.press("Tab")
    _wait_on_page(page, lambda: any("Harbour pilots" in frame for frame in note_echoes))
    assert any("Harbour pilots" in frame for frame in note_echoes), "the save was never echoed"
    for upload in held_uploads:
        upload.continue_()

    page.unroute("**/api/v1/files/**")
    _wait_on_page(page, lambda: stored_uploads(owner, note_id) == [(FILE_NAME, True)])
    assert stored_uploads(owner, note_id) == [(FILE_NAME, True)], (
        "the finished upload never reached the note with its file id"
    )
    open_note(page, note_id)
    file_chip(page, FILE_NAME).click()
    dialog = page.get_by_role("dialog").filter(has_text=FILE_NAME)
    expect(dialog.get_by_text(FILE_TEXT)).to_be_visible()


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
    images = note_images(stored_note(owner, note_id))
    if not images or not images[0].startswith("data:image/"):
        return None
    encoded = images[0].split(",", 1)[1]
    return Image.open(io.BytesIO(base64.b64decode(encoded))).size


def test_a_picture_pasted_into_a_note_is_shrunk_by_the_persons_image_compression(
    page_for, make_user
):
    owner = make_user()
    compression = {"imageCompression": True, "imageCompressionSize": {"width": 24, "height": 16}}
    with owner.client() as client:
        saved = client.post("/api/v1/users/user/settings/update", json={"ui": compression})
    assert saved.status_code == 200, saved.text
    note_id = create_note(owner)
    page = page_for(owner)
    editor = open_note(page, note_id)

    editor.click()
    editor.evaluate(PASTE_IMAGE, buoy_png())

    expect(editor.locator("img")).to_have_count(1)
    eventually(lambda: stored_image_size(owner, note_id), (24, 16))


@pytest.mark.regression
def test_a_picture_picked_with_upload_files_shows_in_the_note(page_for, make_user):
    # open-webui/open-webui#32793, fix 4604f2a3b
    owner = make_user()
    note_id = create_note(owner)
    page = page_for(owner)
    editor = open_note(page, note_id)
    expect(editor.locator("img")).to_have_count(0)

    upload_to_note(page, "buoy.png", base64.b64decode(buoy_png()), "image/png")

    expect(
        editor.locator("img"), "the picture picked with Upload files is not in the note"
    ).to_have_count(1)
    eventually(
        lambda: [src.startswith("data:image/") for src in note_images(stored_note(owner, note_id))],
        [True],
    )
    page.reload()
    expect(editor.locator("img")).to_have_count(1)
