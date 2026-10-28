"""Regression: an image pasted into a note was lost when the note's content was cut and pasted.

Fix `e276d3335` (open-webui/open-webui#31513, issue #31512): the editor stores a pasted image
with a `data://` address, and rebuilding the note from HTML dropped every such image, so cutting
the note's content and pasting it back kept the text and lost the picture. The editor now reads
those images back. The test pastes a picture into a note, waits until the note stores it, cuts the
whole note and pastes it back, and looks for the picture in the editor, in the stored note and
after a reload.

Discriminates: passes on dev a5bc78300; with the frontend change of #31513 reverted, the pasted
back content holds the text but no image.

Since dev b34885efc the note keeps a pasted picture inline as a data URL instead of a `data://`
address beside a file list, and the test reads it there.
"""

from __future__ import annotations

import base64
import io
import re
import time

import pytest
from PIL import Image
from playwright.sync_api import Locator, Page, expect

from harness.actors import Actor
from utils.note_content import note_images

pytestmark = [pytest.mark.regression, pytest.mark.requires_browser, pytest.mark.requires_source]

CLIPBOARD = ["clipboard-read", "clipboard-write"]
TEXT = "the harbour at dawn"

PASTE_IMAGE = """(editor, pngBase64) => {
    const bytes = Uint8Array.from(atob(pngBase64), (c) => c.charCodeAt(0));
    const clipboard = new DataTransfer();
    clipboard.items.add(new File([bytes], 'square.png', { type: 'image/png' }));
    editor.dispatchEvent(
        new ClipboardEvent('paste', { clipboardData: clipboard, bubbles: true, cancelable: true })
    );
}"""


def _red_square_png() -> str:
    buffer = io.BytesIO()
    Image.new("RGB", (160, 160), (255, 0, 0)).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


def _stored_note(author: Actor, note_id: str) -> dict:
    with author.client() as client:
        return client.get(f"/api/v1/notes/{note_id}").json()


def _holds_text_and_image(note: dict) -> bool:
    markdown = str(((note.get("data") or {}).get("content") or {}).get("md"))
    images = note_images(note)
    return TEXT in markdown and len(images) == 1 and images[0].startswith("data:image/png")


def _wait_until_stored_image(author: Actor, note_id: str, timeout: float = 15.0) -> None:
    deadline = time.monotonic() + timeout
    note = _stored_note(author, note_id)
    while not _holds_text_and_image(note) and time.monotonic() < deadline:
        time.sleep(0.2)
        note = _stored_note(author, note_id)
    assert _holds_text_and_image(note), f"the note never stored the image: {note.get('data')}"


def _note_with_pasted_image(page: Page, author: Actor) -> tuple[str, Locator]:
    page.goto("/notes")
    page.get_by_role("main").get_by_role("button", name="Create", exact=True).click()
    expect(page).to_have_url(re.compile(r"/notes/[0-9a-f-]+$"))
    note_id = page.url.rsplit("/", 1)[1]
    editor = page.get_by_role("main").get_by_label("Write something...")
    editor.click()
    page.keyboard.type(TEXT)
    page.keyboard.press("Enter")
    editor.evaluate(PASTE_IMAGE, _red_square_png())
    expect(editor.locator("img")).to_have_count(1)
    _wait_until_stored_image(author, note_id)
    return note_id, editor


def test_a_pasted_image_survives_cutting_and_pasting_the_whole_note(page_for, make_user):
    author = make_user()
    page = page_for(author, permissions=CLIPBOARD)
    note_id, editor = _note_with_pasted_image(page, author)

    editor.click()
    page.keyboard.press("Control+a")
    page.keyboard.press("Control+x")
    expect(editor.locator("img")).to_have_count(0)
    page.keyboard.press("Control+v")

    expect(editor).to_contain_text(TEXT)
    expect(
        editor.locator("img"),
        "the image is gone after cutting and pasting the note (open-webui/open-webui#31512)",
    ).to_have_count(1)
    _wait_until_stored_image(author, note_id)
    page.reload()
    expect(editor.locator("img")).to_have_count(1)
