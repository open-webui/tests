"""Journey: a note downloaded as a PDF carries the image pasted into it.

Open WebUI 0.11.5 draws a note's pasted images into its PDF (open-webui/open-webui#30975, issue
#30974); before that the PDF held only the title and the text. A pasted image is stored with the
note as a data URL and the note's HTML points at it by file id, which the PDF export now resolves.
The export renders the note to one picture per page, so the test pastes a solid red square and
counts red pixels on the page; the title shows as the file name and the text as dark ink.

Discriminates: passes on dev ef67cc3fa; with the frontend change of #30975 reverted the pasted
note's PDF has no red pixels, while the text-only note passes on both.

The pasted image test is red now and then on dev 22102e4a2: when an earlier save of the note is
echoed back to the tab after the paste, the paste is stored with an empty file list and the PDF
has no picture (open-webui/open-webui#32123).
"""

from __future__ import annotations

import base64
import io
import re
import time

import numpy
import pytest
from PIL import Image
from playwright.sync_api import Page, expect
from pypdf import PdfReader

from harness.actors import Actor

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

TEXT = "the harbour at dawn"
# a red square large enough to count, small enough to stay unscaled on the page
SQUARE_SIZE = 160

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
    Image.new("RGB", (SQUARE_SIZE, SQUARE_SIZE), (255, 0, 0)).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


def _stored_note(author: Actor, note_id: str) -> dict:
    with author.client() as client:
        return client.get(f"/api/v1/notes/{note_id}").json()


def _wait_until_stored(
    author: Actor, note_id: str, text: str, image_count: int, timeout: float = 15.0
) -> None:
    deadline = time.monotonic() + timeout
    data: dict = {}
    while time.monotonic() < deadline:
        data = _stored_note(author, note_id).get("data") or {}
        html = str((data.get("content") or {}).get("html"))
        pasted = html.count('src="data://')
        if text in html and pasted == image_count and len(data.get("files") or []) == image_count:
            return
        time.sleep(0.2)
    raise AssertionError(
        f"the note never stored {text!r} with {image_count} images; it holds {data}"
    )


def _write_note(page: Page, author: Actor, title: str, with_image: bool) -> str:
    page.goto("/notes")
    page.get_by_role("main").get_by_role("button", name="Create", exact=True).click()
    expect(page).to_have_url(re.compile(r"/notes/[0-9a-f-]+$"))
    note_id = page.url.rsplit("/", 1)[1]

    editor_page = page.get_by_role("main")
    editor_page.get_by_role("textbox", name="Title").fill(title)
    editor = editor_page.get_by_label("Write something...")
    editor.click()
    page.keyboard.type(TEXT)
    page.keyboard.press("Enter")
    if with_image:
        editor.evaluate(PASTE_IMAGE, _red_square_png())
        expect(editor.locator("img")).to_have_count(1)
    _wait_until_stored(author, note_id, TEXT, image_count=1 if with_image else 0)
    return note_id


def _download_pdf(page: Page, title: str) -> tuple[str, Image.Image]:
    page.goto("/notes")
    card = page.get_by_role("main").get_by_role("button", name="Open note").filter(has_text=title)
    card.get_by_role("button", name="Note Menu").first.click()
    menu = page.get_by_role("menu")
    menu.get_by_role("button", name="Download").hover()
    with page.expect_download() as download_info:
        page.get_by_role("button", name="PDF document (.pdf)").click()
    download = download_info.value
    reader = PdfReader(download.path())
    rendered = reader.pages[0].images[0].image.convert("RGB")
    return download.suggested_filename, rendered


def _red_pixels(picture: Image.Image) -> int:
    red, green, blue = numpy.asarray(picture, dtype=numpy.int16).transpose(2, 0, 1)
    return int(((red > 200) & (green < 70) & (blue < 70)).sum())


def _ink_pixels(picture: Image.Image) -> int:
    return int((numpy.asarray(picture).max(axis=2) < 90).sum())


def test_a_pasted_image_is_drawn_into_the_note_pdf(page_for, make_user):
    author = make_user()
    page = page_for(author)
    title = "Harbour sketch"
    _write_note(page, author, title, with_image=True)

    filename, rendered = _download_pdf(page, title)

    assert filename == f"{title}.pdf"
    red_pixels = _red_pixels(rendered)
    # the square renders at twice its size; allow for JPEG edges
    assert red_pixels > (SQUARE_SIZE * 2) ** 2 // 2, f"only {red_pixels} red pixels in the PDF"
    assert _ink_pixels(rendered) > 200, "the PDF shows no title or text"


# ---------------------------------------------------------------- nearby


def test_a_text_only_note_pdf_keeps_its_title_and_text(page_for, make_user):
    author = make_user()
    page = page_for(author)
    title = "Harbour notes"
    _write_note(page, author, title, with_image=False)

    filename, rendered = _download_pdf(page, title)

    assert filename == f"{title}.pdf"
    assert _ink_pixels(rendered) > 200, "the PDF shows no title or text"
    assert _red_pixels(rendered) == 0
