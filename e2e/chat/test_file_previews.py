"""Journey: opening a file attached to a chat shows what it holds, before and after sending.

A file attached in the composer, and the same file on the sent message, opens a dialog with its
name, its size and the text the server read out of it. Its Preview tab shows a CSV as a table of
its rows, each sheet of a spreadsheet on a tab of its own, a Word document as its rendered page and
Markdown with its formatting. A PDF shows its extracted text; the Preview tab's PDF viewer says
"Failed to load PDF." in the suite's Chromium 131 for any PDF, so it is not driven here. An
attached picture opens full size from the sent message and downloads as the bytes that were
uploaded. Switching a long document to Using Entire Document in its dialog sends the model the
whole text where focused retrieval sends a few pieces of it.

Discriminates: passes on the dev ebc6add67 build. In a frontend build whose file dialog shows
"No content" for the text, no longer takes a CSV for a spreadsheet, hides the sheet tabs, never
loads a Word document, shows Markdown as plain text and ignores the Using Entire Document switch,
and whose picture preview never opens, every test fails.
"""

from __future__ import annotations

import io
import json
import uuid
from pathlib import Path

import pytest
from PIL import Image
from playwright.sync_api import Locator, Page, expect

from harness import upstream as reply
from integration.retrieval.test_document_loaders import (
    DOCX,
    XLSX,
    docx_bytes,
    pdf_bytes,
)
from utils.cached_chat import attach
from utils.chat_ui import conversation, expect_reply, send

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

PROCESSING_TIMEOUT_MS = 30_000


def file_chip(scope: Locator, name: str) -> Locator:
    return scope.get_by_role("button").filter(has_text=name).first


def open_file(page: Page, chip: Locator, name: str) -> Locator:
    """Click a file's chip and return its dialog once the server's reading shows."""
    chip.click()
    dialog = page.get_by_role("dialog").filter(has_text=name)
    expect(dialog.get_by_text("Extracted Lines")).to_be_visible(timeout=PROCESSING_TIMEOUT_MS)
    return dialog


def attach_and_open(page: Page, name: str, content: str | bytes, mime_type: str) -> Locator:
    attach(page, name, content, mime_type)
    chip = file_chip(page.locator("form"), name)
    expect(chip).to_be_visible()
    # the chip spins until the server has read the file; its dialog loads the text only once
    expect(chip.locator(".spinner_ajPY")).to_have_count(0, timeout=PROCESSING_TIMEOUT_MS)
    return open_file(page, chip, name)


def show_preview(dialog: Locator) -> None:
    dialog.get_by_role("button", name="Preview", exact=True).click()


def ask_with_attachments(page: Page, upstream, question: str) -> None:
    upstream.queue(reply.text("Noted.", match=reply.answering(question)))
    send(page, question)
    expect_reply(page, "Noted.")


def test_a_text_file_opens_with_its_size_and_text_in_the_composer_and_the_message(
    page_for, make_user, upstream
):
    page = page_for(make_user())
    text = "The harbour master is Ingrid.\nThe ferry leaves at seven."

    dialog = attach_and_open(page, "harbour.txt", text, "text/plain")
    expect(dialog.get_by_text(f"{len(text.encode())}.0 B")).to_be_visible()
    expect(dialog.get_by_text("2 Extracted Lines")).to_be_visible()
    expect(dialog.get_by_text("The ferry leaves at seven.")).to_be_visible()
    page.keyboard.press("Escape")
    expect(dialog).to_have_count(0)

    ask_with_attachments(page, upstream, f"who runs the harbour? {uuid.uuid4().hex[:6]}")
    sent = open_file(page, file_chip(conversation(page), "harbour.txt"), "harbour.txt")
    expect(sent.get_by_text("The harbour master is Ingrid.")).to_be_visible()


def test_a_csv_previews_as_a_table_of_its_rows(page_for, make_user):
    page = page_for(make_user())
    rows = "route,departs\nNorth pier,07:15\nSouth pier,08:40\n"

    dialog = attach_and_open(page, "ferries.csv", rows, "text/csv")
    show_preview(dialog)

    expect(dialog.get_by_text("3 Rows")).to_be_visible()
    expect(dialog.get_by_role("row").filter(has_text="South pier")).to_contain_text("08:40")
    expect(dialog.get_by_role("row").filter(has_text="North pier")).to_contain_text("07:15")


def test_each_sheet_of_a_spreadsheet_previews_on_its_own_tab(page_for, make_user):
    openpyxl = pytest.importorskip("openpyxl")
    workbook = openpyxl.Workbook()
    workbook.active.title = "Tides"
    workbook.active.append(["Monday", "high at noon"])
    workbook.create_sheet("Ferries").append(["North pier", "07:15"])
    buffer = io.BytesIO()
    workbook.save(buffer)
    page = page_for(make_user())

    dialog = attach_and_open(page, "harbour.xlsx", buffer.getvalue(), XLSX)
    show_preview(dialog)

    expect(dialog.get_by_role("row").filter(has_text="Monday")).to_contain_text("high at noon")
    dialog.get_by_role("button", name="Ferries", exact=True).click()
    expect(dialog.get_by_role("row").filter(has_text="North pier")).to_contain_text("07:15")
    expect(dialog.get_by_text("Monday")).to_have_count(0)


def test_a_word_document_previews_as_its_page(page_for, make_user):
    page = page_for(make_user())
    letter = docx_bytes("Dear harbour master,", "The buoys need paint before May.")

    dialog = attach_and_open(page, "letter.docx", letter, DOCX)
    show_preview(dialog)

    page_view = dialog.locator("section.docx")
    expect(page_view).to_contain_text("The buoys need paint before May.")
    expect(page_view).to_contain_text("Dear harbour master,")


def test_markdown_previews_with_its_formatting(page_for, make_user):
    page = page_for(make_user())
    markdown = "# Tide table\n\n- **Monday**: high at noon\n- Tuesday: high at one\n"

    dialog = attach_and_open(page, "tides.md", markdown, "text/markdown")
    show_preview(dialog)

    expect(dialog.get_by_role("heading", name="Tide table")).to_be_visible()
    expect(dialog.get_by_role("listitem")).to_have_count(2)
    expect(dialog.locator("strong").filter(has_text="Monday")).to_be_visible()


def test_a_pdf_shows_the_text_read_out_of_it(page_for, make_user):
    page = page_for(make_user())

    dialog = attach_and_open(
        page, "chart.pdf", pdf_bytes("Soundings near the reef"), "application/pdf"
    )

    expect(dialog.get_by_text("1 Extracted Lines")).to_be_visible()
    expect(dialog.get_by_text("Soundings near the reef")).to_be_visible()
    expect(dialog.get_by_role("button", name="Preview", exact=True)).to_be_visible()


def red_buoy_png() -> bytes:
    picture = io.BytesIO()
    Image.new("RGB", (40, 30), (200, 30, 30)).save(picture, format="PNG")
    return picture.getvalue()


def test_a_sent_picture_opens_full_size_and_downloads_as_uploaded(page_for, make_user, upstream):
    page = page_for(make_user())
    picture = red_buoy_png()
    attach(page, "buoy.png", picture, "image/png")
    expect(page.get_by_role("button", name="Show image preview")).to_be_visible()
    ask_with_attachments(page, upstream, f"what colour is the buoy? {uuid.uuid4().hex[:6]}")

    conversation(page).get_by_role("button", name="Show image preview").click()
    download_button = page.get_by_role("button", name="Download", exact=True)
    expect(download_button).to_be_visible()
    shown = page.locator(".modal img").last
    expect(shown).to_be_visible()
    assert shown.evaluate("image => [image.naturalWidth, image.naturalHeight]") == [40, 30]

    with page.expect_download() as downloaded:
        download_button.click()
    assert Path(downloaded.value.path()).read_bytes() == picture


def long_report() -> tuple[str, list[str]]:
    """Six paragraphs of about a chunk each, every one naming its own landmark."""
    landmarks = ["Hochosterwitz", "Landskron", "Griffen", "Taggenbrunn", "Glanegg", "Liebenfels"]
    paragraphs = [f"{name} castle. " + "The walls stand on the hill. " * 32 for name in landmarks]
    return "\n\n".join(paragraphs), landmarks


def landmarks_sent(upstream, question: str, landmarks: list[str]) -> list[str]:
    [request] = [body for body in upstream.chat_requests() if reply.answering(question)(body)]
    sent = json.dumps(request["messages"])
    return [name for name in landmarks if name in sent]


def test_using_the_entire_document_sends_the_model_all_of_it(page_for, make_user, upstream):
    page = page_for(make_user())
    report, landmarks = long_report()
    dialog = attach_and_open(page, "castles.txt", report, "text/plain")
    expect(dialog.get_by_text("Using Focused Retrieval")).to_be_visible()
    page.keyboard.press("Escape")
    focused_question = f"which castles? {uuid.uuid4().hex[:6]}"
    ask_with_attachments(page, upstream, focused_question)
    assert len(landmarks_sent(upstream, focused_question, landmarks)) < len(landmarks)

    page.goto("/")
    dialog = attach_and_open(page, "castles.txt", report, "text/plain")
    dialog.get_by_role("switch").click()
    expect(dialog.get_by_text("Using Entire Document")).to_be_visible()
    page.keyboard.press("Escape")
    whole_question = f"which castles, all of them? {uuid.uuid4().hex[:6]}"
    ask_with_attachments(page, upstream, whole_question)

    assert landmarks_sent(upstream, whole_question, landmarks) == landmarks
