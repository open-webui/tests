"""Journey: opening a file attached to a chat shows what it holds, before and after sending.

A file attached in the composer, and the same file on the sent message, opens a dialog with its
name, its size and the text the server read out of it. Its Preview tab shows a CSV as a table of
its rows, each sheet of a spreadsheet on a tab of its own, a Word document as its rendered page and
Markdown with its formatting. A PowerPoint slide draws its title where the file aligns it: left,
right or, when the file leaves it open, centred. A PDF shows its extracted text; the Preview tab's
PDF viewer says "Failed to load PDF." in the suite's Chromium 131 for any PDF, so it is not driven
here. An attached picture opens full size from the sent message and downloads as the bytes that were
uploaded. Switching a long document to Using Entire Document in its dialog sends the model the
whole text where focused retrieval sends a few pieces of it, and a file's switch shows its own
mode after another attachment is removed. Since 98f7a6146 a file under the Full Text Threshold goes
whole without a switch, so those two tests set the threshold to 0 and Top K back to 3, and since
bd6fc15be a model in native function calling reads attached files through its tools unless File
Context is ticked. A file under the threshold, or any file with Always Use Full Text on, shows
Using Entire Document with no switch and says why on hover (c81aca931).

Every test runs in the main chat and in a side chat (`utils/chat_surface.py`). On dev c15f28649
the Entire Document test is red in a side chat: the further New Chat tab it opens never appears,
as the main chat's message box stays hidden after its tab is selected; it passes on dev e791cbdde.

Discriminates: passes on the dev ebc6add67 build. In a frontend build whose file dialog shows
"No content" for the text, no longer takes a CSV for a spreadsheet, hides the sheet tabs, never
loads a Word document, shows Markdown as plain text and ignores the Using Entire Document switch,
and whose picture preview never opens, every test fails. In the pre-fix build of 0.12.4
(a944564c2 reverted) the remaining file shows the removed file's Using Entire Document. The
PowerPoint title test fails on the dev 75358502c build (#32873: the left-aligned title starts 42%
across the slide) and passes with the fix 8fa77c828 (#32876). Both whole-file dialog cases fail in
a dev f659c0d12 backend copy whose `/api/config` leaves out the threshold (no mode shown at all)
and the first one in a copy that reads Always Use Full Text as on for every account.
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
from harness.web_retrieval import RETRIEVAL_CONFIG
from integration.retrieval.test_document_loaders import (
    DOCX,
    PPTX,
    XLSX,
    docx_bytes,
    pdf_bytes,
)
from utils.cached_chat import attach
from utils.chat_surface import ChatSurface
from utils.chat_ui import chat_scope, conversation, expect_reply, send, start_another_chat

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


def attach_and_open(chat: ChatSurface, name: str, content: str | bytes, mime_type: str) -> Locator:
    attach(chat, name, content, mime_type)
    chip = file_chip(chat_scope(chat).locator("form"), name)
    expect(chip).to_be_visible()
    # the chip spins until the server has read the file; its dialog loads the text only once
    expect(chip.locator(".spinner_ajPY")).to_have_count(0, timeout=PROCESSING_TIMEOUT_MS)
    return open_file(chat.page, chip, name)


def show_preview(dialog: Locator) -> None:
    dialog.get_by_role("button", name="Preview", exact=True).click()


def ask_with_attachments(chat: ChatSurface, upstream, question: str) -> None:
    upstream.queue(reply.text("Noted.", match=reply.answering(question)))
    send(chat, question)
    expect_reply(chat, "Noted.")


def test_a_text_file_opens_with_its_size_and_text_in_the_composer_and_the_message(
    chat_for, make_user, upstream
):
    chat = chat_for(make_user())
    text = "The harbour master is Ingrid.\nThe ferry leaves at seven."

    dialog = attach_and_open(chat, "harbour.txt", text, "text/plain")
    expect(dialog.get_by_text(f"{len(text.encode())}.0 B")).to_be_visible()
    expect(dialog.get_by_text("2 Extracted Lines")).to_be_visible()
    expect(dialog.get_by_text("The ferry leaves at seven.")).to_be_visible()
    chat.page.keyboard.press("Escape")
    expect(dialog).to_have_count(0)

    ask_with_attachments(chat, upstream, f"who runs the harbour? {uuid.uuid4().hex[:6]}")
    sent = open_file(chat.page, file_chip(conversation(chat), "harbour.txt"), "harbour.txt")
    expect(sent.get_by_text("The harbour master is Ingrid.")).to_be_visible()


def test_a_csv_previews_as_a_table_of_its_rows(chat_for, make_user):
    chat = chat_for(make_user())
    rows = "route,departs\nNorth pier,07:15\nSouth pier,08:40\n"

    dialog = attach_and_open(chat, "ferries.csv", rows, "text/csv")
    show_preview(dialog)

    expect(dialog.get_by_text("3 Rows")).to_be_visible()
    expect(dialog.get_by_role("row").filter(has_text="South pier")).to_contain_text("08:40")
    expect(dialog.get_by_role("row").filter(has_text="North pier")).to_contain_text("07:15")


def test_each_sheet_of_a_spreadsheet_previews_on_its_own_tab(chat_for, make_user):
    openpyxl = pytest.importorskip("openpyxl")
    workbook = openpyxl.Workbook()
    workbook.active.title = "Tides"
    workbook.active.append(["Monday", "high at noon"])
    workbook.create_sheet("Ferries").append(["North pier", "07:15"])
    buffer = io.BytesIO()
    workbook.save(buffer)
    chat = chat_for(make_user())

    dialog = attach_and_open(chat, "harbour.xlsx", buffer.getvalue(), XLSX)
    show_preview(dialog)

    expect(dialog.get_by_role("row").filter(has_text="Monday")).to_contain_text("high at noon")
    dialog.get_by_role("tab", name="Ferries", exact=True).click()
    expect(dialog.get_by_role("row").filter(has_text="North pier")).to_contain_text("07:15")
    expect(dialog.get_by_text("Monday")).to_have_count(0)


def test_a_word_document_previews_as_its_page(chat_for, make_user):
    chat = chat_for(make_user())
    letter = docx_bytes("Dear harbour master,", "The buoys need paint before May.")

    dialog = attach_and_open(chat, "letter.docx", letter, DOCX)
    show_preview(dialog)

    page_view = dialog.locator("section.docx")
    expect(page_view).to_contain_text("The buoys need paint before May.")
    expect(page_view).to_contain_text("Dear harbour master,")


def titled_deck(*alignments: str | None) -> bytes:
    """One title slide per entry, its title set to that alignment or left to the layout."""
    pptx = pytest.importorskip("pptx")
    from pptx.enum.text import PP_ALIGN

    deck = pptx.Presentation()
    for alignment in alignments:
        slide = deck.slides.add_slide(deck.slide_layouts[0])
        slide.shapes.title.text = "Harbour"
        if alignment:
            slide.shapes.title.text_frame.paragraphs[0].alignment = getattr(PP_ALIGN, alignment)
    buffer = io.BytesIO()
    deck.save(buffer)
    return buffer.getvalue()


def title_ink(dialog: Locator, number: int) -> tuple[float, float]:
    """Where the title's dark pixels start and end on a slide, as % of the slide width."""
    dialog.get_by_role("button", name=f"Slide {number}", exact=True).click()
    slide = dialog.get_by_role("img", name=f"Slide {number}", exact=True)
    expect(slide).to_be_visible()
    return tuple(
        slide.evaluate(
            """async image => {
                await image.decode();
                const canvas = document.createElement('canvas');
                canvas.width = image.naturalWidth;
                canvas.height = image.naturalHeight;
                const context = canvas.getContext('2d');
                context.drawImage(image, 0, 0);
                const { width, height } = canvas;
                const { data } = context.getImageData(0, 0, width, height);
                let first = width, last = -1;
                for (let y = Math.floor(height * 0.28); y < height * 0.48; y++) {
                    for (let x = 0; x < width; x++) {
                        const i = (y * width + x) * 4;
                        if (data[i] + data[i + 1] + data[i + 2] < 300) {
                            first = Math.min(first, x);
                            last = Math.max(last, x);
                        }
                    }
                }
                return [first / width * 100, last / width * 100];
            }"""
        )
    )


def test_a_slide_title_previews_with_the_alignment_the_file_gives_it(chat_for, make_user):
    """#32873: a title the file aligns left was drawn centred, as if it had no alignment."""
    chat = chat_for(make_user())
    deck = titled_deck("LEFT", None, "RIGHT")

    dialog = attach_and_open(chat, "harbour.pptx", deck, PPTX)
    show_preview(dialog)

    left_start, _ = title_ink(dialog, 1)
    assert left_start < 20, f"a left-aligned title starts {left_start:.0f}% across the slide"
    unset_start, unset_end = title_ink(dialog, 2)
    assert unset_start < unset_end and abs((unset_start + unset_end) / 2 - 50) < 5, (
        f"a title without alignment spans {unset_start:.0f}% to {unset_end:.0f}%, not centred"
    )
    _, right_end = title_ink(dialog, 3)
    assert right_end > 80, f"a right-aligned title ends {right_end:.0f}% across the slide"


def test_markdown_previews_with_its_formatting(chat_for, make_user):
    chat = chat_for(make_user())
    markdown = "# Tide table\n\n- **Monday**: high at noon\n- Tuesday: high at one\n"

    dialog = attach_and_open(chat, "tides.md", markdown, "text/markdown")
    show_preview(dialog)

    expect(dialog.get_by_role("heading", name="Tide table")).to_be_visible()
    expect(dialog.get_by_role("listitem")).to_have_count(2)
    expect(dialog.locator("strong").filter(has_text="Monday")).to_be_visible()


def test_a_pdf_shows_the_text_read_out_of_it(chat_for, make_user):
    chat = chat_for(make_user())

    dialog = attach_and_open(
        chat, "chart.pdf", pdf_bytes("Soundings near the reef"), "application/pdf"
    )

    expect(dialog.get_by_text("1 Extracted Lines")).to_be_visible()
    expect(dialog.get_by_text("Soundings near the reef")).to_be_visible()
    expect(dialog.get_by_role("button", name="Preview", exact=True)).to_be_visible()


def red_buoy_png() -> bytes:
    picture = io.BytesIO()
    Image.new("RGB", (40, 30), (200, 30, 30)).save(picture, format="PNG")
    return picture.getvalue()


def test_a_sent_picture_opens_full_size_and_downloads_as_uploaded(chat_for, make_user, upstream):
    chat = chat_for(make_user())
    picture = red_buoy_png()
    attach(chat, "buoy.png", picture, "image/png")
    expect(chat_scope(chat).get_by_role("button", name="Show image preview")).to_be_visible()
    ask_with_attachments(chat, upstream, f"what colour is the buoy? {uuid.uuid4().hex[:6]}")

    conversation(chat).get_by_role("button", name="Show image preview").click()
    download_button = chat.page.get_by_role("button", name="Download", exact=True)
    expect(download_button).to_be_visible()
    shown = chat.page.locator(".modal img").last
    expect(shown).to_be_visible()
    assert shown.evaluate("image => [image.naturalWidth, image.naturalHeight]") == [40, 30]

    with chat.page.expect_download() as downloaded:
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


@pytest.fixture
def retrieved_in_pieces(admin, preserve):
    """Every file sent as its three closest pieces until switched to Using Entire Document.

    Since 98f7a6146 a file under the Full Text Threshold goes whole and its dialog shows no switch,
    and Top K defaults to 10, more pieces than the long report has.
    """
    preserve(RETRIEVAL_CONFIG)
    with admin.client() as client:
        saved = client.post(
            RETRIEVAL_CONFIG[1], json={"RAG_FULL_TEXT_CHAR_THRESHOLD": 0, "TOP_K": 3}
        )
    assert saved.status_code == 200, saved.text


@pytest.mark.usefixtures("retrieved_in_pieces")
def test_a_files_mode_switch_shows_its_own_mode_after_another_file_is_removed(chat_for, make_user):
    """#32699 (fix a944564c2): the remaining file's switch showed the removed file's mode."""
    chat = chat_for(make_user())
    report, _ = long_report()
    dialog = attach_and_open(chat, "castles.txt", report, "text/plain")
    dialog.get_by_role("switch").click()
    expect(dialog.get_by_text("Using Entire Document")).to_be_visible()
    chat.page.keyboard.press("Escape")
    expect(dialog).to_have_count(0)
    dialog = attach_and_open(chat, "rivers.txt", "The Drau flows east.", "text/plain")
    expect(dialog.get_by_text("Using Focused Retrieval")).to_be_visible()
    chat.page.keyboard.press("Escape")
    expect(dialog).to_have_count(0)

    removed = file_chip(chat_scope(chat).locator("form"), "castles.txt")
    removed.hover()
    removed.locator("xpath=..").get_by_role("button", name="Remove File").click()
    expect(chat_scope(chat).locator("form").get_by_text("castles.txt")).to_have_count(0)

    dialog = open_file(
        chat.page, file_chip(chat_scope(chat).locator("form"), "rivers.txt"), "rivers.txt"
    )
    expect(dialog.get_by_text("Using Focused Retrieval")).to_be_visible()
    expect(dialog.get_by_role("switch")).to_have_attribute("aria-checked", "false")


@pytest.mark.usefixtures("retrieved_in_pieces", "file_context_on")
def test_using_the_entire_document_sends_the_model_all_of_it(chat_for, make_user, upstream):
    chat = chat_for(make_user())
    report, landmarks = long_report()
    dialog = attach_and_open(chat, "castles.txt", report, "text/plain")
    expect(dialog.get_by_text("Using Focused Retrieval")).to_be_visible()
    chat.page.keyboard.press("Escape")
    focused_question = f"which castles? {uuid.uuid4().hex[:6]}"
    ask_with_attachments(chat, upstream, focused_question)
    assert len(landmarks_sent(upstream, focused_question, landmarks)) < len(landmarks)

    chat = start_another_chat(chat)
    dialog = attach_and_open(chat, "castles.txt", report, "text/plain")
    dialog.get_by_role("switch").click()
    expect(dialog.get_by_text("Using Entire Document")).to_be_visible()
    chat.page.keyboard.press("Escape")
    whole_question = f"which castles, all of them? {uuid.uuid4().hex[:6]}"
    ask_with_attachments(chat, upstream, whole_question)

    assert landmarks_sent(upstream, whole_question, landmarks) == landmarks


@pytest.mark.parametrize(
    ("retrieval", "explanation"),
    [
        (
            {"RAG_FULL_TEXT_CHAR_THRESHOLD": 50000, "RAG_FULL_CONTEXT": False},
            "This attachment fits within the full text threshold, so its entire text is included "
            "automatically.",
        ),
        (
            {"RAG_FULL_TEXT_CHAR_THRESHOLD": 0, "RAG_FULL_CONTEXT": True},
            "Full text is always enabled by your administrator.",
        ),
    ],
    ids=["under the threshold", "always use full text"],
)
def test_a_file_the_settings_send_whole_says_so_and_offers_no_switch(
    chat_for, make_user, admin, preserve, retrieval, explanation
):
    preserve(RETRIEVAL_CONFIG)
    with admin.client() as client:
        saved = client.post(
            RETRIEVAL_CONFIG[1], json={**retrieval, "BYPASS_EMBEDDING_AND_RETRIEVAL": False}
        )
    assert saved.status_code == 200, saved.text
    chat = chat_for(make_user())

    dialog = attach_and_open(chat, "ferry.txt", "The ferry leaves at seven.", "text/plain")

    mode = dialog.get_by_text("Using Entire Document")
    expect(mode).to_be_visible()
    expect(dialog.get_by_role("switch")).to_have_count(0)
    mode.hover()
    expect(chat.page.get_by_text(explanation)).to_be_visible()
