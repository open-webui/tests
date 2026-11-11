"""Journey: a Word document and a presentation open in the file browser, with their comments.

A terminal that has LibreOffice converts a .docx or .pptx to a PDF for the preview; one that has
none hands over the file and the page draws it itself. Both are driven here. The first on a
scripted terminal that answers the preview request with a PDF of its own: the page shows that
PDF's pages, a presentation one slide at a time, and still reads the comments from the original
file. The second on a machine of a real Terminals orchestrator, whose Open Terminal image has no
LibreOffice: the page lays the document out and draws the slides itself, each with the
reviewer's comment, its author and text. A file that only carries the .docx name says it failed
to load and that its comments could not be read.

Discriminates: all pass on dev c15f28649 with Terminals 0.4.1 and Open Terminal 0.17.7. On
frontend builds of it with these mutations each named test fails: the terminal's PDF never asked
for (both converted tests); the page's own drawing of a .docx skipped, and on another build its
comments not handed to the drawing (laid-out Word document); the page's own drawing of a .pptx
skipped (drawn presentation); the failure of a broken document swallowed (broken document).
"""

from __future__ import annotations

import uuid

import pytest
from playwright.sync_api import Locator, expect

from harness import sample_documents as samples
from harness import slow_machine
from harness.sample_documents import Note
from harness.terminal_files import HOME, serving_files
from harness.terminal_server import TERMINAL_SERVERS_CONFIG
from utils.cached_chat import choose_terminal, open_files_tab
from utils.file_views import comment_shown_by, files_of, open_file

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

# the page unpacks the document and lays it out, or renders a PDF, on first use of each viewer
DOCUMENT_LOAD_TIMEOUT_MS = slow_machine.scaled(8_000)
LETTER = ["Dear harbour master,", "the cider boat lands on Friday."]
LETTER_NOTE = Note("Ines Calder", "Name the quay she lands at")
DECK = ["Harvest review", "Sales by market"]
DECK_NOTE = Note("Ines Calder", "A bigger title here")


@pytest.fixture
def person(make_user):
    return make_user()


@pytest.fixture
def converting_terminal():
    """A scripted terminal that converts a document to a PDF of its own for the preview."""
    with serving_files() as server:
        yield server


def _asked_for_a_pdf(terminal, name: str) -> bool:
    return any(
        name in request.path and "preview=true" in request.path
        for request in terminal.requests_to("/files/view")
    )


def test_a_word_document_shows_as_the_pdf_the_terminal_made_of_it(
    converting_terminal, person, page_for, admin, preserve
):
    converting_terminal.files["letter.docx"] = samples.docx(LETTER, LETTER_NOTE)
    converting_terminal.converted["letter.docx"] = samples.pdf(
        ["The letter as the terminal printed it", "Its second sheet"]
    )
    preserve(TERMINAL_SERVERS_CONFIG)
    browser = files_of(page_for, admin, person, converting_terminal)

    open_file(browser, "letter.docx")

    viewer = browser.get_by_role("application", name="Page viewer")
    expect(viewer.get_by_text("The letter as the terminal printed it")).to_be_visible(
        timeout=DOCUMENT_LOAD_TIMEOUT_MS
    )
    expect(browser.get_by_role("complementary", name="Pages").get_by_role("button")).to_have_count(
        2
    )
    # the document's own text is nowhere: the page did not draw it a second time
    expect(browser.get_by_text(LETTER[0])).to_have_count(0)
    assert _asked_for_a_pdf(converting_terminal, "letter.docx"), (
        "the terminal was never asked to convert the document"
    )
    # the comment is in the document, which the page reads beside the terminal's PDF
    browser.get_by_role("button", name="Document comments").click()
    comment = browser.page.get_by_role("dialog", name="Comment", exact=True)
    expect(comment.get_by_text(LETTER_NOTE.author, exact=True)).to_be_visible()
    expect(comment.get_by_text(LETTER_NOTE.text, exact=True)).to_be_visible()


def test_a_presentation_shows_the_terminals_pdf_one_slide_at_a_time(
    converting_terminal, person, page_for, admin, preserve
):
    converting_terminal.files["deck.pptx"] = samples.pptx(DECK, DECK_NOTE)
    converting_terminal.converted["deck.pptx"] = samples.pdf(
        ["First slide as printed", "Second slide as printed"]
    )
    preserve(TERMINAL_SERVERS_CONFIG)
    browser = files_of(page_for, admin, person, converting_terminal)

    open_file(browser, "deck.pptx")

    viewer = browser.get_by_role("application", name="Page viewer")
    expect(viewer.get_by_text("First slide as printed")).to_be_visible(
        timeout=DOCUMENT_LOAD_TIMEOUT_MS
    )
    expect(viewer.get_by_text("Second slide as printed")).to_have_count(0)
    expect(browser.get_by_text("1 / 2", exact=True)).to_be_visible()
    comment = comment_shown_by(
        browser.page, viewer.get_by_role("button", name=f"Comment: {DECK_NOTE.author}")
    )
    expect(comment.get_by_text(DECK_NOTE.text, exact=True)).to_be_visible()

    browser.page.keyboard.press("Escape")
    browser.get_by_role("button", name="Next page").click()

    expect(viewer.get_by_text("Second slide as printed")).to_be_visible()
    expect(viewer.get_by_text("First slide as printed")).to_have_count(0)
    expect(browser.get_by_text("2 / 2", exact=True)).to_be_visible()
    # the comment belongs to the first slide
    expect(viewer.get_by_role("button", name=f"Comment: {DECK_NOTE.author}")).to_have_count(0)


@pytest.fixture
def machine_files(person, shared_terminal, page_for):
    """`machine_files({name: bytes})` uploads the files into the person's machine of a terminal
    on the Terminals orchestrator and opens its file browser in their chat."""

    def open_browser(files: dict[str, bytes]) -> Locator:
        name = f"Workbench {uuid.uuid4().hex[:6]}"
        terminal_id = shared_terminal(person, name=name)
        with person.client() as client:
            for file_name, content in files.items():
                uploaded = client.post(
                    f"/api/v1/terminals/{terminal_id}/files/upload",
                    params={"directory": HOME},
                    files={"file": (file_name, content, "application/octet-stream")},
                )
                assert uploaded.status_code == 200, f"uploading {file_name}: {uploaded.text}"
        page = page_for(person)
        choose_terminal(page, name)
        return open_files_tab(page)

    return open_browser


def test_a_word_document_is_laid_out_by_the_page_where_the_terminal_cannot_convert_it(
    machine_files,
):
    browser = machine_files({"letter.docx": samples.docx(LETTER, LETTER_NOTE)})

    open_file(browser, "letter.docx")

    expect(browser.get_by_text(LETTER[0], exact=True)).to_be_visible(
        timeout=DOCUMENT_LOAD_TIMEOUT_MS
    )
    expect(browser.get_by_text(LETTER[1], exact=True)).to_be_visible()
    expect(browser.get_by_role("application", name="Page viewer")).to_have_count(0)
    # the commented words are marked, and the pointer resting on them opens the comment
    expect(browser.get_by_role("button", name="Comment 0").first).to_be_visible()
    comment = comment_shown_by(browser.page, browser.get_by_text(LETTER[0], exact=True))
    expect(comment.get_by_text(LETTER_NOTE.author, exact=True)).to_be_visible()
    expect(comment.get_by_text(LETTER_NOTE.text, exact=True)).to_be_visible()


def test_a_presentation_is_drawn_by_the_page_where_the_terminal_cannot_convert_it(machine_files):
    browser = machine_files({"deck.pptx": samples.pptx(DECK, DECK_NOTE)})

    open_file(browser, "deck.pptx")

    slides = browser.get_by_role("application", name="Slide preview")
    expect(slides).to_be_visible(timeout=DOCUMENT_LOAD_TIMEOUT_MS)
    thumbnails = slides.get_by_role("complementary").get_by_role("button")
    expect(thumbnails).to_have_count(2)
    expect(thumbnails.first).to_have_attribute("aria-current", "true")
    comment = comment_shown_by(
        browser.page, slides.get_by_role("button", name=f"Comment: {DECK_NOTE.author}")
    )
    expect(comment.get_by_text(DECK_NOTE.text, exact=True)).to_be_visible()

    browser.page.keyboard.press("Escape")
    slides.get_by_role("button", name="Next slide").click()

    expect(thumbnails.nth(1)).to_have_attribute("aria-current", "true")
    expect(slides.get_by_role("button", name=f"Comment: {DECK_NOTE.author}")).to_have_count(0)


def test_a_file_that_only_has_a_word_documents_name_says_it_failed_to_load(machine_files):
    browser = machine_files(
        {"letter.docx": samples.docx(LETTER), "torn.docx": b"this was never a Word document"}
    )
    open_file(browser, "letter.docx")
    expect(browser.get_by_text(LETTER[0], exact=True)).to_be_visible(
        timeout=DOCUMENT_LOAD_TIMEOUT_MS
    )
    expect(browser.get_by_role("button", name="Document comments")).to_have_count(0)

    browser.get_by_role("button", name="Home", exact=True).click()
    open_file(browser, "torn.docx")

    expect(
        browser.get_by_text("Failed to load DOCX file. Please try downloading it instead.")
    ).to_be_visible(timeout=DOCUMENT_LOAD_TIMEOUT_MS)
    browser.get_by_role("button", name="Document comments").click()
    comment = browser.page.get_by_role("dialog", name="Comment", exact=True)
    expect(
        comment.get_by_text("Could not read comments. Download the original file to review them.")
    ).to_be_visible()
