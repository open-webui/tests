"""Journey: citations of files that are not PDFs, from the reply's marker to the cited passage.

A person attaches files or a knowledge base, the model answers with `[n]` markers, and each
marker shows as a chip named after the source it points to, at the place it was written. Clicking
it, or pressing Enter on it, opens that source with the cited passage on screen and marked. Every
cited file opens in a tab beside the chat, so the answer stays readable: a text file (Markdown,
CSV, HTML) shows its whole text rendered, and a Word document opens scrolled to the cited
passage, with only that passage highlighted. Since e0eb2297d a marker highlights only the lines of
the passage that the answer text before it draws on, and the whole passage when it draws on none
of them; resting the pointer on the highlight shows
the retrieved passage, in a card at the cited text. The highlights stay on the cited text when the
window is widened or narrowed. The same holds after a reload and on a phone. A picture sent to the
model is no source, and a marker pointing past the sources shows nothing.

Dev cc2e033b0 opens a cited Word, PowerPoint or Excel file beside the chat and marks its cited
passages (9b6241e54 started the rework in the citation dialog). Dev 072d0b51d opens text files
there too, where the source dialog showed them before; the dialog gave the original file by its
name, and the tab offers no download (the download test, open-webui/open-webui#34832).

Red on dev c1f3b6fd1, each with its issue:
- the source list under a reply opens a cited Word document in the plain dialog, not beside the
  chat as its marker does (open-webui/open-webui#34831);
- an opened cited document offers no way to download the original file
  (open-webui/open-webui#34832);
- a passage cited from a long text file shows alone, never where it sits in the file
  (open-webui/open-webui#34833);
- once the window is widened, the Word document's page moves to the middle of the wider tab but
  its highlights stay where the page was, off the page to the left of the cited text, and resting
  the pointer on the cited text shows no card (fixed on dev by c3747e844).

Discriminates: the text file and knowledge base tests pass on dev c1f3b6fd1. In copies of that
build: a Word file left out of the marker's open-beside-the-chat check turned the knowledge base
test red at the tab; no cited line kept for marking turned it red at the first highlight, with the
text file test green there as the control; a marker opening the wrong source turned the text file
test red at its first dialog. On the dev d331f062e build, before a marker marked only the lines
its sentence draws on, the Word document test fails at its second marker and the long text file
test at its second cited line.
"""

from __future__ import annotations

import base64
import io
import re
import time
import uuid
from typing import Callable
from urllib.parse import urlsplit

import pytest
from playwright.sync_api import Locator, Page, expect

from harness import sample_documents
from harness import upstream as reply
from harness.knowledge_bases import knowledge_base
from harness.web_retrieval import RETRIEVAL_CONFIG
from utils.chat_ui import chat_input, conversation, expect_reply, last_reply, listed_source, send
from utils.citation_marks import expect_card_at, expect_marks_on_cited_text
from utils.citations_files import (
    DOCX_TYPE,
    attach,
    expect_in_view,
    expect_marked,
    file_panel,
    file_tab,
    hover_mark_over,
    is_marked,
    source_ids_sent,
)
from utils.phone import PHONE

pytestmark = [
    pytest.mark.journey,
    pytest.mark.requires_browser,
    pytest.mark.requires_source,
    # retrieval puts the file text into the prompt only with File Context ticked
    pytest.mark.usefixtures("file_context_on"),
]

MARKDOWN = ("notes.md", "text/markdown", b"# Herons\n\nGrey **herons** wait in the reeds.\n")
CSV = ("table.csv", "text/csv", b"animal,habitat\nheron,reeds\notter,river bank\n")
HTML = (
    "page.html",
    "text/html",
    b"<html><body><h1>Otters</h1><p>Otters slide down the muddy bank.</p></body></html>",
)
PICTURE = ("picture.png", "image/png", sample_documents.png())
LEDGER = [
    f"Section {number}: ledger entry {number} records {number * 7} crates of apples."
    for number in range(60)
]
SECTION = re.compile(r"Section (\d+):")


@pytest.fixture
def one_passage_retrieval(admin, preserve):
    """Files searched by their chunks and only the best chunk cited, so a reply cites one part."""
    preserve(RETRIEVAL_CONFIG)
    with admin.client() as client:
        saved = client.post(RETRIEVAL_CONFIG[1], json={"RAG_FULL_TEXT_THRESHOLD": 0, "TOP_K": 1})
    assert saved.status_code == 200, f"setting retrieval to one chunk failed: {saved.text}"


def _ask(page: Page, upstream, answer: str | Callable[[dict], str], last_words: str) -> str:
    question = f"what do my files say? {uuid.uuid4().hex[:6]}"
    upstream.queue(reply.text(answer, match=reply.answering(question)))
    send(page, question)
    expect_reply(page, last_words)
    return question


def _cited_passage(account, page: Page, name: str) -> str:
    """The passage of `name` the stored reply cites, as retrieval picked it."""
    chat_id = page.url.rsplit("/", 1)[-1]
    answer: dict = {}
    deadline = time.monotonic() + 15
    with account.client() as client:
        # the page shows the reply before the server has stored it as done
        while not answer.get("done") and time.monotonic() < deadline:
            stored = client.get(f"/api/v1/chats/{chat_id}")
            assert stored.status_code == 200, stored.text
            messages = stored.json()["chat"]["history"]["messages"].values()
            answer = next((message for message in messages if message["role"] == "assistant"), {})
            time.sleep(0.1)
    assert answer.get("done"), f"the reply was never stored as done: {answer}"
    passages = [
        text
        for source in answer.get("sources", [])
        for text, metadata in zip(source["document"], source["metadata"])
        if metadata.get("name") == name
    ]
    assert len(passages) == 1, f"expected one cited passage of {name}, the reply has {passages}"
    return passages[0]


def _upload(client, name: str, kind: str, content: bytes) -> str:
    uploaded = client.post(
        "/api/v1/files/",
        params={"process": "true", "process_in_background": "false"},
        files={"file": (name, content, kind)},
    )
    assert uploaded.status_code == 200, f"uploading {name} failed: {uploaded.text}"
    return uploaded.json()["id"]


def _expect_file_tab(page: Page, name: str, shows: list[str]) -> Locator:
    """`name` is open in a tab beside the chat with each text of `shows`; returns what it shows."""
    tab = file_tab(page, name)
    expect(tab, f"{name} did not open in a tab").to_have_attribute("aria-selected", "true")
    panel = page.locator(f"[id='{tab.get_attribute('aria-controls')}']")
    # a web page is shown in a frame of its own
    shown = panel.frame_locator("iframe").locator("body") if name.endswith(".html") else panel
    for text in shows:
        expect(shown, f"the tab of {name} lacks its text").to_contain_text(text)
    expect(chat_input(page), "the opened file hides the chat").to_be_visible()
    expect(page.get_by_role("dialog")).to_have_count(0)
    return shown


def test_cited_text_files_open_their_cited_text(page_for, make_user, upstream):
    account = make_user()
    page = page_for(account)
    attach(page, page, [MARKDOWN, CSV, HTML, PICTURE])
    expect(page.get_by_role("button", name=re.compile("picture.png"))).to_be_visible()
    question = _ask(
        page,
        upstream,
        "Herons wait [1]. Otters live by the river [2]. Otters slide [3]. Both [2][3]. "
        "The picture [4]. Herons again [1]. End of files.",
        "End of files.",
    )
    sources = source_ids_sent(upstream, question)
    assert sources == {"1": "notes.md", "2": "table.csv", "3": "page.html"}, (
        f"the model was told of other sources than the three text files: {sources}"
    )

    shown = last_reply(page)
    # each marker shows as its source's name where it was written; [4] points past the sources
    expect(shown).to_have_text(
        re.compile(
            r"Herons wait\s*notes\.md\s*\. Otters live by the river\s*table\.csv\s*\. "
            r"Otters slide\s*page\.html\s*\. Both\s*table\.csv\s*\+1\s*\. The picture\s*\. "
            r"Herons again\s*notes\.md\s*\. End of files\."
        )
    )
    expect(shown).not_to_contain_text("[")

    shows = {
        "notes.md": ["Grey herons wait in the reeds."],
        "table.csv": ["heron", "reeds", "otter", "river bank"],
        "page.html": ["Otters slide down the muddy bank."],
    }
    for name, texts in shows.items():
        shown.get_by_role("button", name=f"View source: {name}", exact=True).first.click()
        opened = _expect_file_tab(page, name, texts)
        if name == "notes.md":
            expect(opened.get_by_role("heading", name="Herons")).to_be_visible()
            expect(opened.locator("strong", has_text="herons")).to_be_visible()
        if name == "page.html":
            expect(opened.get_by_role("heading", name="Otters")).to_be_visible()
            expect(opened).not_to_contain_text("<h1>")
        page.get_by_role("button", name=f"Close {name}", exact=True).click()
        expect(file_tab(page, name)).to_have_count(0)

    # the second marker of the same file opens that file too
    shown.get_by_role("button", name="View source: notes.md", exact=True).last.click()
    _expect_file_tab(page, "notes.md", shows["notes.md"])

    # the marker of two sources shows each in its card, and each opens its own text
    group = shown.get_by_role("button", name="table.csv +1 more sources").last
    group.hover()
    card = page.locator("[data-link-preview-content]")
    expect(card.get_by_role("button", name="View source: table.csv")).to_be_visible()
    card.get_by_role("button", name="Next").click()
    card.get_by_role("button", name="View source: page.html").click()
    _expect_file_tab(page, "page.html", shows["page.html"])
    expect(card, "the card stays open over the opened file").to_have_count(0)

    # by keyboard: Enter on a marker opens it
    marker = shown.get_by_role("button", name="View source: table.csv", exact=True)
    marker.focus()
    page.keyboard.press("Enter")
    _expect_file_tab(page, "table.csv", shows["table.csv"])

    # the source list numbers the three files and opens the same text
    conversation(page).get_by_role("button", name="Toggle 3 sources").click()
    for number, name in sources.items():
        entry = listed_source(page, name)
        expect(entry).to_have_accessible_name(f"View source: {number} {name}")
        entry.click()
        _expect_file_tab(page, name, shows[name])

    page.reload()
    expect_reply(page, "End of files.")
    last_reply(page).get_by_role("button", name="View source: page.html", exact=True).click()
    _expect_file_tab(page, "page.html", shows["page.html"])


def test_a_cited_word_document_opens_beside_the_chat_at_the_cited_passage(
    page_for, make_user, upstream, one_passage_retrieval
):
    account = make_user(role="admin")
    with account.client() as client, knowledge_base(client, name="Ledger") as knowledge_id:
        file_id = _upload(client, "ledger.docx", DOCX_TYPE, sample_documents.docx(LEDGER))
        added = client.post(f"/api/v1/knowledge/{knowledge_id}/file/add", json={"file_id": file_id})
        assert added.status_code == 200, f"adding the file failed: {added.text}"

        page = page_for(account)
        chat_input(page).click()
        page.keyboard.type("#Ledger")
        page.get_by_role("tooltip").get_by_role("button", name="Ledger", exact=True).click()
        question = _ask(page, upstream, _ledger_answer, "End.")
        passage = _cited_passage(account, page, "ledger.docx")
        sent = "\n".join(
            str(message.get("content"))
            for request in upstream.chat_requests()
            if reply.answering(question)(request)
            for message in request["messages"]
        )
        assert passage.strip() in sent, "the model was not sent the passage the reply cites"
        cited = sorted(int(number) for number in SECTION.findall(passage))
        assert cited, f"the cited passage names no section: {passage!r}"
        uncited = next(number for number in range(60) if number not in cited)
        claimed = [cited[0], cited[-1]]

        shown = last_reply(page)
        expect(shown).to_have_text(
            re.compile(
                rf"Shipments went north\s*ledger\.docx\s*\. Entry {cited[0]} records .* crates"
                rf"\s*ledger\.docx"
            )
        )
        markers = shown.get_by_role("button", name="View source: ledger.docx", exact=True)
        expect(markers).to_have_count(2)

        def section(number: int):
            return file_panel(page, "ledger.docx").get_by_text(LEDGER[number], exact=True)

        def expect_opened_at_the_passage(marked: list[int]) -> None:
            expect(file_tab(page, "ledger.docx")).to_have_attribute("aria-selected", "true")
            # the answer stays readable beside the document
            expect(chat_input(page)).to_be_visible()
            expect(page.get_by_role("dialog")).to_have_count(0)
            expect_in_view(section(cited[0]))
            for number in marked:
                expect_marked(section(number))
            for number in [uncited, *(number for number in cited if number not in marked)]:
                expect(section(number)).to_be_attached()
                assert not is_marked(section(number)), f"section {number} is marked, uncited"

        # the first marker's sentence draws on no line of its own, so it cites the whole passage
        markers.first.click()
        expect_opened_at_the_passage(cited)

        # resting on the highlight shows the passage retrieval cited
        hover_mark_over(page, section(cited[-1]))
        card = page.get_by_role("tooltip").filter(has_text=LEDGER[cited[0]])
        expect(card).to_be_visible()
        expect(card).to_contain_text(LEDGER[cited[-1]])
        expect_card_at(card, section(cited[-1]))
        page.mouse.move(0, 0)

        # the highlights stay on the cited text as the window is widened and narrowed
        panel = file_panel(page, "ledger.docx")
        cited_lines = [section(number) for number in cited]
        expect_marks_on_cited_text(panel, *cited_lines)
        page.set_viewport_size({"width": 2600, "height": 1080})
        expect_marks_on_cited_text(panel, *cited_lines)
        hover_mark_over(page, section(cited[-1]))
        expect_card_at(card, section(cited[-1]))
        page.mouse.move(0, 0)
        page.set_viewport_size({"width": 1280, "height": 1080})
        expect_marks_on_cited_text(panel, *cited_lines)
        page.set_viewport_size({"width": 1920, "height": 1080})

        page.get_by_role("button", name="Close ledger.docx").click()
        expect(file_tab(page, "ledger.docx")).to_have_count(0)
        expect(markers.last).to_be_visible()

        # the second marker, by keyboard, marks only the two entries its sentence names
        markers.last.focus()
        page.keyboard.press("Enter")
        expect_opened_at_the_passage(claimed)
        page.get_by_role("button", name="Close ledger.docx").click()

        page.reload()
        expect_reply(page, "End.")
        last_reply(page).get_by_role("button", name="View source: ledger.docx").first.click()
        expect_opened_at_the_passage(cited)

        phone = page_for(account, path=urlsplit(page.url).path, **PHONE)
        expect_reply(phone, "End.")
        last_reply(phone).get_by_role("button", name="View source: ledger.docx").first.tap()
        phone_section = file_panel(phone, "ledger.docx").get_by_text(LEDGER[cited[0]], exact=True)
        expect_marked(phone_section)
        expect_in_view(phone_section)
        phone.get_by_role("button", name="Close ledger.docx").tap()
        expect(last_reply(phone)).to_be_visible()

        # once the file is gone the tab shows what was retrieved from it (since 072d0b51d; it
        # said "Preview failed" before), it does not load forever
        deleted = client.delete(f"/api/v1/files/{file_id}")
        assert deleted.status_code == 200, deleted.text
        page.reload()
        expect_reply(page, "End.")
        last_reply(page).get_by_role("button", name="View source: ledger.docx").first.click()
        expect(file_panel(page, "ledger.docx")).to_contain_text(LEDGER[cited[0]])


def _ledger_answer(request: dict) -> str:
    """A reply on the ledger passage the model was sent, its second claim naming two entries."""
    sent = "\n".join(str(message.get("content")) for message in request["messages"])
    entries = [number for number, line in enumerate(LEDGER) if line in sent]
    first, last = (entries[0], entries[-1]) if entries else (0, 0)
    return (
        f"Shipments went north [1]. Entry {first} records {first * 7} crates, and entry {last} "
        f"records {last * 7} crates [1]. End."
    )


REPORT = ["The heron colony nests near the old mill.", "Otters were seen by the weir in May."]


def _attached_report(page_for, make_user, upstream) -> Page:
    page = page_for(make_user())
    attach(page, page, [("report.docx", DOCX_TYPE, sample_documents.docx(REPORT))])
    expect(page.get_by_role("button", name=re.compile("report.docx"))).to_be_visible()
    _ask(page, upstream, "Herons nest by the mill [1]. End.", "End.")
    return page


def test_the_source_list_opens_a_cited_word_document_as_its_marker_does(
    page_for, make_user, upstream
):
    """Red on dev c1f3b6fd1: open-webui/open-webui#34831; red on dev 4dc058bbb: #35165."""
    page = _attached_report(page_for, make_user, upstream)
    last_reply(page).get_by_role("button", name="View source: report.docx").click()
    line = file_panel(page, "report.docx").get_by_text(REPORT[0], exact=True)
    expect_marked(line)
    page.get_by_role("button", name="Close report.docx").click()
    expect(file_tab(page, "report.docx")).to_have_count(0)

    conversation(page).get_by_role("button", name="Toggle 1 source").click()
    listed_source(page, "report.docx").click()
    expect(file_tab(page, "report.docx")).to_have_attribute("aria-selected", "true")
    expect_marked(file_panel(page, "report.docx").get_by_text(REPORT[0], exact=True))


FETCH_BYTES = """async (url) => {
    const response = await fetch(url);
    const bytes = new Uint8Array(await response.arrayBuffer());
    let text = '';
    for (const byte of bytes) text += String.fromCharCode(byte);
    return [response.status, btoa(text)];
}"""


def test_an_opened_cited_document_downloads_the_original_file(page_for, make_user, upstream):
    """Red on dev c1f3b6fd1: open-webui/open-webui#34832, fixed by e1855444a with an Open file
    link on the document view that leads to the original file."""
    page = _attached_report(page_for, make_user, upstream)
    last_reply(page).get_by_role("button", name="View source: report.docx").click()
    panel = file_panel(page, "report.docx")
    expect(panel.get_by_text(REPORT[0], exact=True)).to_be_visible()
    open_file = panel.get_by_role("link", name="Open file")
    expect(open_file, "the opened document offers no way to the original file").to_be_visible()
    href = open_file.get_attribute("href")
    assert href, "the Open file link leads nowhere"
    # what the browser gets when the link is followed, with the account's own session
    status, body = page.evaluate(FETCH_BYTES, href)
    assert status == 200, f"opening the original file got {status}"
    from docx import Document

    served = [
        paragraph.text for paragraph in Document(io.BytesIO(base64.b64decode(body))).paragraphs
    ]
    assert served == REPORT, f"the link serves another document: {served}"


TRAIL = [
    f"Trail note {number}: the path past marker {number} climbs {number * 3} metres."
    for number in range(60)
]


def _trail_answer(request: dict) -> str:
    """A reply on the first trail note of the passage the model was sent."""
    sent = "\n".join(str(message.get("content")) for message in request["messages"])
    number = next((number for number, line in enumerate(TRAIL) if line in sent), 0)
    return f"Past marker {number} the path climbs {number * 3} metres [1]. End."


def test_a_passage_cited_from_a_long_text_file_shows_where_it_sits_in_the_file(
    page_for, make_user, upstream, one_passage_retrieval
):
    """Red on dev c1f3b6fd1: open-webui/open-webui#34833."""
    account = make_user()
    page = page_for(account)
    attach(page, page, [("trail.md", "text/markdown", "\n\n".join(TRAIL).encode())])
    expect(page.get_by_role("button", name=re.compile("trail.md"))).to_be_visible()
    _ask(page, upstream, _trail_answer, "End.")
    passage = _cited_passage(account, page, "trail.md")
    cited = [line for line in TRAIL if line in passage]
    assert cited and len(cited) < len(TRAIL), f"the reply cites no part of the file: {passage!r}"
    around = (
        TRAIL[TRAIL.index(cited[0]) - 1]
        if TRAIL.index(cited[0])
        else TRAIL[TRAIL.index(cited[-1]) + 1]
    )

    last_reply(page).get_by_role("button", name="View source: trail.md").click()
    # the file shows around the passage, and only the cited line the reply draws on is marked
    around_line = page.get_by_text(around).first
    expect(around_line, "the text around the cited passage is not shown").to_be_attached()
    expect_marked(page.get_by_text(cited[0]).first)
    assert not is_marked(around_line), f"the uncited line {around!r} is marked"
    for other in cited[1:2]:
        assert not is_marked(page.get_by_text(other).first), f"{other!r} is marked, never drawn on"
