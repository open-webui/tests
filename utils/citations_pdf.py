"""A PDF a reply cites, and the viewer it opens in beside the chat.

`upload_pdf` stores a PDF with one line of text per page for an account, unprocessed, and
`pdf_source` is a reply's source on it as retrieval stores one: each cited passage with the page it
was read from and its relevance. `cited_pdf_tab` and `cited_pdf_panel` are the tab a cited PDF
opens in and its panel, `pdf_viewer` the scrolling page viewer in it and `page_strip` a page in the
previews beside it. `expect_marked` checks which lines the viewer highlights, `expect_page_shown`
which page it is scrolled to, and `passage_card` rests the pointer on a passage and returns the
card that names it. `fetch_as_account` reads a link as the page's account.
"""

from __future__ import annotations

import re
import uuid

import httpx
from playwright.sync_api import Locator, Page, expect
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

from harness import sample_documents as samples
from harness import slow_machine

# the page renders a PDF and its text on first use of the viewer
PDF_LOAD_TIMEOUT_MS = slow_machine.scaled(8_000)

# the lines under a highlight, in page order; marks carry no text of their own
MARKED_TEXT = """(viewer) => {
    const marks = [...viewer.querySelectorAll('[data-quote]')]
        .map((mark) => mark.getBoundingClientRect());
    const marked = [];
    for (const span of viewer.querySelectorAll('.textLayer span')) {
        const box = span.getBoundingClientRect();
        if (!span.textContent.trim() || box.width === 0) continue;
        const x = box.left + box.width / 2, y = box.top + box.height / 2;
        if (marks.some((m) => x >= m.left && x <= m.right && y >= m.top && y <= m.bottom))
            marked.push(span.textContent.trim());
    }
    return marked.join(' ');
}"""


def upload_pdf(client: httpx.Client, name: str, pages: list[str]) -> str:
    """Store a PDF with one line per page as `name`; returns the file id."""
    uploaded = client.post(
        "/api/v1/files/",
        params={"process": "false"},
        files={"file": (name, samples.pdf(pages), "application/pdf")},
    )
    assert uploaded.status_code == 200, f"uploading {name} failed: {uploaded.text}"
    return uploaded.json()["id"]


def pdf_source(file_id: str, name: str, passages: list[tuple[str, int, float]]) -> dict:
    """A reply's source on the PDF: each passage as (text, page number from 1, relevance)."""
    return {
        "source": {"type": "file", "id": file_id, "name": name},
        "document": [text for text, _, _ in passages],
        "metadata": [
            {
                "file_id": file_id,
                "name": name,
                "source": name,
                "page": page - 1,
                "page_label": str(page),
                "content_type": "application/pdf",
            }
            for _, page, _ in passages
        ],
        "distances": [relevance for _, _, relevance in passages],
    }


# what the signed-in account gets from a link, its status and bytes
FETCH_AS_ACCOUNT = """async (url) => {
    const answer = await fetch(url, { headers: { Authorization: `Bearer ${localStorage.token}` } });
    return [answer.status, Array.from(new Uint8Array(await answer.arrayBuffer()))];
}"""


def fetch_as_account(page: Page, url: str) -> tuple[int, bytes]:
    """Fetch `url` in the page as its signed-in account; returns the status and the body."""
    status, body = page.evaluate(FETCH_AS_ACCOUNT, url)
    return status, bytes(body)


def unique_name(stem: str) -> str:
    return f"{stem}-{uuid.uuid4().hex[:6]}.pdf"


def cited_pdf_tab(page: Page, name: str) -> Locator:
    return page.get_by_role("tab", name=re.compile(rf"{re.escape(name)}$"))


def cited_pdf_panel(page: Page, name: str) -> Locator:
    return page.get_by_role("tabpanel", name=re.compile(rf"{re.escape(name)}$"))


def page_strip(panel: Locator, number: int) -> Locator:
    """Page `number` in the strip of page previews beside the viewer."""
    return panel.get_by_role("complementary", name="Pages").get_by_role(
        "button", name=f"Page {number}", exact=True
    )


def pdf_viewer(scope: Locator) -> Locator:
    return scope.get_by_role("application", name="Page viewer")


def marked_text(viewer: Locator) -> str:
    return viewer.evaluate(MARKED_TEXT)


def expect_marked(viewer: Locator, *texts: str) -> None:
    """Exactly the lines `texts`, in page order, are highlighted once the viewer has drawn them."""
    for text in texts:
        expect(viewer.get_by_text(text, exact=True)).to_be_attached(timeout=PDF_LOAD_TIMEOUT_MS)
    expected = " ".join(texts)
    try:
        viewer.page.wait_for_function(
            f"([viewer, text]) => ({MARKED_TEXT})(viewer) === text",
            arg=[viewer.element_handle(), expected],
            timeout=PDF_LOAD_TIMEOUT_MS,
        )
    except PlaywrightTimeoutError:
        raise AssertionError(
            f"the viewer highlights {marked_text(viewer)!r}, not the cited {expected!r}"
        ) from None


def expect_page_shown(viewer: Locator, shown: str, hidden: list[str]) -> None:
    """The viewer shows the page with the line `shown`, scrolled away from the `hidden` lines."""
    line = viewer.get_by_text(shown, exact=True)
    expect(line).to_be_visible(timeout=PDF_LOAD_TIMEOUT_MS)
    expect(line).to_be_in_viewport()
    for other in hidden:
        expect(viewer.get_by_text(other, exact=True)).not_to_be_in_viewport()


def passage_card(viewer: Locator, text: str) -> Locator:
    """Rest the pointer on the passage `text`; returns the card that shows where it is from."""
    viewer.get_by_text(text, exact=True).hover()
    card = viewer.page.get_by_role("tooltip").filter(has_text=text)
    expect(card).to_be_visible()
    return card
