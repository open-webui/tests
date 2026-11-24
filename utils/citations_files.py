"""Cited files as a reply shows them: the files a chat attaches, and the marks over a cited passage.

A cited document that the retrieved text cannot show in its layout (a Word file) opens in a tab
beside the chat, its cited passage marked in yellow over the text. The marks are drawn over the
page and have no role or name, so `marked` and `unmarked` read them by where they lie.
"""

from __future__ import annotations

import re

from playwright.sync_api import Locator, Page, expect
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

from harness import upstream as reply
from utils.chat_ui import Chat, chat_input, open_input_menu

DOCX_TYPE = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
SOURCE_TAG = re.compile(r'<source id="(\d+)" name="([^"]+)"')

# whether a cited-passage mark covers part of the element's box
MARK_OVER = """(element) => {
    const box = element.getBoundingClientRect();
    const scope = element.closest('[role="tabpanel"]') ?? document;
    return Array.from(scope.querySelectorAll('[data-quote]')).some((mark) => {
        const area = mark.getBoundingClientRect();
        return area.width > 0 && area.left < box.right && area.right > box.left
            && area.top < box.bottom - 1 && area.bottom > box.top + 1;
    });
}"""

IN_VIEW_OF_PANEL = """(element) => {
    const box = element.getBoundingClientRect();
    const panel = element.closest('[role="tabpanel"]').getBoundingClientRect();
    return box.height > 0 && box.top >= panel.top && box.bottom <= panel.bottom;
}"""


def attach(chat: Chat, page: Page, files: list[tuple[str, str, bytes]]) -> None:
    """Attach `(name, type, content)` files through the chat input's Upload Files."""
    expect(chat_input(chat)).to_be_visible()
    menu = open_input_menu(chat)
    with page.expect_file_chooser() as chooser:
        menu.get_by_role("menuitem", name="Upload Files").click()
    chooser.value.set_files(
        [{"name": name, "mimeType": kind, "buffer": content} for name, kind, content in files]
    )


def source_ids_sent(upstream, question: str) -> dict[str, str]:
    """The name of each source the model was told about, by its id."""
    [request] = [
        request for request in upstream.chat_requests() if reply.answering(question)(request)
    ]
    context = "\n".join(str(message.get("content")) for message in request["messages"])
    return dict(SOURCE_TAG.findall(context))


def file_tab(page: Page, name: str) -> Locator:
    return page.get_by_role("tab", name=name, exact=True)


def file_panel(page: Page, name: str) -> Locator:
    return page.get_by_role("tabpanel", name=name, exact=True)


def _wait_until(line: Locator, check: str, failure: str) -> None:
    expect(line).to_be_visible()
    try:
        line.page.wait_for_function(f"(element) => ({check})(element)", arg=line.element_handle())
    except PlaywrightTimeoutError:
        raise AssertionError(f"{failure}: {line.text_content()!r}") from None


def expect_marked(line: Locator) -> None:
    """The line shows in the document with a cited-passage mark over it."""
    _wait_until(line, MARK_OVER, "the cited line is not highlighted")


def is_marked(line: Locator) -> bool:
    return line.evaluate(MARK_OVER)


def expect_in_view(line: Locator) -> None:
    """The line lies inside the visible part of its tab, without anyone scrolling."""
    _wait_until(line, IN_VIEW_OF_PANEL, "the cited line is scrolled out of view")


def hover_mark_over(page: Page, line: Locator) -> None:
    """Point at the mark over `line`, as a person rests the pointer on the highlighted text."""
    # a person scrolls to the text first; a line at the pane's edge can sit past the window
    line.scroll_into_view_if_needed()
    box = line.bounding_box()
    assert box, f"{line} has no box to point at"
    page.mouse.move(box["x"] + 20, box["y"] + box["height"] / 2)


def expect_relevance_on_hover(line: Locator, relevance: str | re.Pattern, failure: str) -> None:
    """Resting the pointer on the cited `line` shows the relevance its passage was given."""
    page = line.page
    hover_mark_over(page, line)
    card = page.get_by_role("tooltip").filter(has_text="Relevance")
    expect(card, failure).to_be_visible()
    expect(card, failure).to_contain_text(relevance)


def close_by_escape(page: Page) -> None:
    """Close the open dialog with Escape, once it has taken the focus and listens for keys."""
    dialog = page.get_by_role("dialog")
    expect(dialog.locator(":focus"), "the dialog never took the focus").to_have_count(1)
    page.keyboard.press("Escape")
    expect(dialog).to_be_hidden()
