"""Opening the full-screen chat input with text in it threw an error in the browser, #31465.

Fix commit `b39abad5c`, issue open-webui/open-webui#31465. The rich text editor compares the
value it is given with its own content before replacing it. That content carries ProseMirror
attribute objects without a prototype, which `fast-deep-equal` cannot compare, so the
full-screen view threw "a.valueOf is not a function" as it opened with the message box's text.
The editor now compares with a JSON-only equality that handles them.

Discriminates: passes on dev efe63bd34; with b39abad5c reverted opening the full-screen view
raises the uncaught TypeError; the text reaching the view passes on both.

Every test runs in the main chat and in a side chat (`utils/chat_surface.py`).
"""

from __future__ import annotations

import pytest
from playwright.sync_api import Page, expect

from utils.chat_surface import ChatSurface
from utils.chat_ui import REPLY_TIMEOUT_MS, chat_input

pytestmark = [pytest.mark.regression, pytest.mark.requires_browser, pytest.mark.requires_source]

LINES = ["first line", "second line", "third line", "fourth line"]


def _type_lines(chat: ChatSurface) -> None:
    page = chat.page
    expect(chat_input(chat)).to_be_visible(timeout=REPLY_TIMEOUT_MS)
    chat_input(chat).click()
    for index, line in enumerate(LINES):
        if index:
            page.keyboard.press("Shift+Enter")
        page.keyboard.type(line)


def _full_screen_view(page: Page):
    # the drawer has no role or label; its editor is the one element with an id of its own
    return page.locator("#input-modal")


def test_opening_the_full_screen_input_with_text_raises_no_error(chat_for, make_user):
    chat = chat_for(make_user())
    page = chat.page
    errors: list[str] = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    _type_lines(chat)

    chat.button("Expand input").click()
    expect(_full_screen_view(page).get_by_text("fourth line")).to_be_visible()
    # uncaught errors surface asynchronously after the view mounts
    page.wait_for_timeout(500)

    assert errors == [], f"opening the full-screen input threw (#31465): {errors}"


# ---------------------------------------------------------------- nearby


def test_the_full_screen_input_shows_every_line_typed(chat_for, make_user):
    chat = chat_for(make_user())
    page = chat.page
    _type_lines(chat)

    chat.button("Expand input").click()

    for line in LINES:
        expect(_full_screen_view(page).get_by_text(line, exact=True)).to_be_visible()
