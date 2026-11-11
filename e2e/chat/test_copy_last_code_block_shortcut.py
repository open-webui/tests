"""Regression: the Copy Last Code Block shortcut copied the artifacts pane instead.

Fix PR open-webui/open-webui#31613 (issue open-webui/open-webui#31476) in the frontend. The
shortcut clicks the last Copy button carrying the code block marker on the page, and the
artifacts pane's own Copy button carried that marker too. The pane sits after the messages, so
whenever a reply held an HTML block (which opens the pane by itself) the shortcut put the pane's
whole HTML document on the clipboard and not the last code block of the chat.

A reply holds an HTML block and then a Python block, the pane is opened with Preview (it often
opens by itself, but not on every run) and the shortcut is pressed with the clipboard readable.

Discriminates: passes on the dev a5bc78300 build; with the marker put back on the artifacts pane's
Copy button (the mutation build) the clipboard holds the HTML document and the test goes red.

The test runs in the main chat and in a side chat. The side variant is red on dev c15f28649 (the
code block has no Preview button, open-webui/open-webui#34026) and still red on the side chat
rework build (open-webui/open-webui#34064): there Preview is offered but opens no Artifacts pane
for a side chat's answer (open-webui/open-webui#34026, fix PR open-webui/open-webui#34166).
"""

from __future__ import annotations

import pytest
from playwright.sync_api import expect

from harness import upstream as reply
from utils.chat_ui import expect_reply, expect_split_screen, last_reply, send
from utils.clipboard import wait_for_text

pytestmark = [pytest.mark.regression, pytest.mark.requires_browser, pytest.mark.requires_source]

LAST_BLOCK = "total = 1 + 41"
ANSWER = (
    "Here is a page and a script.\n\n```html\n<h1>artifact card</h1>\n```\n\n"
    f"```python\n{LAST_BLOCK}\n```\n\nThat is all."
)


def test_the_shortcut_copies_the_last_code_block_with_the_artifacts_pane_open(
    chat_for, make_user, upstream
):
    """In a side chat: open-webui/open-webui#34026, fix PR open-webui/open-webui#34166."""
    chat = chat_for(make_user())
    chat.page.context.grant_permissions(["clipboard-read", "clipboard-write"])
    upstream.queue(reply.text(ANSWER, match=reply.answering("page and script")))
    send(chat, "give me a page and script")
    expect_reply(chat, "That is all.")
    box = last_reply(chat)
    box.get_by_role("button", name="Preview", exact=True).click()
    pane = chat.page.locator("#artifacts-container")
    expect(pane.frame_locator("iframe").locator("h1")).to_have_text("artifact card")
    expect_split_screen(chat)

    chat.page.keyboard.press("Control+Shift+;")

    copied = wait_for_text(chat.page)
    assert copied == LAST_BLOCK, (
        f"the shortcut copied something else than the last block: {copied!r}"
    )
