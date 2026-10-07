"""Regression: a pasted HTML email was sent with its text repeated many times.

Issue open-webui/open-webui#24657, fix c5dac0070 (PR open-webui/open-webui#31890). With Rich Text
Input for Chat on, pasting an email or newsletter (tables nested inside other tables) looked fine
in the input, but the sent message repeated the same text as markdown table rows: converting a
table read every row below it, nested ones included, so each inner table's content was counted
again for every table around it. Only the table's own rows and cells are read now, and a plain
table still becomes one markdown table.

Discriminates: passes on the dev b859124f9 build, fails on that build with c5dac0070 reverted
(the model receives the email's text several times).
"""

from __future__ import annotations

import pytest
from playwright.sync_api import Page, expect

from harness import upstream as reply
from utils.chat_ui import chat_input, expect_reply

pytestmark = [pytest.mark.regression, pytest.mark.requires_browser, pytest.mark.requires_source]

NESTED_EMAIL = """
<table><tbody><tr><td>
  <table><tbody><tr><td>
    <table><tbody>
      <tr><td>Quarterly newsletter</td></tr>
      <tr><td>Harbour update for March</td></tr>
    </tbody></table>
  </td></tr></tbody></table>
</td></tr></tbody></table>
"""

PLAIN_TABLE = """
<table><thead><tr><th>Fruit</th><th>Colour</th></tr></thead>
<tbody><tr><td>Plum</td><td>Purple</td></tr><tr><td>Lime</td><td>Green</td></tr></tbody></table>
"""

PASTE = """
(html) => {
    const data = new DataTransfer();
    data.setData('text/html', html);
    const target = document.querySelector('#chat-input');
    target.dispatchEvent(
        new ClipboardEvent('paste', { clipboardData: data, bubbles: true, cancelable: true })
    );
}
"""


def paste_html(page: Page, html: str) -> None:
    expect(chat_input(page)).to_be_visible()
    chat_input(page).click()
    page.evaluate(PASTE, html)


def sent_text(upstream, words: str) -> str:
    [request] = [body for body in upstream.chat_requests() if reply.answering(words)(body)]
    [message] = [entry for entry in request["messages"] if entry["role"] == "user"]
    return message["content"]


def paste_and_send(page: Page, upstream, html: str, words: str) -> str:
    upstream.queue(reply.text("Noted.", match=reply.answering(words)))
    paste_html(page, html)
    page.keyboard.press("Enter")
    expect_reply(page, "Noted.")
    return sent_text(upstream, words)


def test_a_pasted_email_with_nested_tables_is_sent_once(page_for, make_user, upstream):
    page = page_for(make_user())
    page.goto("/")

    sent = paste_and_send(page, upstream, NESTED_EMAIL, "Quarterly newsletter")

    assert sent.count("Quarterly newsletter") == 1, sent
    assert sent.count("Harbour update for March") == 1, sent


def test_a_pasted_plain_table_still_becomes_one_markdown_table(page_for, make_user, upstream):
    page = page_for(make_user())
    page.goto("/")

    sent = paste_and_send(page, upstream, PLAIN_TABLE, "Plum")

    rows = [line.strip() for line in sent.splitlines() if line.strip()]
    assert rows == [
        "| Fruit | Colour |",
        "| --- | --- |",
        "| Plum | Purple |",
        "| Lime | Green |",
    ], sent
